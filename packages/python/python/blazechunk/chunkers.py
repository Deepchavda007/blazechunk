"""High-level chunkers with matching synchronous and asynchronous APIs.

Every chunker in this module exposes the *same four* methods, so once you know
one chunker you know them all:

===========================  =============================================
Method                       What it does
===========================  =============================================
``chunk(text)``              Chunk a single string (blocking).
``chunk_async(text)``        Same, but ``await``-able — the work runs in a
                             worker thread so it never blocks the event loop.
``chunk_batch(texts)``       Chunk many strings, sequentially.
``chunk_batch_async(texts)`` Chunk many strings concurrently on worker
                             threads, with optional back-pressure.
===========================  =============================================

Calling the chunker directly (``chunker(text)``) is shorthand for ``chunk``.

All chunkers return :class:`~blazechunk.Chunk` objects (or a ``list`` of them),
each carrying the chunk ``text`` plus the byte offsets (``start_index`` /
``end_index``) it occupies in the original input and a ``token_count``.

The heavy lifting happens in the compiled Rust extension (``blazechunk._chunk``);
these classes are thin, well-typed, well-documented wrappers around it that add
the async and batch conveniences on top.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any, Literal

from blazechunk._chunk import Chunk
from blazechunk._chunk import CodeChunker as _CodeChunker
from blazechunk._chunk import LateChunk
from blazechunk._chunk import LateChunker as _LateChunker
from blazechunk._chunk import RecursiveChunker as _RecursiveChunker
from blazechunk._chunk import SDPMChunker as _SDPMChunker
from blazechunk._chunk import SemanticChunker as _SemanticChunker
from blazechunk._chunk import SentenceChunker as _SentenceChunker
from blazechunk._chunk import TableChunker as _TableChunker
from blazechunk._chunk import TokenChunker as _TokenChunker

__all__ = [
    "BaseChunker",
    "CodeChunker",
    "LateChunk",
    "LateChunker",
    "RecursiveChunker",
    "SDPMChunker",
    "SemanticChunker",
    "SentenceChunker",
    "TableChunker",
    "TokenChunker",
]

#: An embedding model for the semantic chunkers. Accepted forms:
#:
#: * a **callable** ``f(list[str]) -> 2D`` returning one vector per input text
#:   (a NumPy 2D array, or a list of float lists);
#: * an **object** exposing ``embed_batch(list[str]) -> 2D`` (the Chonkie
#:   ``BaseEmbeddings`` convention); or
#: * an **object** exposing ``encode(list[str]) -> 2D`` (the sentence-transformers
#:   convention).
#:
#: The pure-Rust core has no model of its own — you inject one here and the Rust
#: orchestration calls back into it for embeddings.
EmbeddingModel = Any


def _as_embed_batch(embedding_model: Any) -> Any:
    """Adapt an :data:`EmbeddingModel` into a ``callable(list[str]) -> 2D``."""
    if embedding_model is None:
        raise ValueError("embedding_model is required for semantic chunking")
    embed_batch = getattr(embedding_model, "embed_batch", None)
    if callable(embed_batch):
        return lambda texts: embed_batch(list(texts))
    encode = getattr(embedding_model, "encode", None)
    if callable(encode):
        return lambda texts: encode(list(texts))
    if callable(embedding_model):
        return lambda texts: embedding_model(list(texts))
    raise TypeError(
        "embedding_model must be callable(list[str]) -> 2D, or expose "
        "embed_batch(list[str]) -> 2D or encode(list[str]) -> 2D"
    )


def _as_token_embed_fns(embedding_model: Any) -> tuple[Any, Any]:
    """Adapt a late-chunking model into ``(embed_as_tokens, embed)`` callables.

    Accepts an object exposing both ``embed_as_tokens(text) -> 2D`` and
    ``embed(text) -> 1D``, or a 2-tuple of those callables.
    """
    if isinstance(embedding_model, tuple) and len(embedding_model) == 2:
        embed_tokens, embed = embedding_model
        if callable(embed_tokens) and callable(embed):
            return embed_tokens, embed
    embed_tokens = getattr(embedding_model, "embed_as_tokens", None)
    embed = getattr(embedding_model, "embed", None)
    if callable(embed_tokens) and callable(embed):
        return (lambda text: embed_tokens(text), lambda text: embed(text))
    raise TypeError(
        "LateChunker embedding_model must expose embed_as_tokens(text) -> 2D and "
        "embed(text) -> 1D, or be a (embed_as_tokens, embed) tuple of callables"
    )

#: Where a delimiter/pattern is attached when text is split on it.
#:
#: * ``"prev"`` — keep the delimiter at the end of the preceding chunk (default).
#: * ``"next"`` — move the delimiter to the start of the following chunk.
#: * ``"none"`` — drop the delimiter from both.
IncludeDelim = Literal["prev", "next", "none"]

#: Name of a built-in token counter, or a filesystem path to a HuggingFace
#: ``tokenizer.json`` (the latter only when the extension is built with the
#: ``hf-tokenizer`` feature). Built-in counters:
#:
#: * ``"character"`` — Unicode code points (the default for most chunkers).
#: * ``"word"``      — whitespace-separated words.
#: * ``"byte"``      — raw UTF-8 bytes.
#: * ``"row"``       — newline-separated rows (the default for tables).
Tokenizer = str


class BaseChunker:
    """Common behaviour shared by every chunker: sync, async, and batch.

    You never instantiate ``BaseChunker`` directly — use one of the concrete
    chunkers (:class:`RecursiveChunker`, :class:`SentenceChunker`,
    :class:`TokenChunker`, :class:`TableChunker`, :class:`CodeChunker`). Each of
    those decides *how* text is split; this base class decides *how you call it*.

    Subclasses only need to build their compiled inner chunker in ``__init__``
    and hand it to ``super().__init__(inner)``. Everything else — the async
    offloading, the batching, and the concurrency control — lives here, in one
    place, so the behaviour is identical across all chunkers.

    Concurrency note:
        The async methods offload the (CPU-bound) chunking call to a worker
        thread via :func:`asyncio.to_thread`, which keeps the calling coroutine
        from blocking the event loop. This is exactly what you want inside an
        async web handler or pipeline. Whether the threads also run on multiple
        cores depends on the extension releasing the GIL during chunking; the
        API is correct and future-proof either way.
    """

    __slots__ = ("_inner",)

    def __init__(self, inner: Any) -> None:
        """Store the compiled inner chunker. Called by subclasses, not by users.

        Args:
            inner: An instance of the underlying ``blazechunk._chunk`` chunker
                (e.g. the compiled ``TokenChunker``) that exposes ``.chunk(text)``.
        """
        self._inner = inner

    # -- single text -------------------------------------------------------

    def chunk(self, text: str) -> list[Chunk]:
        """Split ``text`` into chunks using this chunker's configuration.

        Args:
            text: The input to chunk. Whitespace-only input generally yields an
                empty list (see each chunker's docstring for the exact rule).

        Returns:
            A list of :class:`~blazechunk.Chunk` objects, in order. Each chunk
            knows its ``text``, its ``start_index``/``end_index`` byte offsets in
            ``text``, and its ``token_count``.

        Example:
            >>> from blazechunk import TokenChunker
            >>> chunks = TokenChunker(chunk_size=4).chunk("abcdefghij")
            >>> [c.text for c in chunks]
            ['abcd', 'efgh', 'ij']
        """
        return self._inner.chunk(text)

    async def chunk_async(self, text: str) -> list[Chunk]:
        """Asynchronous version of :meth:`chunk`.

        Runs the chunking on a worker thread so ``await``-ing it never blocks the
        event loop. The result is identical to :meth:`chunk`.

        Args:
            text: The input to chunk.

        Returns:
            A list of :class:`~blazechunk.Chunk` objects, in order.

        Example:
            >>> import asyncio
            >>> from blazechunk import SentenceChunker
            >>> async def main() -> None:
            ...     chunks = await SentenceChunker().chunk_async("One. Two. Three.")
            ...     print(len(chunks))
            >>> asyncio.run(main())
            1
        """
        return await asyncio.to_thread(self._inner.chunk, text)

    # -- many texts --------------------------------------------------------

    def chunk_batch(self, texts: Sequence[str]) -> list[list[Chunk]]:
        """Chunk many texts, returning one result list per input.

        Processing is sequential: chunking is CPU-bound, so a plain loop is the
        simplest and most predictable choice. For overlapping work in an async
        program, use :meth:`chunk_batch_async` instead.

        Args:
            texts: A sequence of input strings.

        Returns:
            A list the same length as ``texts``; ``result[i]`` is the chunks for
            ``texts[i]``.

        Example:
            >>> from blazechunk import TokenChunker
            >>> batches = TokenChunker(chunk_size=3).chunk_batch(["abcdef", "xy"])
            >>> [[c.text for c in b] for b in batches]
            [['abc', 'def'], ['xy']]
        """
        return [self._inner.chunk(text) for text in texts]

    async def chunk_batch_async(
        self,
        texts: Sequence[str],
        *,
        max_concurrency: int | None = None,
    ) -> list[list[Chunk]]:
        """Asynchronous version of :meth:`chunk_batch` with bounded concurrency.

        Each text is chunked on a worker thread and the calls are awaited
        together, so the event loop stays responsive while the batch runs.
        Results are returned in the same order as ``texts``.

        Args:
            texts: A sequence of input strings.
            max_concurrency: The maximum number of texts chunked at once. ``None``
                (the default) submits them all together. Pass a positive integer
                to cap in-flight work and limit memory/thread pressure on large
                batches.

        Returns:
            A list the same length as ``texts``; ``result[i]`` is the chunks for
            ``texts[i]``.

        Raises:
            ValueError: If ``max_concurrency`` is not ``None`` and not positive.

        Example:
            >>> import asyncio
            >>> from blazechunk import RecursiveChunker
            >>> async def main() -> list[int]:
            ...     batches = await RecursiveChunker().chunk_batch_async(
            ...         ["First doc.", "Second doc."], max_concurrency=4
            ...     )
            ...     return [len(b) for b in batches]
            >>> asyncio.run(main())
            [1, 1]
        """
        if max_concurrency is not None and max_concurrency < 1:
            raise ValueError("max_concurrency must be a positive integer or None")

        if max_concurrency is None:
            tasks = [asyncio.to_thread(self._inner.chunk, text) for text in texts]
            return list(await asyncio.gather(*tasks))

        semaphore = asyncio.Semaphore(max_concurrency)

        async def _chunk_one(text: str) -> list[Chunk]:
            async with semaphore:
                return await asyncio.to_thread(self._inner.chunk, text)

        return list(await asyncio.gather(*(_chunk_one(text) for text in texts)))

    # -- ergonomics --------------------------------------------------------

    def __call__(self, text: str) -> list[Chunk]:
        """Shorthand for :meth:`chunk` — ``chunker(text)`` chunks one string."""
        return self.chunk(text)

    def __repr__(self) -> str:
        return repr(self._inner)


class RecursiveChunker(BaseChunker):
    """Chunk text by recursively descending a hierarchy of delimiters.

    The recursive chunker tries to split on the most semantically meaningful
    boundary first (by default: paragraphs), and only falls back to finer
    boundaries (sentences, then clauses, then words, then raw tokens) for the
    pieces that are still larger than ``chunk_size``. This keeps chunks close to
    the target size while respecting the natural structure of the text.

    Every chunk's ``text`` is exactly the slice of the input between its
    ``start_index`` and ``end_index`` (the "slice invariant"), and chunks are
    contiguous and cover the whole input.

    Empty input returns ``[]``; unlike the other chunkers, whitespace-only input
    is *not* treated as empty and will still produce a chunk.

    Example:
        >>> from blazechunk import RecursiveChunker
        >>> text = "Para one.\\n\\nPara two is a little longer here."
        >>> chunks = RecursiveChunker(chunk_size=20).chunk(text)
        >>> all(c.text == text[c.start_index:c.end_index] for c in chunks)
        True
    """

    def __init__(
        self,
        *,
        tokenizer: Tokenizer = "character",
        chunk_size: int = 2048,
        min_characters_per_chunk: int = 24,
        rules: list[dict[str, Any]] | None = None,
    ) -> None:
        """Configure a recursive chunker.

        Args:
            tokenizer: How chunk size is measured. A built-in counter name
                (``"character"``, ``"word"``, ``"byte"``, ``"row"``) or a path to
                a HuggingFace ``tokenizer.json`` (requires the ``hf-tokenizer``
                build). Defaults to ``"character"``.
            chunk_size: Target maximum tokens per chunk. Must be > 0.
            min_characters_per_chunk: Fragments shorter than this are merged into
                a neighbour instead of standing alone. Must be > 0.
            rules: Optional custom delimiter hierarchy, replacing the built-in
                five levels. A list of level dicts, applied top to bottom:

                * ``{"delimiters": ["\\n\\n", "\\n"], "include_delim": "prev"}``
                  — split on any of these delimiters. ``include_delim`` is
                  optional and defaults to ``"prev"``.
                * ``{"type": "whitespace"}`` — split on ASCII spaces.
                * ``{"type": "token"}`` — terminal level: hard-split oversized
                  text into fixed-size token groups.

                ``None`` (the default) uses the built-in hierarchy
                (paragraphs → sentences → clauses → words → tokens).

        Raises:
            ValueError: If ``chunk_size`` or ``min_characters_per_chunk`` is 0,
                or if ``rules`` is empty or contains a malformed level.

        Example:
            >>> from blazechunk import RecursiveChunker
            >>> rules = [{"delimiters": ["|"]}, {"type": "token"}]
            >>> chunker = RecursiveChunker(chunk_size=3, min_characters_per_chunk=1,
            ...                            rules=rules)
            >>> [c.text for c in chunker.chunk("a|bb|ccc")]
            ['a|', 'bb|', 'ccc']
        """
        super().__init__(
            _RecursiveChunker(
                tokenizer=tokenizer,
                chunk_size=chunk_size,
                min_characters_per_chunk=min_characters_per_chunk,
                rules=rules,
            )
        )


class SentenceChunker(BaseChunker):
    """Chunk text into groups of whole sentences, with optional overlap.

    Sentences are detected with a set of delimiters (by default ``". "``,
    ``"! "``, ``"? "`` and newline) and then greedily packed into chunks up to
    ``chunk_size`` tokens. Because it never splits mid-sentence, this is a good
    default for retrieval where sentence integrity matters.

    Set ``chunk_overlap`` to repeat the tail sentences of each chunk at the start
    of the next one — useful for keeping context across chunk boundaries. With
    overlap, consecutive chunks overlap each other, but each individual chunk is
    still an exact slice of the input.

    Whitespace-only input returns ``[]``.

    Example:
        >>> from blazechunk import SentenceChunker
        >>> chunks = SentenceChunker(chunk_size=10,
        ...                          min_characters_per_sentence=1).chunk(
        ...     "One. Two. Three. Four."
        ... )
        >>> len(chunks) >= 2
        True
    """

    def __init__(
        self,
        *,
        tokenizer: Tokenizer = "character",
        chunk_size: int = 2048,
        chunk_overlap: int = 0,
        min_sentences_per_chunk: int = 1,
        min_characters_per_sentence: int = 12,
        delim: Sequence[str] | None = None,
        include_delim: IncludeDelim = "prev",
    ) -> None:
        """Configure a sentence chunker.

        Args:
            tokenizer: How chunk size is measured — a built-in counter name or a
                ``tokenizer.json`` path. Defaults to ``"character"``.
            chunk_size: Target maximum tokens per chunk. Must be > 0.
            chunk_overlap: Number of overlap tokens to carry from the end of one
                chunk into the start of the next. Must be < ``chunk_size``.
                ``0`` (the default) disables overlap.
            min_sentences_per_chunk: Minimum sentences per chunk. Must be >= 1.
            min_characters_per_sentence: Sentence fragments shorter than this are
                merged with a neighbour. Must be >= 1.
            delim: The sentence-ending delimiters. ``None`` (the default) uses
                ``[". ", "! ", "? ", "\\n"]``.
            include_delim: Where the delimiter is attached — ``"prev"`` (default),
                ``"next"``, or ``"none"``.

        Raises:
            ValueError: If ``chunk_size`` is 0, ``chunk_overlap`` >= ``chunk_size``,
                ``min_sentences_per_chunk`` < 1, or
                ``min_characters_per_sentence`` < 1.
        """
        super().__init__(
            _SentenceChunker(
                tokenizer=tokenizer,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                min_sentences_per_chunk=min_sentences_per_chunk,
                min_characters_per_sentence=min_characters_per_sentence,
                delim=list(delim) if delim is not None else None,
                include_delim=include_delim,
            )
        )


class TokenChunker(BaseChunker):
    """Chunk text into fixed-size token windows, with optional overlap.

    The simplest chunking strategy: slide a window of ``chunk_size`` tokens over
    the text, stepping by ``chunk_size - overlap`` each time. Boundaries always
    fall on token edges, so multi-byte characters are never split — each chunk's
    ``text`` is an exact slice of the input.

    Whitespace-only input returns ``[]``.

    Example:
        >>> from blazechunk import TokenChunker
        >>> chunks = TokenChunker(chunk_size=4, chunk_overlap=1).chunk("abcdefghij")
        >>> [c.text for c in chunks]  # windows step by chunk_size - overlap = 3
        ['abcd', 'defg', 'ghij']
    """

    def __init__(
        self,
        *,
        tokenizer: Tokenizer = "character",
        chunk_size: int = 2048,
        chunk_overlap: int | float | None = None,
    ) -> None:
        """Configure a token chunker.

        Args:
            tokenizer: How tokens are counted — a built-in counter name or a
                ``tokenizer.json`` path. Defaults to ``"character"`` (one token
                per Unicode code point).
            chunk_size: Number of tokens per window. Must be > 0.
            chunk_overlap: Tokens shared between consecutive windows. Accepts:

                * an ``int`` — an absolute number of tokens (must be
                  < ``chunk_size``);
                * a ``float`` in ``[0, 1)`` — a fraction of ``chunk_size``;
                * ``None`` (the default) — no overlap.

                The resulting step (``chunk_size - overlap``) must be > 0.

        Raises:
            ValueError: If ``chunk_size`` is 0, or the overlap is too large
                (an integer overlap >= ``chunk_size``, or a fraction that leaves
                a non-positive step, e.g. ``1.0``).
        """
        super().__init__(
            _TokenChunker(
                tokenizer=tokenizer,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )


class TableChunker(BaseChunker):
    """Chunk Markdown or HTML tables, repeating the header in every chunk.

    Large tables are split by rows so each chunk stays within ``chunk_size``, and
    the table header (and, for HTML, the closing tags) is re-included in every
    chunk so each one is a valid, self-contained table.

    Because the header is repeated, a table chunk's ``text`` is **not** a plain
    slice of the input — this is the one deliberate exception to the slice
    invariant. Its ``start_index``/``end_index`` still span the original row
    region the chunk was built from.

    Whitespace-only input, or input with no data rows, returns ``[]``.

    Example:
        >>> from blazechunk import TableChunker
        >>> md = "| A | B |\\n|---|---|\\n| 1 | 2 |\\n| 3 | 4 |\\n| 5 | 6 |\\n"
        >>> chunks = TableChunker(chunk_size=2).chunk(md)
        >>> all("|---|---|" in c.text for c in chunks)  # header kept per chunk
        True
    """

    def __init__(
        self,
        *,
        tokenizer: Tokenizer = "row",
        chunk_size: int = 3,
    ) -> None:
        """Configure a table chunker.

        Args:
            tokenizer: How chunk size is measured. Defaults to ``"row"``, so
                ``chunk_size`` counts data rows per chunk. Pass another counter
                (e.g. ``"character"``) to make ``chunk_size`` a token budget
                instead.
            chunk_size: Maximum rows (or tokens, depending on ``tokenizer``) per
                chunk. Must be > 0.

        Raises:
            ValueError: If ``chunk_size`` is 0.
        """
        super().__init__(
            _TableChunker(
                tokenizer=tokenizer,
                chunk_size=chunk_size,
            )
        )


class CodeChunker(BaseChunker):
    """Chunk source code along structural boundaries.

    Code is grouped into logical blocks (by brace depth, indentation, or blank
    lines, depending on the language) and those blocks are merged up to
    ``chunk_size``. A single block larger than the budget is split at line
    boundaries — never mid-line — so chunks stay syntactically readable. Each
    chunk's ``text`` is an exact slice of the input.

    Whitespace-only input returns ``[]``.

    Example:
        >>> from blazechunk import CodeChunker
        >>> code = "fn a() {\\n    let x = 1;\\n}\\n\\nfn b() {\\n    let y = 2;\\n}\\n"
        >>> chunks = CodeChunker(chunk_size=20, language="rust").chunk(code)
        >>> all(c.text == code[c.start_index:c.end_index] for c in chunks)
        True
    """

    def __init__(
        self,
        *,
        tokenizer: Tokenizer = "character",
        chunk_size: int = 2048,
        language: str = "auto",
    ) -> None:
        """Configure a code chunker.

        Args:
            tokenizer: How chunk size is measured — a built-in counter name or a
                ``tokenizer.json`` path. Defaults to ``"character"``.
            chunk_size: Target maximum tokens per chunk. Must be > 0.
            language: Source language, used to pick the block-detection strategy.
                ``"auto"`` (the default) detects it heuristically. Recognised
                names include ``"rust"``, ``"python"``, ``"javascript"``,
                ``"typescript"``, ``"c"``, ``"cpp"``, ``"java"``, ``"go"``,
                ``"ruby"``; anything unknown falls back to generic, blank-line
                based blocks.

        Raises:
            ValueError: If ``chunk_size`` is 0.
        """
        super().__init__(
            _CodeChunker(
                tokenizer=tokenizer,
                chunk_size=chunk_size,
                language=language,
            )
        )


class SemanticChunker(BaseChunker):
    """Chunk text at semantic-similarity troughs between sentence windows.

    Sentences are embedded (via the injected ``embedding_model``) as a sliding
    window and compared to the sentence that follows; a drop in that similarity
    curve marks a topic shift and becomes a chunk boundary. Boundaries are found
    with a Savitzky–Golay filter and kept when they fall below a percentile
    ``threshold``. Sentences between boundaries are grouped, then any group over
    ``chunk_size`` tokens is split at sentence boundaries.

    Chunks partition the input: each chunk's ``text`` is exactly the slice between
    its ``start_index`` and ``end_index``, chunks are contiguous, and together
    they cover the whole input. Whitespace-only input returns ``[]``.

    Example:
        >>> import numpy as np
        >>> from blazechunk import SemanticChunker
        >>> # A toy topic embedder: "cat" vs "finance" direction.
        >>> def embed(texts):
        ...     out = []
        ...     for t in texts:
        ...         low = t.lower()
        ...         out.append([low.count("cat"), low.count("finance"), 0.05])
        ...     return np.array(out, dtype="float32")
        >>> chunker = SemanticChunker(embed, min_characters_per_sentence=1,
        ...                           filter_tolerance=0.5)
        >>> isinstance(chunker.chunk("The cat sat. A cat purrs."), list)
        True
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        *,
        tokenizer: Tokenizer = "character",
        threshold: float = 0.8,
        chunk_size: int = 2048,
        similarity_window: int = 3,
        min_sentences_per_chunk: int = 1,
        min_characters_per_sentence: int = 24,
        delim: Sequence[str] | None = None,
        include_delim: IncludeDelim = "prev",
        skip_window: int = 0,
        filter_window: int = 5,
        filter_polyorder: int = 3,
        filter_tolerance: float = 0.2,
    ) -> None:
        """Configure a semantic chunker.

        Args:
            embedding_model: The embedding source (see :data:`EmbeddingModel`).
            tokenizer: How chunk size is measured — a built-in counter name or a
                ``tokenizer.json`` path. Defaults to ``"character"``.
            threshold: Percentile in ``(0, 1)`` for keeping similarity troughs as
                split points; also the raw cosine cutoff for the skip-merge pass.
            chunk_size: Target maximum tokens per chunk. Must be > 0.
            similarity_window: Number of sentences per comparison window. Must be
                > 0.
            min_sentences_per_chunk: Minimum sentences between split points.
            min_characters_per_sentence: Sentence fragments shorter than this are
                merged with a neighbour.
            delim: Sentence delimiters. ``None`` uses ``[". ", "! ", "? ", "\\n"]``.
            include_delim: Where the delimiter attaches — ``"prev"``/``"next"``/``"none"``.
            skip_window: If > 0, enable the double-pass skip-merge (see
                :class:`SDPMChunker`). ``0`` (the default) disables it.
            filter_window: Savitzky–Golay window length (odd). Must be > 0.
            filter_polyorder: Savitzky–Golay polynomial order. Must be
                ``< filter_window``.
            filter_tolerance: First-derivative zero tolerance for minima, in
                ``(0, 1)``.

        Raises:
            ValueError: If any numeric parameter is out of range.
            TypeError: If ``embedding_model`` is not an accepted form.
        """
        super().__init__(
            _SemanticChunker(
                _as_embed_batch(embedding_model),
                tokenizer=tokenizer,
                threshold=threshold,
                chunk_size=chunk_size,
                similarity_window=similarity_window,
                min_sentences_per_chunk=min_sentences_per_chunk,
                min_characters_per_sentence=min_characters_per_sentence,
                delim=list(delim) if delim is not None else None,
                include_delim=include_delim,
                skip_window=skip_window,
                filter_window=filter_window,
                filter_polyorder=filter_polyorder,
                filter_tolerance=filter_tolerance,
            )
        )


class SDPMChunker(BaseChunker):
    """Semantic chunking with a second, skip-window merge pass (SDPM).

    Semantic Double-Pass Merging first groups sentences at similarity troughs
    exactly like :class:`SemanticChunker`, then makes a second pass that looks
    ahead up to ``skip_window + 1`` groups and merges the current group with the
    most similar one within reach — bridging a short off-topic digression back to
    the topic it interrupted. Only the ``skip_window`` default differs from
    :class:`SemanticChunker`: here it is ``1`` (and must be ``>= 1``), so the
    second pass is always active.

    Chunks partition the input (same invariants as :class:`SemanticChunker`).
    Whitespace-only input returns ``[]``.

    Example:
        >>> import numpy as np
        >>> from blazechunk import SDPMChunker
        >>> def embed(texts):
        ...     return np.array([[t.lower().count("cat"), 0.05] for t in texts],
        ...                     dtype="float32")
        >>> chunker = SDPMChunker(embed, min_characters_per_sentence=1)
        >>> isinstance(chunker.chunk("A cat. A cat. A cat."), list)
        True
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        *,
        tokenizer: Tokenizer = "character",
        threshold: float = 0.8,
        chunk_size: int = 2048,
        similarity_window: int = 3,
        min_sentences_per_chunk: int = 1,
        min_characters_per_sentence: int = 24,
        delim: Sequence[str] | None = None,
        include_delim: IncludeDelim = "prev",
        skip_window: int = 1,
        filter_window: int = 5,
        filter_polyorder: int = 3,
        filter_tolerance: float = 0.2,
    ) -> None:
        """Configure an SDPM chunker.

        Args:
            embedding_model: The embedding source (see :data:`EmbeddingModel`).
            skip_window: Number of groups to look ahead when merging. Must be
                ``>= 1`` (``0`` would reduce this to a plain semantic chunker and
                is rejected). Defaults to ``1``.

        All other arguments match :class:`SemanticChunker`.

        Raises:
            ValueError: If any numeric parameter is out of range, including
                ``skip_window < 1``.
            TypeError: If ``embedding_model`` is not an accepted form.
        """
        super().__init__(
            _SDPMChunker(
                _as_embed_batch(embedding_model),
                tokenizer=tokenizer,
                threshold=threshold,
                chunk_size=chunk_size,
                similarity_window=similarity_window,
                min_sentences_per_chunk=min_sentences_per_chunk,
                min_characters_per_sentence=min_characters_per_sentence,
                delim=list(delim) if delim is not None else None,
                include_delim=include_delim,
                skip_window=skip_window,
                filter_window=filter_window,
                filter_polyorder=filter_polyorder,
                filter_tolerance=filter_tolerance,
            )
        )


class LateChunker(BaseChunker):
    """Chunk recursively, then attach a whole-document ("late") embedding per chunk.

    "Late chunking" embeds the entire document once so every token vector carries
    full-document context, and only then splits into chunks. Boundaries come from
    a recursive delimiter hierarchy (the same strategy as
    :class:`RecursiveChunker`); each chunk's embedding is the mean of the token
    embeddings that fall within its span.

    ``chunk(text)`` returns :class:`~blazechunk.LateChunk` objects — like a normal
    chunk (``text``, ``start_index``, ``end_index``, ``token_count``) but also
    carrying an ``embedding`` (a list of floats). The byte range still satisfies
    the slice invariant. Whitespace-only input returns ``[]``.

    The ``embedding_model`` must provide token-level embeddings: either an object
    exposing ``embed_as_tokens(text) -> 2D`` and ``embed(text) -> 1D``, or a
    ``(embed_as_tokens, embed)`` tuple of callables.

    Example:
        >>> import numpy as np
        >>> from blazechunk import LateChunker
        >>> # A toy token embedder: one vector per character.
        >>> class Toy:
        ...     def embed_as_tokens(self, text):
        ...         return np.array([[ord(c) % 97, i % 13] for i, c in enumerate(text)],
        ...                         dtype="float32")
        ...     def embed(self, text):
        ...         toks = self.embed_as_tokens(text)
        ...         return toks.mean(axis=0) if len(toks) else np.zeros(2, "float32")
        >>> chunks = LateChunker(Toy(), chunk_size=8,
        ...                      min_characters_per_chunk=1).chunk("hello world foo bar")
        >>> all(len(c.embedding) == 2 for c in chunks)
        True
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        *,
        tokenizer: Tokenizer = "character",
        chunk_size: int = 2048,
        min_characters_per_chunk: int = 24,
        rules: list[dict[str, Any]] | None = None,
    ) -> None:
        """Configure a late chunker.

        Args:
            embedding_model: A token-embedding source (see the class docstring).
            tokenizer: How boundary chunk size is measured — a built-in counter
                name or a ``tokenizer.json`` path. Defaults to ``"character"``. For
                best alignment, use the same tokenization the embedder uses.
            chunk_size: Target maximum tokens per chunk. Must be > 0.
            min_characters_per_chunk: Fragments shorter than this merge into a
                neighbour. Must be > 0.
            rules: Optional custom delimiter hierarchy (same format as
                :class:`RecursiveChunker`). ``None`` uses the built-in hierarchy.

        Raises:
            ValueError: If ``chunk_size`` or ``min_characters_per_chunk`` is 0, or
                ``rules`` is malformed.
            TypeError: If ``embedding_model`` is not an accepted form.
        """
        embed_tokens_fn, embed_fn = _as_token_embed_fns(embedding_model)
        super().__init__(
            _LateChunker(
                embed_tokens_fn,
                embed_fn,
                tokenizer=tokenizer,
                chunk_size=chunk_size,
                min_characters_per_chunk=min_characters_per_chunk,
                rules=rules,
            )
        )
