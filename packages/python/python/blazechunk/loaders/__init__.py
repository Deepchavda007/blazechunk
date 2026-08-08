"""Chunk documents — PDF, Word, PowerPoint, Excel, OpenDocument, RTF, EPUB, CSV.

Every RAG pipeline starts with a file, not a string. This module closes that
gap: hand it a path and get back chunks that know where they came from.

    from blazechunk.loaders import DocumentChunker

    result = DocumentChunker().chunk("report.pdf")
    for c in result.chunks:
        print(c.heading_path, c.kind, c.text[:60])

Conversion is done by `anydoc <https://github.com/firecrawl/anydoc>`_, a
pure-Rust converter with no ML and no network calls. Install it with::

    pip install "blazechunk[anydoc]"

Markdown and plain-text input need no extra at all.

What this buys you over converting yourself
-------------------------------------------
Converting a file to Markdown and handing the string to a text chunker throws
away the structure on the way in. The chunker then tries to guess it back from
punctuation, and splits tables mid-row and code blocks mid-function because it
has no idea they are there.

:class:`DocumentChunker` segments the document *first*, then routes each piece
to a chunker that suits it — table rows to :class:`~blazechunk.TableChunker`,
fenced code to :class:`~blazechunk.CodeChunker`, prose to whichever chunker you
chose. Each chunk carries the chain of headings above it, which turns an
otherwise anonymous fragment into something a retriever can place.

Offsets and provenance
----------------------
``md_start`` / ``md_end`` are byte offsets into
:attr:`ChunkedDocument.markdown` — the converted text, which is returned
alongside the chunks — and **not** into the original file. anydoc exposes no
mapping back to source bytes, so a page number for a PDF chunk is not something
this can honestly provide. If you need page-level attribution for audit or
compliance, this is not the tool for that half of the job.

The fields are deliberately not called ``start``/``end``: elsewhere in
blazechunk those index the input you passed in, and here they cannot.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Union

from blazechunk.chunkers import BaseChunker, CodeChunker, RecursiveChunker, TableChunker
from blazechunk.loaders._anydoc_adapter import (
    PASSTHROUGH_FORMATS,
    SUPPORTED_FORMATS,
    convert,
)
from blazechunk.loaders._errors import (
    DocumentError,
    DocumentResourceLimit,
    EncryptedDocument,
    MalformedDocument,
    ScannedDocumentError,
    UnsupportedDocument,
)
from blazechunk.loaders._segments import (
    Segment,
    group_span,
    merge_small_segments,
    segment_markdown,
)

__all__ = [
    "DocumentChunker",
    "ChunkedDocument",
    "DocumentChunk",
    "DocumentError",
    "UnsupportedDocument",
    "ScannedDocumentError",
    "MalformedDocument",
    "EncryptedDocument",
    "DocumentResourceLimit",
    "SUPPORTED_FORMATS",
    "PASSTHROUGH_FORMATS",
]

#: Anything a document can be read from: a path, or the bytes themselves.
DocumentSource = Union[str, "os.PathLike[str]", bytes, bytearray]

#: Distinguishes "the caller passed None to disable this" from "the caller said
#: nothing, so use the default", which a plain ``None`` default cannot express.
_UNSET: Any = object()

#: What a batch does with a document that fails to convert.
#:
#: * ``"raise"``   — propagate the first failure (the default).
#: * ``"skip"``    — leave it out of the results, which makes the returned list
#:   shorter than the input.
#: * ``"collect"`` — keep positions aligned by putting the
#:   :class:`DocumentError` itself in the failing slot.
OnError = Literal["raise", "skip", "collect"]


@dataclass(frozen=True)
class DocumentChunk:
    """One chunk of a document, with the structure it came from.

    Attributes:
        text: The chunk's text. For every chunk with ``is_exact`` true this is
            exactly ``markdown_bytes[md_start:md_end]`` decoded.
        md_start: Start offset, in UTF-8 bytes, into
            :attr:`ChunkedDocument.markdown`. Byte offsets, not code points —
            the same convention as :class:`~blazechunk.Chunk`.
        md_end: End offset in UTF-8 bytes, exclusive.
        kind: ``"prose"``, ``"table"``, ``"code"``, ``"list"``, ``"quote"``,
            ``"heading"`` or ``"rule"``.
        heading_path: The headings enclosing this chunk, outermost first, e.g.
            ``("Methods", "Sample Preparation")``. Empty above the first
            heading. This is the highest-value field here for retrieval: it
            makes a fragment from the middle of a long document self-locating,
            and rerankers can use it directly.
        lang: The info string of a fenced code block, else ``None``.
        source_format: The format the document was read from, e.g. ``"pdf"``.
        token_count: Tokens in this chunk, as counted by the chunker that
            produced it.
        is_exact: Whether ``text`` is a verbatim slice of the Markdown. False
            only for the second and later chunks of a split table, which repeat
            the header row so each chunk stays readable on its own.
    """

    text: str
    md_start: int
    md_end: int
    kind: str
    heading_path: tuple[str, ...]
    lang: str | None
    source_format: str
    token_count: int
    is_exact: bool

    def __len__(self) -> int:
        return len(self.text)

    def __repr__(self) -> str:
        where = "/".join(self.heading_path) or "-"
        return (
            f"DocumentChunk(kind={self.kind!r}, heading_path={where!r}, "
            f"md=[{self.md_start}:{self.md_end}], text={self.text[:40]!r}…)"
        )


@dataclass(frozen=True)
class ChunkedDocument:
    """The result of chunking one document.

    The Markdown comes back with the chunks rather than being thrown away,
    because the offsets are meaningless without the string they index into.
    Returning them separately would invite exactly the mistake the
    ``md_``-prefixed names are there to prevent.

    Attributes:
        chunks: The chunks, in document order.
        markdown: The converted document. Every offset indexes into this.
        format: The format the document was read from, e.g. ``"docx"``.
        warnings: Non-fatal notes about the conversion, e.g. that a table was
            split across chunks.
    """

    chunks: list[DocumentChunk]
    markdown: str
    format: str
    warnings: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.chunks)

    def __iter__(self):
        return iter(self.chunks)

    def __getitem__(self, index: int) -> DocumentChunk:
        return self.chunks[index]

    @property
    def markdown_bytes(self) -> bytes:
        """The Markdown as UTF-8, the buffer ``md_start``/``md_end`` index."""
        return self.markdown.encode("utf-8")

    def __repr__(self) -> str:
        return (
            f"ChunkedDocument(format={self.format!r}, chunks={len(self.chunks)}, "
            f"markdown={len(self.markdown)} chars, warnings={len(self.warnings)})"
        )


def _read(source: DocumentSource) -> tuple[bytes, str | None]:
    """Normalise a source into ``(data, filename)``."""
    if isinstance(source, (bytes, bytearray)):
        return bytes(source), None
    path = Path(os.fspath(source))
    return path.read_bytes(), str(path)


class DocumentChunker:
    """Turn documents into structure-aware chunks.

    Deliberately *not* a :class:`~blazechunk.BaseChunker` subclass. Every
    chunker in blazechunk maps ``str -> list[Chunk]``; this maps
    ``file -> ChunkedDocument``. Inheriting would let a ``DocumentChunker`` be
    passed where a text chunker is expected and fail there, so the two are kept
    as separate types that happen to share method *names*.

    Args:
        chunker: The chunker for prose. Defaults to
            ``RecursiveChunker(chunk_size=2048)``.
        table_chunker: The chunker for tables. Defaults to
            ``TableChunker(chunk_size=3)``, which keeps rows intact and repeats
            the header on every chunk. Pass ``None`` to send tables to
            ``chunker`` like any other text.
        code_chunker: The chunker for fenced code. Defaults to
            ``CodeChunker(chunk_size=2048)``. Pass ``None`` to send code to
            ``chunker``.
        respect_headings: When true (the default) a chunk never spans a heading
            boundary, even if both sides are small.
        merge_small_segments: Whether to combine undersized neighbours, so a
            lone heading or a one-line paragraph does not become its own chunk.
        min_chunk_size: The size, in characters, below which a segment looks for
            a partner to merge forward into.
        max_concurrency: Default bound on documents converted at once by the
            async and batch methods.

    Note:
        :class:`~blazechunk.SemanticChunker`, :class:`~blazechunk.SDPMChunker`
        and :class:`~blazechunk.LateChunker` all work here, but they operate
        *within* a segment and never across one. Structural boundaries win over
        embedding-derived ones — a heading is ground truth where a similarity
        trough is an estimate — which is the right default but is a genuine
        behaviour change if you expected purely semantic segmentation.

    Example:
        >>> from blazechunk import RecursiveChunker
        >>> from blazechunk.loaders import DocumentChunker
        >>> loader = DocumentChunker(chunker=RecursiveChunker(chunk_size=1024))
        >>> result = loader.chunk("report.pdf")            # doctest: +SKIP
        >>> result.chunks[0].heading_path                  # doctest: +SKIP
        ('Methods', 'Sample Preparation')
    """

    __slots__ = (
        "_chunker",
        "_table_chunker",
        "_code_chunker",
        "_respect_headings",
        "_merge",
        "_min_chunk_size",
        "_max_concurrency",
    )

    def __init__(
        self,
        chunker: BaseChunker | None = None,
        *,
        table_chunker: BaseChunker | None = _UNSET,
        code_chunker: BaseChunker | None = _UNSET,
        respect_headings: bool = True,
        merge_small_segments: bool = True,
        min_chunk_size: int = 256,
        max_concurrency: int | None = 8,
    ) -> None:
        if min_chunk_size < 0:
            raise ValueError("min_chunk_size must be >= 0")
        if max_concurrency is not None and max_concurrency < 1:
            raise ValueError("max_concurrency must be a positive integer or None")

        self._chunker = chunker or RecursiveChunker(chunk_size=2048)
        self._table_chunker = (
            TableChunker(chunk_size=3) if table_chunker is _UNSET else table_chunker
        )
        self._code_chunker = (
            CodeChunker(chunk_size=2048) if code_chunker is _UNSET else code_chunker
        )
        self._respect_headings = respect_headings
        self._merge = merge_small_segments
        self._min_chunk_size = min_chunk_size
        self._max_concurrency = max_concurrency

    # -- single document ---------------------------------------------------

    def chunk(
        self,
        source: DocumentSource,
        *,
        format: str | None = None,
    ) -> ChunkedDocument:
        """Convert and chunk one document.

        Args:
            source: A path, or the document's bytes.
            format: The format name, when it cannot be detected. Formats with no
                signature of their own (CSV) need this if ``source`` is bytes
                with no filename to fall back on.

        Returns:
            A :class:`ChunkedDocument`.

        Raises:
            ScannedDocumentError: A PDF with no text layer; it needs OCR first.
            UnsupportedDocument: Unknown format, or nothing extractable.
            MalformedDocument: Structurally unusable.
            EncryptedDocument: Password-protected.
            DocumentResourceLimit: A parser safety limit was crossed.
            OSError: ``source`` is a path that could not be read.
        """
        data, filename = _read(source)
        markdown, resolved = convert(data, format=format, filename=filename)
        return self.chunk_markdown(markdown, format=resolved)

    def chunk_markdown(self, markdown: str, *, format: str = "md") -> ChunkedDocument:
        """Chunk Markdown that has already been converted.

        Useful when the text came from somewhere anydoc does not cover — an OCR
        step, an HTML pipeline, a database column — and you still want the
        structural routing and heading paths.

        Args:
            markdown: A GFM document.
            format: The name to record as the source format.

        Returns:
            A :class:`ChunkedDocument`.
        """
        segments = segment_markdown(markdown)
        if not segments:
            return ChunkedDocument([], markdown, format, [])

        if self._merge and self._min_chunk_size > 0:
            groups = merge_small_segments(
                segments,
                markdown,
                min_size=self._min_chunk_size,
                respect_headings=self._respect_headings,
            )
        else:
            groups = [[seg] for seg in segments]

        md_bytes = markdown.encode("utf-8")
        chunks: list[DocumentChunk] = []
        warnings: list[str] = []
        split_tables = 0

        for group in groups:
            span = group_span(group, markdown)
            produced = self._chunk_segment(span, markdown, md_bytes, format)
            if span.kind == "table" and len(produced) > 1:
                split_tables += 1
            chunks.extend(produced)

        if split_tables:
            warnings.append(
                f"{split_tables} table(s) exceeded the table chunker's size and were "
                "split; each continuation chunk repeats the header row, so its text is "
                "not a verbatim slice of the Markdown (is_exact=False)."
            )

        return ChunkedDocument(chunks, markdown, format, warnings)

    def _pick(self, kind: str) -> BaseChunker:
        """The chunker that suits a segment of this kind."""
        if kind == "table" and self._table_chunker is not None:
            return self._table_chunker
        if kind == "code" and self._code_chunker is not None:
            return self._code_chunker
        return self._chunker

    def _chunk_segment(
        self,
        span: Segment,
        markdown: str,
        md_bytes: bytes,
        source_format: str,
    ) -> list[DocumentChunk]:
        """Chunk one segment and lift its offsets into document coordinates."""
        text = span.text(markdown)
        if not text.strip():
            return []

        pieces = self._pick(span.kind).chunk(text)
        if not pieces:
            # A chunker that declines the text would leave a hole in the
            # document. Emit the segment whole rather than lose it.
            return [
                DocumentChunk(
                    text=text,
                    md_start=span.byte_start,
                    md_end=span.byte_end,
                    kind=span.kind,
                    heading_path=span.heading_path,
                    lang=span.lang,
                    source_format=source_format,
                    token_count=len(text),
                    is_exact=True,
                )
            ]

        out: list[DocumentChunk] = []
        last = len(pieces) - 1
        for position, piece in enumerate(pieces):
            start = span.byte_start + piece.start_index
            end = span.byte_start + piece.end_index
            # Pin the outer edges to the segment so the segment is covered end
            # to end. This matters for tables: TableChunker's offsets span only
            # the data rows, and pulling the first chunk back to the segment
            # start makes its text — header, delimiter row and all — an exact
            # slice again, as well as closing the coverage gap.
            if position == 0:
                start = span.byte_start
            if position == last:
                end = max(end, span.byte_end)
            out.append(
                DocumentChunk(
                    text=piece.text,
                    md_start=start,
                    md_end=end,
                    kind=span.kind,
                    heading_path=span.heading_path,
                    lang=span.lang,
                    source_format=source_format,
                    token_count=piece.token_count,
                    is_exact=md_bytes[start:end] == piece.text.encode("utf-8"),
                )
            )
        return out

    # -- async -------------------------------------------------------------

    async def chunk_async(
        self,
        source: DocumentSource,
        *,
        format: str | None = None,
    ) -> ChunkedDocument:
        """Asynchronous :meth:`chunk`.

        Conversion is CPU-bound Rust and dominates the wall clock, and anydoc
        releases the GIL while it runs, so offloading it to a worker thread
        genuinely overlaps rather than merely deferring.
        """
        data, filename = _read(source)
        markdown, resolved = await asyncio.to_thread(
            convert, data, format=format, filename=filename
        )
        return await asyncio.to_thread(self.chunk_markdown, markdown, format=resolved)

    # -- many documents ----------------------------------------------------

    def chunk_batch(
        self,
        sources: Sequence[DocumentSource],
        *,
        format: str | None = None,
        on_error: OnError = "raise",
    ) -> list[Any]:
        """Chunk many documents, sequentially.

        Args:
            sources: Paths or byte strings.
            format: An explicit format applied to every source.
            on_error: What to do with a document that fails to convert. See
                :data:`OnError`.

        Returns:
            A list of :class:`ChunkedDocument`. With ``on_error="collect"`` a
            failing slot holds the :class:`DocumentError` instead; with
            ``"skip"`` it is absent and the list is shorter than ``sources``.
        """
        results: list[Any] = []
        for source in sources:
            try:
                results.append(self.chunk(source, format=format))
            except DocumentError as error:
                if on_error == "raise":
                    raise
                if on_error == "collect":
                    results.append(error)
        return results

    async def chunk_batch_async(
        self,
        sources: Sequence[DocumentSource],
        *,
        format: str | None = None,
        on_error: OnError = "raise",
        max_concurrency: int | None = None,
    ) -> list[Any]:
        """Asynchronous :meth:`chunk_batch`, with bounded concurrency.

        Args:
            sources: Paths or byte strings.
            format: An explicit format applied to every source.
            on_error: What to do with a failing document. See :data:`OnError`.
            max_concurrency: Documents converted at once. Defaults to the value
                given to the constructor.

        Returns:
            The same shapes :meth:`chunk_batch` returns, in input order.
        """
        limit = self._max_concurrency if max_concurrency is None else max_concurrency
        if limit is not None and limit < 1:
            raise ValueError("max_concurrency must be a positive integer or None")

        semaphore = asyncio.Semaphore(limit) if limit is not None else None

        async def one(source: DocumentSource) -> Any:
            try:
                if semaphore is None:
                    return await self.chunk_async(source, format=format)
                async with semaphore:
                    return await self.chunk_async(source, format=format)
            except DocumentError as error:
                if on_error == "raise":
                    raise
                return error

        settled = await asyncio.gather(*(one(source) for source in sources))
        if on_error == "skip":
            return [item for item in settled if not isinstance(item, DocumentError)]
        return list(settled)

    # -- ergonomics --------------------------------------------------------

    def __call__(self, source: DocumentSource, *, format: str | None = None) -> ChunkedDocument:
        """Shorthand for :meth:`chunk`."""
        return self.chunk(source, format=format)

    def __repr__(self) -> str:
        return (
            f"DocumentChunker(chunker={self._chunker!r}, "
            f"respect_headings={self._respect_headings}, "
            f"min_chunk_size={self._min_chunk_size})"
        )
