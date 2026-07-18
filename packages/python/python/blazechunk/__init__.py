"""blazechunk — the fastest semantic text chunking library.

This package pairs a SIMD-accelerated Rust core with a small, uniform Python API.

Two layers are available:

* **High-level chunkers** — :class:`RecursiveChunker`, :class:`SentenceChunker`,
  :class:`TokenChunker`, :class:`TableChunker`, and :class:`CodeChunker`. Each one
  offers the same four methods — ``chunk`` / ``chunk_async`` and ``chunk_batch`` /
  ``chunk_batch_async`` — and returns :class:`Chunk` objects with byte offsets and
  token counts. Start here.

* **Low-level primitives** — the zero-copy :class:`Chunker` iterator, the
  :func:`chunk` / :func:`chunk_async` convenience helpers, and the
  offset/merge/split functions (:func:`chunk_offsets`, :func:`split_offsets`,
  :func:`merge_splits`, …) for building your own pipeline.

Example:
    >>> from blazechunk import TokenChunker
    >>> chunker = TokenChunker(chunk_size=512, chunk_overlap=64)
    >>> chunks = chunker.chunk("... a long document ...")
    >>> chunks[0].text, chunks[0].start_index, chunks[0].token_count  # doctest: +SKIP
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator, Sequence

from blazechunk._chunk import (
    DEFAULT_DELIMITERS,
    DEFAULT_TARGET_SIZE,
    Chunk,
    Chunker,
    MergeResult,
    PatternSplitter,
    chunk_offsets,
    find_merge_indices,
    merge_splits,
    split_offsets,
    split_pattern_offsets,
)

from blazechunk.chunkers import (
    BaseChunker,
    CodeChunker,
    RecursiveChunker,
    SentenceChunker,
    TableChunker,
    TokenChunker,
)

__version__ = "0.10.2"

__all__ = [
    # High-level chunkers
    "BaseChunker",
    "RecursiveChunker",
    "SentenceChunker",
    "TokenChunker",
    "TableChunker",
    "CodeChunker",
    "Chunk",
    # Low-level fast path
    "Chunker",
    "chunk",
    "chunk_async",
    "chunk_offsets",
    "split_offsets",
    "split_pattern_offsets",
    "find_merge_indices",
    "merge_splits",
    "MergeResult",
    "PatternSplitter",
    "DEFAULT_TARGET_SIZE",
    "DEFAULT_DELIMITERS",
]


def chunk(
    text: str | bytes,
    *,
    size: int = DEFAULT_TARGET_SIZE,
    delimiters: str | bytes | None = None,
    patterns: Sequence[str | bytes] | None = None,
) -> Iterator[memoryview]:
    """Split ``text`` at delimiter boundaries, yielding zero-copy chunks.

    This is the fastest way to chunk raw bytes: offsets are computed in a single
    Rust call and each chunk is returned as a :class:`memoryview` into the
    original buffer, with no copying. Materialise a chunk with ``bytes(chunk)``
    when you need to own it.

    Args:
        text: The input to chunk, as ``str`` (encoded to UTF-8) or ``bytes``.
        size: Target chunk size in bytes. Defaults to ``DEFAULT_TARGET_SIZE``
            (4096).
        delimiters: Single-byte delimiter characters to break on, as ``str`` or
            ``bytes``. Defaults to ``DEFAULT_DELIMITERS`` (``b"\\n.?"``).
        patterns: Optional multi-byte patterns (e.g. CJK punctuation such as
            ``["。", "，"]``). Composable with ``delimiters`` — both stay active.

    Yields:
        :class:`memoryview` slices of the input, in order. Their lengths sum to
        the length of the (UTF-8 encoded) input.

    Example:
        >>> from blazechunk import chunk
        >>> [bytes(c) for c in chunk(b"Hello. World. Test.", size=10, delimiters=b".")]
        [b'Hello.', b' World.', b' Test.']
    """
    if isinstance(text, str):
        text = text.encode("utf-8")

    offsets = chunk_offsets(text, size=size, delimiters=delimiters, patterns=patterns)

    view = memoryview(text)
    for start, end in offsets:
        yield view[start:end]


async def chunk_async(
    text: str | bytes,
    *,
    size: int = DEFAULT_TARGET_SIZE,
    delimiters: str | bytes | None = None,
    patterns: Sequence[str | bytes] | None = None,
) -> list[bytes]:
    """Asynchronous version of :func:`chunk`.

    Runs the split on a worker thread so ``await``-ing it never blocks the event
    loop. Because the resulting chunks outlive the worker, they are returned as
    owned ``bytes`` objects (a list) rather than zero-copy memoryviews.

    Args:
        text: The input to chunk, as ``str`` (encoded to UTF-8) or ``bytes``.
        size: Target chunk size in bytes. Defaults to ``DEFAULT_TARGET_SIZE``.
        delimiters: Single-byte delimiters. Defaults to ``DEFAULT_DELIMITERS``.
        patterns: Optional multi-byte patterns, composable with ``delimiters``.

    Returns:
        A list of ``bytes`` chunks, in order.

    Example:
        >>> import asyncio
        >>> from blazechunk import chunk_async
        >>> asyncio.run(chunk_async(b"Hello. World.", size=10, delimiters=b"."))
        [b'Hello.', b' World.']
    """

    def _run() -> list[bytes]:
        return [
            bytes(view)
            for view in chunk(text, size=size, delimiters=delimiters, patterns=patterns)
        ]

    return await asyncio.to_thread(_run)


# Optional numpy-backed signal-processing helpers. Present only when the
# extension was built with the (default) `numpy-support` feature.
try:
    from blazechunk._chunk import (  # noqa: F401
        filter_split_indices,
        find_local_minima_interpolated,
        savgol_filter,
        windowed_cross_similarity,
    )

    __all__ += [
        "savgol_filter",
        "find_local_minima_interpolated",
        "windowed_cross_similarity",
        "filter_split_indices",
    ]
except ImportError:
    pass
