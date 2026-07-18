"""Type stubs for the compiled Rust extension ``blazechunk._chunk``.

These describe the low-level surface implemented in Rust (via PyO3). Most users
should prefer the typed, documented wrappers re-exported from ``blazechunk``
itself (see ``blazechunk.chunkers``); these stubs exist so the compiled module
type-checks cleanly for the wrappers and for advanced/direct use.
"""

from typing import Any

DEFAULT_TARGET_SIZE: int
DEFAULT_DELIMITERS: bytes

class Chunk:
    """A chunk of text with byte-offset indices into the original input."""

    text: str
    start_index: int
    end_index: int
    token_count: int
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...

class Chunker:
    """Zero-copy, size-based byte chunker. Iterates ``bytes`` chunks."""

    def __init__(
        self,
        text: str | bytes,
        size: int = ...,
        delimiters: str | bytes | None = ...,
        pattern: str | bytes | None = ...,
        patterns: list[str | bytes] | None = ...,
        prefix: bool = ...,
        consecutive: bool = ...,
        forward_fallback: bool = ...,
    ) -> None: ...
    def __iter__(self) -> Chunker: ...
    def __next__(self) -> bytes: ...
    def reset(self) -> None: ...
    def collect_offsets(self) -> list[tuple[int, int]]: ...

class RecursiveChunker:
    def __init__(
        self,
        tokenizer: str | None = ...,
        chunk_size: int = ...,
        min_characters_per_chunk: int = ...,
        rules: list[dict[str, Any]] | None = ...,
    ) -> None: ...
    def chunk(self, text: str) -> list[Chunk]: ...
    def __repr__(self) -> str: ...

class SentenceChunker:
    def __init__(
        self,
        tokenizer: str | None = ...,
        chunk_size: int = ...,
        chunk_overlap: int = ...,
        min_sentences_per_chunk: int = ...,
        min_characters_per_sentence: int = ...,
        delim: list[str] | None = ...,
        include_delim: str | None = ...,
    ) -> None: ...
    def chunk(self, text: str) -> list[Chunk]: ...
    def __repr__(self) -> str: ...

class TokenChunker:
    def __init__(
        self,
        tokenizer: str | None = ...,
        chunk_size: int = ...,
        chunk_overlap: int | float | None = ...,
    ) -> None: ...
    def chunk(self, text: str) -> list[Chunk]: ...
    def __repr__(self) -> str: ...

class TableChunker:
    def __init__(
        self,
        tokenizer: str | None = ...,
        chunk_size: int = ...,
    ) -> None: ...
    def chunk(self, text: str) -> list[Chunk]: ...
    def __repr__(self) -> str: ...

class CodeChunker:
    def __init__(
        self,
        tokenizer: str | None = ...,
        chunk_size: int = ...,
        language: str | None = ...,
    ) -> None: ...
    def chunk(self, text: str) -> list[Chunk]: ...
    def __repr__(self) -> str: ...

class MergeResult:
    merged: list[str]
    token_counts: list[int]
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...

class PatternSplitter:
    def __init__(self, patterns: list[str | bytes]) -> None: ...
    def split(
        self,
        text: str | bytes,
        include_delim: str = ...,
        min_chars: int = ...,
    ) -> list[tuple[int, int]]: ...

def chunk_offsets(
    text: str | bytes,
    size: int = ...,
    delimiters: str | bytes | None = ...,
    pattern: str | bytes | None = ...,
    patterns: list[str | bytes] | None = ...,
    prefix: bool = ...,
    consecutive: bool = ...,
    forward_fallback: bool = ...,
) -> list[tuple[int, int]]: ...
def split_offsets(
    text: str | bytes,
    delimiters: str | bytes | None = ...,
    include_delim: str = ...,
    min_chars: int = ...,
) -> list[tuple[int, int]]: ...
def split_pattern_offsets(
    text: str | bytes,
    patterns: list[str | bytes],
    include_delim: str = ...,
    min_chars: int = ...,
) -> list[tuple[int, int]]: ...
def find_merge_indices(token_counts: list[int], chunk_size: int) -> list[int]: ...
def merge_splits(
    splits: list[str], token_counts: list[int], chunk_size: int
) -> MergeResult: ...

# Available only when built with the (default) `numpy-support` feature.
def savgol_filter(
    data: Any, window_length: int = ..., poly_order: int = ..., deriv: int = ...
) -> Any: ...
def find_local_minima_interpolated(
    data: Any, window_size: int = ..., poly_order: int = ..., tolerance: float = ...
) -> tuple[Any, Any]: ...
def windowed_cross_similarity(embeddings: Any, window_size: int = ...) -> Any: ...
def filter_split_indices(
    indices: Any, values: Any, threshold: float = ..., min_distance: int = ...
) -> tuple[Any, Any]: ...
