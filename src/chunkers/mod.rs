//! High-level chunkers built on the core primitives.
//!
//! Each chunker is a deep module with a small interface — `chunk(text) -> Vec<Chunk>`
//! — over a hidden implementation that reuses the `split`, `merge`, and `savgol`
//! primitives. The tokenizer is injected as a `&dyn TokenCounter`, so the same
//! chunker works with any counter (character, word, byte, row, or a future
//! subword tokenizer) without code changes.

use crate::split::{IncludeDelim, split_at_delimiters, split_at_patterns};

/// A chunk of text, identified by byte offsets into the original input.
///
/// Invariant for every chunker **except** the table chunker: the chunk text is
/// exactly `original[start..end]`. Chunkers may emit overlapping chunks (e.g.
/// [`sentence::SentenceChunker`] with overlap), but each chunk is always a valid,
/// contiguous slice of the original text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Chunk {
    /// Byte offset of the chunk start in the original text.
    pub start: usize,
    /// Byte offset of the chunk end (exclusive) in the original text.
    pub end: usize,
    /// Number of tokens in the chunk, per the injected counter.
    pub token_count: usize,
}

/// Error returned by chunkers whose configuration can be invalid at chunk time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChunkError {
    /// A configuration value is invalid (e.g. a non-positive token stride).
    InvalidConfig(String),
}

impl std::fmt::Display for ChunkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChunkError::InvalidConfig(m) => write!(f, "invalid chunker configuration: {m}"),
        }
    }
}

impl std::error::Error for ChunkError {}

/// Split text at delimiters, routing non-ASCII / multi-byte delimiters through
/// full-pattern matching (fixes upstream #536) and single-byte ASCII delimiters
/// through the fast per-byte path. Returns contiguous byte-offset segments.
///
/// Shared by [`recursive`] and [`sentence`].
pub(crate) fn split_by_delimiters(
    text: &str,
    delimiters: &[String],
    include: IncludeDelim,
    min_chars: usize,
) -> Vec<(usize, usize)> {
    let complex = delimiters.iter().any(|d| d.len() > 1 || !d.is_ascii());
    if complex {
        let pats: Vec<&[u8]> = delimiters.iter().map(|d| d.as_bytes()).collect();
        split_at_patterns(text.as_bytes(), &pats, include, min_chars)
    } else {
        let delim_bytes: Vec<u8> = delimiters
            .iter()
            .filter_map(|d| d.as_bytes().first().copied())
            .collect();
        split_at_delimiters(text.as_bytes(), &delim_bytes, include, min_chars)
    }
}

/// Byte ranges of each line in `text`, including the trailing `'\n'`. The ranges
/// are contiguous and cover the whole text. Shared by [`table`] and [`code`].
pub(crate) fn line_ranges(text: &str) -> Vec<(usize, usize)> {
    let bytes = text.as_bytes();
    let mut ranges = Vec::new();
    let mut start = 0;
    for (i, &b) in bytes.iter().enumerate() {
        if b == b'\n' {
            ranges.push((start, i + 1));
            start = i + 1;
        }
    }
    if start < bytes.len() {
        ranges.push((start, bytes.len()));
    }
    ranges
}

pub mod code;
pub mod late;
pub mod recursive;
pub mod sdpm;
pub mod semantic;
pub mod sentence;
pub mod table;
pub mod token;

pub use code::{CodeChunker, Language};
pub use late::{LateChunk, LateChunker};
pub use recursive::{RecursiveChunker, RecursiveLevel, RecursiveRules};
pub use sdpm::SDPMChunker;
pub use semantic::SemanticChunker;
pub use sentence::SentenceChunker;
pub use table::{TableChunk, TableChunker};
pub use token::{Overlap, TokenChunker};
