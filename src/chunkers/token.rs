//! Fixed-size token chunking.
//!
//! Strides over the text's tokens in windows of `chunk_size` with a configurable
//! overlap. Mirrors Chonkie's `TokenChunker`.
//!
//! Chunk offsets are derived directly from token byte-spans in the **original
//! text** — never by re-decoding token groups and accumulating their lengths.
//! This makes the slice invariant `chunk.text == original[start..end]` hold even
//! when a window boundary lands next to a multi-byte character, fixing upstream
//! bug #629 by construction.

use crate::chunkers::{Chunk, ChunkError};
use crate::token_counter::TokenCounter;

/// How much consecutive chunks overlap.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Overlap {
    /// An absolute number of tokens.
    Tokens(usize),
    /// A fraction of `chunk_size` (e.g. `0.1` = 10%), truncated to whole tokens.
    Fraction(f64),
}

/// Splits text into fixed-size token windows with optional overlap.
#[derive(Debug, Clone)]
pub struct TokenChunker {
    chunk_size: usize,
    chunk_overlap: Overlap,
}

impl Default for TokenChunker {
    fn default() -> Self {
        Self::new()
    }
}

impl TokenChunker {
    /// New chunker with upstream defaults: `chunk_size = 2048`, `chunk_overlap = 0`.
    pub fn new() -> Self {
        Self {
            chunk_size: 2048,
            chunk_overlap: Overlap::Tokens(0),
        }
    }

    /// Set the number of tokens per chunk.
    pub fn chunk_size(mut self, n: usize) -> Self {
        self.chunk_size = n;
        self
    }

    /// Set overlap as an absolute number of tokens.
    pub fn chunk_overlap_tokens(mut self, n: usize) -> Self {
        self.chunk_overlap = Overlap::Tokens(n);
        self
    }

    /// Set overlap as a fraction of `chunk_size`.
    pub fn chunk_overlap_fraction(mut self, frac: f64) -> Self {
        self.chunk_overlap = Overlap::Fraction(frac);
        self
    }

    /// Chunk `text`, enumerating tokens with `counter`.
    ///
    /// Returns an error if the resolved stride (`chunk_size - overlap`) would be
    /// non-positive.
    pub fn chunk(&self, text: &str, counter: &dyn TokenCounter) -> Result<Vec<Chunk>, ChunkError> {
        if self.chunk_size == 0 {
            return Err(ChunkError::InvalidConfig("chunk_size must be > 0".into()));
        }
        let overlap = match self.chunk_overlap {
            Overlap::Tokens(n) => n,
            Overlap::Fraction(f) => {
                if !(0.0..1.0).contains(&f) {
                    return Err(ChunkError::InvalidConfig(
                        "chunk_overlap fraction must be in [0.0, 1.0)".into(),
                    ));
                }
                (f * self.chunk_size as f64) as usize
            }
        };
        if overlap >= self.chunk_size {
            return Err(ChunkError::InvalidConfig(
                "chunk_overlap must be < chunk_size (stride would be non-positive)".into(),
            ));
        }
        let stride = self.chunk_size - overlap;

        if text.trim().is_empty() {
            return Ok(vec![]);
        }

        let spans = counter.token_spans(text);
        let n = spans.len();
        if n == 0 {
            return Ok(vec![]);
        }

        let mut out = Vec::new();
        let mut start = 0usize;
        loop {
            let end = (start + self.chunk_size).min(n);
            out.push(Chunk {
                start: spans[start].0,
                end: spans[end - 1].1,
                token_count: end - start,
            });
            if end >= n {
                break;
            }
            start += stride;
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::token_counter::CharCounter;

    #[test]
    fn empty_and_whitespace_return_none() {
        assert!(
            TokenChunker::new()
                .chunk("", &CharCounter)
                .unwrap()
                .is_empty()
        );
        assert!(
            TokenChunker::new()
                .chunk("   ", &CharCounter)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn strides_with_overlap() {
        let text = "abcdefghij";
        let out = TokenChunker::new()
            .chunk_size(4)
            .chunk_overlap_tokens(1)
            .chunk(text, &CharCounter)
            .unwrap();
        assert_eq!(&text[out[0].start..out[0].end], "abcd");
        assert_eq!(&text[out[1].start..out[1].end], "defg"); // step = 3
        assert_eq!(out[0].token_count, 4);
    }

    #[test]
    fn small_text_single_chunk() {
        let text = "abc";
        let out = TokenChunker::new()
            .chunk_size(10)
            .chunk(text, &CharCounter)
            .unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(&text[out[0].start..out[0].end], "abc");
    }

    #[test]
    fn multibyte_slice_invariant() {
        // Regression for #629: boundaries never split a multi-byte character.
        let text = "a🩺bc🩺de";
        let out = TokenChunker::new()
            .chunk_size(2)
            .chunk(text, &CharCounter)
            .unwrap();
        let mut covered = String::new();
        for c in &out {
            assert!(text.is_char_boundary(c.start) && text.is_char_boundary(c.end));
            let s = &text[c.start..c.end];
            assert!(!s.is_empty());
            covered.push_str(s);
        }
        assert_eq!(covered, text); // no overlap, size divides cleanly-ish → full coverage
    }

    #[test]
    fn zero_stride_rejected_fraction() {
        let err = TokenChunker::new()
            .chunk_size(4)
            .chunk_overlap_fraction(1.0)
            .chunk("abcd", &CharCounter);
        assert!(err.is_err());
    }

    #[test]
    fn overlap_ge_size_rejected() {
        let err = TokenChunker::new()
            .chunk_size(4)
            .chunk_overlap_tokens(4)
            .chunk("abcd", &CharCounter);
        assert!(err.is_err());
    }
}
