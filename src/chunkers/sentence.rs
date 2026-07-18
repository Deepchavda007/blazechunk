//! Sentence-aware chunking.
//!
//! Splits text into sentences at sentence-ending delimiters, then greedily packs
//! whole sentences into chunks that fit the token budget — never sub-splitting a
//! sentence. Supports a minimum number of sentences per chunk and token overlap
//! between consecutive chunks. Mirrors Chonkie's `SentenceChunker`.
//!
//! Chunks may overlap each other (with `chunk_overlap > 0`), but each chunk is
//! always a valid contiguous slice of the original text.

use crate::chunkers::{Chunk, ChunkError, split_by_delimiters};
use crate::merge::find_merge_indices;
use crate::split::IncludeDelim;
use crate::token_counter::TokenCounter;

/// Groups sentences into token-budgeted chunks.
#[derive(Debug, Clone)]
pub struct SentenceChunker {
    chunk_size: usize,
    chunk_overlap: usize,
    min_sentences_per_chunk: usize,
    min_characters_per_sentence: usize,
    delim: Vec<String>,
    include_delim: IncludeDelim,
}

impl Default for SentenceChunker {
    fn default() -> Self {
        Self::new()
    }
}

impl SentenceChunker {
    /// New chunker with upstream defaults: `chunk_size = 2048`, `chunk_overlap = 0`,
    /// `min_sentences_per_chunk = 1`, `min_characters_per_sentence = 12`,
    /// `delim = [". ", "! ", "? ", "\n"]`, `include_delim = Prev`.
    pub fn new() -> Self {
        Self {
            chunk_size: 2048,
            chunk_overlap: 0,
            min_sentences_per_chunk: 1,
            min_characters_per_sentence: 12,
            delim: vec![". ".into(), "! ".into(), "? ".into(), "\n".into()],
            include_delim: IncludeDelim::Prev,
        }
    }

    /// Set the token budget per chunk.
    pub fn chunk_size(mut self, n: usize) -> Self {
        self.chunk_size = n;
        self
    }

    /// Set token overlap between consecutive chunks.
    pub fn chunk_overlap(mut self, n: usize) -> Self {
        self.chunk_overlap = n;
        self
    }

    /// Set the minimum number of sentences per chunk.
    pub fn min_sentences_per_chunk(mut self, n: usize) -> Self {
        self.min_sentences_per_chunk = n;
        self
    }

    /// Set the minimum characters per sentence (shorter fragments merge).
    pub fn min_characters_per_sentence(mut self, n: usize) -> Self {
        self.min_characters_per_sentence = n;
        self
    }

    /// Replace the sentence delimiters.
    pub fn delim(mut self, delim: Vec<String>) -> Self {
        self.delim = delim;
        self
    }

    /// Set where the delimiter attaches (Prev/Next/None).
    pub fn include_delim(mut self, include: IncludeDelim) -> Self {
        self.include_delim = include;
        self
    }

    /// Validate the configuration, returning the same error [`chunk`](Self::chunk)
    /// would. Single source of truth for config validity, shared by `chunk` and
    /// the binding layers so the rules can never drift.
    pub fn validate(&self) -> Result<(), ChunkError> {
        if self.chunk_size == 0 {
            return Err(ChunkError::InvalidConfig("chunk_size must be > 0".into()));
        }
        if self.chunk_overlap >= self.chunk_size {
            return Err(ChunkError::InvalidConfig(
                "chunk_overlap must be < chunk_size".into(),
            ));
        }
        if self.min_sentences_per_chunk == 0 {
            return Err(ChunkError::InvalidConfig(
                "min_sentences_per_chunk must be >= 1".into(),
            ));
        }
        if self.min_characters_per_sentence == 0 {
            return Err(ChunkError::InvalidConfig(
                "min_characters_per_sentence must be >= 1".into(),
            ));
        }
        Ok(())
    }

    /// Chunk `text`, measuring size with `counter`.
    ///
    /// Returns an error if the configuration is invalid (see [`validate`](Self::validate)).
    pub fn chunk(&self, text: &str, counter: &dyn TokenCounter) -> Result<Vec<Chunk>, ChunkError> {
        self.validate()?;
        if text.trim().is_empty() {
            return Ok(vec![]);
        }

        let sentences = split_by_delimiters(
            text,
            &self.delim,
            self.include_delim,
            self.min_characters_per_sentence,
        );
        if sentences.is_empty() {
            return Ok(vec![]);
        }

        let token_counts: Vec<usize> = sentences
            .iter()
            .map(|&(s, e)| counter.count(&text[s..e]))
            .collect();
        let n = sentences.len();

        let mut out = Vec::new();
        let mut pos = 0usize;
        while pos < n {
            // Greedily take as many whole sentences as fit the budget.
            let indices = find_merge_indices(&token_counts[pos..], self.chunk_size);
            let mut split_idx = if indices.is_empty() {
                n
            } else {
                (pos + indices[0]).min(n)
            };
            if split_idx <= pos {
                split_idx = (pos + 1).min(n);
            }

            // Enforce the minimum-sentences-per-chunk floor.
            if split_idx - pos < self.min_sentences_per_chunk {
                let desired = pos + self.min_sentences_per_chunk;
                split_idx = if desired <= n { desired } else { n };
            }

            let start = sentences[pos].0;
            let end = sentences[split_idx - 1].1;
            out.push(Chunk {
                start,
                end,
                // Recompute on the joined text (matches upstream _create_chunk).
                token_count: counter.count(&text[start..end]),
            });

            // Overlap: walk backward re-including tail sentences up to the budget.
            if self.chunk_overlap > 0 && split_idx < n {
                let mut overlap_idx = split_idx - 1;
                let mut overlap_tokens = 0usize;
                while overlap_idx > pos && overlap_tokens < self.chunk_overlap {
                    let next = overlap_tokens + token_counts[overlap_idx] + 1;
                    if next > self.chunk_overlap {
                        break;
                    }
                    overlap_tokens = next;
                    overlap_idx -= 1;
                }
                pos = overlap_idx + 1;
            } else {
                pos = split_idx;
            }
        }

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::token_counter::CharCounter;

    fn valid_slice(text: &str, c: &Chunk) {
        assert!(text.is_char_boundary(c.start) && text.is_char_boundary(c.end));
        assert!(c.start < c.end);
    }

    #[test]
    fn whitespace_only_empty() {
        assert!(
            SentenceChunker::new()
                .chunk("   \n ", &CharCounter)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn packs_sentences_within_budget() {
        let text = "One. Two. Three. Four. Five. ";
        let out = SentenceChunker::new()
            .chunk_size(10)
            .min_characters_per_sentence(1)
            .chunk(text, &CharCounter)
            .unwrap();
        assert!(out.len() >= 2);
        for c in &out {
            valid_slice(text, c);
            assert!(c.token_count <= 10 || (c.end - c.start) <= 10);
        }
    }

    #[test]
    fn min_sentences_respected() {
        let text = "A. B. C. D. E. F. ";
        let out = SentenceChunker::new()
            .chunk_size(3)
            .min_sentences_per_chunk(2)
            .min_characters_per_sentence(1)
            .chunk(text, &CharCounter)
            .unwrap();
        assert!(!out.is_empty());
        // First chunk must hold two sentences ("A. B. " = 6 chars) despite size=3.
        assert_eq!(out[0].end - out[0].start, 6);
    }

    #[test]
    fn overlap_reincludes_tail() {
        let text = "A. B. C. D. E. F. ";
        let out = SentenceChunker::new()
            .chunk_size(9)
            .chunk_overlap(4)
            .min_characters_per_sentence(1)
            .chunk(text, &CharCounter)
            .unwrap();
        assert!(out.len() >= 2);
        // Some consecutive pair overlaps (next starts before prev ends).
        let overlaps = out.windows(2).any(|w| w[1].start < w[0].end);
        assert!(overlaps, "expected overlapping chunks: {:?}", out);
    }

    #[test]
    fn oversized_single_sentence_stays_whole() {
        let text = "Thissentencehasnodelimitersandisverylong. Short. ";
        let out = SentenceChunker::new()
            .chunk_size(5)
            .min_characters_per_sentence(1)
            .chunk(text, &CharCounter)
            .unwrap();
        // The long sentence is emitted whole (never sub-split).
        assert!(out.iter().any(|c| c.token_count > 5));
        for c in &out {
            valid_slice(text, c);
        }
    }

    #[test]
    fn cjk_sentence_delimiter_valid_utf8() {
        let text = "第一句。第二句。第三句。";
        let out = SentenceChunker::new()
            .chunk_size(4)
            .min_characters_per_sentence(1)
            .delim(vec!["。".into()])
            .chunk(text, &CharCounter)
            .unwrap();
        for c in &out {
            assert!(std::str::from_utf8(&text.as_bytes()[c.start..c.end]).is_ok());
        }
    }

    #[test]
    fn zero_chunk_size_errors() {
        let err = SentenceChunker::new()
            .chunk_size(0)
            .chunk("A. B.", &CharCounter);
        assert!(matches!(err, Err(ChunkError::InvalidConfig(_))));
    }

    #[test]
    fn overlap_ge_size_errors() {
        let err = SentenceChunker::new()
            .chunk_size(5)
            .chunk_overlap(5)
            .chunk("A. B.", &CharCounter);
        assert!(matches!(err, Err(ChunkError::InvalidConfig(_))));
    }

    #[test]
    fn zero_min_sentences_errors() {
        let err = SentenceChunker::new()
            .min_sentences_per_chunk(0)
            .chunk("A. B.", &CharCounter);
        assert!(matches!(err, Err(ChunkError::InvalidConfig(_))));
    }

    #[test]
    fn zero_min_characters_errors() {
        let err = SentenceChunker::new()
            .min_characters_per_sentence(0)
            .chunk("A. B.", &CharCounter);
        assert!(matches!(err, Err(ChunkError::InvalidConfig(_))));
    }
}
