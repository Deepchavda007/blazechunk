//! Semantic Double-Pass Merging (SDPM).
//!
//! SDPM is [`SemanticChunker`](crate::SemanticChunker) with the second (skip-window) merge
//! pass always on. The first pass groups sentences at similarity troughs; the second pass
//! looks ahead up to `skip_window + 1` groups and merges the current group with the most
//! similar one within reach — bridging a short off-topic digression back to the topic it
//! interrupted. This chunker shares the exact engine in [`crate::chunkers::semantic`]; the
//! only differences from `SemanticChunker` are the default `skip_window` (1 vs 0) and that
//! `skip_window == 0` is rejected (a zero skip window would make it a plain semantic chunker).
//!
//! Mirrors Chonkie's `SDPMChunker`. The embedder is injected as a `&dyn Embedder`.

use crate::chunkers::semantic::{SemanticParams, run_sdpm, sdpm_defaults, validate_sdpm};
use crate::chunkers::{Chunk, ChunkError};
use crate::embeddings::Embedder;
use crate::split::IncludeDelim;
use crate::token_counter::TokenCounter;

/// Groups sentences semantically, then merges skip-adjacent same-topic groups.
///
/// Mirrors Chonkie's `SDPMChunker`. See the module docs for the two-pass algorithm.
#[derive(Debug, Clone)]
pub struct SDPMChunker {
    params: SemanticParams,
}

impl Default for SDPMChunker {
    fn default() -> Self {
        Self::new()
    }
}

impl SDPMChunker {
    /// New chunker with upstream defaults — identical to [`SemanticChunker`](crate::SemanticChunker)
    /// but with `skip_window = 1` so the double-pass merge is active.
    pub fn new() -> Self {
        Self {
            params: sdpm_defaults(),
        }
    }

    /// Similarity percentile threshold in `(0, 1)`; also the raw cosine cutoff for the
    /// skip-merge pass.
    pub fn threshold(mut self, t: f64) -> Self {
        self.params.threshold = t;
        self
    }

    /// Set the token budget per chunk.
    pub fn chunk_size(mut self, n: usize) -> Self {
        self.params.chunk_size = n;
        self
    }

    /// Number of sentences in each similarity window.
    pub fn similarity_window(mut self, n: usize) -> Self {
        self.params.similarity_window = n;
        self
    }

    /// Minimum sentences between split points.
    pub fn min_sentences_per_chunk(mut self, n: usize) -> Self {
        self.params.min_sentences_per_chunk = n;
        self
    }

    /// Minimum characters per sentence (shorter fragments merge into neighbours).
    pub fn min_characters_per_sentence(mut self, n: usize) -> Self {
        self.params.min_characters_per_sentence = n;
        self
    }

    /// Replace the sentence delimiters.
    pub fn delim(mut self, delim: Vec<String>) -> Self {
        self.params.delim = delim;
        self
    }

    /// Set where the delimiter attaches (Prev/Next/None).
    pub fn include_delim(mut self, include: IncludeDelim) -> Self {
        self.params.include_delim = include;
        self
    }

    /// Number of groups to look ahead when merging (must be `>= 1`).
    pub fn skip_window(mut self, n: usize) -> Self {
        self.params.skip_window = n;
        self
    }

    /// Savitzky–Golay window length (odd; even values disable minima detection gracefully).
    pub fn filter_window(mut self, n: usize) -> Self {
        self.params.filter_window = n;
        self
    }

    /// Savitzky–Golay polynomial order (`< filter_window`).
    pub fn filter_polyorder(mut self, n: usize) -> Self {
        self.params.filter_polyorder = n;
        self
    }

    /// Tolerance for treating the first derivative as zero when detecting minima.
    pub fn filter_tolerance(mut self, t: f64) -> Self {
        self.params.filter_tolerance = t;
        self
    }

    /// Validate the configuration, returning the same error [`chunk`](Self::chunk) would.
    pub fn validate(&self) -> Result<(), ChunkError> {
        validate_sdpm(&self.params)
    }

    /// Chunk `text`, measuring size with `counter` and semantics with `embedder`.
    pub fn chunk(
        &self,
        text: &str,
        counter: &dyn TokenCounter,
        embedder: &dyn Embedder,
    ) -> Result<Vec<Chunk>, ChunkError> {
        run_sdpm(&self.params, text, counter, embedder)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::{HashEmbedder, TopicEmbedder};
    use crate::token_counter::CharCounter;

    fn assert_partition(text: &str, chunks: &[Chunk]) {
        let mut prev = 0;
        for c in chunks {
            assert_eq!(c.start, prev, "chunks not contiguous: {chunks:?}");
            assert!(text.is_char_boundary(c.start) && text.is_char_boundary(c.end));
            prev = c.end;
        }
        assert_eq!(prev, text.len(), "chunks do not cover the whole text");
    }

    #[test]
    fn default_skip_window_is_one() {
        assert_eq!(SDPMChunker::new().params.skip_window, 1);
    }

    #[test]
    fn skip_window_zero_errors() {
        let err = SDPMChunker::new().skip_window(0).chunk(
            "a. b. c.",
            &CharCounter,
            &HashEmbedder::new(8),
        );
        assert!(matches!(err, Err(ChunkError::InvalidConfig(_))));
    }

    #[test]
    fn whitespace_only_is_empty() {
        let out = SDPMChunker::new()
            .chunk("   ", &CharCounter, &HashEmbedder::new(8))
            .unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn partitions_and_preserves_bytes() {
        // A → B → A pattern: an interrupting topic between two same-topic spans.
        let a1 = "The cat sat on the mat here. A cat purrs softly now. My cat naps all day. ";
        let b = "The finance report rose sharply. Finance markets moved today. ";
        let a2 = "Another cat climbs the tall tree. The cat plays with yarn. A cat meows loudly. ";
        let text = format!("{a1}{b}{a2}");
        let out = SDPMChunker::new()
            .min_characters_per_sentence(1)
            .filter_window(3)
            .filter_polyorder(2)
            .chunk(
                &text,
                &CharCounter,
                &TopicEmbedder::new(&["cat", "finance"]),
            )
            .unwrap();
        assert!(!out.is_empty());
        assert_partition(&text, &out);
    }

    #[test]
    fn sdpm_matches_semantic_when_no_merge_applies() {
        // When no groups are similar enough to skip-merge, SDPM and a plain SemanticChunker
        // with the same settings must produce identical chunks (the second pass is a no-op).
        use crate::chunkers::SemanticChunker;
        let text = "The cat sat here now. A cat purrs softly. My cat naps all day. \
                    The finance report rose. Finance markets moved. Budgets were set. \
                    Birds fly south now. The bird sings loud. A bird builds nests.";
        let embedder = TopicEmbedder::new(&["cat", "finance", "bird"]);
        let cfg = |c: SemanticChunker| c.min_characters_per_sentence(1).threshold(0.99);
        let semantic = cfg(SemanticChunker::new())
            .chunk(text, &CharCounter, &embedder)
            .unwrap();
        let sdpm = SDPMChunker::new()
            .min_characters_per_sentence(1)
            .threshold(0.99)
            .skip_window(1)
            .chunk(text, &CharCounter, &embedder)
            .unwrap();
        assert_partition(text, &semantic);
        assert_partition(text, &sdpm);
        // Distinct topics never reach the 0.99 cosine cutoff, so no bridging happens and the
        // two chunkers agree.
        assert_eq!(semantic, sdpm);
    }
}
