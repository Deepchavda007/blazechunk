//! Late chunking: recursive boundaries + whole-document contextual embeddings.
//!
//! "Late chunking" (Günther et al., Jina AI) embeds the **entire document once** so every
//! token vector carries full-document context, and only *then* splits into chunks — the
//! opposite order from embedding each chunk in isolation. This chunker:
//!
//! 1. computes chunk boundaries with a [`RecursiveChunker`](crate::RecursiveChunker)
//!    (byte offsets + per-chunk token counts), then
//! 2. asks the injected [`TokenEmbedder`] for one contextual vector per token across the
//!    whole text, and
//! 3. mean-pools the token vectors that fall within each chunk's token span.
//!
//! Mirrors Chonkie's `LateChunker`. Because the embedding *is* the deliverable, this chunker
//! returns [`LateChunk`] (byte offsets + `token_count` + `embedding`) rather than the plain
//! [`Chunk`], parallel to how [`TableChunker`](crate::TableChunker) has its own
//! [`TableChunk`](crate::TableChunk). The byte range still satisfies the slice invariant:
//! `late_chunk.text == original[start..end]`.

use crate::chunkers::recursive::{RecursiveChunker, RecursiveRules};
use crate::chunkers::{Chunk, ChunkError};
use crate::embeddings::TokenEmbedder;
use crate::token_counter::TokenCounter;

/// A chunk carrying its late-interaction embedding.
///
/// `start`/`end` are byte offsets into the original text (slice invariant holds).
/// `token_count` is the number of token embeddings pooled into `embedding` — after the
/// alignment/rebalance step it may differ slightly from the raw recursive token count when
/// the counter and token embedder disagree on tokenization (see the module docs).
#[derive(Debug, Clone, PartialEq)]
pub struct LateChunk {
    /// Byte offset of the chunk start in the original text.
    pub start: usize,
    /// Byte offset of the chunk end (exclusive) in the original text.
    pub end: usize,
    /// Number of token embeddings pooled into `embedding`.
    pub token_count: usize,
    /// Mean-pooled contextual embedding for this chunk's token span.
    pub embedding: Vec<f32>,
}

/// Chunks recursively, then attaches a mean-pooled whole-document ("late") embedding to
/// each chunk. Mirrors Chonkie's `LateChunker`. The token embedder is injected.
#[derive(Debug, Clone)]
pub struct LateChunker {
    chunk_size: usize,
    min_characters_per_chunk: usize,
    rules: RecursiveRules,
}

impl Default for LateChunker {
    fn default() -> Self {
        Self::new()
    }
}

impl LateChunker {
    /// New chunker with upstream defaults: `chunk_size = 2048`,
    /// `min_characters_per_chunk = 24`, and the default recursive rules.
    pub fn new() -> Self {
        Self {
            chunk_size: 2048,
            min_characters_per_chunk: 24,
            rules: RecursiveRules::default(),
        }
    }

    /// Set the token budget per chunk.
    pub fn chunk_size(mut self, n: usize) -> Self {
        self.chunk_size = n;
        self
    }

    /// Set the minimum characters per split (shorter fragments merge into neighbours).
    pub fn min_characters_per_chunk(mut self, n: usize) -> Self {
        self.min_characters_per_chunk = n;
        self
    }

    /// Replace the recursive delimiter hierarchy used for boundary detection.
    pub fn rules(mut self, rules: RecursiveRules) -> Self {
        self.rules = rules;
        self
    }

    /// Validate the configuration, returning the same error [`chunk`](Self::chunk) would.
    pub fn validate(&self) -> Result<(), ChunkError> {
        if self.chunk_size == 0 {
            return Err(ChunkError::InvalidConfig("chunk_size must be > 0".into()));
        }
        if self.min_characters_per_chunk == 0 {
            return Err(ChunkError::InvalidConfig(
                "min_characters_per_chunk must be >= 1".into(),
            ));
        }
        Ok(())
    }

    /// The [`RecursiveChunker`] this late chunker uses for boundary detection.
    fn recursive(&self) -> RecursiveChunker {
        RecursiveChunker::new()
            .chunk_size(self.chunk_size)
            .min_characters_per_chunk(self.min_characters_per_chunk)
            .rules(self.rules.clone())
    }

    /// Chunk `text`: recursive boundaries measured by `counter`, then late embeddings from
    /// `embedder`.
    ///
    /// Errors on invalid config, or if the token stream is unusably inconsistent with the
    /// recursive token counts (the sum of counts exceeds the number of token embeddings even
    /// after the fallback path).
    pub fn chunk(
        &self,
        text: &str,
        counter: &dyn TokenCounter,
        embedder: &dyn TokenEmbedder,
    ) -> Result<Vec<LateChunk>, ChunkError> {
        self.validate()?;
        if text.trim().is_empty() {
            return Ok(vec![]);
        }

        // 1. Recursive boundaries (byte offsets + per-chunk token counts).
        let chunks: Vec<Chunk> = self.recursive().chunk(text, counter)?;
        if chunks.is_empty() {
            return Ok(vec![]);
        }

        // 2. Whole-document contextual token embeddings.
        let mut token_embeddings = embedder.embed_as_tokens(text);
        let mut token_counts: Vec<usize> = chunks.iter().map(|c| c.token_count).collect();
        let mut total: usize = token_counts.iter().sum();

        // 3. Fallback: the token stream is shorter than the recursive counts imply (e.g. the
        //    embedder's tokenizer differs from the counter). Re-embed each chunk as a single
        //    vector and treat each chunk as one "token" (matches upstream).
        if token_embeddings.len() < total {
            token_embeddings = chunks
                .iter()
                .map(|c| embedder.embed(&text[c.start..c.end]))
                .collect();
            token_counts = vec![1; chunks.len()];
            total = chunks.len();
        }

        let m = token_embeddings.len();
        if total > m {
            return Err(ChunkError::InvalidConfig(format!(
                "token count sum ({total}) exceeds number of token embeddings ({m})"
            )));
        }
        // 4. Absorb any leftover tokens into the first and last chunk (matches upstream).
        if total < m {
            let diff = m - total;
            token_counts[0] += diff / 2;
            let last = token_counts.len() - 1;
            token_counts[last] += diff - diff / 2;
            total = m;
        }
        debug_assert_eq!(total, m);

        // 5. Mean-pool token embeddings within each chunk's token span.
        let dim = token_embeddings.iter().map(Vec::len).max().unwrap_or(0);
        let mut out = Vec::with_capacity(chunks.len());
        let mut cursor = 0usize;
        for (i, c) in chunks.iter().enumerate() {
            let span = token_counts[i];
            let embedding = mean_pool(&token_embeddings[cursor..cursor + span], dim);
            cursor += span;
            out.push(LateChunk {
                start: c.start,
                end: c.end,
                token_count: span,
                embedding,
            });
        }
        Ok(out)
    }
}

/// Mean of a slice of equal-dimension vectors. Returns a zero vector of length `dim` for an
/// empty span (a chunk with no tokens, which the rebalance step should already prevent).
fn mean_pool(vectors: &[Vec<f32>], dim: usize) -> Vec<f32> {
    if vectors.is_empty() || dim == 0 {
        return vec![0.0; dim];
    }
    let mut acc = vec![0.0f32; dim];
    for v in vectors {
        for (j, &x) in v.iter().enumerate() {
            acc[j] += x;
        }
    }
    let n = vectors.len() as f32;
    for x in &mut acc {
        *x /= n;
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::CharTokenEmbedder;
    use crate::token_counter::CharCounter;

    fn approx_eq(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-4)
    }

    #[test]
    fn whitespace_only_is_empty() {
        let out = LateChunker::new()
            .chunk("   \n ", &CharCounter, &CharTokenEmbedder::new(8))
            .unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn slice_invariant_and_contiguous() {
        let text = "Para one here.\n\nPara two is a little longer. It has two sentences.\n\nThird paragraph.";
        let embedder = CharTokenEmbedder::new(8);
        let out = LateChunker::new()
            .chunk_size(20)
            .min_characters_per_chunk(1)
            .chunk(text, &CharCounter, &embedder)
            .unwrap();
        assert!(out.len() > 1, "expected multiple chunks: {out:?}");
        let mut prev = 0;
        for c in &out {
            assert_eq!(c.start, prev, "not contiguous");
            assert!(text.is_char_boundary(c.start) && text.is_char_boundary(c.end));
            prev = c.end;
        }
        assert_eq!(prev, text.len(), "chunks do not cover the whole text");
    }

    #[test]
    fn embedding_is_mean_of_span_tokens() {
        // With CharCounter + CharTokenEmbedder the token stream length equals the character
        // count exactly, so no rebalancing occurs and each chunk's embedding must equal the
        // mean of that chunk's per-character token vectors.
        let text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda";
        let embedder = CharTokenEmbedder::new(8);
        let out = LateChunker::new()
            .chunk_size(8)
            .min_characters_per_chunk(1)
            .chunk(text, &CharCounter, &embedder)
            .unwrap();
        assert!(out.len() > 1);

        let all_tokens = embedder.embed_as_tokens(text);
        // token_counts should be exactly the char counts (no rebalance) → sum == chars.
        assert_eq!(
            out.iter().map(|c| c.token_count).sum::<usize>(),
            all_tokens.len()
        );

        // Rebuild the per-chunk mean directly from the chunk's char span and compare.
        let mut cursor = 0usize;
        for c in &out {
            let expected = mean_pool(&all_tokens[cursor..cursor + c.token_count], 8);
            assert!(
                approx_eq(&c.embedding, &expected),
                "embedding mismatch: {:?} vs {:?}",
                c.embedding,
                expected
            );
            cursor += c.token_count;
        }
    }

    #[test]
    fn small_text_single_chunk_has_embedding() {
        let text = "Just one short line.";
        let embedder = CharTokenEmbedder::new(8);
        let out = LateChunker::new()
            .min_characters_per_chunk(1)
            .chunk(text, &CharCounter, &embedder)
            .unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].start, 0);
        assert_eq!(out[0].end, text.len());
        assert_eq!(out[0].embedding.len(), 8);
        assert_eq!(out[0].token_count, text.chars().count());
    }

    #[test]
    fn fallback_when_token_stream_too_short() {
        // A token embedder that returns FEWER token vectors than the char count forces the
        // fallback path: each chunk is re-embedded as a single vector and counts become 1.
        struct ShortEmbedder;
        impl TokenEmbedder for ShortEmbedder {
            fn embed_as_tokens(&self, _text: &str) -> Vec<Vec<f32>> {
                // Deliberately far too few tokens.
                vec![vec![1.0, 2.0]]
            }
            fn embed(&self, text: &str) -> Vec<f32> {
                vec![text.len() as f32, 0.0]
            }
        }
        let text = "First sentence here. Second sentence here. Third sentence here.";
        let out = LateChunker::new()
            .chunk_size(5)
            .min_characters_per_chunk(1)
            .chunk(text, &CharCounter, &ShortEmbedder)
            .unwrap();
        assert!(!out.is_empty());
        // Every chunk got a fallback single-vector embedding of dim 2 and token_count 1.
        for c in &out {
            assert_eq!(c.token_count, 1);
            assert_eq!(c.embedding.len(), 2);
            assert!((c.embedding[0] - (c.end - c.start) as f32).abs() < 1e-6);
        }
    }

    #[test]
    fn rebalance_absorbs_extra_tokens() {
        // A token embedder that returns MORE tokens than the counter's count triggers the
        // rebalance branch (extra tokens split between first and last chunk).
        struct LongEmbedder {
            per_char: usize,
        }
        impl TokenEmbedder for LongEmbedder {
            fn embed_as_tokens(&self, text: &str) -> Vec<Vec<f32>> {
                // `per_char` vectors per character → total = chars * per_char > chars.
                let mut v = Vec::new();
                for (i, _) in text.chars().enumerate() {
                    for k in 0..self.per_char {
                        v.push(vec![(i + k) as f32]);
                    }
                }
                v
            }
            fn embed(&self, _text: &str) -> Vec<f32> {
                vec![0.0]
            }
        }
        let text = "one two three four five six seven eight";
        let embedder = LongEmbedder { per_char: 2 };
        let out = LateChunker::new()
            .chunk_size(6)
            .min_characters_per_chunk(1)
            .chunk(text, &CharCounter, &embedder)
            .unwrap();
        assert!(out.len() > 1);
        // token_counts sum to the (larger) token-embedding count, not the char count.
        let total: usize = out.iter().map(|c| c.token_count).sum();
        assert_eq!(total, text.chars().count() * 2);
        for c in &out {
            assert_eq!(c.embedding.len(), 1);
        }
    }

    #[test]
    fn zero_chunk_size_errors() {
        let err = LateChunker::new().chunk_size(0).chunk(
            "hi there friend",
            &CharCounter,
            &CharTokenEmbedder::new(8),
        );
        assert!(matches!(err, Err(ChunkError::InvalidConfig(_))));
    }
}
