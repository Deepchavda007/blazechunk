//! Embedding injection for the semantic chunkers.
//!
//! The [`SemanticChunker`](crate::SemanticChunker), [`SDPMChunker`](crate::SDPMChunker),
//! and [`LateChunker`](crate::LateChunker) turn text into vectors and reason about their
//! similarity — but the core crate has **no ML model**. The embedder is the *injected*
//! dependency that defines what a vector is, exactly like [`TokenCounter`] defines what a
//! token is: chunkers accept `&dyn Embedder` / `&dyn TokenEmbedder` and never construct a
//! model themselves.
//!
//! [`TokenCounter`]: crate::TokenCounter
//!
//! Two traits, because the two use cases are genuinely different:
//! - [`Embedder`]: one vector per *text* (a sentence or a window of sentences). Drives the
//!   window-vs-sentence similarity curve in Semantic/SDPM.
//! - [`TokenEmbedder`]: one vector per *token* across a whole document (contextual "late
//!   interaction" embeddings), mean-pooled per chunk in Late.
//!
//! The bindings implement these traits over a Python callable / JS function; the core and
//! its tests implement them deterministically (see the test embedders in this module).

/// Cosine similarity between two vectors.
///
/// Returns `0.0` when the lengths differ or either vector has zero norm — the same
/// degenerate-case handling `savgol::windowed_cross_similarity` uses internally, so the
/// similarity curve never contains `NaN`.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na <= 0.0 || nb <= 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// Produces a dense embedding vector for each input text.
///
/// Injected into [`SemanticChunker`](crate::SemanticChunker) and
/// [`SDPMChunker`](crate::SDPMChunker). Implementations must return one vector per input,
/// in order, all of the same dimension. The core never loads a model — the binding layer
/// (or a test) supplies one.
pub trait Embedder {
    /// Embed each text into a dense vector. `embed_batch(texts).len() == texts.len()`, and
    /// every returned vector has the same length.
    fn embed_batch(&self, texts: &[&str]) -> Vec<Vec<f32>>;

    /// Embed a single text. Defaults to a one-element [`embed_batch`](Embedder::embed_batch).
    fn embed(&self, text: &str) -> Vec<f32> {
        self.embed_batch(std::slice::from_ref(&text))
            .into_iter()
            .next()
            .unwrap_or_default()
    }

    /// Similarity between two embeddings. Defaults to [`cosine_similarity`].
    fn similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        cosine_similarity(a, b)
    }
}

/// Produces one contextual embedding per token for a whole document.
///
/// Injected into [`LateChunker`](crate::LateChunker). "Late chunking" embeds the entire
/// document once so each token vector carries full-document context, then mean-pools the
/// token vectors that fall within each chunk. `embed_as_tokens(text)` returns the per-token
/// vectors in document order; `embed` is the single-vector fallback used when the token
/// stream is shorter than the chunkers' token budget could account for.
pub trait TokenEmbedder {
    /// One embedding vector per token across the whole `text`, in document order.
    fn embed_as_tokens(&self, text: &str) -> Vec<Vec<f32>>;

    /// Single-vector embedding for a text span (fallback / re-embed path).
    fn embed(&self, text: &str) -> Vec<f32>;
}

// A blanket-ish convenience: anything that is an `Embedder` reference can also act as a
// no-op is intentionally NOT provided — the two traits are distinct on purpose so a caller
// must decide which contract it satisfies.

// ============================================================================
// Deterministic embedders (available to the whole crate's tests, and handy as
// dependency-free defaults). Kept `pub` so binding-crate tests can reuse them.
// ============================================================================

/// A deterministic, dependency-free [`Embedder`] that hashes character trigrams into a
/// fixed-dimension vector.
///
/// Not semantically meaningful, but stable and non-degenerate — ideal for asserting
/// structural invariants (byte-preservation, contiguity, slice invariant) without a model.
#[derive(Debug, Clone, Copy)]
pub struct HashEmbedder {
    dim: usize,
}

impl HashEmbedder {
    /// New hash embedder producing `dim`-dimensional vectors.
    pub fn new(dim: usize) -> Self {
        Self { dim: dim.max(1) }
    }
}

impl Default for HashEmbedder {
    fn default() -> Self {
        Self::new(64)
    }
}

impl Embedder for HashEmbedder {
    fn embed_batch(&self, texts: &[&str]) -> Vec<Vec<f32>> {
        texts
            .iter()
            .map(|t| {
                let mut v = vec![0.0f32; self.dim];
                let bytes = t.as_bytes();
                if bytes.is_empty() {
                    // Avoid an all-zero (zero-norm) vector so cosine stays defined.
                    v[0] = 1.0;
                    return v;
                }
                // Hash overlapping byte trigrams (with padding) into buckets.
                let n = bytes.len();
                for i in 0..n {
                    let b0 = bytes[i] as u64;
                    let b1 = *bytes.get(i + 1).unwrap_or(&0) as u64;
                    let b2 = *bytes.get(i + 2).unwrap_or(&0) as u64;
                    let h =
                        (b0.wrapping_mul(31).wrapping_add(b1).wrapping_mul(31)).wrapping_add(b2);
                    let idx = (h % self.dim as u64) as usize;
                    v[idx] += 1.0;
                }
                v
            })
            .collect()
    }
}

/// A deterministic [`Embedder`] whose vectors reflect which *marker keywords* appear in a
/// text, so tests can construct documents with a known semantic structure.
///
/// Each keyword owns one dimension. A text's vector counts keyword hits per dimension;
/// texts sharing keywords point in near-parallel directions (high cosine), texts with
/// disjoint keywords are near-orthogonal (low cosine). A tiny uniform bias keeps
/// keyword-free texts from being zero-norm. This lets a test assert that a two-topic
/// document splits at the topic boundary and that SDPM re-merges skip-adjacent same-topic
/// groups.
#[derive(Debug, Clone)]
pub struct TopicEmbedder {
    keywords: Vec<String>,
}

impl TopicEmbedder {
    /// New topic embedder. Each keyword (lowercased, substring-matched) owns one dimension.
    pub fn new(keywords: &[&str]) -> Self {
        Self {
            keywords: keywords.iter().map(|k| k.to_lowercase()).collect(),
        }
    }
}

impl Embedder for TopicEmbedder {
    fn embed_batch(&self, texts: &[&str]) -> Vec<Vec<f32>> {
        let dim = self.keywords.len() + 1;
        texts
            .iter()
            .map(|t| {
                let lower = t.to_lowercase();
                let mut v = vec![0.0f32; dim];
                // Uniform bias in the final dimension keeps every vector non-zero.
                v[dim - 1] = 0.05;
                for (k, kw) in self.keywords.iter().enumerate() {
                    let mut count = 0.0f32;
                    let mut start = 0;
                    while let Some(pos) = lower[start..].find(kw.as_str()) {
                        count += 1.0;
                        start += pos + kw.len().max(1);
                        if start >= lower.len() {
                            break;
                        }
                    }
                    v[k] = count;
                }
                v
            })
            .collect()
    }
}

/// A deterministic [`TokenEmbedder`] that emits one vector per Unicode code point, so its
/// token stream length equals [`CharCounter`](crate::CharCounter)'s token count exactly —
/// giving LateChunker tests exact token/embedding alignment.
///
/// Each token vector encodes the byte value of the code point's first byte plus its
/// position, so pooled chunk embeddings differ across chunks and can be checked against a
/// hand-computed mean.
#[derive(Debug, Clone, Copy)]
pub struct CharTokenEmbedder {
    dim: usize,
}

impl CharTokenEmbedder {
    /// New per-code-point token embedder producing `dim`-dimensional vectors.
    pub fn new(dim: usize) -> Self {
        Self { dim: dim.max(2) }
    }
}

impl Default for CharTokenEmbedder {
    fn default() -> Self {
        Self::new(8)
    }
}

impl TokenEmbedder for CharTokenEmbedder {
    fn embed_as_tokens(&self, text: &str) -> Vec<Vec<f32>> {
        text.chars()
            .enumerate()
            .map(|(i, ch)| {
                let mut v = vec![0.0f32; self.dim];
                v[0] = (ch as u32 % 97) as f32;
                v[1] = (i % 13) as f32;
                v
            })
            .collect()
    }

    fn embed(&self, text: &str) -> Vec<f32> {
        // Mean of the per-token vectors (or a non-zero sentinel for empty text).
        let toks = self.embed_as_tokens(text);
        if toks.is_empty() {
            let mut v = vec![0.0f32; self.dim];
            v[0] = 1.0;
            return v;
        }
        let mut acc = vec![0.0f32; self.dim];
        for t in &toks {
            for (j, &x) in t.iter().enumerate() {
                acc[j] += x;
            }
        }
        for x in &mut acc {
            *x /= toks.len() as f32;
        }
        acc
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_identical_is_one() {
        let a = vec![1.0, 2.0, 3.0];
        assert!((cosine_similarity(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_orthogonal_is_zero() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn cosine_degenerate_cases_are_zero() {
        assert_eq!(cosine_similarity(&[], &[]), 0.0);
        assert_eq!(cosine_similarity(&[1.0], &[1.0, 2.0]), 0.0);
        assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
    }

    #[test]
    fn hash_embedder_is_deterministic_and_nonzero() {
        let e = HashEmbedder::new(32);
        let a = e.embed("hello world");
        let b = e.embed("hello world");
        assert_eq!(a, b);
        assert_eq!(a.len(), 32);
        assert!(a.iter().any(|&x| x > 0.0));
        // Empty text still yields a defined (non-zero-norm) vector.
        let empty = e.embed("");
        assert!(cosine_similarity(&empty, &empty).abs() > 0.0);
    }

    #[test]
    fn embed_default_matches_batch() {
        let e = HashEmbedder::new(16);
        let single = e.embed("abc");
        let batched = e.embed_batch(&["abc"]);
        assert_eq!(single, batched[0]);
    }

    #[test]
    fn topic_embedder_separates_topics() {
        let e = TopicEmbedder::new(&["cat", "finance"]);
        let cat1 = e.embed("the cat sat");
        let cat2 = e.embed("a cat purrs");
        let fin = e.embed("the finance report");
        // Same topic → high similarity; different topic → low.
        assert!(cosine_similarity(&cat1, &cat2) > 0.9);
        assert!(cosine_similarity(&cat1, &fin) < 0.3);
    }

    #[test]
    fn char_token_embedder_aligns_with_char_count() {
        use crate::token_counter::{CharCounter, TokenCounter};
        let text = "héllo 世界"; // mixed ASCII + multibyte
        let e = CharTokenEmbedder::new(8);
        let toks = e.embed_as_tokens(text);
        assert_eq!(toks.len(), CharCounter.count(text));
        assert_eq!(toks.len(), CharCounter.token_spans(text).len());
    }
}
