//! Semantic chunking via similarity-trough detection.
//!
//! Splits text into sentences, embeds a sliding *window* of sentences and the sentence that
//! immediately follows it, and measures their cosine similarity. A drop in that similarity
//! curve marks a topic shift. Local minima are found with a Savitzky–Golay filter
//! (`savgol::find_local_minima_interpolated`) and kept when they fall below a percentile
//! threshold (`savgol::filter_split_indices`) — the exact primitives upstream Chonkie's
//! `SemanticChunker` already delegates to this crate for. Sentences between split points are
//! grouped, optionally merged across a *skip window* (the double-pass merge that powers
//! [`SDPMChunker`](crate::SDPMChunker)), and finally split so no group exceeds the token
//! budget.
//!
//! The embedder is injected as a `&dyn Embedder`, mirroring how `TokenCounter` is injected —
//! the core never loads a model. Chunks partition the text (no overlap), so byte-preservation,
//! contiguity, and the slice invariant (`chunk.text == original[start..end]`) all hold by
//! construction.

use crate::chunkers::{Chunk, ChunkError, split_by_delimiters};
use crate::embeddings::Embedder;
use crate::savgol::{filter_split_indices, find_local_minima_interpolated};
use crate::split::IncludeDelim;
use crate::token_counter::TokenCounter;

/// Shared configuration + engine for the semantic chunkers.
///
/// Both [`SemanticChunker`] and [`SDPMChunker`](crate::SDPMChunker) hold one of these; the
/// only difference between them is the default `skip_window` (0 vs 1) and SDPM's stricter
/// validation. Keeping the algorithm here means the double-pass logic lives in exactly one
/// place.
#[derive(Debug, Clone)]
pub(crate) struct SemanticParams {
    pub threshold: f64,
    pub chunk_size: usize,
    pub similarity_window: usize,
    pub min_sentences_per_chunk: usize,
    pub min_characters_per_sentence: usize,
    pub delim: Vec<String>,
    pub include_delim: IncludeDelim,
    pub skip_window: usize,
    pub filter_window: usize,
    pub filter_polyorder: usize,
    pub filter_tolerance: f64,
}

impl SemanticParams {
    fn defaults() -> Self {
        Self {
            threshold: 0.8,
            chunk_size: 2048,
            similarity_window: 3,
            min_sentences_per_chunk: 1,
            min_characters_per_sentence: 24,
            delim: vec![". ".into(), "! ".into(), "? ".into(), "\n".into()],
            include_delim: IncludeDelim::Prev,
            skip_window: 0,
            filter_window: 5,
            filter_polyorder: 3,
            filter_tolerance: 0.2,
        }
    }

    /// Validate the configuration. `require_skip` (SDPM) additionally rejects `skip_window == 0`.
    fn validate(&self, require_skip: bool) -> Result<(), ChunkError> {
        if self.chunk_size == 0 {
            return Err(ChunkError::InvalidConfig("chunk_size must be > 0".into()));
        }
        if self.similarity_window == 0 {
            return Err(ChunkError::InvalidConfig(
                "similarity_window must be > 0".into(),
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
        if !(self.threshold > 0.0 && self.threshold < 1.0) {
            return Err(ChunkError::InvalidConfig(
                "threshold must be between 0 and 1 (exclusive)".into(),
            ));
        }
        if self.filter_window == 0 {
            return Err(ChunkError::InvalidConfig(
                "filter_window must be > 0".into(),
            ));
        }
        if self.filter_polyorder >= self.filter_window {
            return Err(ChunkError::InvalidConfig(
                "filter_polyorder must be < filter_window".into(),
            ));
        }
        if !(self.filter_tolerance > 0.0 && self.filter_tolerance < 1.0) {
            return Err(ChunkError::InvalidConfig(
                "filter_tolerance must be between 0 and 1 (exclusive)".into(),
            ));
        }
        if require_skip && self.skip_window == 0 {
            return Err(ChunkError::InvalidConfig(
                "skip_window must be >= 1 for SDPM".into(),
            ));
        }
        Ok(())
    }

    /// Run the full semantic pipeline.
    fn run(
        &self,
        text: &str,
        counter: &dyn TokenCounter,
        embedder: &dyn Embedder,
    ) -> Result<Vec<Chunk>, ChunkError> {
        if text.trim().is_empty() {
            return Ok(vec![]);
        }

        // 1. Sentences as a contiguous byte-span partition (multibyte-safe: #536).
        let sentences = split_by_delimiters(
            text,
            &self.delim,
            self.include_delim,
            self.min_characters_per_sentence,
        );
        if sentences.is_empty() {
            return Ok(vec![]);
        }
        let n = sentences.len();

        // Per-sentence token counts (summed to size groups, matching upstream).
        let sent_tokens: Vec<usize> = sentences
            .iter()
            .map(|&(s, e)| counter.count(&text[s..e]))
            .collect();

        // 2. Too few sentences to form a window → everything is one chunk.
        if n <= self.similarity_window {
            return Ok(vec![self.make_chunk(&sentences, &sent_tokens, 0, n)]);
        }

        // 3. Similarity curve: window (sentences[i..i+w]) vs the next sentence.
        let similarities = self.similarities(text, &sentences, embedder);

        // 4. Boundaries between sentence groups.
        let boundaries = self.split_boundaries(&similarities, n);

        // 5. Group sentences between consecutive boundaries (contiguous ranges).
        let mut groups: Vec<(usize, usize)> = boundaries
            .windows(2)
            .map(|w| (w[0], w[1]))
            .filter(|&(a, b)| b > a)
            .collect();

        // 6. Optional double-pass skip-merge.
        if self.skip_window > 0 && groups.len() > 1 {
            groups = self.skip_and_merge(text, &sentences, &groups, embedder);
        }

        // 7. Split any group that exceeds the token budget, at sentence boundaries.
        let final_groups = self.split_oversized(&groups, &sent_tokens);

        // 8. Emit chunks.
        Ok(final_groups
            .into_iter()
            .map(|(gs, ge)| self.make_chunk(&sentences, &sent_tokens, gs, ge))
            .collect())
    }

    /// Cosine similarity between each sentence window and the sentence that follows it.
    /// Length is `n - similarity_window`.
    fn similarities(
        &self,
        text: &str,
        sentences: &[(usize, usize)],
        embedder: &dyn Embedder,
    ) -> Vec<f64> {
        let n = sentences.len();
        let w = self.similarity_window;
        let count = n - w;

        // Window i = concatenation of sentences[i..i+w]; since sentences are contiguous
        // byte spans, that is exactly text[sentences[i].0 .. sentences[i+w-1].1].
        let window_texts: Vec<&str> = (0..count)
            .map(|i| &text[sentences[i].0..sentences[i + w - 1].1])
            .collect();
        let sentence_texts: Vec<&str> = (0..count)
            .map(|i| &text[sentences[i + w].0..sentences[i + w].1])
            .collect();

        let window_embs = embedder.embed_batch(&window_texts);
        let sentence_embs = embedder.embed_batch(&sentence_texts);

        (0..count)
            .map(|i| {
                let a = window_embs.get(i).map(Vec::as_slice).unwrap_or(&[]);
                let b = sentence_embs.get(i).map(Vec::as_slice).unwrap_or(&[]);
                embedder.similarity(a, b) as f64
            })
            .collect()
    }

    /// Turn the similarity curve into sentence-index boundaries `[0, .., n]`.
    fn split_boundaries(&self, similarities: &[f64], n: usize) -> Vec<usize> {
        let internal = self.internal_split_indices(similarities);

        let mut boundaries = Vec::with_capacity(internal.len() + 2);
        boundaries.push(0);
        for &b in &internal {
            let b = b.min(n);
            if b > *boundaries.last().unwrap() && b < n {
                boundaries.push(b);
            }
        }
        boundaries.push(n);
        boundaries
    }

    /// Internal split points (in sentence-index space): similarity minima that pass the
    /// percentile threshold, offset by `similarity_window`. Empty when there are too few
    /// samples for the filter, no minima are found, or all are filtered out.
    fn internal_split_indices(&self, similarities: &[f64]) -> Vec<usize> {
        // Need at least `filter_window` samples for the Savitzky–Golay filter.
        if similarities.len() < self.filter_window {
            return Vec::new();
        }
        let Some(minima) = find_local_minima_interpolated(
            similarities,
            self.filter_window,
            self.filter_polyorder,
            self.filter_tolerance,
        ) else {
            return Vec::new();
        };
        if minima.indices.is_empty() {
            return Vec::new();
        }
        let filtered = filter_split_indices(
            &minima.indices,
            &minima.values,
            self.threshold,
            self.min_sentences_per_chunk,
        );
        filtered
            .indices
            .iter()
            .map(|&i| i + self.similarity_window)
            .collect()
    }

    /// Best-candidate skip-merge (upstream `main` `_skip_and_merge`).
    ///
    /// For each group, look ahead up to `skip_window + 1` groups and merge with the single
    /// most-similar group whose cosine similarity meets `threshold`, absorbing everything in
    /// between. This reconnects semantically related content separated by a short digression.
    fn skip_and_merge(
        &self,
        text: &str,
        sentences: &[(usize, usize)],
        groups: &[(usize, usize)],
        embedder: &dyn Embedder,
    ) -> Vec<(usize, usize)> {
        let group_text = |g: &(usize, usize)| &text[sentences[g.0].0..sentences[g.1 - 1].1];
        let texts: Vec<&str> = groups.iter().map(group_text).collect();
        let embs = embedder.embed_batch(&texts);

        let mut out = Vec::new();
        let mut i = 0usize;
        let len = groups.len();
        while i < len {
            if i == len - 1 {
                out.push(groups[i]);
                break;
            }
            let skip_index = (i + self.skip_window + 1).min(len - 1);
            let mut best_sim = -1.0f32;
            let mut best_idx: Option<usize> = None;
            for j in (i + 1)..=skip_index {
                let sim = embedder.similarity(&embs[i], &embs[j]);
                if sim as f64 >= self.threshold && sim > best_sim {
                    best_sim = sim;
                    best_idx = Some(j);
                }
            }
            match best_idx {
                Some(j) => {
                    // Merge groups[i..=j] into one contiguous sentence range.
                    out.push((groups[i].0, groups[j].1));
                    i = j + 1;
                }
                None => {
                    out.push(groups[i]);
                    i += 1;
                }
            }
        }
        out
    }

    /// Split groups whose summed token count exceeds `chunk_size`, greedily, at sentence
    /// boundaries (upstream `_split_groups`). Groups stay contiguous.
    fn split_oversized(
        &self,
        groups: &[(usize, usize)],
        sent_tokens: &[usize],
    ) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        for &(gs, ge) in groups {
            let total: usize = sent_tokens[gs..ge].iter().sum();
            if total <= self.chunk_size {
                out.push((gs, ge));
                continue;
            }
            let mut cur_start = gs;
            let mut cur_tokens = 0usize;
            // Absolute sentence indices matter here (they become chunk ranges), so index
            // `sent_tokens` directly rather than enumerating a sub-slice.
            #[allow(clippy::needless_range_loop)]
            for s in gs..ge {
                if cur_tokens + sent_tokens[s] <= self.chunk_size || s == cur_start {
                    // Always keep at least one sentence per sub-group, even if a single
                    // sentence alone exceeds the budget (never sub-split a sentence).
                    cur_tokens += sent_tokens[s];
                } else {
                    out.push((cur_start, s));
                    cur_start = s;
                    cur_tokens = sent_tokens[s];
                }
            }
            if cur_start < ge {
                out.push((cur_start, ge));
            }
        }
        out
    }

    /// Build a chunk spanning `sentences[gs..ge]` with a summed token count.
    fn make_chunk(
        &self,
        sentences: &[(usize, usize)],
        sent_tokens: &[usize],
        gs: usize,
        ge: usize,
    ) -> Chunk {
        Chunk {
            start: sentences[gs].0,
            end: sentences[ge - 1].1,
            token_count: sent_tokens[gs..ge].iter().sum(),
        }
    }
}

/// Groups sentences into token-budgeted chunks at semantic-similarity troughs.
///
/// Mirrors Chonkie's `SemanticChunker`. The embedder is injected; see the module docs.
#[derive(Debug, Clone)]
pub struct SemanticChunker {
    params: SemanticParams,
}

impl Default for SemanticChunker {
    fn default() -> Self {
        Self::new()
    }
}

impl SemanticChunker {
    /// New chunker with upstream defaults: `threshold = 0.8`, `chunk_size = 2048`,
    /// `similarity_window = 3`, `min_sentences_per_chunk = 1`,
    /// `min_characters_per_sentence = 24`, `delim = [". ", "! ", "? ", "\n"]`,
    /// `include_delim = Prev`, `skip_window = 0`, `filter_window = 5`,
    /// `filter_polyorder = 3`, `filter_tolerance = 0.2`.
    pub fn new() -> Self {
        Self {
            params: SemanticParams::defaults(),
        }
    }

    /// Similarity percentile threshold in `(0, 1)`; lower keeps fewer split points.
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

    /// Enable the double-pass skip-merge (0 disables). See [`SDPMChunker`](crate::SDPMChunker).
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
        self.params.validate(false)
    }

    /// Chunk `text`, measuring size with `counter` and semantics with `embedder`.
    pub fn chunk(
        &self,
        text: &str,
        counter: &dyn TokenCounter,
        embedder: &dyn Embedder,
    ) -> Result<Vec<Chunk>, ChunkError> {
        self.params.validate(false)?;
        self.params.run(text, counter, embedder)
    }
}

/// Fresh defaults for the SDPM chunker (Semantic defaults + `skip_window = 1`).
pub(crate) fn sdpm_defaults() -> SemanticParams {
    let mut p = SemanticParams::defaults();
    p.skip_window = 1;
    p
}

/// Run the engine on SDPM's behalf with SDPM's stricter validation.
pub(crate) fn run_sdpm(
    params: &SemanticParams,
    text: &str,
    counter: &dyn TokenCounter,
    embedder: &dyn Embedder,
) -> Result<Vec<Chunk>, ChunkError> {
    params.validate(true)?;
    params.run(text, counter, embedder)
}

/// Validate on SDPM's behalf.
pub(crate) fn validate_sdpm(params: &SemanticParams) -> Result<(), ChunkError> {
    params.validate(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::{HashEmbedder, TopicEmbedder};
    use crate::token_counter::CharCounter;

    /// Chunks partition the text: contiguous, cover everything, valid UTF-8 slices.
    fn assert_partition(text: &str, chunks: &[Chunk]) {
        let mut prev = 0;
        for c in chunks {
            assert_eq!(c.start, prev, "chunks not contiguous: {chunks:?}");
            assert!(text.is_char_boundary(c.start) && text.is_char_boundary(c.end));
            prev = c.end;
        }
        assert_eq!(prev, text.len(), "chunks do not cover the whole text");
    }

    fn topic_doc() -> String {
        // Two clearly separated topics, each with enough sentences to form windows and to
        // give the Savitzky–Golay filter (window 5) enough samples.
        let cats = [
            "The cat sat on the warm windowsill. ",
            "A cat loves to nap in the sun. ",
            "My cat chases the red laser toy. ",
            "The cat purrs when it is happy. ",
            "Every cat needs a scratching post. ",
        ];
        let finance = [
            "The finance report shows rising revenue. ",
            "Quarterly finance targets were exceeded. ",
            "Investors watch the finance markets closely. ",
            "The finance team approved the new budget. ",
            "Strong finance growth drove the stock up. ",
        ];
        let mut s = String::new();
        for c in cats {
            s.push_str(c);
        }
        for f in finance {
            s.push_str(f);
        }
        s
    }

    #[test]
    fn whitespace_only_is_empty() {
        let out = SemanticChunker::new()
            .chunk("   \n\t ", &CharCounter, &HashEmbedder::new(32))
            .unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn few_sentences_single_chunk() {
        let text = "One sentence only here.";
        let out = SemanticChunker::new()
            .chunk(text, &CharCounter, &HashEmbedder::new(32))
            .unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].start, 0);
        assert_eq!(out[0].end, text.len());
    }

    #[test]
    fn partitions_and_preserves_bytes() {
        let text = topic_doc();
        let out = SemanticChunker::new()
            .min_characters_per_sentence(1)
            .chunk(&text, &CharCounter, &HashEmbedder::new(64))
            .unwrap();
        assert!(!out.is_empty());
        assert_partition(&text, &out);
        // Slice invariant.
        for c in &out {
            let _ = &text[c.start..c.end];
        }
    }

    #[test]
    fn splits_at_topic_boundary() {
        // The five cat sentences then five finance sentences give a similarity curve with a
        // sharp trough exactly at the topic change. `filter_tolerance` is raised to 0.5 here
        // because this synthetic topic embedder produces a much sharper V than real
        // embeddings (whose gentler valleys the 0.2 default is tuned for); the algorithm is
        // unchanged. The detected minimum lands at similarity index 2, i.e. sentence
        // boundary 2 + similarity_window(3) = 5 — precisely the cat→finance split.
        let text = topic_doc();
        let out = SemanticChunker::new()
            .min_characters_per_sentence(1)
            .filter_tolerance(0.5)
            .chunk(
                &text,
                &CharCounter,
                &TopicEmbedder::new(&["cat", "finance"]),
            )
            .unwrap();
        assert_partition(&text, &out);
        assert_eq!(
            out.len(),
            2,
            "expected exactly one topic split, got {out:?}"
        );
        // First chunk is all-cat, second is all-finance — no topic bleed.
        let first = text[out[0].start..out[0].end].to_lowercase();
        let second = text[out[1].start..out[1].end].to_lowercase();
        assert!(first.contains("cat") && !first.contains("finance"));
        assert!(second.contains("finance") && !second.contains("cat"));
    }

    #[test]
    fn respects_token_budget_by_splitting_groups() {
        let text = topic_doc();
        let out = SemanticChunker::new()
            .min_characters_per_sentence(1)
            .chunk_size(20) // small budget → oversized groups get split
            .chunk(
                &text,
                &CharCounter,
                &TopicEmbedder::new(&["cat", "finance"]),
            )
            .unwrap();
        assert_partition(&text, &out);
        // Every multi-sentence group must respect the budget; a lone oversized sentence may
        // still exceed it (never sub-split a sentence), so allow that case.
        for c in &out {
            let sentences = text[c.start..c.end].split_inclusive(". ").count();
            assert!(
                c.token_count <= 20 || sentences == 1,
                "chunk exceeds budget without being a single sentence: {c:?}"
            );
        }
    }

    #[test]
    fn cjk_sentences_stay_valid_utf8() {
        let text = "这是第一句关于猫的句子。第二句也是关于猫的。第三句仍然是猫。\
                    这是关于金融的第一句。第二句是金融相关。第三句依旧金融。";
        let out = SemanticChunker::new()
            .min_characters_per_sentence(1)
            .delim(vec!["。".into()])
            .filter_window(3)
            .filter_polyorder(2)
            .chunk(text, &CharCounter, &HashEmbedder::new(64))
            .unwrap();
        for c in &out {
            assert!(text.get(c.start..c.end).is_some());
        }
        assert_partition(text, &out);
    }

    #[test]
    fn skip_and_merge_bridges_digression() {
        // Directly exercise the double-pass merge: an A|B|A group layout where the two A
        // groups are identical in topic and B is a different topic. With skip_window >= 1,
        // A0 should merge with A2 across B, collapsing all three groups into one.
        let text = "cat cat cat. finance money here. cat cat cat. ";
        let sentences = split_by_delimiters(text, &[". ".into()], IncludeDelim::Prev, 1);
        assert_eq!(sentences.len(), 3, "sentences: {sentences:?}");
        let groups = vec![(0usize, 1usize), (1, 2), (2, 3)];

        let mut p = SemanticParams::defaults();
        p.skip_window = 1;
        p.threshold = 0.8;
        let embedder = TopicEmbedder::new(&["cat", "finance"]);
        let merged = p.skip_and_merge(text, &sentences, &groups, &embedder);
        assert_eq!(merged, vec![(0, 3)], "A0 and A2 should merge across B");

        // With skip_window = 0 the method is a no-op (guarded by the caller, but assert it).
        p.skip_window = 0;
        let unmerged = p.skip_and_merge(text, &sentences, &groups, &embedder);
        assert_eq!(unmerged, groups);
    }

    #[test]
    fn skip_and_merge_leaves_dissimilar_groups_alone() {
        let text = "cat cat cat. finance money here. bird bird bird. ";
        let sentences = split_by_delimiters(text, &[". ".into()], IncludeDelim::Prev, 1);
        let groups = vec![(0usize, 1usize), (1, 2), (2, 3)];
        let mut p = SemanticParams::defaults();
        p.skip_window = 2;
        p.threshold = 0.8;
        let embedder = TopicEmbedder::new(&["cat", "finance", "bird"]);
        // No two groups are similar enough to merge → unchanged.
        let out = p.skip_and_merge(text, &sentences, &groups, &embedder);
        assert_eq!(out, groups);
    }

    #[test]
    fn zero_chunk_size_errors() {
        let err = SemanticChunker::new().chunk_size(0).chunk(
            "a. b.",
            &CharCounter,
            &HashEmbedder::new(8),
        );
        assert!(matches!(err, Err(ChunkError::InvalidConfig(_))));
    }

    #[test]
    fn bad_threshold_errors() {
        for t in [0.0, 1.0, 1.5, -0.1] {
            let err = SemanticChunker::new().threshold(t).chunk(
                "a. b.",
                &CharCounter,
                &HashEmbedder::new(8),
            );
            assert!(matches!(err, Err(ChunkError::InvalidConfig(_))), "t={t}");
        }
    }

    #[test]
    fn bad_filter_params_error() {
        let err = SemanticChunker::new()
            .filter_polyorder(5)
            .filter_window(5)
            .chunk("a. b.", &CharCounter, &HashEmbedder::new(8));
        assert!(matches!(err, Err(ChunkError::InvalidConfig(_))));
        let err = SemanticChunker::new().filter_tolerance(1.0).chunk(
            "a. b.",
            &CharCounter,
            &HashEmbedder::new(8),
        );
        assert!(matches!(err, Err(ChunkError::InvalidConfig(_))));
        let err = SemanticChunker::new().similarity_window(0).chunk(
            "a. b.",
            &CharCounter,
            &HashEmbedder::new(8),
        );
        assert!(matches!(err, Err(ChunkError::InvalidConfig(_))));
    }
}
