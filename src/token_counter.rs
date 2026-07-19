//! Token counting for chunkers.
//!
//! Chunkers measure chunk size in *tokens*. The [`TokenCounter`] trait is the
//! injected dependency that defines what a "token" is — chunkers never construct
//! a tokenizer themselves. Built-in counters cover the fast, dependency-free
//! cases; enable the `hf-tokenizer` feature for a real HuggingFace subword
//! tokenizer ([`HfTokenCounter`]) that satisfies the same trait — no chunker
//! code changes required.

/// Counts tokens in text, and enumerates token byte-spans for token-based chunking.
///
/// `token_spans` returns a contiguous partition of the text (`spans[i].1 ==
/// spans[i+1].0`), so any contiguous run of spans slices the original text into
/// valid UTF-8. The default is one span per Unicode code point.
pub trait TokenCounter {
    /// Number of tokens in `text`.
    fn count(&self, text: &str) -> usize;

    /// Contiguous byte-span partition of `text` into tokens.
    fn token_spans(&self, text: &str) -> Vec<(usize, usize)> {
        text.char_indices()
            .map(|(i, ch)| (i, i + ch.len_utf8()))
            .collect()
    }

    /// Whether this counter measures whole rows. The table chunker uses this to
    /// pick row mode (fixed data rows per chunk) over token mode (token budget
    /// per chunk). All counters except [`RowCounter`] return `false`.
    fn is_row_counter(&self) -> bool {
        false
    }
}

/// Counts Unicode code points — upstream `"character"` tokenizer, the default.
pub struct CharCounter;

impl TokenCounter for CharCounter {
    fn count(&self, text: &str) -> usize {
        text.chars().count()
    }
}

/// Counts whitespace-separated words — upstream `"word"`.
pub struct WordCounter;

impl TokenCounter for WordCounter {
    fn count(&self, text: &str) -> usize {
        // Derive the count from the span partition so the two can never drift.
        // A naive `text.split(' ').count()` over-counts runs of consecutive
        // spaces (each extra space becomes a phantom empty token that has no
        // corresponding span), which would break token-based chunking offsets.
        self.token_spans(text).len()
    }

    fn token_spans(&self, text: &str) -> Vec<(usize, usize)> {
        // Each token is a run of non-space bytes plus its trailing spaces, keeping
        // the spans a contiguous partition. Space is ASCII (0x20) and never appears
        // inside a multi-byte UTF-8 sequence, so every boundary is a char boundary.
        let bytes = text.as_bytes();
        let mut spans = Vec::new();
        let mut i = 0;
        while i < bytes.len() {
            let start = i;
            while i < bytes.len() && bytes[i] != b' ' {
                i += 1;
            }
            while i < bytes.len() && bytes[i] == b' ' {
                i += 1;
            }
            spans.push((start, i));
        }
        spans
    }
}

/// Counts UTF-8 bytes — upstream `"byte"` tokenizer.
///
/// `token_spans` inherits the per-code-point default (not per-byte) so that
/// token-based chunking never slices through a multi-byte character.
pub struct ByteCounter;

impl TokenCounter for ByteCounter {
    fn count(&self, text: &str) -> usize {
        text.len()
    }
}

/// Counts newline-separated rows — upstream `"row"` tokenizer, TableChunker's
/// default. Empty text counts as `0` rows.
pub struct RowCounter;

impl TokenCounter for RowCounter {
    fn count(&self, text: &str) -> usize {
        if text.is_empty() {
            0
        } else {
            text.split('\n').count()
        }
    }

    fn is_row_counter(&self) -> bool {
        true
    }

    fn token_spans(&self, text: &str) -> Vec<(usize, usize)> {
        // One span per line, including its trailing '\n'.
        let bytes = text.as_bytes();
        let mut spans = Vec::new();
        let mut start = 0;
        for (i, &b) in bytes.iter().enumerate() {
            if b == b'\n' {
                spans.push((start, i + 1));
                start = i + 1;
            }
        }
        if start < bytes.len() {
            spans.push((start, bytes.len()));
        }
        spans
    }
}

/// Real subword token counts via a HuggingFace [`tokenizers::Tokenizer`].
///
/// Gated behind the `hf-tokenizer` cargo feature. Construction is **local-file
/// (or bytes) only** — Hub downloads are intentionally out of scope so the
/// optional dependency stays free of `http`/`hf-hub`.
///
/// `token_spans` derives a contiguous, byte-boundary-safe partition from
/// `Encoding::get_offsets()` so [`crate::TokenChunker`]'s slice invariant
/// (and the #629 fix) still holds when this counter is plugged in.
#[cfg(feature = "hf-tokenizer")]
#[derive(Clone)]
pub struct HfTokenCounter {
    tokenizer: std::sync::Arc<tokenizers::Tokenizer>,
}

#[cfg(feature = "hf-tokenizer")]
impl HfTokenCounter {
    /// Load a tokenizer from a local `tokenizer.json` path.
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self, String> {
        let tokenizer = tokenizers::Tokenizer::from_file(path.as_ref()).map_err(|e| {
            format!(
                "failed to load tokenizer from {}: {e}",
                path.as_ref().display()
            )
        })?;
        Ok(Self {
            tokenizer: std::sync::Arc::new(tokenizer),
        })
    }

    /// Load a tokenizer from raw `tokenizer.json` bytes.
    pub fn from_bytes(bytes: impl AsRef<[u8]>) -> Result<Self, String> {
        let tokenizer = tokenizers::Tokenizer::from_bytes(bytes.as_ref())
            .map_err(|e| format!("failed to load tokenizer from bytes: {e}"))?;
        Ok(Self {
            tokenizer: std::sync::Arc::new(tokenizer),
        })
    }

    /// Encode without special tokens; on failure treat the whole text as one token.
    fn encode(&self, text: &str) -> Option<tokenizers::Encoding> {
        self.tokenizer.encode(text, false).ok()
    }
}

#[cfg(feature = "hf-tokenizer")]
impl TokenCounter for HfTokenCounter {
    fn count(&self, text: &str) -> usize {
        if text.is_empty() {
            return 0;
        }
        match self.encode(text) {
            Some(enc) => enc.get_ids().len(),
            None => 1,
        }
    }

    fn token_spans(&self, text: &str) -> Vec<(usize, usize)> {
        if text.is_empty() {
            return vec![];
        }
        let Some(enc) = self.encode(text) else {
            return vec![(0, text.len())];
        };
        let offsets = enc.get_offsets();
        if offsets.is_empty() {
            return vec![(0, text.len())];
        }

        // Turn possibly gapped/empty HF offsets into a contiguous byte partition
        // with exactly one span per encoding token (so count == spans.len()).
        let n = offsets.len();
        let mut ends = vec![0usize; n];
        let mut cursor = 0usize;
        for (i, &(start, end)) in offsets.iter().enumerate() {
            let start = start.min(text.len());
            let end = end.min(text.len()).max(start);
            // Absorb any gap before this token into it; empty offsets still
            // advance by keeping the cursor (filled in on the next real end,
            // or by the final extension to text.len()).
            let span_end = if end > cursor { end } else { cursor };
            ends[i] = span_end;
            cursor = span_end;
        }
        // Guarantee the partition covers the whole input.
        if let Some(last) = ends.last_mut() {
            *last = text.len();
        }
        // Snap every boundary down to a char boundary (HF offsets are bytes,
        // but defensive in case a tokenizer reports a mid-codepoint index).
        for e in &mut ends {
            while *e > 0 && *e < text.len() && !text.is_char_boundary(*e) {
                *e -= 1;
            }
        }
        // Rebuild starts from ends so spans stay contiguous even after snapping.
        let mut spans = Vec::with_capacity(n);
        let mut prev = 0usize;
        for &e in &ends {
            let end = e.max(prev);
            spans.push((prev, end));
            prev = end;
        }
        if let Some(last) = spans.last_mut() {
            last.1 = text.len();
        }
        spans
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn char_counts_code_points() {
        assert_eq!(CharCounter.count("héllo"), 5);
    }

    #[test]
    fn char_counts_cjk() {
        assert_eq!(CharCounter.count("你好"), 2);
    }

    #[test]
    fn word_counts_spaces() {
        assert_eq!(WordCounter.count("a b c"), 3);
    }

    #[test]
    fn byte_counts_utf8() {
        assert_eq!(ByteCounter.count("你好"), 6);
    }

    #[test]
    fn row_counts_lines() {
        assert_eq!(RowCounter.count("a\nb\nc"), 3);
        assert_eq!(RowCounter.count(""), 0);
    }

    #[test]
    fn char_spans_partition_text() {
        let text = "a🩺b";
        let spans = CharCounter.token_spans(text);
        assert_eq!(spans.len(), 3);
        // contiguous and valid UTF-8 slices
        let mut prev = 0;
        for (s, e) in &spans {
            assert_eq!(*s, prev);
            assert!(text.get(*s..*e).is_some());
            prev = *e;
        }
        assert_eq!(prev, text.len());
    }

    #[test]
    fn word_spans_are_contiguous() {
        let text = "a b c";
        let spans = WordCounter.token_spans(text);
        assert_eq!(spans.len(), 3);
        assert_eq!(&text[spans[0].0..spans[0].1], "a ");
        assert_eq!(&text[spans[2].0..spans[2].1], "c");
    }

    #[test]
    fn dyn_dispatch_works() {
        let counter: &dyn TokenCounter = &CharCounter;
        assert_eq!(counter.count("abc"), 3);
    }

    #[test]
    fn word_count_ignores_consecutive_spaces() {
        // Regression: `"a  b".split(' ')` yields ["a", "", "b"] (a phantom empty
        // token). Deriving count from token_spans keeps it at 2, matching spans.
        assert_eq!(WordCounter.count("a  b"), 2);
        assert_eq!(WordCounter.count("a   b   c"), 3);
        assert_eq!(WordCounter.count(""), 0);
    }

    #[test]
    fn count_matches_token_spans_len() {
        // The span partition is the single source of truth for how many tokens a
        // counter reports. Inputs are ASCII with no trailing newline so the
        // invariant is meaningful for every counter: ByteCounter measures UTF-8
        // bytes (== code points only for ASCII) and RowCounter measures lines (a
        // trailing '\n' would add a phantom empty row with no span).
        let inputs = [
            "the quick brown fox",
            "a  b   c",           // consecutive spaces (the WordCounter drift case)
            "line one\nline two", // interior newline for RowCounter
            "single",
            "",
        ];
        let counters: [&dyn TokenCounter; 4] =
            [&CharCounter, &WordCounter, &ByteCounter, &RowCounter];
        for text in inputs {
            for c in counters {
                assert_eq!(
                    c.count(text),
                    c.token_spans(text).len(),
                    "count/token_spans drift for input {text:?}"
                );
            }
        }
    }

    /// Minimal WordLevel+Whitespace `tokenizer.json` for `hf-tokenizer` tests.
    #[cfg(feature = "hf-tokenizer")]
    const TEST_TOKENIZER_JSON: &str = r#"{
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [
            {
                "id": 0,
                "content": "[UNK]",
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": true
            }
        ],
        "normalizer": null,
        "pre_tokenizer": { "type": "Whitespace" },
        "post_processor": null,
        "decoder": null,
        "model": {
            "type": "WordLevel",
            "vocab": {
                "[UNK]": 0,
                "hello": 1,
                "world": 2,
                "foo": 3,
                "bar": 4,
                "the": 5,
                "quick": 6,
                "brown": 7,
                "fox": 8
            },
            "unk_token": "[UNK]"
        }
    }"#;

    #[cfg(feature = "hf-tokenizer")]
    fn test_hf_counter() -> HfTokenCounter {
        HfTokenCounter::from_bytes(TEST_TOKENIZER_JSON.as_bytes()).expect("test tokenizer")
    }

    #[cfg(feature = "hf-tokenizer")]
    #[test]
    fn hf_count_known_strings() {
        let c = test_hf_counter();
        assert_eq!(c.count("hello world"), 2);
        assert_eq!(c.count("foo bar"), 2);
        assert_eq!(c.count("hello"), 1);
        assert_eq!(c.count(""), 0);
        // OOV still produces one UNK token per whitespace word.
        assert_eq!(c.count("zzz yyy"), 2);
    }

    #[cfg(feature = "hf-tokenizer")]
    #[test]
    fn hf_spans_are_contiguous_byte_partition() {
        let c = test_hf_counter();
        let text = "hello world foo";
        let spans = c.token_spans(text);
        assert_eq!(spans.len(), c.count(text));
        let mut prev = 0;
        for &(s, e) in &spans {
            assert_eq!(s, prev);
            assert!(text.is_char_boundary(s) && text.is_char_boundary(e));
            assert!(text.get(s..e).is_some());
            prev = e;
        }
        assert_eq!(prev, text.len());
    }

    #[cfg(feature = "hf-tokenizer")]
    #[test]
    fn hf_token_chunker_slice_invariant() {
        // Plugging HfTokenCounter into TokenChunker must preserve the #629
        // invariant: every chunk is an exact byte slice of the original text.
        use crate::chunkers::TokenChunker;
        let c = test_hf_counter();
        let text = "hello world foo bar the quick brown fox";
        let chunks = TokenChunker::new()
            .chunk_size(2)
            .chunk(text, &c)
            .expect("chunk");
        assert!(!chunks.is_empty());
        for chunk in &chunks {
            assert!(text.is_char_boundary(chunk.start) && text.is_char_boundary(chunk.end));
            let sliced = &text[chunk.start..chunk.end];
            assert!(!sliced.is_empty() || text.is_empty());
            // Contiguous windows may overlap with overlap>0; here overlap=0 so
            // we only check each individual slice is valid UTF-8 from original.
            assert_eq!(sliced, &text[chunk.start..chunk.end]);
        }
    }

    #[cfg(feature = "hf-tokenizer")]
    #[test]
    fn hf_from_file_roundtrip() {
        let dir = std::env::temp_dir().join("chunk_hf_tokenizer_test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("tokenizer.json");
        std::fs::write(&path, TEST_TOKENIZER_JSON).unwrap();
        let c = HfTokenCounter::from_file(&path).expect("from_file");
        assert_eq!(c.count("hello world"), 2);
        let _ = std::fs::remove_file(&path);
    }
}
