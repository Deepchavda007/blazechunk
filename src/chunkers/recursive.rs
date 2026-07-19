//! Recursive chunking through a hierarchy of delimiter levels.
//!
//! Splits text at the coarsest level that fits the token budget (paragraphs →
//! sentences → pauses → words → tokens), recursing into any merged group that is
//! still too large. Mirrors Chonkie's `RecursiveChunker`, reusing the crate's
//! `split`/`merge` primitives. All offsets are byte offsets into the original text.

use crate::chunkers::{Chunk, ChunkError, split_by_delimiters};
use crate::merge::find_merge_indices;
use crate::split::{IncludeDelim, split_at_delimiters};
use crate::token_counter::TokenCounter;

/// One level of the recursive hierarchy: split by delimiters, by whitespace, or
/// (as a last resort) by raw token groups.
#[derive(Debug, Clone)]
pub enum RecursiveLevel {
    /// Split at the given delimiters. Non-ASCII or multi-byte delimiters are
    /// matched as whole patterns (never per-byte), which keeps multibyte text
    /// intact (upstream bug #536).
    Delimiters {
        delimiters: Vec<String>,
        include_delim: IncludeDelim,
    },
    /// Split on ASCII spaces.
    Whitespace { include_delim: IncludeDelim },
    /// Terminal level: hard-split into groups of `chunk_size` tokens.
    Token,
}

/// The ordered levels the recursive chunker walks.
#[derive(Debug, Clone)]
pub struct RecursiveRules {
    pub levels: Vec<RecursiveLevel>,
}

impl RecursiveRules {
    /// The default five-level hierarchy (paragraphs, sentences, pauses, words, tokens).
    pub fn default_rules() -> Self {
        let strs = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        Self {
            levels: vec![
                RecursiveLevel::Delimiters {
                    delimiters: strs(&["\n\n", "\r\n", "\n", "\r"]),
                    include_delim: IncludeDelim::Prev,
                },
                RecursiveLevel::Delimiters {
                    delimiters: strs(&[". ", "! ", "? "]),
                    include_delim: IncludeDelim::Prev,
                },
                RecursiveLevel::Delimiters {
                    delimiters: strs(&[
                        "{", "}", "\"", "[", "]", "<", ">", "(", ")", ":", ";", ",", "—", "|", "~",
                        "-", "...", "`", "'",
                    ]),
                    include_delim: IncludeDelim::Prev,
                },
                RecursiveLevel::Whitespace {
                    include_delim: IncludeDelim::Prev,
                },
                RecursiveLevel::Token,
            ],
        }
    }

    /// A single delimiter level — handy for tests and simple use.
    pub fn single(delimiters: Vec<String>) -> Self {
        Self {
            levels: vec![RecursiveLevel::Delimiters {
                delimiters,
                include_delim: IncludeDelim::Prev,
            }],
        }
    }
}

impl Default for RecursiveRules {
    fn default() -> Self {
        Self::default_rules()
    }
}

/// Recursively chunks text through a delimiter hierarchy to fit a token budget.
#[derive(Debug, Clone)]
pub struct RecursiveChunker {
    chunk_size: usize,
    min_characters_per_chunk: usize,
    rules: RecursiveRules,
}

impl Default for RecursiveChunker {
    fn default() -> Self {
        Self::new()
    }
}

impl RecursiveChunker {
    /// New chunker with upstream defaults: `chunk_size = 2048`,
    /// `min_characters_per_chunk = 24`, and the default five-level rules.
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

    /// Set the minimum characters per split (shorter fragments merge into neighbors).
    pub fn min_characters_per_chunk(mut self, n: usize) -> Self {
        self.min_characters_per_chunk = n;
        self
    }

    /// Replace the delimiter hierarchy.
    pub fn rules(mut self, rules: RecursiveRules) -> Self {
        self.rules = rules;
        self
    }

    /// Validate the configuration, returning the same error [`chunk`](Self::chunk)
    /// would. This is the single source of truth for config validity, shared by
    /// `chunk` and the binding layers so the rules can never drift.
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

    /// Chunk `text`, measuring size with `counter`.
    ///
    /// Returns an error if `chunk_size` or `min_characters_per_chunk` is zero.
    pub fn chunk(&self, text: &str, counter: &dyn TokenCounter) -> Result<Vec<Chunk>, ChunkError> {
        self.validate()?;
        let mut out = Vec::new();
        self.recurse(text, 0, 0, counter, &mut out);
        Ok(out)
    }

    fn recurse(
        &self,
        text: &str,
        base: usize,
        level: usize,
        counter: &dyn TokenCounter,
        out: &mut Vec<Chunk>,
    ) {
        // Recursive short-circuits only on truly empty text (matches upstream).
        if text.is_empty() {
            return;
        }
        if level >= self.rules.levels.len() {
            out.push(Chunk {
                start: base,
                end: base + text.len(),
                token_count: counter.count(text),
            });
            return;
        }

        let rule = &self.rules.levels[level];
        let splits: Vec<(usize, usize)> = match rule {
            RecursiveLevel::Delimiters {
                delimiters,
                include_delim,
            } => split_by_delimiters(
                text,
                delimiters,
                *include_delim,
                self.min_characters_per_chunk,
            ),
            RecursiveLevel::Whitespace { include_delim } => split_at_delimiters(
                text.as_bytes(),
                b" ",
                *include_delim,
                self.min_characters_per_chunk,
            ),
            RecursiveLevel::Token => token_groups(text, self.chunk_size.max(1), counter),
        };

        if splits.is_empty() {
            // A rule produced no splits for non-empty text (e.g. an injected
            // counter whose `token_spans` is empty). Never drop text: emit it as
            // a terminal chunk so byte-preservation and contiguity still hold.
            out.push(Chunk {
                start: base,
                end: base + text.len(),
                token_count: counter.count(text),
            });
            return;
        }

        let token_counts: Vec<usize> = splits
            .iter()
            .map(|&(s, e)| counter.count(&text[s..e]))
            .collect();

        // Token level is terminal: every group already fits, emit directly.
        if matches!(rule, RecursiveLevel::Token) {
            for (i, &(s, e)) in splits.iter().enumerate() {
                out.push(Chunk {
                    start: base + s,
                    end: base + e,
                    token_count: token_counts[i],
                });
            }
            return;
        }

        // Merge consecutive splits greedily into groups that fit the budget.
        let indices = find_merge_indices(&token_counts, self.chunk_size);
        let mut gstart = 0usize;
        for &gend in &indices {
            let gend = gend.min(splits.len());
            if gend <= gstart {
                continue;
            }
            let local_start = splits[gstart].0;
            let local_end = splits[gend - 1].1;
            let group_tc: usize = token_counts[gstart..gend].iter().sum();

            if group_tc > self.chunk_size {
                // Still too big — recurse into the next finer level.
                self.recurse(
                    &text[local_start..local_end],
                    base + local_start,
                    level + 1,
                    counter,
                    out,
                );
            } else {
                out.push(Chunk {
                    start: base + local_start,
                    end: base + local_end,
                    token_count: group_tc,
                });
            }
            gstart = gend;
        }
    }
}

/// Group token spans into contiguous byte ranges of at most `size` tokens each.
fn token_groups(text: &str, size: usize, counter: &dyn TokenCounter) -> Vec<(usize, usize)> {
    let spans = counter.token_spans(text);
    if spans.is_empty() {
        return vec![];
    }
    let mut groups = Vec::new();
    let mut i = 0;
    while i < spans.len() {
        let end = (i + size).min(spans.len());
        groups.push((spans[i].0, spans[end - 1].1));
        i = end;
    }
    groups
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::token_counter::CharCounter;

    fn assert_contiguous(text: &str, chunks: &[Chunk]) {
        let mut prev = 0;
        for c in chunks {
            assert_eq!(c.start, prev, "chunks not contiguous");
            assert!(text.is_char_boundary(c.start) && text.is_char_boundary(c.end));
            assert_eq!(&text[c.start..c.end].len(), &(c.end - c.start));
            prev = c.end;
        }
        assert_eq!(prev, text.len(), "chunks do not cover the whole text");
    }

    #[test]
    fn empty_returns_none() {
        assert!(
            RecursiveChunker::new()
                .chunk("", &CharCounter)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn small_text_one_chunk() {
        let out = RecursiveChunker::new()
            .chunk("Hello world.", &CharCounter)
            .unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].start, 0);
        assert_eq!(out[0].end, "Hello world.".len());
        assert_eq!(out[0].token_count, 12);
    }

    #[test]
    fn slice_invariant_and_contiguous() {
        let text =
            "Para one.\n\nPara two is a bit longer. It has two sentences.\n\nThird one here.";
        let out = RecursiveChunker::new()
            .chunk_size(20)
            .chunk(text, &CharCounter)
            .unwrap();
        assert!(out.len() > 1);
        assert_contiguous(text, &out);
    }

    #[test]
    fn cjk_delimiters_no_panic_valid_utf8() {
        // Regression for #536: a CJK delimiter is one code point but three bytes.
        let text = "第一句。第二句。第三句。";
        let rules = RecursiveRules::single(vec!["。".into()]);
        let out = RecursiveChunker::new()
            .chunk_size(4)
            .min_characters_per_chunk(1)
            .rules(rules)
            .chunk(text, &CharCounter)
            .unwrap();
        for c in &out {
            assert!(text.get(c.start..c.end).is_some());
        }
        assert_eq!(
            out.iter().map(|c| c.end - c.start).sum::<usize>(),
            text.len()
        );
    }

    #[test]
    fn oversized_unbroken_text_hard_splits_at_token_level() {
        let text = "abcdefghijklmnopqrstuvwxyz"; // no delimiters, no spaces
        let out = RecursiveChunker::new()
            .chunk_size(10)
            .chunk(text, &CharCounter)
            .unwrap();
        assert_contiguous(text, &out);
        for c in &out {
            assert!(c.token_count <= 10);
        }
    }

    #[test]
    fn realistic_doc_preserves_bytes() {
        let text = "The quick brown fox. Jumps over? The lazy dog!\n\nSecond paragraph here with more words to force splitting across levels.";
        let out = RecursiveChunker::new()
            .chunk_size(15)
            .chunk(text, &CharCounter)
            .unwrap();
        assert_contiguous(text, &out);
    }

    #[test]
    fn zero_chunk_size_errors() {
        let err = RecursiveChunker::new()
            .chunk_size(0)
            .chunk("hi", &CharCounter);
        assert!(matches!(err, Err(ChunkError::InvalidConfig(_))));
    }

    #[test]
    fn zero_min_characters_errors() {
        let err = RecursiveChunker::new()
            .min_characters_per_chunk(0)
            .chunk("hi", &CharCounter);
        assert!(matches!(err, Err(ChunkError::InvalidConfig(_))));
    }

    #[test]
    fn custom_rule_absent_delimiter_preserves_bytes() {
        // A custom top-level delimiter that never appears must not drop text: the
        // whole input falls through to a terminal chunk instead of vanishing.
        let text = "hello world foo bar baz";
        let rules = RecursiveRules::single(vec!["ZZZ".into()]);
        let out = RecursiveChunker::new()
            .chunk_size(2)
            .min_characters_per_chunk(1)
            .rules(rules)
            .chunk(text, &CharCounter)
            .unwrap();
        assert!(!out.is_empty());
        assert_contiguous(text, &out);
    }

    #[test]
    fn empty_splits_fall_back_to_terminal_chunk() {
        // A pathological counter whose `token_spans` is empty makes the token
        // level produce zero splits. The fallback must emit the text as a single
        // terminal chunk rather than silently dropping it.
        struct EmptySpanCounter;
        impl TokenCounter for EmptySpanCounter {
            fn count(&self, _text: &str) -> usize {
                // Larger than chunk_size so every level recurses down to Token.
                100
            }
            fn token_spans(&self, _text: &str) -> Vec<(usize, usize)> {
                vec![]
            }
        }
        let text = "abcdefgh"; // no delimiters, no spaces → reaches the token level
        let out = RecursiveChunker::new()
            .chunk_size(2)
            .min_characters_per_chunk(1)
            .chunk(text, &EmptySpanCounter)
            .unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].start, 0);
        assert_eq!(out[0].end, text.len());
    }
}
