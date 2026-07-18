//! Structural code chunking (heuristic, dependency-free).
//!
//! Groups source lines into logical blocks — using brace depth for C-family
//! languages, top-level indentation for Python-family languages, and blank-line
//! separation as a fallback — then merges blocks up to a byte budget derived
//! from the token budget. A single block larger than the budget is split at line
//! boundaries. Lines are never split mid-way, so the slice invariant holds.
//!
//! This is the hybrid approach from the design: fast and dependency-free, with
//! an internal seam left open to add tree-sitter behind a cargo feature later.
//! It mirrors the *intent* of Chonkie's `CodeChunker` without the AST.

use crate::chunkers::{Chunk, line_ranges};
use crate::token_counter::TokenCounter;

/// Source language family. `Auto` detects a family heuristically.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Language {
    Auto,
    Rust,
    JavaScript,
    TypeScript,
    C,
    Cpp,
    Java,
    Go,
    Python,
    Ruby,
    /// Unknown language: blank-line separation only.
    Generic,
}

impl Language {
    fn is_brace(self) -> bool {
        matches!(
            self,
            Language::Rust
                | Language::JavaScript
                | Language::TypeScript
                | Language::C
                | Language::Cpp
                | Language::Java
                | Language::Go
        )
    }

    fn is_indent(self) -> bool {
        matches!(self, Language::Python | Language::Ruby)
    }

    /// Map a lowercase language name (or file hint) to a variant. Unknown → Generic.
    pub fn from_name(name: &str) -> Language {
        match name.to_lowercase().as_str() {
            "auto" => Language::Auto,
            "rust" | "rs" => Language::Rust,
            "javascript" | "js" | "jsx" => Language::JavaScript,
            "typescript" | "ts" | "tsx" => Language::TypeScript,
            "c" | "h" => Language::C,
            "cpp" | "c++" | "cc" | "hpp" => Language::Cpp,
            "java" => Language::Java,
            "go" => Language::Go,
            "python" | "py" => Language::Python,
            "ruby" | "rb" => Language::Ruby,
            _ => Language::Generic,
        }
    }
}

/// Splits source code into structure-aware, byte-budgeted chunks.
#[derive(Debug, Clone)]
pub struct CodeChunker {
    chunk_size: usize,
    language: Language,
}

impl Default for CodeChunker {
    fn default() -> Self {
        Self::new()
    }
}

impl CodeChunker {
    /// New chunker with defaults: `chunk_size = 2048`, `language = Auto`.
    pub fn new() -> Self {
        Self {
            chunk_size: 2048,
            language: Language::Auto,
        }
    }

    /// Set the token budget per chunk.
    pub fn chunk_size(mut self, n: usize) -> Self {
        self.chunk_size = n;
        self
    }

    /// Set the source language (or `Auto`).
    pub fn language(mut self, lang: Language) -> Self {
        self.language = lang;
        self
    }

    /// Chunk `text`, measuring size with `counter`.
    pub fn chunk(&self, text: &str, counter: &dyn TokenCounter) -> Vec<Chunk> {
        if text.trim().is_empty() || self.chunk_size == 0 {
            return vec![];
        }

        let lang = match self.language {
            Language::Auto => detect_language(text),
            other => other,
        };

        // Convert the token budget into a byte budget.
        let tokens = counter.count(text).max(1);
        let bytes_per_token = (text.len() / tokens).max(1);
        let max_bytes = self.chunk_size.saturating_mul(bytes_per_token).max(1);

        let blocks = segment_blocks(text, lang);

        let mut out = Vec::new();
        let mut cur: Option<(usize, usize)> = None;

        for (s, e) in blocks {
            let len = e - s;
            if len > max_bytes {
                if let Some((cs, ce)) = cur.take() {
                    push_chunk(text, cs, ce, counter, &mut out);
                }
                for (ps, pe) in line_split(text, s, e, max_bytes) {
                    push_chunk(text, ps, pe, counter, &mut out);
                }
                continue;
            }
            match cur {
                None => cur = Some((s, e)),
                Some((cs, ce)) => {
                    if (ce - cs) + len <= max_bytes {
                        cur = Some((cs, e));
                    } else {
                        push_chunk(text, cs, ce, counter, &mut out);
                        cur = Some((s, e));
                    }
                }
            }
        }
        if let Some((cs, ce)) = cur {
            push_chunk(text, cs, ce, counter, &mut out);
        }
        out
    }
}

fn push_chunk(
    text: &str,
    start: usize,
    end: usize,
    counter: &dyn TokenCounter,
    out: &mut Vec<Chunk>,
) {
    if start >= end {
        return;
    }
    out.push(Chunk {
        start,
        end,
        token_count: counter.count(&text[start..end]),
    });
}

/// Leading-whitespace width of a line (spaces and tabs).
fn indent_width(line: &str) -> usize {
    line.bytes()
        .take_while(|&b| b == b' ' || b == b'\t')
        .count()
}

/// Split text into logical blocks (contiguous, covering the whole text).
fn segment_blocks(text: &str, lang: Language) -> Vec<(usize, usize)> {
    let lines = line_ranges(text);
    if lines.is_empty() {
        return vec![];
    }

    let mut blocks = Vec::new();
    let mut block_start = lines[0].0;
    let mut depth: i32 = 0;

    for i in 0..lines.len() {
        let (s, e) = lines[i];
        let line = &text[s..e];
        let this_blank = line.trim().is_empty();

        if lang.is_brace() {
            for b in line.bytes() {
                match b {
                    b'{' => depth += 1,
                    b'}' => depth -= 1,
                    _ => {}
                }
            }
        }

        let boundary_after = match lines.get(i + 1) {
            None => false,
            Some(&(ns, ne)) => {
                let next = &text[ns..ne];
                let next_blank = next.trim().is_empty();
                // Blank-run separation applies to every language.
                if this_blank && !next_blank {
                    true
                } else if lang.is_brace() {
                    // End of a top-level construct, next line begins new content.
                    depth <= 0 && !this_blank && !next_blank
                } else if lang.is_indent() {
                    // Next line begins a new top-level (column-0) construct.
                    !next_blank && indent_width(next) == 0
                } else {
                    false
                }
            }
        };

        if boundary_after {
            blocks.push((block_start, e));
            block_start = e;
        }
    }
    blocks.push((block_start, text.len()));
    blocks
}

/// Split byte range `[s, e)` at line boundaries into pieces of at most
/// `max_bytes` (a single over-long line becomes its own piece).
fn line_split(text: &str, s: usize, e: usize, max_bytes: usize) -> Vec<(usize, usize)> {
    let sub = &text[s..e];
    let mut pieces = Vec::new();
    let mut cur_start = s;
    let mut cur_end = s;
    for (ls, le) in line_ranges(sub) {
        let (abs_s, abs_e) = (s + ls, s + le);
        let line_len = abs_e - abs_s;
        if cur_end == cur_start {
            cur_start = abs_s;
            cur_end = abs_e;
        } else if (cur_end - cur_start) + line_len <= max_bytes {
            cur_end = abs_e;
        } else {
            pieces.push((cur_start, cur_end));
            cur_start = abs_s;
            cur_end = abs_e;
        }
    }
    if cur_end > cur_start {
        pieces.push((cur_start, cur_end));
    }
    pieces
}

/// Heuristically detect a language family from content.
fn detect_language(text: &str) -> Language {
    let braces = text.bytes().filter(|&b| b == b'{').count();
    let mut python_signals = 0usize;
    for line in text.lines() {
        let t = line.trim_start();
        if t.starts_with("def ")
            || t.starts_with("class ")
            || t.starts_with("import ")
            || t.starts_with("from ")
        {
            python_signals += 1;
        }
        if line.trim_end().ends_with(':') {
            python_signals += 1;
        }
    }
    if braces >= 2 && braces >= python_signals {
        Language::Rust // representative brace family
    } else if python_signals > 0 {
        Language::Python // representative indent family
    } else {
        Language::Generic
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::token_counter::CharCounter;

    const CODE: &str = "fn a() {\n    let x = 1;\n}\n\nfn b() {\n    let y = 2;\n}\n\nfn c() {\n    let z = 3;\n}\n";

    fn assert_contiguous_lines(text: &str, chunks: &[Chunk]) {
        let mut prev = 0;
        for c in chunks {
            assert_eq!(c.start, prev, "not contiguous");
            assert!(
                c.start == 0 || text.as_bytes()[c.start - 1] == b'\n',
                "mid-line split"
            );
            assert!(text.is_char_boundary(c.start) && text.is_char_boundary(c.end));
            prev = c.end;
        }
        assert_eq!(prev, text.len(), "does not cover whole text");
    }

    #[test]
    fn empty_returns_none() {
        assert!(CodeChunker::new().chunk("  \n ", &CharCounter).is_empty());
    }

    #[test]
    fn slice_invariant_and_no_midline_split() {
        let out = CodeChunker::new()
            .chunk_size(20)
            .language(Language::Rust)
            .chunk(CODE, &CharCounter);
        assert!(out.len() > 1);
        assert_contiguous_lines(CODE, &out);
    }

    #[test]
    fn groups_blocks_within_budget() {
        let out = CodeChunker::new()
            .chunk_size(200)
            .language(Language::Rust)
            .chunk(CODE, &CharCounter);
        assert_eq!(out.len(), 1); // whole snippet fits
        assert_eq!(&CODE[out[0].start..out[0].end], CODE);
    }

    #[test]
    fn python_indentation_blocks() {
        let code = "def a():\n    return 1\n\ndef b():\n    return 2\n\ndef c():\n    return 3\n";
        let out = CodeChunker::new()
            .chunk_size(15)
            .language(Language::Python)
            .chunk(code, &CharCounter);
        assert!(out.len() > 1);
        assert_contiguous_lines(code, &out);
    }

    #[test]
    fn auto_detects_and_chunks() {
        let out = CodeChunker::new().chunk_size(20).chunk(CODE, &CharCounter);
        assert_contiguous_lines(CODE, &out);
    }

    #[test]
    fn single_long_line_becomes_its_own_chunk() {
        let code = "this_is_one_very_long_line_with_no_breaks_at_all_exceeding_budget = 1\n";
        let out = CodeChunker::new()
            .chunk_size(5)
            .language(Language::Generic)
            .chunk(code, &CharCounter);
        assert_contiguous_lines(code, &out);
        assert_eq!(out.len(), 1); // cannot split a single line
    }
}
