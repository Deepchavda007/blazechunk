//! Table chunking for markdown and HTML tables.
//!
//! Splits a large table into smaller tables, **re-emitting the header (and HTML
//! footer) in every chunk** so each chunk is a self-contained, renderable table.
//! Mirrors Chonkie's `TableChunker`.
//!
//! Because each chunk's text re-includes the header, [`TableChunk`] carries its
//! own `text` and is the one documented exception to the crate's slice invariant:
//! `start`/`end` span the original **data-row region**, while `text` is
//! `header + rows (+ footer)`.
//!
//! The markdown header block is detected by locating the separator row
//! (`|---|`), not by assuming it is the second line — this fixes upstream bug
//! #582 (multi-line / non-standard headers).

use crate::chunkers::{ChunkError, line_ranges};
use crate::token_counter::TokenCounter;

/// A self-contained table chunk. Unlike [`super::Chunk`], it owns its `text`
/// because the header is re-included in every chunk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TableChunk {
    /// Byte offset of the covered data-row region start in the original text.
    pub start: usize,
    /// Byte offset of the covered data-row region end (exclusive).
    pub end: usize,
    /// Rows (row mode) or token count of `text` (token mode).
    pub token_count: usize,
    /// The self-contained table text: `header + rows (+ footer)`.
    pub text: String,
}

/// Splits markdown or HTML tables while preserving the header in each chunk.
#[derive(Debug, Clone)]
pub struct TableChunker {
    chunk_size: usize,
}

impl Default for TableChunker {
    fn default() -> Self {
        Self::new()
    }
}

impl TableChunker {
    /// New chunker with the upstream default `chunk_size = 3` (data rows per chunk
    /// with a row counter, otherwise a token budget).
    pub fn new() -> Self {
        Self { chunk_size: 3 }
    }

    /// Set the chunk size (rows in row mode, tokens in token mode).
    pub fn chunk_size(mut self, n: usize) -> Self {
        self.chunk_size = n;
        self
    }

    /// Validate the configuration, returning the same error [`chunk`](Self::chunk)
    /// would. Single source of truth for config validity, shared by `chunk` and
    /// the binding layers so the rules can never drift.
    pub fn validate(&self) -> Result<(), ChunkError> {
        if self.chunk_size == 0 {
            return Err(ChunkError::InvalidConfig("chunk_size must be > 0".into()));
        }
        Ok(())
    }

    /// Chunk a table. `counter` selects row mode ([`RowCounter`](crate::RowCounter))
    /// or token mode (any other counter).
    ///
    /// Returns an error if `chunk_size` is zero.
    pub fn chunk(
        &self,
        text: &str,
        counter: &dyn TokenCounter,
    ) -> Result<Vec<TableChunk>, ChunkError> {
        self.validate()?;
        if text.trim().is_empty() {
            return Ok(vec![]);
        }

        let parsed = if text.to_lowercase().contains("<table") {
            parse_html(text)
        } else {
            parse_markdown(text)
        };
        let Some(table) = parsed else {
            return Ok(vec![]);
        };
        if table.rows.is_empty() {
            return Ok(vec![]);
        }

        Ok(if counter.is_row_counter() {
            self.chunk_rows(text, &table)
        } else {
            self.chunk_tokens(text, &table, counter)
        })
    }

    /// Row mode: at most `chunk_size` data rows per chunk.
    fn chunk_rows(&self, text: &str, table: &Table) -> Vec<TableChunk> {
        if table.rows.len() <= self.chunk_size {
            return vec![TableChunk {
                start: 0,
                end: text.len(),
                token_count: table.rows.len(),
                text: text.to_string(),
            }];
        }
        let mut out = Vec::new();
        for group in table.rows.chunks(self.chunk_size) {
            out.push(self.build_chunk(text, table, group));
        }
        out
    }

    /// Token mode: accumulate rows until adding one would reach `chunk_size`.
    fn chunk_tokens(
        &self,
        text: &str,
        table: &Table,
        counter: &dyn TokenCounter,
    ) -> Vec<TableChunk> {
        if counter.count(text.trim()) <= self.chunk_size {
            return vec![TableChunk {
                start: 0,
                end: text.len(),
                token_count: counter.count(text.trim()),
                text: text.to_string(),
            }];
        }

        let base_tokens = counter.count(&table.header) + counter.count(&table.footer);
        let mut out = Vec::new();
        let mut group: Vec<(usize, usize)> = Vec::new();
        let mut group_tokens = base_tokens;

        for &(s, e) in &table.rows {
            let row_tokens = counter.count(&text[s..e]);
            if group_tokens + row_tokens >= self.chunk_size && !group.is_empty() {
                out.push(self.build_chunk(text, table, &group));
                group = Vec::new();
                group_tokens = base_tokens;
            }
            group.push((s, e));
            group_tokens += row_tokens;
        }
        if !group.is_empty() {
            out.push(self.build_chunk(text, table, &group));
        }
        // Fill token_count from the built text.
        for c in &mut out {
            c.token_count = counter.count(&c.text);
        }
        out
    }

    fn build_chunk(&self, text: &str, table: &Table, group: &[(usize, usize)]) -> TableChunk {
        let start = group[0].0;
        let end = group[group.len() - 1].1;
        let mut body =
            String::with_capacity(table.header.len() + (end - start) + table.footer.len());
        body.push_str(&table.header);
        for &(s, e) in group {
            body.push_str(&text[s..e]);
        }
        body.push_str(&table.footer);
        TableChunk {
            start,
            end,
            token_count: group.len(),
            text: body,
        }
    }
}

/// A parsed table: the header block, footer, and data-row byte ranges.
struct Table {
    header: String,
    footer: String,
    rows: Vec<(usize, usize)>,
}

/// A markdown separator row: only `| - : space`, with at least one `-`.
fn is_md_separator(line: &str) -> bool {
    let t = line.trim();
    if t.is_empty() {
        return false;
    }
    let mut has_dash = false;
    for ch in t.chars() {
        match ch {
            '-' => has_dash = true,
            '|' | ':' | ' ' => {}
            _ => return false,
        }
    }
    has_dash
}

fn parse_markdown(text: &str) -> Option<Table> {
    let lines = line_ranges(text);
    // First non-empty line begins the header.
    let h0 = lines
        .iter()
        .position(|&(s, e)| !text[s..e].trim().is_empty())?;
    // Separator row at or after the header line.
    let sep_idx = lines
        .iter()
        .enumerate()
        .skip(h0)
        .find(|(_, r)| is_md_separator(&text[r.0..r.1]))
        .map(|(i, _)| i)?;

    let header = text[lines[h0].0..lines[sep_idx].1].to_string();
    let rows: Vec<(usize, usize)> = lines[sep_idx + 1..]
        .iter()
        .copied()
        .filter(|&(s, e)| !text[s..e].trim().is_empty())
        .collect();
    if rows.is_empty() {
        return None;
    }
    Some(Table {
        header,
        footer: String::new(),
        rows,
    })
}

// -------------------------------------------------------------------------
// HTML scanning (dependency-free, tag-boundary-aware).
//
// A naive substring search misidentifies tags: `<table` matches `<tablet>`,
// `<tr` matches `<track>`, `</tr>` misses `</tr >`/`</TR>`, nested tables fold
// their rows into the outer table, and `<!-- <tr> -->` comments count as rows.
// The helpers below scan with tag-name boundaries, tolerant closing tags,
// nesting depth, and comment skipping — enough to parse real-world tables
// without pulling in a full HTML5 parser. All tag bytes are ASCII, so byte
// offsets remain valid char boundaries.
// -------------------------------------------------------------------------

/// A byte ends an HTML tag name when it is `>`, `/`, or ASCII whitespace.
fn is_tag_boundary(b: u8) -> bool {
    b == b'>' || b == b'/' || b.is_ascii_whitespace()
}

/// If a `<name` opening tag (case-insensitive) with a proper name boundary
/// starts at `pos`, return the index just past the name. `name` is lowercase.
fn match_open_tag(bytes: &[u8], pos: usize, name: &[u8]) -> Option<usize> {
    if bytes.get(pos) != Some(&b'<') {
        return None;
    }
    let name_end = pos + 1 + name.len();
    let candidate = bytes.get(pos + 1..name_end)?;
    if candidate
        .iter()
        .zip(name)
        .all(|(a, b)| a.to_ascii_lowercase() == *b)
        && bytes.get(name_end).is_some_and(|&b| is_tag_boundary(b))
    {
        Some(name_end)
    } else {
        None
    }
}

/// If a `</name ... >` closing tag (case-insensitive, tolerant of whitespace
/// before `>`) starts at `pos`, return the index just past `>`.
fn match_close_tag(bytes: &[u8], pos: usize, name: &[u8]) -> Option<usize> {
    if bytes.get(pos) != Some(&b'<') || bytes.get(pos + 1) != Some(&b'/') {
        return None;
    }
    let name_end = pos + 2 + name.len();
    let candidate = bytes.get(pos + 2..name_end)?;
    if !candidate
        .iter()
        .zip(name)
        .all(|(a, b)| a.to_ascii_lowercase() == *b)
    {
        return None;
    }
    let mut i = name_end;
    while bytes.get(i).is_some_and(|b| b.is_ascii_whitespace()) {
        i += 1;
    }
    (bytes.get(i) == Some(&b'>')).then_some(i + 1)
}

/// Index just past the `>` closing the tag that starts at `pos`, honoring
/// quoted attribute values so `<tr title="a>b">` isn't cut short. None if the
/// tag never closes.
fn end_of_tag(bytes: &[u8], pos: usize) -> Option<usize> {
    let mut i = pos;
    let mut quote: Option<u8> = None;
    while i < bytes.len() {
        let b = bytes[i];
        match quote {
            Some(q) if b == q => quote = None,
            Some(_) => {}
            None if b == b'"' || b == b'\'' => quote = Some(b),
            None if b == b'>' => return Some(i + 1),
            None => {}
        }
        i += 1;
    }
    None
}

/// If an HTML comment starts at `pos`, return the index just past `-->`
/// (or end of input for an unterminated comment).
fn skip_comment(bytes: &[u8], pos: usize) -> Option<usize> {
    if !bytes[pos..].starts_with(b"<!--") {
        return None;
    }
    let mut i = pos + 4;
    while i + 3 <= bytes.len() {
        if &bytes[i..i + 3] == b"-->" {
            return Some(i + 3);
        }
        i += 1;
    }
    Some(bytes.len())
}

/// Find the next `<name>` opening tag at or after `from` (boundary-aware,
/// skipping comments), returning the index of its `<`.
fn find_open_tag(bytes: &[u8], from: usize, name: &[u8]) -> Option<usize> {
    let mut i = from;
    while i < bytes.len() {
        if let Some(end) = skip_comment(bytes, i) {
            i = end;
            continue;
        }
        if match_open_tag(bytes, i, name).is_some() {
            return Some(i);
        }
        i += 1;
    }
    None
}

/// Whether a boundary-aware `</name>` closing tag exists at or after `from`.
fn has_close_tag(bytes: &[u8], from: usize, name: &[u8]) -> bool {
    let mut i = from;
    while i < bytes.len() {
        if let Some(end) = skip_comment(bytes, i) {
            i = end;
            continue;
        }
        if match_close_tag(bytes, i, name).is_some() {
            return true;
        }
        i += 1;
    }
    false
}

/// Given `pos` at a `<table` open tag, return the index just past its matching
/// `</table>`, skipping nested tables and comments. None if unclosed.
fn skip_table(bytes: &[u8], pos: usize) -> Option<usize> {
    let mut i = end_of_tag(bytes, pos)?;
    let mut depth = 1usize;
    while i < bytes.len() {
        if let Some(end) = skip_comment(bytes, i) {
            i = end;
            continue;
        }
        if match_open_tag(bytes, i, b"table").is_some() {
            depth += 1;
            i = end_of_tag(bytes, i)?;
            continue;
        }
        if let Some(end) = match_close_tag(bytes, i, b"table") {
            depth -= 1;
            i = end;
            if depth == 0 {
                return Some(i);
            }
            continue;
        }
        i += 1;
    }
    None
}

/// Given `pos` at a `<tr` open tag, return the index just past its matching
/// `</tr>`, skipping nested tables (and their inner rows) and comments. None if
/// the row never closes.
fn scan_row(bytes: &[u8], pos: usize) -> Option<usize> {
    let open_end = end_of_tag(bytes, pos)?;
    // A self-closing `<tr/>` is a complete (empty) row.
    if bytes[pos..open_end].ends_with(b"/>") {
        return Some(open_end);
    }
    let mut i = open_end;
    while i < bytes.len() {
        if let Some(end) = skip_comment(bytes, i) {
            i = end;
            continue;
        }
        if match_open_tag(bytes, i, b"table").is_some() {
            i = skip_table(bytes, i)?;
            continue;
        }
        if let Some(end) = match_close_tag(bytes, i, b"tr") {
            return Some(end);
        }
        i += 1;
    }
    None
}

fn parse_html(text: &str) -> Option<Table> {
    let bytes = text.as_bytes();
    // Header ends after <tbody ...> if present, else before the first <tr>.
    let (header_end, footer) = match find_open_tag(bytes, 0, b"tbody") {
        Some(tb) => {
            let gt = end_of_tag(bytes, tb)?;
            let footer = if has_close_tag(bytes, 0, b"tbody") {
                "</tbody></table>"
            } else {
                "</table>"
            };
            (gt, footer.to_string())
        }
        None => {
            let first_tr = find_open_tag(bytes, 0, b"tr")?;
            (first_tr, "</table>".to_string())
        }
    };

    let header = text[..header_end].to_string();
    let mut rows = Vec::new();
    let mut pos = header_end;
    while pos < bytes.len() {
        // Comments and nested tables between rows are skipped, never counted.
        if let Some(end) = skip_comment(bytes, pos) {
            pos = end;
            continue;
        }
        // The end of our own table stops row collection.
        if match_close_tag(bytes, pos, b"table").is_some() {
            break;
        }
        if match_open_tag(bytes, pos, b"table").is_some() {
            match skip_table(bytes, pos) {
                Some(end) => pos = end,
                None => break,
            }
            continue;
        }
        if match_open_tag(bytes, pos, b"tr").is_some() {
            match scan_row(bytes, pos) {
                Some(end) => {
                    rows.push((pos, end));
                    pos = end;
                }
                None => break, // unclosed row: stop gracefully
            }
            continue;
        }
        pos += 1;
    }
    if rows.is_empty() {
        return None;
    }
    Some(Table {
        header,
        footer,
        rows,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::token_counter::{CharCounter, RowCounter};

    const MD: &str = "| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n| 5 | 6 |\n";

    #[test]
    fn row_mode_splits_and_reincludes_header() {
        let out = TableChunker::new()
            .chunk_size(2)
            .chunk(MD, &RowCounter)
            .unwrap();
        assert!(out.len() >= 2);
        for c in &out {
            assert!(c.text.contains("| A | B |"), "missing header: {:?}", c.text);
            assert!(
                c.text.contains("|---|---|"),
                "missing separator: {:?}",
                c.text
            );
        }
    }

    #[test]
    fn small_table_single_chunk() {
        let out = TableChunker::new()
            .chunk_size(10)
            .chunk(MD, &RowCounter)
            .unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].text, MD);
        assert_eq!(out[0].token_count, 3);
    }

    #[test]
    fn multiline_header_keeps_separator() {
        // #582: separator detected explicitly, not assumed at a fixed index.
        let md = "| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n";
        let out = TableChunker::new()
            .chunk_size(1)
            .chunk(md, &RowCounter)
            .unwrap();
        assert_eq!(out.len(), 2);
        for c in &out {
            assert!(c.text.contains("|---|---|"));
        }
    }

    #[test]
    fn data_region_offsets_cover_rows() {
        let out = TableChunker::new()
            .chunk_size(2)
            .chunk(MD, &RowCounter)
            .unwrap();
        // Each chunk's [start,end] slices the original into the covered rows.
        for c in &out {
            let region = &MD[c.start..c.end];
            assert!(region.contains("|") && region.trim_start().starts_with('|'));
        }
    }

    #[test]
    fn empty_returns_none() {
        assert!(
            TableChunker::new()
                .chunk("   ", &RowCounter)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn not_a_table_returns_none() {
        assert!(
            TableChunker::new()
                .chunk("just some prose\nno pipes here", &RowCounter)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn token_mode_splits_by_budget() {
        let out = TableChunker::new()
            .chunk_size(20)
            .chunk(MD, &CharCounter)
            .unwrap();
        assert!(!out.is_empty());
        for c in &out {
            assert!(c.text.contains("| A | B |"));
        }
    }

    #[test]
    fn html_table_reincludes_header() {
        let html = "<table><thead><tr><th>A</th></tr></thead><tbody><tr><td>1</td></tr><tr><td>2</td></tr><tr><td>3</td></tr></tbody></table>";
        let out = TableChunker::new()
            .chunk_size(2)
            .chunk(html, &RowCounter)
            .unwrap();
        assert!(!out.is_empty());
        for c in &out {
            assert!(c.text.starts_with("<table"));
            assert!(c.text.ends_with("</table>"));
        }
    }

    #[test]
    fn zero_chunk_size_errors() {
        let err = TableChunker::new().chunk_size(0).chunk(MD, &RowCounter);
        assert!(matches!(err, Err(ChunkError::InvalidConfig(_))));
    }

    #[test]
    fn html_attributed_and_case_variant_tags() {
        // Attributes, whitespace, and mixed case on both open and close tags.
        let html = "<table><tbody>\n<TR class=\"x\"><td>1</td></tr>\n<tr ><td>2</td></TR >\n</tbody></table>";
        let out = TableChunker::new()
            .chunk_size(1)
            .chunk(html, &RowCounter)
            .unwrap();
        assert_eq!(out.len(), 2);
        for c in &out {
            assert!(c.text.starts_with("<table"));
            assert!(c.text.ends_with("</table>"));
        }
    }

    #[test]
    fn html_rejects_lookalike_tags() {
        // <track> must not be mistaken for <tr>; both rows are real <tr> rows.
        let html = "<table><tbody><tr><td><track/></td></tr><tr><td>ok</td></tr></tbody></table>";
        let out = TableChunker::new()
            .chunk_size(1)
            .chunk(html, &RowCounter)
            .unwrap();
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn html_nested_table_rows_not_folded() {
        // The inner table's <tr> must not be collected as an outer-table row.
        let html = "<table><tbody>\
            <tr><td><table><tr><td>inner</td></tr></table></td></tr>\
            <tr><td>outer2</td></tr>\
            </tbody></table>";
        let out = TableChunker::new()
            .chunk_size(1)
            .chunk(html, &RowCounter)
            .unwrap();
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn html_comment_fake_row_ignored() {
        // The fake <tr> inside the comment must not be counted as a row. With two
        // real rows and chunk_size 1 the table splits into exactly two chunks
        // (not three), and each rebuilt chunk excludes the commented-out row.
        let html = "<table><tbody><!-- <tr><td>fake</td></tr> --><tr><td>r1</td></tr><tr><td>r2</td></tr></tbody></table>";
        let out = TableChunker::new()
            .chunk_size(1)
            .chunk(html, &RowCounter)
            .unwrap();
        assert_eq!(out.len(), 2);
        for c in &out {
            assert!(
                !c.text.contains("fake"),
                "comment leaked into chunk: {:?}",
                c.text
            );
        }
    }

    #[test]
    fn html_unclosed_row_is_graceful() {
        // A malformed, unclosed trailing row must not panic; the valid row stays.
        let html = "<table><tbody><tr><td>1</td></tr><tr><td>2";
        let out = TableChunker::new()
            .chunk_size(1)
            .chunk(html, &RowCounter)
            .unwrap();
        assert_eq!(out.len(), 1);
        assert!(out[0].text.contains("1"));
    }
}
