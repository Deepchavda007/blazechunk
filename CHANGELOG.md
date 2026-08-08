# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and the project follows
[Semantic Versioning](https://semver.org/).

## [0.15.0] - 2026-08-08

### Added
- **Document input** — `blazechunk.loaders.DocumentChunker` chunks PDF, Word, PowerPoint,
  Excel, OpenDocument, RTF, EPUB and CSV files directly, via
  [anydoc](https://github.com/firecrawl/anydoc). Install with
  `pip install "blazechunk[anydoc]"`. Markdown and plain text need no extra.
  - Accepts a path, a `Path`, or `bytes` (with `format=` for signature-less formats
    such as CSV). Returns a `ChunkedDocument` carrying the chunks, the canonical
    Markdown they index into, the source format, and any warnings.
  - Each `DocumentChunk` carries `heading_path` (the chain of headings above it),
    `kind` (`prose`/`table`/`code`/`list`/`quote`/`heading`/`rule`), `lang` for fenced
    code, `source_format`, and `token_count`.
  - **Structure-aware routing** — the document is segmented before it is chunked, so
    tables go to `TableChunker` (never split mid-row, header repeated per chunk), fenced
    code to `CodeChunker` (fences intact), and prose to the chunker you configure.
    Undersized segments merge *forward*, so a heading joins the prose it introduces.
  - Sync, async and batch throughout (`chunk`, `chunk_async`, `chunk_batch`,
    `chunk_batch_async`), with `on_error="raise"|"skip"|"collect"` for batch runs over a
    directory and bounded conversion concurrency.
  - `chunk_markdown()` applies the same structural routing to Markdown you already have —
    useful for feeding OCR output back in.
- **Named load errors** — `DocumentError` and its subclasses `UnsupportedDocument`,
  `MalformedDocument`, `EncryptedDocument`, `DocumentResourceLimit`, and
  `ScannedDocumentError` for the common case of an image-only PDF that needs OCR
  upstream. An unreadable *file* still raises `OSError`.

### Notes
- Chunk offsets from `DocumentChunker` are named `md_start` / `md_end` and index the
  **converted Markdown** (returned as `ChunkedDocument.markdown`), not the original file.
  anydoc exposes no mapping back to source bytes, so page-level attribution for PDFs is
  not available and no approximation is provided.
- The embedding-based chunkers (`SemanticChunker`, `SDPMChunker`, `LateChunker`) work with
  `DocumentChunker`, but operate *within* a structural segment and never across one.
- Scoped the `1 TB/s` figure in the README to the raw SIMD primitive it measures, rather
  than presenting it as end-to-end throughput.

## [0.13.0] - 2026-07-19

### Added
- **Three embedding-based chunkers**, bringing the total to nine:
  - `SemanticChunker` — splits at semantic-similarity troughs between sentence windows,
    using Savitzky–Golay minima detection (`threshold`, `similarity_window`,
    `filter_window`/`filter_polyorder`/`filter_tolerance`, optional `skip_window`).
  - `SDPMChunker` — Semantic Double-Pass Merging: the semantic pass plus a skip-window
    second pass that re-merges related, non-adjacent sentence groups (`skip_window`,
    default `1`).
  - `LateChunker` — "late chunking": recursive boundaries plus a whole-document,
    mean-pooled embedding per chunk. Returns `LateChunk` objects carrying an `embedding`.
- **Injected embedders** — the pure-Rust core ships no model; embeddings are injected via
  the new `Embedder` / `TokenEmbedder` traits (mirroring `TokenCounter`). In Python, pass a
  callable `embed_batch(list[str]) -> 2D` or any object exposing `embed_batch` / `encode`
  (e.g. sentence-transformers, model2vec); `LateChunker` takes a token-level embedder.
  `cosine_similarity` and deterministic test embedders are exposed from the Rust crate.

## [0.11.0]

### Added
- **LangChain integration** — `blazechunk.integrations.langchain.BlazechunkTextSplitter`,
  a `TextSplitter` backed by any blazechunk chunker. Sync (`split_text`) and async
  (`asplit_text`, `atransform_documents`). Install with `pip install "blazechunk[langchain]"`.
- **Agno integration** — `blazechunk.integrations.agno.BlazechunkChunking`, a
  `ChunkingStrategy` backed by any blazechunk chunker. Sync (`chunk`) and async
  (`achunk`). Install with `pip install "blazechunk[agno]"`.

## [0.10.3]

### Changed
- Linked the documentation site from the README and the PyPI project metadata
  (added `Documentation` / `Issues` project URLs).
- Fixed PyPI README links to absolute URLs so they resolve on the project page;
  refreshed badges.

## [0.10.2]

### Added
- Initial public release. Six chunkers — `Chunker`, `RecursiveChunker`,
  `SentenceChunker`, `TokenChunker`, `TableChunker`, `CodeChunker` — each with
  matching synchronous and asynchronous methods (`chunk`/`chunk_async`,
  `chunk_batch`/`chunk_batch_async`), a SIMD-accelerated Rust core, and typed
  Python bindings (`py.typed` + stubs).
