# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and the project follows
[Semantic Versioning](https://semver.org/).

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
