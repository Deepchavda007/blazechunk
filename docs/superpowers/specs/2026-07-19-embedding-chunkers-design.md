# Design: Three embedding-based chunkers (Semantic, SDPM, Late)

**Date:** 2026-07-19
**Status:** Approved (design), implementation in progress
**Author:** Deep Chavda (with Claude)
**Builds on:** `2026-07-17-rust-chunkers-design.md` (the five-chunker core)

## Goal

Add three high-level, embedding-based chunkers to the Rust `chunk` crate and expose
them through the `blazechunk` (PyO3) bindings:

1. `SemanticChunker` — splits on semantic-similarity troughs between sentence windows.
2. `SDPMChunker` — Semantic Double-Pass Merging (Semantic + a skip-window merge pass).
3. `LateChunker` — recursive boundaries + whole-document contextual token embeddings,
   mean-pooled per chunk ("late chunking").

**No LLM/generative chunker** (SlumberChunker) — explicitly out of scope per request.

## Key discovery (why this is a relocation, not an invention)

Upstream Chonkie's `SemanticChunker` **already delegates its signal processing to this
crate**: it calls `chonkie_core.find_local_minima_interpolated` and
`chonkie_core.filter_split_indices` — the exact functions in `src/savgol.rs`, already
`pub` and re-exported in `src/lib.rs`. This work relocates the *orchestration* (sentence
split → similarity → boundary detection → skip-merge → size split) from Python into the
Rust core, exactly as the five-chunker effort did. `src/savgol.rs` was built as the
groundwork for precisely this.

## The new seam: inject the embedder (never construct it)

The core is pure Rust with **no ML model**. Embeddings are injected via traits, mirroring
exactly how `TokenCounter` is injected into every existing chunker ("accept dependencies,
don't create them"). New module `src/embeddings.rs`:

```rust
/// Cosine similarity; 0.0 if either vector is zero-norm or lengths differ.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32;

/// Dense text embeddings. Injected into Semantic/SDPM. The core never loads a model.
pub trait Embedder {
    /// Embed each text into a dense vector. All returned vectors share one dimension.
    fn embed_batch(&self, texts: &[&str]) -> Vec<Vec<f32>>;
    /// Embed a single text (default: delegates to `embed_batch`).
    fn embed(&self, text: &str) -> Vec<f32> { /* default */ }
    /// Similarity between two vectors (default: cosine).
    fn similarity(&self, a: &[f32], b: &[f32]) -> f32 { cosine_similarity(a, b) }
}

/// Contextual per-token embeddings for a whole document. Injected into LateChunker.
pub trait TokenEmbedder {
    /// One embedding vector per token, in document order, for the whole text.
    fn embed_as_tokens(&self, text: &str) -> Vec<Vec<f32>>;
    /// Single-vector fallback embedding for a text span.
    fn embed(&self, text: &str) -> Vec<f32>;
}
```

Chunkers take `&dyn Embedder` / `&dyn TokenEmbedder`, matching the `&dyn TokenCounter`
pattern. Tests inject deterministic mock embedders (below) just as they inject
`CharCounter`.

## Global conventions (unchanged from the five-chunker design)

- **Byte offsets everywhere.** `Chunk.start/end` are byte offsets into the original text.
- **Slice invariant** holds for Semantic and SDPM: `chunk.text == original[start..end]`.
  Semantic/SDPM chunks *partition* the text (no overlap): sentences from
  `split_by_delimiters` are a contiguous byte-span partition, and every operation
  (grouping, skip-merge, size-split) only concatenates or subdivides consecutive
  sentences, so byte-preservation + contiguity + slice invariant all hold by construction.
- **LateChunker** returns a distinct `LateChunk { start, end, token_count, embedding }`
  (parallel to `TableChunk`). Its `start/end` come from `RecursiveChunker`, so the slice
  invariant holds for the byte range; the `embedding` is the pooled late-interaction
  vector — the chunker's whole point. This keeps the core `Chunk` free of an `embedding`
  field (which the five-chunker design put out of scope), while still returning it where
  it is the deliverable.
- **Empty/whitespace:** all three treat `text.trim().is_empty()` as `[]` (they match the
  four non-recursive chunkers; Late's *internal* recursion still only short-circuits on
  truly empty, but the outer guard returns `[]` for whitespace-only, matching upstream
  `SemanticChunker`/`LateChunker`).

## Per-chunker specifications (from upstream source, verbatim behavior)

### 1. SemanticChunker
Upstream defaults (`chonkie/chunker/semantic.py`, main):

| Param | Default | Meaning |
|---|---|---|
| `threshold` | `0.8` | Percentile (0–1, exclusive) passed to `filter_split_indices` |
| `chunk_size` | `2048` | Max tokens per chunk |
| `similarity_window` | `3` | # sentences in each comparison window |
| `min_sentences_per_chunk` | `1` | Min distance between splits (`filter_split_indices` min_distance) |
| `min_characters_per_sentence` | `24` | Short fragments merge into neighbours |
| `delim` | `[". ", "! ", "? ", "\n"]` | Sentence delimiters |
| `include_delim` | `Prev` | Delimiter attaches to previous sentence |
| `skip_window` | `0` | 0 = no skip-merge; >0 enables it |
| `filter_window` | `5` | Savitzky–Golay window (odd) |
| `filter_polyorder` | `3` | Savitzky–Golay poly order (`0 <= p < window`) |
| `filter_tolerance` | `0.2` | 1st-derivative zero tolerance for minima |

Validation (mirrors upstream): `chunk_size>0`, `similarity_window>0`,
`min_sentences_per_chunk>0`, `skip_window>=0`, `0<threshold<1`, `filter_window>0`,
`0<=filter_polyorder<filter_window`, `0<filter_tolerance<1`.

Algorithm (`chunk(text, counter, embedder)`):
1. `text.trim().is_empty()` → `[]`.
2. `sentences` = `split_by_delimiters(text, delim, include_delim, min_characters_per_sentence)`
   (contiguous byte-span partition). Empty → `[]`.
3. `n = sentences.len()`. If `n <= similarity_window` → one chunk spanning all sentences.
4. **Similarities** (`_get_similarity`), length `n - similarity_window`: for
   `i in 0..(n - similarity_window)`:
   - `window_text = concat(sentences[i .. i+similarity_window])`,
     `sentence_text = sentences[i + similarity_window]`
   - `sim[i] = embedder.similarity(embed(window_text), embed(sentence_text))`
   (batch all window texts and all sentence texts in two `embed_batch` calls).
5. **Split indices** (`_get_split_indices`):
   - `len(sim) < filter_window` → no internal splits (one group).
   - `minima = find_local_minima_interpolated(sim, filter_window, filter_polyorder, filter_tolerance)`;
     `None`/empty → no internal splits.
   - `filtered = filter_split_indices(minima.indices, minima.values, threshold, min_sentences_per_chunk)`.
   - boundaries = `[0] ++ [i + similarity_window for i in filtered.indices] ++ [n]`.
6. **Group** sentences between consecutive boundaries (contiguous ranges).
7. **Skip-merge** if `skip_window > 0` (see shared `skip_and_merge`, best-candidate variant
   from main).
8. **Size split** (`_split_groups`): any group whose summed token_count exceeds `chunk_size`
   is greedily re-split at sentence boundaries.
9. **Emit** one `Chunk` per final group: `start = first.start`, `end = last.end`,
   `token_count = counter.count(text[start..end])`.

`skip_and_merge(groups, embedder, threshold, skip_window)` (main-branch, best-candidate):
- `groups.len() <= 1 || skip_window == 0` → return unchanged.
- Embed each group's joined text once.
- `i = 0`; loop: last group → push, done. Else `skip_index = min(i+skip_window+1, len-1)`;
  scan `j in (i+1)..=skip_index`, pick the `j` with the highest `similarity(emb[i],emb[j])`
  that is `>= threshold`. If found → merge `groups[i..=j]`, `i = j+1`; else push `groups[i]`,
  `i += 1`. (Here `threshold` is used as a *raw cosine* cutoff, not a percentile — matches
  upstream's dual use of the same field.)

### 2. SDPMChunker
Semantic Double-Pass Merging = `SemanticChunker` with the skip-merge pass on. Distinct
public type (own `sdpm.rs`) that delegates to the shared semantic engine with
`skip_window >= 1` (default `1`), so the double pass is always active. Same params as
Semantic plus `skip_window` (default `1`, validated `>= 1`). All offset/invariant
properties are identical to Semantic.

### 3. LateChunker
Upstream defaults (`chonkie/chunker/late.py`): `chunk_size=2048`,
`rules=RecursiveRules::default()`, `min_characters_per_chunk=24`, embedding_model injected.

Algorithm (`chunk(text, counter, token_embedder) -> Vec<LateChunk>`):
1. `text.trim().is_empty()` → `[]`.
2. `chunks = RecursiveChunker::new().chunk_size(cs).min_characters_per_chunk(mc).rules(r)
   .chunk(text, counter)?` — byte-offset chunks + per-chunk `token_count`.
3. `token_embeddings = token_embedder.embed_as_tokens(text)`; `m = token_embeddings.len()`.
4. `token_counts = [c.token_count for c in chunks]`; `total = sum`.
5. **Fallback**: if `m < total`, re-embed each chunk as one vector
   (`token_embedder.embed(chunk_text)`), set every `token_count = 1`, `m = chunks.len()`.
6. If `sum(token_counts) > m` → `ChunkError::InvalidConfig` (counts exceed tokens).
7. If `sum(token_counts) < m`: `diff = m - sum`; `token_counts[0] += diff/2`,
   `token_counts[last] += diff - diff/2` (rebalance, matches upstream).
8. Assert `sum == m`.
9. Mean-pool: `emb[i] = mean(token_embeddings[cum[i]..cum[i+1]])` over the token axis.
10. Emit `LateChunk { start, end, token_count (adjusted), embedding }`.

Alignment note: upstream uses the embedding model's own tokenizer for both counting and
token embeddings, so counts align exactly; the rebalance step absorbs small drift. In our
injected design the caller should pass a `counter` consistent with the token embedder; the
rebalance keeps it robust otherwise.

## Test embedders (deterministic, like `CharCounter`)

`src/embeddings.rs` ships (behind `#[cfg(test)]`, or `pub` for cross-module reuse):
- **`HashEmbedder`** — hashes character trigrams into a fixed-dim vector; deterministic,
  dependency-free. Used for invariant/robustness tests.
- **`TopicEmbedder`** — maps each of a few marker keywords to a basis direction; sentences
  sharing keywords get near-parallel vectors. Lets tests *construct* a document with a
  known semantic boundary and assert the split lands there.
- **`CharTokenEmbedder`** — `embed_as_tokens` returns one deterministic vector per code
  point (so `m == CharCounter` token total exactly, giving exact alignment); `embed`
  returns the mean. Used for LateChunker tests, including asserting a chunk's pooled
  embedding equals the mean of its span's per-char vectors.

## Testing strategy (per chunker, test-first)

Every module asserts, at minimum:
- **byte-preservation** (Semantic/SDPM: chunks sum to `text.len()`), **contiguity**, and the
  **slice invariant** (`chunk.text == text[start..end]`);
- **defaults** match upstream;
- **config validation** errors (each invalid field);
- **whitespace-only → `[]`**, **too-few-sentences → one chunk**;
- **semantic behavior** via `TopicEmbedder`: a two-topic document splits at the topic
  boundary; SDPM re-merges skip-adjacent same-topic groups that Semantic left separate;
- **CJK/multibyte** sentences stay valid UTF-8 (reuses the #536-safe `split_by_delimiters`);
- **Late**: pooled embedding equals the mean over the chunk's token span; the
  rebalance/fallback branches are exercised; `LateChunk` byte ranges satisfy the slice
  invariant.

Rust: `cargo test` (co-located `#[cfg(test)] mod tests`), `cargo fmt --check`,
`cargo clippy -- -D warnings`.

## Bindings plan (after the Rust core is green)

- **PyO3** (`packages/python/src/lib.rs`): a `PyEmbedder`/`PyTokenEmbedder` that wraps a
  Python callable/object (`embed_batch(list[str]) -> list[list[float]] | np.ndarray`,
  `embed_as_tokens(str) -> np.ndarray`) and implements the core traits by re-acquiring the
  GIL. Three `#[pyclass]` wrappers (`SemanticChunker`, `SDPMChunker`, `LateChunker`)
  registered in the module; `LateChunker` returns chunks carrying an `embedding` (list of
  floats). `blazechunk/chunkers.py` gains matching Python wrappers (sync + async,
  type-hinted) that accept an `embedding_model` (any object exposing `embed_batch`/`encode`,
  or a plain callable).
- **WASM**: accept a JS embedding function (or precomputed vectors) — lower priority; a
  precomputed-embedding entry point is the minimum.

## Build order (one chunker at a time, test-first)
1. `src/embeddings.rs` — traits, `cosine_similarity`, test embedders. `cargo test`.
2. `SemanticChunker` (`src/chunkers/semantic.rs`) + tests. `cargo test semantic`.
3. `SDPMChunker` (`src/chunkers/sdpm.rs`, shares the semantic engine) + tests.
4. `LateChunker` (`src/chunkers/late.rs`) + `LateChunk` + tests.
5. Wire `mod`/`lib` re-exports; full `cargo test`, `fmt`, `clippy`.
6. PyO3 bindings + `blazechunk/chunkers.py` + `.pyi`; `maturin develop`; `pytest`.
7. `main.py` demos, README/CHANGELOG updates, version bump.

## Out of scope
- SlumberChunker (LLM/generative).
- Bundling an embedding model in the Rust core (trait seam left open; a model2vec-style
  static embedder could satisfy `Embedder` later without chunker changes).
- `from_recipe`/Hub-backed recipes; batch multiprocessing.
