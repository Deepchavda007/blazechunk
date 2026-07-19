<p>
  <div align="center">
  <h1>
    blazechunk<br /> <br />
    <a href="https://pypi.org/project/blazechunk/">
      <img
        src="https://img.shields.io/pypi/v/blazechunk"
        alt="Python Package"
      />
    </a>
    <a href="https://blazechunk-documentation.vercel.app/">
      <img
        src="https://img.shields.io/badge/docs-blazechunk-3498db"
        alt="Documentation"
      />
    </a>
    <a href="https://github.com/Deepchavda007/blazechunk/actions/workflows/ci.yml">
      <img
        src="https://github.com/Deepchavda007/blazechunk/actions/workflows/ci.yml/badge.svg"
        alt="CI"
      />
    </a>
    <a href="https://pypi.org/project/blazechunk/">
      <img
        src="https://img.shields.io/pypi/pyversions/blazechunk"
        alt="Python Versions"
      />
    </a>
    <a href="https://github.com/psf/black">
      <img
        src="https://img.shields.io/badge/code%20style-black-000000.svg"
        alt="The Uncompromising Code Formatter"
      />
    </a>
    <a href="https://github.com/Deepchavda007/blazechunk/blob/main/CONTRIBUTING.md">
      <img
        src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg"
        alt="PRs Welcome"
      />
    </a>
    <a href="https://opensource.org/licenses/MIT">
      <img
        src="https://img.shields.io/badge/License-MIT%2FApache--2.0-blue.svg"
        alt="License: MIT/Apache-2.0"
      />
    </a>
  </h1>
  <em>the fastest semantic text chunking library — up to 1 TB/s throughput</em>
  </div>
</p>

**blazechunk** splits text at semantic boundaries and does it stupid fast: a SIMD-accelerated
Rust core with a small, uniform Python API. It ships nine chunkers, and every high-level chunker
offers **matching synchronous and asynchronous** methods with full type hints and docstrings.

📖 **Documentation:** https://blazechunk-documentation.vercel.app/

### Features

- ⚡ **SIMD-accelerated Rust core** — up to ~1 TB/s on the raw chunking primitive.
- 🧩 **Nine chunkers** — a zero-copy byte `Chunker` plus `RecursiveChunker`, `SentenceChunker`,
  `TokenChunker`, `TableChunker`, `CodeChunker`, and the embedding-based `SemanticChunker`,
  `SDPMChunker`, and `LateChunker`.
- 🔁 **Sync *and* async** — every chunker has `chunk` / `chunk_async` and
  `chunk_batch` / `chunk_batch_async`; async work runs off the event loop.
- 🔤 **Pluggable tokenizers** — count by character, word, byte, or table row out of the box,
  or point at a HuggingFace `tokenizer.json` (with the `hf-tokenizer` build).
- 🔌 **Framework integrations** — drop-in adapters for **LangChain** and **Agno**
  (`pip install "blazechunk[langchain]"` / `"blazechunk[agno]"`).
- 🧵 **Typed & documented** — ships `py.typed` and type stubs; every method has a docstring.

### Installation

Install using pip

```bash
$ pip install -U blazechunk
```

### Usage

Every high-level chunker exposes the same four methods, so once you know one you know them all.

```python
from blazechunk import TokenChunker

chunker = TokenChunker(chunk_size=512, chunk_overlap=64)

# chunk a single document
for c in chunker.chunk("... a long document ..."):
    print(c.text, c.start_index, c.end_index, c.token_count)

# chunk many documents
batches = chunker.chunk_batch(["doc one ...", "doc two ..."])
```

#### Async

```python
import asyncio
from blazechunk import RecursiveChunker

async def main() -> None:
    chunker = RecursiveChunker(chunk_size=2048)

    # await a single document — the work runs off the event loop
    chunks = await chunker.chunk_async("... a long document ...")

    # await many, with optional back-pressure
    batches = await chunker.chunk_batch_async(
        ["doc one ...", "doc two ..."], max_concurrency=8
    )

asyncio.run(main())
```

#### The nine chunkers

| Chunker            | Splits on                                             |
|--------------------|-------------------------------------------------------|
| `Chunker`          | byte-size windows at delimiter boundaries (zero-copy) |
| `RecursiveChunker` | a hierarchy: paragraphs → sentences → … → tokens      |
| `SentenceChunker`  | whole sentences, with optional overlap                |
| `TokenChunker`     | fixed-size token windows, with optional overlap       |
| `TableChunker`     | Markdown/HTML table rows (header repeated per chunk)  |
| `CodeChunker`      | structural code blocks (brace/indent aware)           |
| `SemanticChunker`  | semantic-similarity troughs between sentence windows  |
| `SDPMChunker`      | semantic + a skip-window double-pass merge            |
| `LateChunker`      | recursive boundaries + mean-pooled "late" embeddings  |

```python
from blazechunk import SentenceChunker, TableChunker, CodeChunker

SentenceChunker(chunk_size=2048, chunk_overlap=128).chunk(prose)
TableChunker(chunk_size=3).chunk(markdown_or_html_table)
CodeChunker(chunk_size=2048, language="python").chunk(source_code)
```

#### Embedding-based chunkers

`SemanticChunker`, `SDPMChunker`, and `LateChunker` need vectors — and the pure-Rust core
ships **no model**. You *inject* an embedder, exactly like you inject a tokenizer; the Rust
orchestration calls back into it. Pass any callable `embed_batch(list[str]) -> 2D`, or an
object exposing `embed_batch` / `encode` (e.g. sentence-transformers or model2vec):

```python
from sentence_transformers import SentenceTransformer
from blazechunk import SemanticChunker, SDPMChunker, LateChunker

model = SentenceTransformer("all-MiniLM-L6-v2")

# SemanticChunker / SDPMChunker: one vector per sentence window.
semantic = SemanticChunker(model, threshold=0.8, chunk_size=2048)
chunks = semantic.chunk(prose)          # -> list[Chunk], partitions the text

# SDPM adds a skip-window second pass that re-merges related, non-adjacent groups.
sdpm = SDPMChunker(model, skip_window=1).chunk(prose)

# LateChunker embeds the whole document once, then mean-pools per chunk. It needs
# token-level embeddings: an object with embed_as_tokens(text) and embed(text)
# (or a (embed_as_tokens, embed) tuple). Each result is a LateChunk with `.embedding`.
late = LateChunker(token_model, chunk_size=2048).chunk(document)
vec = late[0].embedding                 # the chunk's late-interaction vector
```

#### Zero-copy fast path

The `Chunker` primitive and the `chunk()` helper yield zero-copy `memoryview` slices for
maximum throughput:

```python
from blazechunk import chunk

for view in chunk(b"Hello. World. Test.", size=10, delimiters=b"."):
    print(bytes(view))
```

### Integrations

blazechunk plugs into popular RAG frameworks — install the matching extra.

**LangChain**

```bash
pip install "blazechunk[langchain]"
```

```python
from blazechunk import TokenChunker
from blazechunk.integrations.langchain import BlazechunkTextSplitter

splitter = BlazechunkTextSplitter(TokenChunker(chunk_size=512, chunk_overlap=64))
docs = splitter.create_documents([text])
```

**Agno**

```bash
pip install "blazechunk[agno]"
```

```python
from blazechunk import TokenChunker
from blazechunk.integrations.agno import BlazechunkChunking

strategy = BlazechunkChunking(TokenChunker(chunk_size=512, chunk_overlap=64))
# TextKnowledgeBase(path="docs", vector_db=..., chunking_strategy=strategy)
```

Both adapters also provide async variants — LangChain `asplit_text` / `atransform_documents`
and Agno `achunk` — so they run off the event loop in async ingestion pipelines.

### Benchmarks

Throughput of the raw SIMD size-based chunking primitive, measured on enwik8/enwik9
(Wikipedia extracts) on an Apple Silicon MacBook:

| Input           | Chunk size | Throughput |
|-----------------|------------|------------|
| enwik8 (100 MB) | 32 KB      | **1.7 TB/s** |
| enwik8 (100 MB) | 16 KB      | 680 GB/s   |
| enwik8 (100 MB) | 4 KB       | 190 GB/s   |
| enwik9 (1 GB)   | 32 KB      | 1 TB/s     |
| enwik9 (1 GB)   | 4 KB       | 50 GB/s    |

The five high-level chunkers do more work (tokenize + split + merge), so their throughput is
naturally lower — reported per-chunker so the numbers stay honest:

| Chunker            | Input                | Throughput  |
|--------------------|----------------------|-------------|
| `RecursiveChunker` | 1 MB prose           | ~1.0 GiB/s  |
| `TableChunker`     | 2000-row markdown    | ~1.7 GiB/s  |
| `CodeChunker`      | ~25 KB Rust source   | ~800 MiB/s  |
| `SentenceChunker`  | 50 KB prose          | ~670 MiB/s  |
| `TokenChunker`     | 50 KB prose          | ~640 MiB/s  |

Reproduce with `cargo bench` (see [benches/README.md](benches/README.md) for the full table
and methodology).

### Contributing

blazechunk is open source and **contributions are very welcome** — a bug report, a new
chunker, a performance win, or a docs fix all help.

- 🐛 Found a bug or want a feature? [Open an issue](https://github.com/Deepchavda007/blazechunk/issues).
- 🔧 Want to send a change? See **[CONTRIBUTING.md](CONTRIBUTING.md)** for the dev setup
  (Rust + maturin), how to run the tests, and the PR checklist.

By contributing you agree your work is dual-licensed under MIT and Apache-2.0, matching
the project.

<a href="https://github.com/Deepchavda007/blazechunk/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Deepchavda007/blazechunk" alt="Contributors" />
</a>

### Acknowledgements

blazechunk is a fork of the excellent [chonkie-inc/chunk](https://github.com/chonkie-inc/chunk)
project and builds on its SIMD chunking core.

### License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
