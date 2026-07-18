<p>
  <div align="center">
  <h1>
    blazechunk<br /> <br />
    <a href="https://pypi.python.org/pypi/blazechunk">
      <img
        src="https://img.shields.io/pypi/v/blazechunk"
        alt="Python Package"
      />
    </a>
    <a href="https://github.com/Deepchavda007/blazechunk/actions/workflows/ci.yml">
      <img
        src="https://github.com/Deepchavda007/blazechunk/actions/workflows/ci.yml/badge.svg"
        alt="CI"
      />
    </a>
    <a href="https://pypi.python.org/pypi/blazechunk">
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
Rust core with a small, uniform Python API. It ships six chunkers, and every high-level chunker
offers **matching synchronous and asynchronous** methods with full type hints and docstrings.

### Features

- ⚡ **SIMD-accelerated Rust core** — up to ~1 TB/s on the raw chunking primitive.
- 🧩 **Six chunkers** — a zero-copy byte `Chunker` plus `RecursiveChunker`, `SentenceChunker`,
  `TokenChunker`, `TableChunker`, and `CodeChunker`.
- 🔁 **Sync *and* async** — every chunker has `chunk` / `chunk_async` and
  `chunk_batch` / `chunk_batch_async`; async work runs off the event loop.
- 🔤 **Pluggable tokenizers** — count by character, word, byte, or table row out of the box,
  or point at a HuggingFace `tokenizer.json` (with the `hf-tokenizer` build).
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

#### The six chunkers

| Chunker            | Splits on                                             |
|--------------------|-------------------------------------------------------|
| `Chunker`          | byte-size windows at delimiter boundaries (zero-copy) |
| `RecursiveChunker` | a hierarchy: paragraphs → sentences → … → tokens      |
| `SentenceChunker`  | whole sentences, with optional overlap                |
| `TokenChunker`     | fixed-size token windows, with optional overlap       |
| `TableChunker`     | Markdown/HTML table rows (header repeated per chunk)  |
| `CodeChunker`      | structural code blocks (brace/indent aware)           |

```python
from blazechunk import SentenceChunker, TableChunker, CodeChunker

SentenceChunker(chunk_size=2048, chunk_overlap=128).chunk(prose)
TableChunker(chunk_size=3).chunk(markdown_or_html_table)
CodeChunker(chunk_size=2048, language="python").chunk(source_code)
```

#### Zero-copy fast path

The `Chunker` primitive and the `chunk()` helper yield zero-copy `memoryview` slices for
maximum throughput:

```python
from blazechunk import chunk

for view in chunk(b"Hello. World. Test.", size=10, delimiters=b"."):
    print(bytes(view))
```

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
