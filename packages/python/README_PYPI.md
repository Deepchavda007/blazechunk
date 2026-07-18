<h1 align="center">blazechunk</h1>

<p align="center">
  <em>the fastest semantic text chunking library — up to 1 TB/s throughput</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/blazechunk"><img src="https://img.shields.io/pypi/v/blazechunk?color=e67e22" alt="PyPI"></a>
  <a href="https://pypi.org/project/blazechunk"><img src="https://img.shields.io/pypi/pyversions/blazechunk?color=3498db" alt="Python versions"></a>
  <a href="https://blazechunk-documentation.vercel.app/"><img src="https://img.shields.io/badge/docs-blazechunk-3498db" alt="Documentation"></a>
  <a href="https://github.com/Deepchavda007/blazechunk"><img src="https://img.shields.io/badge/github-blazechunk-3498db" alt="GitHub"></a>
  <a href="https://github.com/Deepchavda007/blazechunk/blob/main/LICENSE-MIT"><img src="https://img.shields.io/badge/license-MIT%2FApache--2.0-9b59b6.svg" alt="License"></a>
</p>

---

**blazechunk** splits text at semantic boundaries and does it stupid fast: a SIMD-accelerated
Rust core with a small, uniform Python API. It ships six chunkers — a zero-copy byte `Chunker`
plus `RecursiveChunker`, `SentenceChunker`, `TokenChunker`, `TableChunker`, and `CodeChunker` —
and every high-level chunker offers **matching synchronous and asynchronous** methods.

📖 **Full documentation:** https://blazechunk-documentation.vercel.app/

## 📦 installation

```bash
pip install blazechunk
```

## 🚀 usage

### High-level chunkers (sync + async)

Every chunker exposes the same four methods, so once you know one you know them all:
`chunk` / `chunk_async` and `chunk_batch` / `chunk_batch_async`.

```python
from blazechunk import TokenChunker

chunker = TokenChunker(chunk_size=512, chunk_overlap=64)

# synchronous
chunks = chunker.chunk("... a long document ...")
for c in chunks:
    print(c.text, c.start_index, c.end_index, c.token_count)

# many documents at once
batches = chunker.chunk_batch(["doc one ...", "doc two ..."])
```

```python
import asyncio
from blazechunk import RecursiveChunker

async def main() -> None:
    chunker = RecursiveChunker(chunk_size=2048)

    # await a single document — the work runs off the event loop
    chunks = await chunker.chunk_async("... a long document ...")

    # await many documents concurrently, with optional back-pressure
    batches = await chunker.chunk_batch_async(
        ["doc one ...", "doc two ..."], max_concurrency=8
    )

asyncio.run(main())
```

Other chunkers follow the same shape:

```python
from blazechunk import SentenceChunker, TableChunker, CodeChunker

SentenceChunker(chunk_size=2048, chunk_overlap=128).chunk(prose)
TableChunker(chunk_size=3).chunk(markdown_or_html_table)   # header repeated per chunk
CodeChunker(chunk_size=2048, language="python").chunk(source_code)
```

### Low-level byte chunker (zero-copy)

The `Chunker` primitive and the `chunk()` helper yield zero-copy `memoryview` slices for
maximum throughput:

```python
from blazechunk import chunk, chunk_async

# synchronous generator of zero-copy memoryviews
for view in chunk(b"Hello. World. Test.", size=10, delimiters=b"."):
    print(bytes(view))

# async variant returns owned bytes
chunks = await chunk_async(b"Hello. World.", size=10, delimiters=b".")
```

## 🙏 acknowledgements

blazechunk is a fork of the excellent [chonkie-inc/chunk](https://github.com/chonkie-inc/chunk)
project, and builds on its SIMD chunking core. Licensed under either of
[Apache License, Version 2.0](https://github.com/Deepchavda007/blazechunk/blob/main/LICENSE-APACHE)
or [MIT license](https://github.com/Deepchavda007/blazechunk/blob/main/LICENSE-MIT) at your option.
