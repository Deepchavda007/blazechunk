<h1 align="center">blazechunk</h1>

<p align="center">
  <em>the fastest semantic text chunking library — now reads PDFs, Word, PowerPoint and Excel</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/blazechunk"><img src="https://badgen.net/badge/pypi/v0.15.0/blue" alt="PyPI version"></a>
  <a href="https://pypi.org/project/blazechunk"><img src="https://img.shields.io/pypi/pyversions/blazechunk" alt="Python versions"></a>
  <a href="https://blazechunk-documentation.vercel.app/"><img src="https://img.shields.io/badge/docs-blazechunk-3498db" alt="Documentation"></a>
  <a href="https://github.com/Deepchavda007/blazechunk"><img src="https://img.shields.io/badge/github-blazechunk-3498db" alt="GitHub"></a>
  <a href="https://github.com/Deepchavda007/blazechunk/blob/main/LICENSE-MIT"><img src="https://img.shields.io/badge/license-MIT%2FApache--2.0-9b59b6.svg" alt="License"></a>
</p>

---

**blazechunk** splits text at semantic boundaries and does it stupid fast: a SIMD-accelerated
Rust core with a small, uniform Python API. It ships nine chunkers — a zero-copy byte `Chunker`
plus `RecursiveChunker`, `SentenceChunker`, `TokenChunker`, `TableChunker`, `CodeChunker`, and the
embedding-based `SemanticChunker`, `SDPMChunker` and `LateChunker` — and every high-level chunker
offers **matching synchronous and asynchronous** methods.

**New in 0.15:** `DocumentChunker` reads PDF, Word, PowerPoint, Excel, OpenDocument, RTF, EPUB
and CSV files directly, and produces chunks that know which heading they came from.

📖 **Full documentation:** https://blazechunk-documentation.vercel.app/

## 📦 installation

```bash
pip install blazechunk

pip install "blazechunk[anydoc]"   # + PDF / Word / PowerPoint / Excel / EPUB / CSV
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

## 📄 documents (PDF, Word, PowerPoint, Excel, …)

Every RAG pipeline starts with a file, not a string. `DocumentChunker` takes the file.

```bash
pip install "blazechunk[anydoc]"
```

```python
from blazechunk.loaders import DocumentChunker

result = DocumentChunker().chunk("report.pdf")

for c in result.chunks:
    print(c.heading_path, c.kind, c.text[:60])
    # ('Methods', 'Sample Preparation')  prose  'We sampled two hundred sites across …'
```

Conversion is handled by [anydoc](https://github.com/firecrawl/anydoc) — a pure-Rust converter
from Firecrawl with no ML and no network calls.

| Format | Extensions |
|---|---|
| PDF | `.pdf` |
| Word | `.doc`, `.docx`, `.docm` |
| PowerPoint | `.ppt`, `.pptx`, `.pptm`, `.pps`, `.ppsx`, `.pot` |
| Excel | `.xls`, `.xlsx`, `.xlsm`, `.xlsb` |
| OpenDocument | `.odt`, `.ods`, `.odp` |
| RTF / EPUB / CSV | `.rtf`, `.epub`, `.csv` |
| Markdown / text | `.md`, `.txt` — no extra required |

### Why not just convert and chunk?

Converting a file to Markdown and handing the string to a text chunker throws the structure away
on the way in. The chunker then guesses it back from punctuation, and splits tables mid-row and
functions mid-body because it has no idea they are there.

`DocumentChunker` segments the document **first**, then routes each piece to a chunker that suits
it — table rows to `TableChunker`, fenced code to `CodeChunker`, prose to whichever chunker you
picked:

```python
from blazechunk import RecursiveChunker, TableChunker, CodeChunker
from blazechunk.loaders import DocumentChunker

loader = DocumentChunker(
    chunker=RecursiveChunker(chunk_size=2048),   # prose
    table_chunker=TableChunker(chunk_size=3),    # rows stay whole, header repeated
    code_chunker=CodeChunker(chunk_size=2048),   # fences stay intact
    respect_headings=True,                       # never merge across a heading
    min_chunk_size=256,                          # merge undersized neighbours
)

result = loader.chunk("handbook.docx")
result = loader.chunk(pdf_bytes, format="pdf")   # bytes work too
```

Async and batch mirror the rest of the library:

```python
result  = await loader.chunk_async("report.pdf")
results = await loader.chunk_batch_async(paths, max_concurrency=8)

# skip the files that cannot be read instead of stopping the run
results = loader.chunk_batch(paths, on_error="skip")
```

### heading_path is the point

Each chunk carries the chain of headings above it, which is what turns an anonymous fragment into
something a retriever can place — and what a reranker can use directly:

```python
for c in result.chunks:
    store.add(
        text=c.text,
        metadata={
            "section": " > ".join(c.heading_path),   # "Methods > Sample Preparation"
            "kind": c.kind,                          # prose | table | code | list | quote
            "format": c.source_format,               # pdf, docx, …
        },
    )
```

### Offsets and provenance

`md_start` / `md_end` are byte offsets into `result.markdown` — **the converted Markdown, which
is returned alongside the chunks** — and not into the original file. They are named `md_*` rather
than `start`/`end` precisely so they are not mistaken for offsets into your input.

```python
data = result.markdown_bytes
assert data[c.md_start:c.md_end].decode() == c.text   # for every chunk with c.is_exact
```

anydoc exposes no mapping back to source bytes, so **a page number for a PDF chunk is not
something this can honestly provide.** If you need page-level attribution for audit or compliance,
this path does not give you it, and no approximation is shipped in its place.

Two guarantees hold, and are enforced by the test suite over thousands of generated documents:

- **Exact reconstruction** — a chunk with `is_exact` is byte-for-byte the slice its offsets name.
  The only chunks where this is false are the second and later chunks of a split table, which
  repeat the header row so each one reads on its own.
- **Full coverage** — the chunks' spans tile the document in order, without overlap, and
  everything they leave out is whitespace. Nothing is silently dropped.

### Scanned PDFs

anydoc reads the text layer of a PDF; it does not do OCR. A scanned or image-only PDF opens fine
in any reader but has no text to extract, so it raises a named error rather than a puzzling
"unsupported format":

```python
from blazechunk.loaders import DocumentChunker, ScannedDocumentError, DocumentError

try:
    result = DocumentChunker().chunk("scan.pdf")
except ScannedDocumentError:
    ...      # run OCR upstream, then pass the text back in
except DocumentError:
    ...      # malformed, encrypted, unsupported — catches every load failure
```

Run OCR first (or use Firecrawl Parse, the hosted API that adds OCR models), then feed the result
back through `chunk_markdown` to keep the same structural routing:

```python
result = DocumentChunker().chunk_markdown(text_from_ocr)
```

## 🔌 integrations

blazechunk plugs into popular RAG frameworks — install the matching extra.

```bash
pip install "blazechunk[langchain]"   # LangChain
pip install "blazechunk[agno]"        # Agno
```

```python
# LangChain
from blazechunk import TokenChunker
from blazechunk.integrations.langchain import BlazechunkTextSplitter

splitter = BlazechunkTextSplitter(TokenChunker(chunk_size=512, chunk_overlap=64))
docs = splitter.create_documents([text])

# Agno
from blazechunk.integrations.agno import BlazechunkChunking

strategy = BlazechunkChunking(TokenChunker(chunk_size=512, chunk_overlap=64))
```

## 🙏 acknowledgements

blazechunk is a fork of the excellent [chonkie-inc/chunk](https://github.com/chonkie-inc/chunk)
project, and builds on its SIMD chunking core. Licensed under either of
[Apache License, Version 2.0](https://github.com/Deepchavda007/blazechunk/blob/main/LICENSE-APACHE)
or [MIT license](https://github.com/Deepchavda007/blazechunk/blob/main/LICENSE-MIT) at your option.
