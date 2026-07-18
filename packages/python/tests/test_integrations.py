"""Tests for the optional framework adapters in blazechunk.integrations.

Each test skips unless the target framework is installed (install the matching
extra: `pip install "blazechunk[langchain]"` / `"blazechunk[agno]"`).
"""

import pytest

from blazechunk import TokenChunker

SAMPLE = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump! "
) * 20


class TestLangChainAdapter:
    def test_split_text_returns_strings(self):
        pytest.importorskip("langchain_text_splitters")
        from blazechunk.integrations.langchain import BlazechunkTextSplitter

        splitter = BlazechunkTextSplitter(TokenChunker(chunk_size=64, chunk_overlap=8))
        pieces = splitter.split_text(SAMPLE)

        assert len(pieces) > 1
        assert all(isinstance(p, str) and p for p in pieces)
        # Every piece is an exact substring of the source (blazechunk invariant).
        assert all(p in SAMPLE for p in pieces)

    def test_create_documents(self):
        pytest.importorskip("langchain_text_splitters")
        from blazechunk.integrations.langchain import BlazechunkTextSplitter

        splitter = BlazechunkTextSplitter(TokenChunker(chunk_size=64))
        docs = splitter.create_documents([SAMPLE])

        assert len(docs) > 1
        assert all(d.page_content for d in docs)

    def test_default_chunker(self):
        pytest.importorskip("langchain_text_splitters")
        from blazechunk.integrations.langchain import BlazechunkTextSplitter

        splitter = BlazechunkTextSplitter()  # defaults to RecursiveChunker
        assert splitter.split_text(SAMPLE)

    def test_async_split_text(self):
        pytest.importorskip("langchain_text_splitters")
        import asyncio

        from blazechunk.integrations.langchain import BlazechunkTextSplitter

        splitter = BlazechunkTextSplitter(TokenChunker(chunk_size=64, chunk_overlap=8))
        sync_pieces = splitter.split_text(SAMPLE)
        async_pieces = asyncio.run(splitter.asplit_text(SAMPLE))
        assert async_pieces == sync_pieces

    def test_async_transform_documents(self):
        pytest.importorskip("langchain_text_splitters")
        import asyncio

        from langchain_core.documents import Document

        from blazechunk.integrations.langchain import BlazechunkTextSplitter

        splitter = BlazechunkTextSplitter(TokenChunker(chunk_size=64))
        docs = asyncio.run(
            splitter.atransform_documents([Document(page_content=SAMPLE, metadata={"s": 1})])
        )
        assert len(docs) > 1
        assert all(d.page_content for d in docs)
        assert all(d.metadata.get("s") == 1 for d in docs)


class TestAgnoAdapter:
    def test_chunk_returns_documents(self):
        pytest.importorskip("agno")
        from blazechunk.integrations.agno import BlazechunkChunking

        try:
            from agno.knowledge.document.base import Document
        except ImportError:
            from agno.document.base import Document  # type: ignore

        strategy = BlazechunkChunking(TokenChunker(chunk_size=64, chunk_overlap=8))
        doc = Document(content=SAMPLE, name="sample", meta_data={"source": "test"})
        chunks = strategy.chunk(doc)

        assert len(chunks) > 1
        assert all(c.content for c in chunks)
        # Source metadata is carried through, chunk index is added.
        assert all(c.meta_data.get("source") == "test" for c in chunks)
        assert [c.meta_data["chunk"] for c in chunks] == list(range(1, len(chunks) + 1))

    def test_empty_document_passthrough(self):
        pytest.importorskip("agno")
        from blazechunk.integrations.agno import BlazechunkChunking

        try:
            from agno.knowledge.document.base import Document
        except ImportError:
            from agno.document.base import Document  # type: ignore

        strategy = BlazechunkChunking()
        doc = Document(content="", name="empty")
        assert strategy.chunk(doc) == [doc]

    def test_async_chunk(self):
        pytest.importorskip("agno")
        import asyncio

        from blazechunk.integrations.agno import BlazechunkChunking

        try:
            from agno.knowledge.document.base import Document
        except ImportError:
            from agno.document.base import Document  # type: ignore

        strategy = BlazechunkChunking(TokenChunker(chunk_size=64, chunk_overlap=8))
        doc = Document(content=SAMPLE, name="sample", meta_data={"source": "test"})
        sync_chunks = strategy.chunk(doc)
        async_chunks = asyncio.run(strategy.achunk(doc))
        assert [c.content for c in async_chunks] == [c.content for c in sync_chunks]
        assert len(async_chunks) > 1
