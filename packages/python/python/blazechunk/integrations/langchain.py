"""LangChain adapter for blazechunk.

Exposes :class:`BlazechunkTextSplitter`, a LangChain ``TextSplitter`` backed by
any blazechunk chunker. Requires the ``langchain`` extra::

    pip install "blazechunk[langchain]"
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

from langchain_core.documents import Document
from langchain_text_splitters import TextSplitter

from blazechunk import BaseChunker, RecursiveChunker

__all__ = ["BlazechunkTextSplitter"]


class BlazechunkTextSplitter(TextSplitter):
    """A LangChain ``TextSplitter`` backed by a blazechunk chunker.

    Chunk sizing is governed entirely by the blazechunk chunker you pass in
    (its ``chunk_size`` / ``chunk_overlap`` / tokenizer settings). This splitter
    overrides ``split_text`` completely, so LangChain's own ``chunk_size`` /
    ``chunk_overlap`` constructor arguments are **not** used for splitting —
    configure the chunker instead. Other base arguments (e.g. ``add_start_index``)
    still apply to the ``Document`` objects produced by ``create_documents`` /
    ``split_documents``.

    Because blazechunk guarantees ``chunk.text == original[start:end]`` (chunks
    are exact substrings), LangChain's ``add_start_index=True`` locates each
    chunk reliably.

    Args:
        chunker: Any blazechunk chunker (``TokenChunker``, ``SentenceChunker``,
            ``RecursiveChunker``, ``TableChunker``, ``CodeChunker``). Defaults to
            ``RecursiveChunker(chunk_size=2048)``.
        **kwargs: Forwarded to ``langchain_text_splitters.TextSplitter``.

    Example:
        >>> from blazechunk import TokenChunker
        >>> from blazechunk.integrations.langchain import BlazechunkTextSplitter
        >>> splitter = BlazechunkTextSplitter(TokenChunker(chunk_size=512, chunk_overlap=64))
        >>> pieces = splitter.split_text("...long document...")
        >>> docs = splitter.create_documents(["...long document..."])
    """

    def __init__(self, chunker: Optional[BaseChunker] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._chunker: BaseChunker = chunker or RecursiveChunker(chunk_size=2048)

    def split_text(self, text: str) -> List[str]:
        """Split ``text`` into a list of chunk strings, in order."""
        return [chunk.text for chunk in self._chunker.chunk(text)]

    async def asplit_text(self, text: str) -> List[str]:
        """Async ``split_text`` — chunks off the event loop via ``chunk_async``."""
        return [chunk.text for chunk in await self._chunker.chunk_async(text)]

    async def atransform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> List[Document]:
        """Async document transform — chunks all documents concurrently.

        Uses the chunker's ``chunk_batch_async`` so the event loop stays free.
        Each source document's metadata is copied onto its chunks. (Unlike the
        sync path, ``add_start_index`` is not applied here.)
        """
        docs = list(documents)
        batches = await self._chunker.chunk_batch_async([d.page_content for d in docs])
        out: List[Document] = []
        for source, chunks in zip(docs, batches):
            for chunk in chunks:
                out.append(Document(page_content=chunk.text, metadata=dict(source.metadata)))
        return out
