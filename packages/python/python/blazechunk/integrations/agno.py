"""Agno adapter for blazechunk.

Exposes :class:`BlazechunkChunking`, an Agno ``ChunkingStrategy`` backed by any
blazechunk chunker. Requires the ``agno`` extra::

    pip install "blazechunk[agno]"
"""

from __future__ import annotations

from typing import List, Optional

try:  # Agno >= ~1.5 (current module layout)
    from agno.knowledge.chunking.strategy import ChunkingStrategy
    from agno.knowledge.document.base import Document
except ImportError:  # older Agno
    from agno.document.chunking.strategy import ChunkingStrategy  # type: ignore
    from agno.document.base import Document  # type: ignore

from blazechunk import BaseChunker, Chunk, RecursiveChunker

__all__ = ["BlazechunkChunking"]


class BlazechunkChunking(ChunkingStrategy):
    """An Agno ``ChunkingStrategy`` backed by a blazechunk chunker.

    Chunk sizing is governed by the blazechunk chunker you pass in. Each returned
    ``Document`` carries the source document's ``name`` and metadata, plus a
    ``chunk`` index and ``chunk_size`` (character count) in its ``meta_data``.

    Provides both the synchronous ``chunk`` and the asynchronous ``achunk`` hooks
    Agno calls, so it works in both sync and async ingestion pipelines.

    Args:
        chunker: Any blazechunk chunker (``TokenChunker``, ``SentenceChunker``,
            ``RecursiveChunker``, ``TableChunker``, ``CodeChunker``). Defaults to
            ``RecursiveChunker(chunk_size=5000)``.

    Example:
        >>> from blazechunk import TokenChunker
        >>> from blazechunk.integrations.agno import BlazechunkChunking
        >>> strategy = BlazechunkChunking(TokenChunker(chunk_size=512, chunk_overlap=64))
        >>> # pass to a knowledge base / reader:
        >>> #   TextKnowledgeBase(path="docs", vector_db=..., chunking_strategy=strategy)
    """

    def __init__(self, chunker: Optional[BaseChunker] = None) -> None:
        self.chunker: BaseChunker = chunker or RecursiveChunker(chunk_size=5000)

    def chunk(self, document: Document) -> List[Document]:
        """Split an Agno ``Document`` into a list of chunked ``Document`` objects."""
        if not document.content:
            return [document]
        pieces = self.chunker.chunk(self.clean_text(document.content))
        return self._to_documents(document, pieces)

    async def achunk(self, document: Document) -> List[Document]:
        """Async ``chunk`` — chunks off the event loop via ``chunk_async``."""
        if not document.content:
            return [document]
        pieces = await self.chunker.chunk_async(self.clean_text(document.content))
        return self._to_documents(document, pieces)

    def _to_documents(self, document: Document, pieces: List[Chunk]) -> List[Document]:
        chunked: List[Document] = []
        for index, piece in enumerate(pieces, start=1):
            meta_data = dict(document.meta_data) if document.meta_data else {}
            meta_data["chunk"] = index
            meta_data["chunk_size"] = len(piece.text)
            chunked.append(
                Document(
                    id=f"{document.id}_{index}" if document.id else None,
                    name=document.name,
                    meta_data=meta_data,
                    content=piece.text,
                )
            )
        return chunked
