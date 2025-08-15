"""Simplified wrapper to adapt PgaiStore to VectorStore interface."""

from typing import List, Dict, Any, Optional
from loguru import logger

from ...interfaces.chunk_store import ChunkStoreInterface
from .pgai_store import PgaiStore
from ...rag.encode.base import EncoderBase
from ...rag.chunk.base import ChunkData
from ...models.core import Query


class PgaiVectorWrapper(ChunkStoreInterface):
    """Lightweight adapter that wraps PgaiStore to implement VectorStore interface.

    Uses dynamic method forwarding for ChunkStoreInterface methods while providing
    data transformation for VectorStore compatibility methods.
    """

    def __init__(self, pgai_store: PgaiStore, encoder: EncoderBase, cache_size: int = 1000):
        """Initialize the wrapper.

        Args:
            pgai_store: The underlying PgaiStore instance
            encoder: Encoder for compatibility (not used by pgai)
            cache_size: Cache size for compatibility
        """
        self.pgai_store = pgai_store
        self.encoder = encoder  # Keep for compatibility
        self.cache_size = cache_size
        self.initialized = False

    async def initialize(self) -> bool:
        """Initialize the wrapper and underlying store."""
        if self.initialized:
            return True

        # Ensure pgai store is initialized
        if not self.pgai_store.initialized:
            await self.pgai_store.initialize()

        self.initialized = True
        logger.debug("PgaiVectorWrapper initialized")
        return True

    # ChunkStoreInterface abstract methods - minimal forwarding implementation
    async def add(self, chunks: List[ChunkData]) -> List[str]:
        return await self.pgai_store.add(chunks)

    async def read(self, chunk_ids: List[str], filters: Optional[Dict[str, Any]] = None) -> List[Optional[ChunkData]]:
        return await self.pgai_store.read(chunk_ids, filters)

    async def update(self, chunk_id: str, chunk: ChunkData) -> bool:
        return await self.pgai_store.update(chunk_id, chunk)

    async def delete(self, chunk_ids: List[str]) -> List[bool]:
        return await self.pgai_store.delete(chunk_ids)

    async def query(self, query: Query, top_k: int = 5) -> List[ChunkData]:
        return await self.pgai_store.query(query, top_k)

    async def retrieve(self, query: str, top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """Retrieve method for compatibility with multi-path retrieval system."""
        # Convert string query to Query object
        query_obj = Query(text=query, metadata=kwargs)
        chunks = await self.pgai_store.query(query_obj, top_k)
        return [{'id': c.chunk_id, 'content': c.content, 'metadata': c.metadata, 'score': 1.0}
                for c in chunks]

    async def count(self) -> int:
        return await self.pgai_store.count()

    async def clear(self) -> bool:
        return await self.pgai_store.clear()

    async def get_chunks_by_session(self, session_id: str) -> List[ChunkData]:
        return await self.pgai_store.get_chunks_by_session(session_id)

    async def get_chunks_by_round(self, round_id: str) -> List[ChunkData]:
        return await self.pgai_store.get_chunks_by_round(round_id)

    async def get_chunks_by_user(self, user_id: str) -> List[ChunkData]:
        return await self.pgai_store.get_chunks_by_user(user_id)

    async def get_chunks_by_strategy(self, strategy_type: str) -> List[ChunkData]:
        return await self.pgai_store.get_chunks_by_strategy(strategy_type)

    async def get_chunks_stats(self, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return await self.pgai_store.get_chunks_stats(filters)

    def __getattr__(self, name):
        """Forward any other methods to pgai_store."""
        if hasattr(self.pgai_store, name):
            return getattr(self.pgai_store, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    # VectorStore interface methods that require data transformation
    async def add_items(self, items: List[Dict[str, Any]]) -> List[str]:
        """Add items to the store (VectorStore interface)."""
        chunks = [ChunkData(content=item.get('content', ''), chunk_id=item.get('id', ''),
                            metadata=item.get('metadata', {})) for item in items]
        return await self.pgai_store.add(chunks)

    async def search(self, query_text: str, top_k: int = 5,
                     filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for similar items (VectorStore interface)."""
        chunks = await self.pgai_store.query(Query(text=query_text), top_k)
        return [{'id': c.chunk_id, 'content': c.content, 'metadata': c.metadata, 'score': 1.0}
                for c in chunks if not filters or all(c.metadata.get(k) == v for k, v in filters.items())]

    async def get_item(self, item_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific item by ID (VectorStore interface)."""
        chunks = await self.pgai_store.read([item_id])
        if chunks and chunks[0]:
            c = chunks[0]
            return {'id': c.chunk_id, 'content': c.content, 'metadata': c.metadata}
        return None

    async def update_item(self, item_id: str, item: Dict[str, Any]) -> bool:
        """Update an existing item (VectorStore interface)."""
        chunk = ChunkData(content=item.get('content', ''), chunk_id=item_id,
                          metadata=item.get('metadata', {}))
        return await self.pgai_store.update(item_id, chunk)

    async def get_stats(self) -> Dict[str, Any]:
        """Get store statistics (VectorStore interface)."""
        stats = await self.pgai_store.get_chunks_stats()
        return {'total_items': stats.get('total_chunks', 0), 'by_session': stats.get('by_session', {}),
                'by_user': stats.get('by_user', {}), 'by_strategy': stats.get('by_strategy', {}),
                'storage_size': stats.get('storage_size', 'N/A'), 'backend': 'pgai'}

    async def search_by_embedding(self, embedding: List[float], top_k: int = 5,
                                  filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search by embedding vector (VectorStore interface)."""
        results = await self.pgai_store.search_similar(embedding, top_k, **filters or {})
        return [{'id': r.get('id'), 'content': r.get('content'), 'metadata': r.get('metadata', {}),
                 'score': 1.0 - r.get('distance', 0.0)} for r in results]

    async def get_items_by_session(self, session_id: str) -> List[Dict[str, Any]]:
        """Get items by session ID (business logic method)."""
        chunks = await self.pgai_store.get_chunks_by_session(session_id)
        return [{'id': c.chunk_id, 'content': c.content, 'metadata': c.metadata} for c in chunks]

    async def get_items_by_user(self, user_id: str) -> List[Dict[str, Any]]:
        """Get items by user ID (business logic method)."""
        chunks = await self.pgai_store.get_chunks_by_user(user_id)
        return [{'id': c.chunk_id, 'content': c.content, 'metadata': c.metadata} for c in chunks]

    async def close(self):
        """Close the wrapper and underlying store."""
        if self.pgai_store:
            await self.pgai_store.close()

        self.initialized = False
        logger.debug("PgaiVectorWrapper closed")

    def __del__(self):
        """Cleanup when wrapper is destroyed."""
        if self.initialized:
            logger.debug("PgaiVectorWrapper being destroyed")
