"""Wrapper to adapt PgaiStore to VectorStore interface."""

from typing import List, Dict, Any, Optional, Tuple
from loguru import logger

from ..interfaces.chunk_store import ChunkStoreInterface
from .pgai_store import PgaiStore
from ..rag.encode.base import EncoderBase
from ..rag.chunk.base import ChunkData
from ..models import Query


class PgaiVectorWrapper(ChunkStoreInterface):
    """Wrapper that adapts PgaiStore to implement VectorStore interface.
    
    This allows PgaiStore to be used wherever VectorStore is expected,
    maintaining compatibility with existing MemFuse architecture.
    """
    
    def __init__(self, pgai_store: PgaiStore, encoder: EncoderBase, cache_size: int = 1000):
        """Initialize the wrapper.
        
        Args:
            pgai_store: The underlying PgaiStore instance
            encoder: Encoder for compatibility (not used by pgai)
            cache_size: Cache size for compatibility
        """
        self.pgai_store = pgai_store
        self.encoder = encoder
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

    # ChunkStoreInterface CRUD methods
    async def add(self, chunks: List[ChunkData]) -> List[str]:
        """Add chunks to the store (ChunkStoreInterface)."""
        return await self.pgai_store.add(chunks)

    async def read(self, chunk_ids: List[str], filters: Optional[Dict[str, Any]] = None) -> List[Optional[ChunkData]]:
        """Read chunks by IDs (ChunkStoreInterface)."""
        return await self.pgai_store.read(chunk_ids, filters)

    async def update(self, chunk_id: str, chunk: ChunkData) -> bool:
        """Update a chunk (ChunkStoreInterface)."""
        return await self.pgai_store.update(chunk_id, chunk)

    async def delete(self, chunk_ids: List[str]) -> List[bool]:
        """Delete chunks by IDs (ChunkStoreInterface)."""
        return await self.pgai_store.delete(chunk_ids)

    async def query(self, query: Query, top_k: int = 5) -> List[ChunkData]:
        """Query for similar chunks (ChunkStoreInterface)."""
        return await self.pgai_store.query(query, top_k)

    async def count(self) -> int:
        """Get total chunk count (ChunkStoreInterface)."""
        return await self.pgai_store.count()

    async def clear(self) -> bool:
        """Clear all chunks (ChunkStoreInterface)."""
        return await self.pgai_store.clear()

    # ChunkStoreInterface business methods
    async def get_chunks_by_session(self, session_id: str) -> List[ChunkData]:
        """Get chunks by session ID (ChunkStoreInterface)."""
        return await self.pgai_store.get_chunks_by_session(session_id)

    async def get_chunks_by_round(self, round_id: str) -> List[ChunkData]:
        """Get chunks by round ID (ChunkStoreInterface)."""
        return await self.pgai_store.get_chunks_by_round(round_id)

    async def get_chunks_by_user(self, user_id: str) -> List[ChunkData]:
        """Get chunks by user ID (ChunkStoreInterface)."""
        return await self.pgai_store.get_chunks_by_user(user_id)

    async def get_chunks_by_strategy(self, strategy_type: str) -> List[ChunkData]:
        """Get chunks by strategy type (ChunkStoreInterface)."""
        return await self.pgai_store.get_chunks_by_strategy(strategy_type)

    async def get_chunks_stats(self, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get chunk statistics (ChunkStoreInterface)."""
        return await self.pgai_store.get_chunks_stats(filters)
    
    async def add_items(self, items: List[Dict[str, Any]]) -> List[str]:
        """Add items to the store (VectorStore interface).
        
        Args:
            items: List of items with 'id', 'content', and 'metadata' keys
            
        Returns:
            List of added item IDs
        """
        # Convert items to ChunkData
        chunks = []
        for item in items:
            chunk = ChunkData(
                content=item.get('content', ''),
                chunk_id=item.get('id', ''),
                metadata=item.get('metadata', {})
            )
            chunks.append(chunk)
        
        # Use pgai store's add method
        return await self.pgai_store.add(chunks)
    
    async def search(self, query_text: str, top_k: int = 5, 
                    filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for similar items (VectorStore interface).
        
        Args:
            query_text: Text to search for
            top_k: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            List of search results with 'id', 'content', 'metadata', and 'score' keys
        """
        # Create Query object
        query = Query(text=query_text)
        
        # Use pgai store's query method
        chunks = await self.pgai_store.query(query, top_k)
        
        # Convert ChunkData back to VectorStore format
        results = []
        for chunk in chunks:
            # Apply filters if provided
            if filters:
                match = True
                for key, value in filters.items():
                    if chunk.metadata.get(key) != value:
                        match = False
                        break
                if not match:
                    continue
            
            result = {
                'id': chunk.chunk_id,
                'content': chunk.content,
                'metadata': chunk.metadata,
                'score': 1.0  # pgai doesn't provide scores in query method
            }
            results.append(result)
        
        return results
    
    async def get_item(self, item_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific item by ID (VectorStore interface).
        
        Args:
            item_id: ID of the item to retrieve
            
        Returns:
            Item data or None if not found
        """
        chunks = await self.pgai_store.read([item_id])
        
        if chunks and chunks[0] is not None:
            chunk = chunks[0]
            return {
                'id': chunk.chunk_id,
                'content': chunk.content,
                'metadata': chunk.metadata
            }
        
        return None
    
    async def delete_items(self, item_ids: List[str]) -> List[bool]:
        """Delete items by their IDs (VectorStore interface).
        
        Args:
            item_ids: List of item IDs to delete
            
        Returns:
            List of deletion success flags
        """
        return await self.pgai_store.delete(item_ids)
    
    async def update_item(self, item_id: str, item: Dict[str, Any]) -> bool:
        """Update an existing item (VectorStore interface).
        
        Args:
            item_id: ID of the item to update
            item: New item data
            
        Returns:
            True if successful, False if item not found
        """
        chunk = ChunkData(
            content=item.get('content', ''),
            chunk_id=item_id,
            metadata=item.get('metadata', {})
        )
        
        return await self.pgai_store.update(item_id, chunk)
    
    async def count_items(self) -> int:
        """Get total number of items in the store (VectorStore interface).
        
        Returns:
            Number of items
        """
        return await self.pgai_store.count()
    
    async def clear(self) -> bool:
        """Clear all items from the store (VectorStore interface).
        
        Returns:
            True if successful
        """
        return await self.pgai_store.clear()
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get store statistics (VectorStore interface).
        
        Returns:
            Dictionary containing store statistics
        """
        stats = await self.pgai_store.get_chunks_stats()
        
        # Convert to VectorStore format
        return {
            'total_items': stats.get('total_chunks', 0),
            'by_session': stats.get('by_session', {}),
            'by_user': stats.get('by_user', {}),
            'by_strategy': stats.get('by_strategy', {}),
            'storage_size': stats.get('storage_size', 'N/A'),
            'backend': 'pgai'
        }
    
    # Additional methods for compatibility with existing VectorStore usage
    async def search_by_embedding(self, embedding: List[float], top_k: int = 5,
                                 filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search by embedding vector (VectorStore interface).
        
        Args:
            embedding: Query embedding vector
            top_k: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            List of search results
        """
        # Use pgai store's search_similar method
        results = await self.pgai_store.search_similar(embedding, top_k, **filters or {})
        
        # Convert to VectorStore format
        vector_results = []
        for result in results:
            vector_result = {
                'id': result.get('id'),
                'content': result.get('content'),
                'metadata': result.get('metadata', {}),
                'score': 1.0 - result.get('distance', 0.0)  # Convert distance to similarity score
            }
            vector_results.append(vector_result)
        
        return vector_results
    
    async def get_items_by_session(self, session_id: str) -> List[Dict[str, Any]]:
        """Get items by session ID (business logic method).
        
        Args:
            session_id: Session ID to filter by
            
        Returns:
            List of items for the session
        """
        chunks = await self.pgai_store.get_chunks_by_session(session_id)
        
        return [
            {
                'id': chunk.chunk_id,
                'content': chunk.content,
                'metadata': chunk.metadata
            }
            for chunk in chunks
        ]
    
    async def get_items_by_user(self, user_id: str) -> List[Dict[str, Any]]:
        """Get items by user ID (business logic method).
        
        Args:
            user_id: User ID to filter by
            
        Returns:
            List of items for the user
        """
        chunks = await self.pgai_store.get_chunks_by_user(user_id)
        
        return [
            {
                'id': chunk.chunk_id,
                'content': chunk.content,
                'metadata': chunk.metadata
            }
            for chunk in chunks
        ]
    
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
