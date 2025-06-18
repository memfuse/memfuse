"""Base keyword store module for MemFuse server."""

from loguru import logger
from abc import abstractmethod
from typing import List, Optional, Dict, Any

from ..base import StoreBase
from ...models.core import Query, StoreType
from ...rag.chunk.base import ChunkData
from ...interfaces.chunk_store import ChunkStoreInterface, StorageError


class KeywordStore(StoreBase, ChunkStoreInterface):
    """Base class for keyword store implementations.
    
    Implements ChunkStoreInterface for unified chunk handling across all store types.
    """

    def __init__(
        self,
        data_dir: str,
        cache_size: int = 100,
        **kwargs
    ):
        """Initialize the keyword store.

        Args:
            data_dir: Directory to store data
            cache_size: Size of the query cache
            **kwargs: Additional arguments
        """
        super().__init__(data_dir, **kwargs)
        self.cache_size = cache_size

        # Initialize query cache
        self.query_cache = {}

    @property
    def store_type(self) -> StoreType:
        """Get the store type.

        Returns:
            Store type
        """
        return StoreType.KEYWORD

    # ChunkStoreInterface implementation

    # CRUD Operations
    async def add(self, chunks: List[ChunkData]) -> List[str]:
        """Create: Add chunks to the keyword store.

        Args:
            chunks: List of chunks to add to the store

        Returns:
            List of added chunk IDs

        Raises:
            StorageError: If chunks cannot be added
        """
        if not chunks:
            return []

        try:
            # Invalidate query cache
            self.query_cache = {}

            return await self.add_chunks_to_index(chunks)
        except Exception as e:
            logger.error(f"Error adding chunks to keyword store: {e}")
            raise StorageError(
                f"Failed to add chunks: {e}",
                store_type="keyword",
                operation="add"
            )

    @abstractmethod
    async def add_chunks_to_index(self, chunks: List[ChunkData]) -> List[str]:
        """Add chunks to the keyword index (implementation specific).
        
        Args:
            chunks: Chunks to add to the index
            
        Returns:
            List of added chunk IDs
        """
        pass

    async def read(self, chunk_ids: List[str], filters: Optional[Dict[str, Any]] = None) -> List[Optional[ChunkData]]:
        """Read: Get chunks by their IDs with optional metadata filters.

        This is a database-level read operation that retrieves chunks by exact IDs
        and optionally filters by metadata conditions.

        Args:
            chunk_ids: List of chunk IDs to retrieve
            filters: Optional metadata filters (e.g., {"user_id": "123", "type": "chunk"})

        Returns:
            List of ChunkData objects, None for chunks not found or filtered out

        Raises:
            StorageError: If chunks cannot be retrieved
        """
        try:
            chunks = await self.get_chunks_by_ids(chunk_ids)

            # Apply metadata filters if provided
            if filters:
                filtered_chunks = []
                for chunk in chunks:
                    if chunk is None:
                        filtered_chunks.append(None)
                        continue

                    # Check if chunk matches all filter conditions
                    matches = True
                    for key, value in filters.items():
                        if chunk.metadata.get(key) != value:
                            matches = False
                            break

                    if matches:
                        filtered_chunks.append(chunk)
                    else:
                        filtered_chunks.append(None)

                return filtered_chunks

            return chunks

        except Exception as e:
            logger.error(f"Error reading chunks from keyword store: {e}")
            raise StorageError(
                f"Failed to read chunks: {e}",
                store_type="keyword",
                operation="read"
            )

    async def update(self, chunk_id: str, chunk: ChunkData) -> bool:
        """Update: Modify an existing chunk.

        Args:
            chunk_id: ID of the chunk to update
            chunk: New chunk data

        Returns:
            True if successful, False if chunk not found

        Raises:
            StorageError: If chunk cannot be updated
        """
        try:
            # Update chunk in index
            success = await self.update_chunk_in_index(chunk_id, chunk)

            if success:
                # Invalidate query cache
                self.query_cache = {}

            return success

        except Exception as e:
            logger.error(f"Error updating chunk in keyword store: {e}")
            raise StorageError(
                f"Failed to update chunk: {e}",
                store_type="keyword",
                operation="update"
            )

    @abstractmethod
    async def update_chunk_in_index(self, chunk_id: str, chunk: ChunkData) -> bool:
        """Update a chunk in the keyword index (implementation specific).

        Args:
            chunk_id: ID of the chunk to update
            chunk: New chunk data

        Returns:
            True if successful, False if chunk not found
        """
        pass

    @abstractmethod
    async def get_chunks_by_ids(self, chunk_ids: List[str]) -> List[Optional[ChunkData]]:
        """Get chunks by their IDs (implementation specific).
        
        Args:
            chunk_ids: List of chunk IDs to retrieve
            
        Returns:
            List of ChunkData objects, None for chunks not found
        """
        pass

    async def query(self, query: Query, top_k: int = 5) -> List[ChunkData]:
        """Query relevant chunks based on the query.
        
        Args:
            query: Query object containing search text and metadata
            top_k: Maximum number of results to return
            
        Returns:
            List of relevant ChunkData objects, sorted by relevance score
            
        Raises:
            StorageError: If query cannot be executed
        """
        try:
            # Add user_id to cache key if present
            cache_key = f"{query.text}:{top_k}"
            user_id = query.metadata.get("user_id") if query.metadata else None
            if user_id:
                cache_key += f":{user_id}"

            if cache_key in self.query_cache:
                return self.query_cache[cache_key]

            # Perform search
            results = await self.search_chunks(query.text, top_k, user_id)
            
            # Apply filters
            if query.metadata:
                include_messages = query.metadata.get("include_messages", True)
                include_knowledge = query.metadata.get("include_knowledge", True)
                include_chunks = query.metadata.get("include_chunks", True)

                filtered_results = []
                for chunk in results:
                    chunk_type = chunk.metadata.get("type")
                    if (
                        (chunk_type == "message" and include_messages) or
                        (chunk_type == "knowledge" and include_knowledge) or
                        (chunk_type == "chunk" and include_chunks)
                    ):
                        filtered_results.append(chunk)

                results = filtered_results[:top_k]

            # Cache results
            self.query_cache[cache_key] = results

            # Limit cache size
            if len(self.query_cache) > self.cache_size:
                # Remove oldest entry (first key)
                oldest_key = next(iter(self.query_cache))
                del self.query_cache[oldest_key]

            return results
            
        except Exception as e:
            logger.error(f"Error querying chunks from keyword store: {e}")
            raise StorageError(
                f"Failed to query chunks: {e}",
                store_type="keyword",
                operation="query"
            )

    @abstractmethod
    async def search_chunks(self, query_text: str, top_k: int = 5, user_id: Optional[str] = None) -> List[ChunkData]:
        """Search chunks using keyword matching (implementation specific).

        Args:
            query_text: Query text
            top_k: Number of results to return
            user_id: User ID to filter by (optional)

        Returns:
            List of ChunkData objects sorted by relevance
        """
        pass

    async def delete(self, chunk_ids: List[str]) -> List[bool]:
        """Delete chunks by their IDs.
        
        Args:
            chunk_ids: List of chunk IDs to delete
            
        Returns:
            List of deletion success flags (True if deleted, False if not found)
            
        Raises:
            StorageError: If chunks cannot be deleted
        """
        try:
            # Invalidate query cache
            self.query_cache = {}
            
            return await self.delete_chunks_by_ids(chunk_ids)
        except Exception as e:
            logger.error(f"Error deleting chunks from keyword store: {e}")
            raise StorageError(
                f"Failed to delete chunks: {e}",
                store_type="keyword",
                operation="delete"
            )

    @abstractmethod
    async def delete_chunks_by_ids(self, chunk_ids: List[str]) -> List[bool]:
        """Delete chunks by their IDs (implementation specific).
        
        Args:
            chunk_ids: List of chunk IDs to delete
            
        Returns:
            List of deletion success flags
        """
        pass

    async def count(self) -> int:
        """Get the total number of chunks in the store.
        
        Returns:
            Total number of chunks stored
            
        Raises:
            StorageError: If count cannot be retrieved
        """
        try:
            return await self.get_chunk_count()
        except Exception as e:
            logger.error(f"Error counting chunks in keyword store: {e}")
            raise StorageError(
                f"Failed to count chunks: {e}",
                store_type="keyword",
                operation="count"
            )

    @abstractmethod
    async def get_chunk_count(self) -> int:
        """Get the total number of chunks in the store (implementation specific).
        
        Returns:
            Total number of chunks stored
        """
        pass

    async def clear(self) -> bool:
        """Clear all chunks from the store.
        
        Returns:
            True if successful, False otherwise
            
        Raises:
            StorageError: If store cannot be cleared
        """
        try:
            # Clear query cache
            self.query_cache = {}
            
            return await self.clear_all_chunks()
        except Exception as e:
            logger.error(f"Error clearing keyword store: {e}")
            raise StorageError(
                f"Failed to clear store: {e}",
                store_type="keyword",
                operation="clear"
            )

    @abstractmethod
    async def clear_all_chunks(self) -> bool:
        """Clear all chunks from the store (implementation specific).
        
        Returns:
            True if successful, False otherwise
        """
        pass
