"""Unified chunk storage interface for MemFuse stores.

This module defines the standard interface that all store types (Vector, Keyword, Graph)
must implement to handle chunk data consistently. This eliminates the need for 
ChunkData -> Item/Node conversions and provides a unified API.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from ..rag.chunk.base import ChunkData
from ..models import Query


class ChunkStoreInterface(ABC):
    """Unified chunk storage interface for all store types.
    
    This interface standardizes how all stores (Vector, Keyword, Graph) 
    handle chunk data, eliminating the need for ChunkData -> Item/Node conversions.
    All stores should implement this interface to provide consistent chunk operations.
    """
    
    # CRUD Operations
    @abstractmethod
    async def add(self, chunks: List[ChunkData]) -> List[str]:
        """Create: Add chunks to storage.

        Args:
            chunks: List of chunks to add to the store

        Returns:
            List of added chunk IDs

        Raises:
            StorageError: If chunks cannot be added
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    async def delete(self, chunk_ids: List[str]) -> List[bool]:
        """Delete: Remove chunks by their IDs.

        Args:
            chunk_ids: List of chunk IDs to delete

        Returns:
            List of deletion success flags (True if deleted, False if not found)

        Raises:
            StorageError: If chunks cannot be deleted
        """
        pass

    # Semantic Query Operations
    @abstractmethod
    async def query(self, query: Query, top_k: int = 5) -> List[ChunkData]:
        """Query: Semantic search for relevant chunks based on query text.

        This is a semantic-level query operation that finds chunks similar to
        the query text using vector similarity, keyword matching, or graph traversal.

        Args:
            query: Query object containing search text and metadata
            top_k: Maximum number of results to return

        Returns:
            List of relevant ChunkData objects, sorted by relevance score

        Raises:
            StorageError: If query cannot be executed
        """
        pass
    
    @abstractmethod
    async def count(self) -> int:
        """Get the total number of chunks in the store.
        
        Returns:
            Total number of chunks stored
            
        Raises:
            StorageError: If count cannot be retrieved
        """
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """Clear all chunks from the store.
        
        Returns:
            True if successful, False otherwise
            
        Raises:
            StorageError: If store cannot be cleared
        """
        pass


class StorageError(Exception):
    """Exception raised when storage operations fail."""
    
    def __init__(self, message: str, store_type: str = None, operation: str = None):
        """Initialize storage error.
        
        Args:
            message: Error message
            store_type: Type of store where error occurred
            operation: Operation that failed
        """
        self.store_type = store_type
        self.operation = operation
        super().__init__(message)
    
    def __str__(self):
        """String representation of the error."""
        parts = [super().__str__()]
        if self.store_type:
            parts.append(f"Store: {self.store_type}")
        if self.operation:
            parts.append(f"Operation: {self.operation}")
        return " | ".join(parts)
