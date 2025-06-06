"""Base vector store module for MemFuse server."""

from abc import abstractmethod
from loguru import logger
import time
from typing import List, Optional, Dict, Any

import numpy as np

from ..base import StoreBase
from ...models.core import Query, StoreType
from ...rag.encode.base import EncoderBase
from ...rag.encode.MiniLM import MiniLMEncoder
from ...rag.chunk.base import ChunkData
from ...interfaces.chunk_store import ChunkStoreInterface, StorageError


class VectorStore(StoreBase, ChunkStoreInterface):
    """Base class for vector store implementations.

    This class provides a common interface and default implementations for vector stores.
    Subclasses should implement the abstract methods and can override the default
    implementations if needed.

    The registry pattern has been moved to the factory for better separation of concerns.

    Implements ChunkStoreInterface for unified chunk handling across all store types.
    """

    def __init__(
        self,
        data_dir: str,
        encoder: Optional[EncoderBase] = None,
        model_name: str = "all-MiniLM-L6-v2",
        cache_size: int = 100,
        buffer_size: int = 10,
        **kwargs
    ):
        """Initialize the vector store.

        Args:
            data_dir: Directory to store data
            encoder: Encoder to use (if None, a MiniLMEncoder will be created)
            model_name: Name of the embedding model (used if encoder is None)
            cache_size: Size of the query cache
            buffer_size: Size of the write buffer
            **kwargs: Additional arguments
        """
        super().__init__(data_dir, **kwargs)
        self.model_name = model_name
        self.cache_size = cache_size
        self.buffer_size = buffer_size

        # Initialize encoder
        self.encoder = encoder or MiniLMEncoder(
            model_name=model_name,
            cache_size=cache_size
        )

        # Initialize query cache
        self.query_cache = {}

        # Performance metrics
        self.metrics = {
            "embedding_time": 0.0,
            "embedding_count": 0,
            "query_time": 0.0,
            "query_count": 0,
            "add_time": 0.0,
            "add_count": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }

    @property
    def store_type(self) -> StoreType:
        """Get the store type.

        Returns:
            Store type
        """
        return StoreType.VECTOR

    async def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        start_time = time.time()
        try:
            embedding = await self.encoder.encode_text(text)
            self.metrics["embedding_time"] += time.time() - start_time
            self.metrics["embedding_count"] += 1
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            # Return a zero vector as fallback
            if hasattr(self, 'embedding_dim'):
                return np.zeros(self.embedding_dim)
            else:
                # Try to infer dimension from encoder
                try:
                    sample_embedding = await self.encoder.encode_text("test")
                    return np.zeros(len(sample_embedding))
                except Exception:
                    # Last resort: return a 384-dimensional vector
                    return np.zeros(384)

    async def _generate_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts.

        Args:
            texts: Texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        start_time = time.time()
        try:
            embeddings = await self.encoder.encode_texts(texts)
            self.metrics["embedding_time"] += time.time() - start_time
            self.metrics["embedding_count"] += len(texts)
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            # Return zero vectors as fallback
            if hasattr(self, 'embedding_dim'):
                return [np.zeros(self.embedding_dim) for _ in texts]
            else:
                # Try to infer dimension from encoder
                try:
                    sample_embedding = await self.encoder.encode_text("test")
                    return [np.zeros(len(sample_embedding)) for _ in texts]
                except Exception:
                    # Last resort: return 384-dimensional vectors
                    return [np.zeros(384) for _ in texts]

    async def add(self, chunks: List[ChunkData]) -> List[str]:
        """Add chunks to the vector store.

        Args:
            chunks: List of chunks to add to the store

        Returns:
            List of added chunk IDs

        Raises:
            StorageError: If chunks cannot be added
        """
        if not chunks:
            return []

        start_time = time.time()
        try:
            # Generate embeddings for all chunks
            contents = [chunk.content for chunk in chunks]
            embeddings = await self._generate_embeddings(contents)

            # Add chunks with embeddings
            result = await self.add_with_embeddings(chunks, embeddings)

            # Update metrics
            self.metrics["add_time"] += time.time() - start_time
            self.metrics["add_count"] += len(chunks)

            # Invalidate query cache
            self.query_cache = {}

            return result

        except Exception as e:
            logger.error(f"Error adding chunks to vector store: {e}")
            raise StorageError(
                f"Failed to add chunks: {e}",
                store_type="vector",
                operation="add"
            )

    @abstractmethod
    async def add_with_embeddings(self, chunks: List[ChunkData], embeddings: List[np.ndarray]) -> List[str]:
        """Add chunks with pre-computed embeddings.

        Args:
            chunks: Chunks to add
            embeddings: Pre-computed embeddings

        Returns:
            List of IDs of the added chunks
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
            logger.error(f"Error reading chunks from vector store: {e}")
            raise StorageError(
                f"Failed to read chunks: {e}",
                store_type="vector",
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
            # Generate new embedding for updated content
            embedding = await self._generate_embedding(chunk.content)

            # Update chunk with new embedding
            success = await self.update_chunk_with_embedding(chunk_id, chunk, embedding)

            if success:
                # Invalidate query cache
                self.query_cache = {}

            return success

        except Exception as e:
            logger.error(f"Error updating chunk in vector store: {e}")
            raise StorageError(
                f"Failed to update chunk: {e}",
                store_type="vector",
                operation="update"
            )

    @abstractmethod
    async def update_chunk_with_embedding(self, chunk_id: str, chunk: ChunkData, embedding: np.ndarray) -> bool:
        """Update a chunk with pre-computed embedding (implementation specific).

        Args:
            chunk_id: ID of the chunk to update
            chunk: New chunk data
            embedding: Pre-computed embedding

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

    @abstractmethod
    async def get_embedding(self, chunk_id: str) -> Optional[np.ndarray]:
        """Get the embedding for a chunk.

        Args:
            chunk_id: ID of the chunk

        Returns:
            Embedding if found, None otherwise
        """
        pass

    @abstractmethod
    async def query_by_embedding_chunks(self, embedding: np.ndarray, top_k: int = 5, query: Optional[Query] = None) -> List[ChunkData]:
        """Query the store by embedding and return chunks.

        Args:
            embedding: Query embedding
            top_k: Number of results to return
            query: Original query object for filtering (optional)

        Returns:
            List of ChunkData objects
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
            logger.error(f"Error counting chunks in vector store: {e}")
            raise StorageError(
                f"Failed to count chunks: {e}",
                store_type="vector",
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
            return await self.clear_all_chunks()
        except Exception as e:
            logger.error(f"Error clearing vector store: {e}")
            raise StorageError(
                f"Failed to clear store: {e}",
                store_type="vector",
                operation="clear"
            )

    @abstractmethod
    async def clear_all_chunks(self) -> bool:
        """Clear all chunks from the store (implementation specific).

        Returns:
            True if successful, False otherwise
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
            return await self.delete_chunks_by_ids(chunk_ids)
        except Exception as e:
            logger.error(f"Error deleting chunks from vector store: {e}")
            raise StorageError(
                f"Failed to delete chunks: {e}",
                store_type="vector",
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
        start_time = time.time()
        try:
            # Add user_id to cache key if present
            cache_key = f"{query.text}:{top_k}"
            user_id = None
            if query.metadata and "user_id" in query.metadata:
                user_id = query.metadata["user_id"]
                cache_key += f":{user_id}"

            if cache_key in self.query_cache:
                self.metrics["cache_hits"] += 1
                return self.query_cache[cache_key]

            self.metrics["cache_misses"] += 1

            # Generate embedding
            embedding = await self._generate_embedding(query.text)

            # Query by embedding
            chunks = await self.query_by_embedding_chunks(embedding, top_k, query)
            logger.debug(
                f"Retrieved {len(chunks)} chunks with user_id filter: {user_id}")

            # Apply user_id filter
            if query.metadata and "user_id" in query.metadata:
                user_id = query.metadata["user_id"]
                filtered_chunks = []
                for chunk in chunks:
                    chunk_user_id = chunk.metadata.get("user_id")
                    if chunk_user_id == user_id:
                        filtered_chunks.append(chunk)
                    else:
                        logger.debug(
                            f"Post-filtering: Removing chunk with user_id={chunk_user_id}, expected {user_id}")

                chunks = filtered_chunks

            # Apply type filters
            if query.metadata:
                include_messages = query.metadata.get("include_messages", True)
                include_knowledge = query.metadata.get("include_knowledge", True)
                include_chunks = query.metadata.get("include_chunks", True)

                filtered_chunks = []
                for chunk in chunks:
                    chunk_type = chunk.metadata.get("type")
                    if ((chunk_type == "message" and include_messages) or
                        (chunk_type == "knowledge" and include_knowledge) or
                            (chunk_type == "chunk" and include_chunks)):
                        filtered_chunks.append(chunk)

                chunks = filtered_chunks[:top_k]

            # Cache results
            self.query_cache[cache_key] = chunks

            # Limit cache size
            if len(self.query_cache) > self.cache_size:
                # Remove oldest entry (first key)
                oldest_key = next(iter(self.query_cache))
                del self.query_cache[oldest_key]

            # Update metrics
            self.metrics["query_time"] += time.time() - start_time
            self.metrics["query_count"] += 1

            return chunks

        except Exception as e:
            logger.error(f"Error querying chunks from vector store: {e}")
            raise StorageError(
                f"Failed to query chunks: {e}",
                store_type="vector",
                operation="query"
            )

    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics.

        Returns:
            Dictionary of metrics
        """
        metrics = self.metrics.copy()

        # Calculate averages
        if metrics["embedding_count"] > 0:
            metrics["avg_embedding_time"] = metrics["embedding_time"] / \
                metrics["embedding_count"]
        else:
            metrics["avg_embedding_time"] = 0

        if metrics["query_count"] > 0:
            metrics["avg_query_time"] = metrics["query_time"] / \
                metrics["query_count"]
        else:
            metrics["avg_query_time"] = 0

        if metrics["add_count"] > 0:
            metrics["avg_add_time"] = metrics["add_time"] / metrics["add_count"]
        else:
            metrics["avg_add_time"] = 0

        # Calculate cache hit rate
        total_cache_accesses = metrics["cache_hits"] + metrics["cache_misses"]
        if total_cache_accesses > 0:
            metrics["cache_hit_rate"] = metrics["cache_hits"] / \
                total_cache_accesses
        else:
            metrics["cache_hit_rate"] = 0

        return metrics
