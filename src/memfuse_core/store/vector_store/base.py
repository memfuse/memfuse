"""Base vector store module for MemFuse server."""

from abc import abstractmethod
from loguru import logger
import time
from typing import List, Optional, Dict, Any

import numpy as np

from ..base import StoreBase
from ...models.core import Item, Query, QueryResult
from ...models.core import StoreType
from ...rag.encode.base import EncoderBase
from ...rag.encode.MiniLM import MiniLMEncoder


class VectorStore(StoreBase):
    """Base class for vector store implementations.

    This class provides a common interface and default implementations for vector stores.
    Subclasses should implement the abstract methods and can override the default
    implementations if needed.

    The registry pattern has been moved to the factory for better separation of concerns.
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

    async def add(self, item: Item) -> str:
        """Add an item to the store.

        Args:
            item: Item to add

        Returns:
            ID of the added item
        """
        start_time = time.time()
        try:
            # Generate embedding
            embedding = await self._generate_embedding(item.content)

            # Add with embedding
            result = await self.add_with_embedding(item, embedding)

            # Update metrics
            self.metrics["add_time"] += time.time() - start_time
            self.metrics["add_count"] += 1

            # Invalidate query cache
            self.query_cache = {}

            return result
        except Exception as e:
            logger.error(f"Error adding item: {e}")
            raise

    async def add_batch(self, items: List[Item]) -> List[str]:
        """Add multiple items to the store.

        Args:
            items: Items to add

        Returns:
            List of IDs of the added items
        """
        if not items:
            return []

        start_time = time.time()
        try:
            # Generate embeddings
            contents = [item.content for item in items]
            embeddings = await self._generate_embeddings(contents)

            # Add with embeddings
            result = await self.add_batch_with_embeddings(items, embeddings)

            # Update metrics
            self.metrics["add_time"] += time.time() - start_time
            self.metrics["add_count"] += len(items)

            # Invalidate query cache
            self.query_cache = {}

            return result
        except Exception as e:
            logger.error(f"Error adding batch: {e}")
            raise

    @abstractmethod
    async def add_with_embedding(self, item: Item, embedding: np.ndarray) -> str:
        """Add an item with a pre-computed embedding.

        Args:
            item: Item to add
            embedding: Pre-computed embedding

        Returns:
            ID of the added item
        """
        pass

    @abstractmethod
    async def add_batch_with_embeddings(self, items: List[Item], embeddings: List[np.ndarray]) -> List[str]:
        """Add multiple items with pre-computed embeddings.

        Args:
            items: Items to add
            embeddings: Pre-computed embeddings

        Returns:
            List of IDs of the added items
        """
        pass

    @abstractmethod
    async def get_embedding(self, item_id: str) -> Optional[np.ndarray]:
        """Get the embedding for an item.

        Args:
            item_id: ID of the item

        Returns:
            Embedding if found, None otherwise
        """
        pass

    @abstractmethod
    async def query_by_embedding(self, embedding: np.ndarray, top_k: int = 5, query: Optional[Query] = None) -> List[QueryResult]:
        """Query the store by embedding.

        Args:
            embedding: Query embedding
            top_k: Number of results to return
            query: Original query object for filtering (optional)

        Returns:
            List of query results
        """
        pass

    @abstractmethod
    async def get_nearest_neighbors(self, item_id: str, top_k: int = 5) -> List[QueryResult]:
        """Get the nearest neighbors of an item.

        Args:
            item_id: ID of the item
            top_k: Number of results to return

        Returns:
            List of query results
        """
        pass

    async def update(self, item_id: str, item: Item) -> bool:
        """Update an item.

        Args:
            item_id: ID of the item to update
            item: New item data

        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate embedding
            embedding = await self._generate_embedding(item.content)

            # Update with embedding
            if hasattr(self, 'update_with_embedding'):
                return await self.update_with_embedding(item_id, item, embedding)
            else:
                # Default implementation: delete and add
                await self.delete(item_id)
                await self.add_with_embedding(item, embedding)
                return True
        except Exception as e:
            logger.error(f"Error updating item: {e}")
            return False

    async def update_batch(self, item_ids: List[str], items: List[Item]) -> List[bool]:
        """Update multiple items.

        Args:
            item_ids: IDs of the items to update
            items: New item data

        Returns:
            List of success flags
        """
        if not items:
            return []

        try:
            # Generate embeddings
            contents = [item.content for item in items]
            embeddings = await self._generate_embeddings(contents)

            # Update with embeddings
            if hasattr(self, 'update_batch_with_embeddings'):
                return await self.update_batch_with_embeddings(item_ids, items, embeddings)
            else:
                # Default implementation: update one by one
                results = []
                for item_id, item, embedding in zip(item_ids, items, embeddings):
                    if hasattr(self, 'update_with_embedding'):
                        result = await self.update_with_embedding(item_id, item, embedding)
                    else:
                        # Delete and add
                        await self.delete(item_id)
                        await self.add_with_embedding(item, embedding)
                        result = True
                    results.append(result)
                return results
        except Exception as e:
            logger.error(f"Error updating batch: {e}")
            return [False] * len(items)

    async def query(self, query: Query, top_k: int = 5) -> List[QueryResult]:
        """Query the store.

        Args:
            query: Query to execute
            top_k: Number of results to return

        Returns:
            List of query results
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

            # Query by embedding with the original query object for filtering
            # The query object contains user_id which will be used for filtering at the database level
            results = await self.query_by_embedding(embedding, top_k, query)
            logger.debug(
                f"Retrieved results with user_id filter: {user_id}, got {len(results)} results")

            # Apply user_id filter as a post-processing step
            if query.metadata and "user_id" in query.metadata:
                user_id = query.metadata["user_id"]
                filtered_results = []
                for result in results:
                    result_user_id = result.metadata.get("user_id")
                    if result_user_id == user_id:
                        filtered_results.append(result)
                    else:
                        logger.debug(
                            f"Post-filtering: Removing result with user_id={result_user_id}, expected {user_id}")

                results = filtered_results

            # Apply other filters (include_messages, include_knowledge)
            if query.metadata:
                include_messages = query.metadata.get("include_messages", True)
                include_knowledge = query.metadata.get(
                    "include_knowledge", True)

                filtered_results = []
                for result in results:
                    item_type = result.metadata.get("type")
                    if (item_type == "message" and include_messages) or (item_type == "knowledge" and include_knowledge):
                        filtered_results.append(result)

                results = filtered_results[:top_k]

            # Cache results
            self.query_cache[cache_key] = results

            # Limit cache size
            if len(self.query_cache) > self.cache_size:
                # Remove oldest entry (first key)
                oldest_key = next(iter(self.query_cache))
                del self.query_cache[oldest_key]

            # Update metrics
            self.metrics["query_time"] += time.time() - start_time
            self.metrics["query_count"] += 1

            return results
        except Exception as e:
            logger.error(f"Error querying store: {e}")
            return []

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
