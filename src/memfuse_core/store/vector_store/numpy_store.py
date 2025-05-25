"""NumPy vector store implementation."""

import os
import json
import numpy as np
from typing import Dict, List, Optional, Tuple

from ...utils.config import (
    get_embedding_dim,
    get_top_k,
    get_similarity_threshold,
    get_vector_items_file,
    get_vector_embeddings_file,
)
from ...models.core import Item, Query, QueryResult
from ...models.core import StoreType
from .base import VectorStore
from ...utils.path_manager import PathManager


class NumpyVectorStore(VectorStore):
    """NumPy-based vector store implementation.

    This implementation uses NumPy arrays for storing and retrieving embeddings.
    It provides high-performance vector operations with minimal dependencies.
    """

    def __init__(
        self,
        data_dir: str,
        model_name: str = "all-MiniLM-L6-v2",
        embedding_dim: Optional[int] = None,
        buffer_size: int = 10,
        cache_size: int = 100,
        **kwargs
    ):
        """Initialize the NumPy vector store.

        Args:
            data_dir: Directory to store data
            model_name: Name of the embedding model
            embedding_dim: Dimension of the embeddings (optional)
            buffer_size: Size of the write buffer
            cache_size: Size of the query cache
            **kwargs: Additional arguments
        """
        # Call parent constructor
        super().__init__(
            data_dir=data_dir,
            model_name=model_name,
            cache_size=cache_size,
            buffer_size=buffer_size,
            **kwargs
        )

        # Get embedding dimension from configuration if not provided
        if embedding_dim is None:
            self.embedding_dim = get_embedding_dim(model_name)
        else:
            self.embedding_dim = embedding_dim

        # Create vector store directory
        self.vector_dir = os.path.join(data_dir, "vector_store")

        # Get file names from configuration
        vector_items_file = get_vector_items_file()
        vector_embeddings_file = get_vector_embeddings_file()

        # Initialize items and embeddings file paths
        self.items_file = os.path.join(self.vector_dir, vector_items_file)
        self.embeddings_file = os.path.join(
            self.vector_dir, vector_embeddings_file)

        # Initialize items and embeddings dictionaries
        self.items = {}
        self.embeddings = {}

        # Initialize write buffer
        self.write_buffer_items = []
        self.write_buffer_embeddings = []

        # Initialize normalized embeddings cache
        self.normalized_embeddings = None

    async def initialize(self) -> bool:
        """Initialize the vector store.

        Returns:
            True if successful, False otherwise
        """
        # Create vector store directory
        PathManager.ensure_directory(self.vector_dir)

        # Load items and embeddings if they exist
        self.items = self._load_items()
        self.embeddings = self._load_embeddings()

        self.initialized = True
        return True

    def _load_items(self) -> Dict[str, Item]:
        """Load items from file.

        Returns:
            Dictionary of items
        """
        if os.path.exists(self.items_file):
            with open(self.items_file, "r") as f:
                items_data = json.load(f)
                return {item_id: Item(**item) for item_id, item in items_data.items()}
        return {}

    def _save_items(self):
        """Save items to file."""
        items_data = {item_id: item.__dict__ for item_id,
                      item in self.items.items()}
        with open(self.items_file, "w") as f:
            json.dump(items_data, f)

    def _load_embeddings(self) -> Dict[str, np.ndarray]:
        """Load embeddings from file.

        Returns:
            Dictionary of embeddings
        """
        if os.path.exists(self.embeddings_file):
            with open(self.embeddings_file, "r") as f:
                embeddings_data = json.load(f)
                return {item_id: np.array(embedding) for item_id, embedding in embeddings_data.items()}
        return {}

    def _save_embeddings(self):
        """Save embeddings to file."""
        embeddings_data = {item_id: embedding.tolist()
                           for item_id, embedding in self.embeddings.items()}
        with open(self.embeddings_file, "w") as f:
            json.dump(embeddings_data, f)

    def _flush_buffer(self):
        """Flush the write buffer to disk."""
        if not self.write_buffer_items and not self.write_buffer_embeddings:
            return

        # Add buffered items to items dictionary
        for item in self.write_buffer_items:
            self.items[item.id] = item

        # Add buffered embeddings to embeddings dictionary
        for item_id, embedding in self.write_buffer_embeddings:
            self.embeddings[item_id] = embedding

        # Clear the buffer
        self.write_buffer_items = []
        self.write_buffer_embeddings = []

        # Save to disk
        self._save_items()
        self._save_embeddings()

        # Invalidate normalized embeddings cache
        self.normalized_embeddings = None

    def _get_normalized_embeddings(self) -> Tuple[List[str], np.ndarray]:
        """Get normalized embeddings for similarity calculations.

        Returns:
            Tuple of (item_ids, normalized_embeddings)
        """
        if self.normalized_embeddings is None:
            # Get all item IDs and embeddings
            item_ids = list(self.embeddings.keys())
            if not item_ids:
                return [], np.array([])

            # Stack embeddings into a single array
            embeddings_array = np.stack(
                [self.embeddings[item_id] for item_id in item_ids])

            # Normalize embeddings
            norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            normalized = embeddings_array / norms

            # Cache the result
            self.normalized_embeddings = (item_ids, normalized)

        return self.normalized_embeddings

    async def add(self, item: Item) -> str:
        """Add an item to the store.

        Args:
            item: Item to add

        Returns:
            ID of the added item
        """
        # Generate embedding
        embedding = await self._generate_embedding(item.content)

        # Add to buffer
        self.write_buffer_items.append(item)
        self.write_buffer_embeddings.append((item.id, embedding))

        # Flush buffer if it's full
        if len(self.write_buffer_items) >= self.buffer_size:
            self._flush_buffer()

        # Invalidate query cache
        self.query_cache = {}

        return item.id

    async def add_batch(self, items: List[Item]) -> List[str]:
        """Add multiple items to the store.

        Args:
            items: Items to add

        Returns:
            List of IDs of the added items
        """
        if not items:
            return []

        # Generate embeddings
        contents = [item.content for item in items]
        embeddings = await self._generate_embeddings(contents)

        # Add to buffer
        for item, embedding in zip(items, embeddings):
            self.write_buffer_items.append(item)
            self.write_buffer_embeddings.append((item.id, embedding))

        # Flush buffer if it's full
        if len(self.write_buffer_items) >= self.buffer_size:
            self._flush_buffer()

        # Invalidate query cache
        self.query_cache = {}

        return [item.id for item in items]

    async def add_with_embedding(self, item: Item, embedding: np.ndarray) -> str:
        """Add an item with a pre-computed embedding.

        Args:
            item: Item to add
            embedding: Pre-computed embedding

        Returns:
            ID of the added item
        """
        # Add to buffer
        self.write_buffer_items.append(item)
        self.write_buffer_embeddings.append((item.id, embedding))

        # Flush buffer if it's full
        if len(self.write_buffer_items) >= self.buffer_size:
            self._flush_buffer()

        # Invalidate query cache
        self.query_cache = {}

        return item.id

    async def add_batch_with_embeddings(self, items: List[Item], embeddings: List[np.ndarray]) -> List[str]:
        """Add multiple items with pre-computed embeddings.

        Args:
            items: Items to add
            embeddings: Pre-computed embeddings

        Returns:
            List of IDs of the added items
        """
        if not items:
            return []

        # Add to buffer
        for item, embedding in zip(items, embeddings):
            self.write_buffer_items.append(item)
            self.write_buffer_embeddings.append((item.id, embedding))

        # Flush buffer if it's full
        if len(self.write_buffer_items) >= self.buffer_size:
            self._flush_buffer()

        # Invalidate query cache
        self.query_cache = {}

        return [item.id for item in items]

    async def get(self, item_id: str) -> Optional[Item]:
        """Get an item by ID.

        Args:
            item_id: ID of the item to get

        Returns:
            Item if found, None otherwise
        """
        # Check buffer first
        for item in self.write_buffer_items:
            if item.id == item_id:
                return item

        # Check items dictionary
        return self.items.get(item_id)

    async def get_batch(self, item_ids: List[str]) -> List[Optional[Item]]:
        """Get multiple items by ID.

        Args:
            item_ids: IDs of the items to get

        Returns:
            List of items (None for items not found)
        """
        results = []
        for item_id in item_ids:
            item = await self.get(item_id)
            results.append(item)
        return results

    async def get_embedding(self, item_id: str) -> Optional[np.ndarray]:
        """Get the embedding for an item.

        Args:
            item_id: ID of the item

        Returns:
            Embedding if found, None otherwise
        """
        # Check buffer first
        for buffer_id, embedding in self.write_buffer_embeddings:
            if buffer_id == item_id:
                return embedding

        # Check embeddings dictionary
        return self.embeddings.get(item_id)

    async def get_batch_embeddings(self, item_ids: List[str]) -> List[Optional[np.ndarray]]:
        """Get embeddings for multiple items.

        Args:
            item_ids: IDs of the items

        Returns:
            List of embeddings (None for items not found)
        """
        results = []
        for item_id in item_ids:
            embedding = await self.get_embedding(item_id)
            results.append(embedding)
        return results

    async def update(self, item_id: str, item: Item) -> bool:
        """Update an item.

        Args:
            item_id: ID of the item to update
            item: New item data

        Returns:
            True if successful, False otherwise
        """
        # Check if item exists
        if item_id not in self.items and not any(buffer_item.id == item_id for buffer_item in self.write_buffer_items):
            return False

        # Generate embedding
        embedding = await self._generate_embedding(item.content)

        # Update item and embedding
        self.items[item_id] = item
        self.embeddings[item_id] = embedding

        # Save to disk
        self._save_items()
        self._save_embeddings()

        # Invalidate normalized embeddings cache
        self.normalized_embeddings = None

        # Invalidate query cache
        self.query_cache = {}

        return True

    async def update_batch(self, item_ids: List[str], items: List[Item]) -> List[bool]:
        """Update multiple items.

        Args:
            item_ids: IDs of the items to update
            items: New item data

        Returns:
            List of success flags
        """
        results = []
        for item_id, item in zip(item_ids, items):
            result = await self.update(item_id, item)
            results.append(result)
        return results

    async def update_with_embedding(self, item_id: str, item: Item, embedding: np.ndarray) -> bool:
        """Update an item with a pre-computed embedding.

        Args:
            item_id: ID of the item to update
            item: New item data
            embedding: Pre-computed embedding

        Returns:
            True if successful, False otherwise
        """
        # Check if item exists
        if item_id not in self.items and not any(buffer_item.id == item_id for buffer_item in self.write_buffer_items):
            return False

        # Update item and embedding
        self.items[item_id] = item
        self.embeddings[item_id] = embedding

        # Save to disk
        self._save_items()
        self._save_embeddings()

        # Invalidate normalized embeddings cache
        self.normalized_embeddings = None

        # Invalidate query cache
        self.query_cache = {}

        return True

    async def update_batch_with_embeddings(self, item_ids: List[str], items: List[Item], embeddings: List[np.ndarray]) -> List[bool]:
        """Update multiple items with pre-computed embeddings.

        Args:
            item_ids: IDs of the items to update
            items: New item data
            embeddings: Pre-computed embeddings

        Returns:
            List of success flags
        """
        results = []
        for item_id, item, embedding in zip(item_ids, items, embeddings):
            result = await self.update_with_embedding(item_id, item, embedding)
            results.append(result)
        return results

    async def delete(self, item_id: str) -> bool:
        """Delete an item.

        Args:
            item_id: ID of the item to delete

        Returns:
            True if successful, False otherwise
        """
        # Check if item exists
        if item_id not in self.items and not any(buffer_item.id == item_id for buffer_item in self.write_buffer_items):
            return False

        # Remove from buffer
        self.write_buffer_items = [
            item for item in self.write_buffer_items if item.id != item_id]
        self.write_buffer_embeddings = [
            (i, e) for i, e in self.write_buffer_embeddings if i != item_id]

        # Remove from dictionaries
        if item_id in self.items:
            del self.items[item_id]
        if item_id in self.embeddings:
            del self.embeddings[item_id]

        # Save to disk
        self._save_items()
        self._save_embeddings()

        # Invalidate normalized embeddings cache
        self.normalized_embeddings = None

        # Invalidate query cache
        self.query_cache = {}

        return True

    async def delete_batch(self, item_ids: List[str]) -> List[bool]:
        """Delete multiple items.

        Args:
            item_ids: IDs of the items to delete

        Returns:
            List of success flags
        """
        results = []
        for item_id in item_ids:
            result = await self.delete(item_id)
            results.append(result)
        return results

    async def query(self, query: Query, top_k: Optional[int] = None) -> List[QueryResult]:
        """Query the store.

        Args:
            query: Query to execute
            top_k: Number of results to return (optional)

        Returns:
            List of query results
        """
        # Get top_k from configuration if not provided
        if top_k is None:
            top_k = get_top_k()

        # Check cache
        cache_key = (query.text, top_k)
        if cache_key in self.query_cache:
            return self.query_cache[cache_key]

        # Generate embedding
        query_embedding = await self._generate_embedding(query.text)

        # Get results
        results = await self.query_by_embedding(query_embedding, top_k)

        # Filter results based on metadata
        if query.metadata:
            include_messages = query.metadata.get("include_messages", True)
            include_knowledge = query.metadata.get("include_knowledge", True)

            filtered_results = []
            for result in results:
                item_type = result.metadata.get("type")
                if (item_type == "message" and include_messages) or (item_type == "knowledge" and include_knowledge):
                    filtered_results.append(result)

            results = filtered_results[:top_k]

        # Cache results
        if len(self.query_cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.query_cache))
            del self.query_cache[oldest_key]

        self.query_cache[cache_key] = results

        return results

    async def query_by_embedding(
        self,
        embedding: np.ndarray,
        top_k: Optional[int] = None
    ) -> List[QueryResult]:
        """Query the store by embedding.

        Args:
            embedding: Query embedding
            top_k: Number of results to return (optional)

        Returns:
            List of query results
        """
        # Get top_k from configuration if not provided
        if top_k is None:
            top_k = get_top_k()

        # Flush buffer to ensure all items are included in the search
        self._flush_buffer()

        # Get normalized embeddings
        item_ids, normalized_embeddings = self._get_normalized_embeddings()
        if not item_ids:
            return []

        # Normalize query embedding
        query_norm = np.linalg.norm(embedding)
        if query_norm > 0:
            normalized_query = embedding / query_norm
        else:
            normalized_query = embedding

        # Calculate similarities
        similarities = normalized_embeddings @ normalized_query

        # Get top-k indices
        if len(similarities) <= top_k:
            top_indices = np.argsort(similarities)[::-1]
        else:
            top_indices = np.argpartition(similarities, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(
                similarities[top_indices])[::-1]]

        # Get similarity threshold from configuration
        similarity_threshold = get_similarity_threshold()

        # Create results
        results = []
        for idx in top_indices:
            item_id = item_ids[idx]
            item = self.items[item_id]
            score = float(similarities[idx])

            # Skip items with low similarity
            if score < similarity_threshold:
                continue

            results.append(
                QueryResult(
                    id=item_id,
                    content=item.content,
                    score=score,
                    metadata=item.metadata,
                    store_type=StoreType.VECTOR,
                )
            )

        return results

    async def get_nearest_neighbors(
        self,
        item_id: str,
        top_k: Optional[int] = None
    ) -> List[QueryResult]:
        """Get the nearest neighbors of an item.

        Args:
            item_id: ID of the item
            top_k: Number of results to return (optional)

        Returns:
            List of query results
        """
        # Get top_k from configuration if not provided
        if top_k is None:
            top_k = get_top_k()

        # Get item embedding
        embedding = await self.get_embedding(item_id)
        if embedding is None:
            return []

        # Query by embedding
        results = await self.query_by_embedding(embedding, top_k + 1)

        # Remove the item itself from results
        return [result for result in results if result.id != item_id][:top_k]

    async def clear(self) -> bool:
        """Clear the store.

        Returns:
            True if successful, False otherwise
        """
        # Clear dictionaries
        self.items = {}
        self.embeddings = {}

        # Clear buffer
        self.write_buffer_items = []
        self.write_buffer_embeddings = []

        # Clear cache
        self.query_cache = {}
        self.normalized_embeddings = None

        # Remove files
        if os.path.exists(self.items_file):
            os.remove(self.items_file)
        if os.path.exists(self.embeddings_file):
            os.remove(self.embeddings_file)

        return True

    async def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate an embedding for a text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        from memfuse_core.common.utils.embeddings import create_embedding

        # Use the embeddings utility to create an embedding
        embedding = await create_embedding(text, self.model_name)
        return np.array(embedding)

    async def _generate_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts.

        Args:
            texts: Texts to embed

        Returns:
            List of embedding vectors
        """
        from memfuse_core.common.utils.embeddings import create_batch_embeddings

        # Use the embeddings utility to create batch embeddings
        embeddings = await create_batch_embeddings(texts, self.model_name)
        return [np.array(embedding) for embedding in embeddings]
