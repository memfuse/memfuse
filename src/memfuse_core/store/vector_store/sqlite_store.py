"""SQLite vector store implementation."""

import os
import json
import sqlite3
import uuid
import numpy as np
import asyncio
from typing import List, Optional

from ...utils.config import (
    get_embedding_dim,
    get_top_k,
    get_similarity_threshold,
)
from ...models.core import Item, QueryResult
from .base import VectorStore
from ...utils.path_manager import PathManager


class SQLiteVectorStore(VectorStore):
    """SQLite-based vector store implementation.

    This implementation uses SQLite for storing and retrieving embeddings.
    It provides efficient vector operations with minimal dependencies.
    """

    def __init__(
        self,
        data_dir: str,
        model_name: str = "all-MiniLM-L6-v2",
        embedding_dim: Optional[int] = None,
        cache_size: int = 100,
        buffer_size: int = 10,
        **kwargs
    ):
        """Initialize the SQLite vector store.

        Args:
            data_dir: Directory to store data
            model_name: Name of the embedding model
            embedding_dim: Dimension of the embeddings (optional)
            cache_size: Size of the query cache
            buffer_size: Size of the write buffer
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
        self.vector_dir = os.path.join(data_dir, "sqlite_vector_store")

        # Initialize database path
        self.db_path = os.path.join(self.vector_dir, "vector_store.db")

        # Initialize connection and cursor
        self.conn = None
        self.cursor = None

        # Initialize lock for thread safety
        self._init_lock = asyncio.Lock()

    async def initialize(self) -> bool:
        """Initialize the vector store.

        Returns:
            True if successful, False otherwise
        """
        async with self._init_lock:
            if self.initialized:
                return True

            try:
                # Create vector store directory
                PathManager.ensure_directory(self.vector_dir)

                # Initialize SQLite database
                self.conn = sqlite3.connect(self.db_path)
                self.conn.row_factory = sqlite3.Row
                self.cursor = self.conn.cursor()

                # Create tables
                self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS items (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                ''')

                self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS embeddings (
                    id TEXT PRIMARY KEY,
                    embedding BLOB NOT NULL,
                    FOREIGN KEY (id) REFERENCES items(id) ON DELETE CASCADE
                )
                ''')

                # Create index on items
                self.cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_items_id ON items(id)
                ''')

                # Enable foreign keys
                self.cursor.execute('PRAGMA foreign_keys = ON')

                # Commit changes
                self.conn.commit()

                self.initialized = True
                return True
            except Exception as e:
                print(f"Error initializing SQLiteVectorStore: {e}")
                return False

    def _item_from_row(self, row) -> Item:
        """Create an Item from a database row.

        Args:
            row: Database row

        Returns:
            Item object
        """
        return Item(
            id=row['id'],
            content=row['content'],
            metadata=json.loads(row['metadata'])
        )

    def _embedding_to_bytes(self, embedding: np.ndarray) -> bytes:
        """Convert embedding to bytes for store.

        Args:
            embedding: Embedding vector

        Returns:
            Bytes representation
        """
        return embedding.tobytes()

    def _bytes_to_embedding(self, data: bytes) -> np.ndarray:
        """Convert bytes to embedding.

        Args:
            data: Bytes representation

        Returns:
            Embedding vector
        """
        return np.frombuffer(data, dtype=np.float32)

    async def add_with_embedding(self, item: Item, embedding: np.ndarray) -> str:
        """Add an item with a pre-computed embedding.

        Args:
            item: Item to add
            embedding: Pre-computed embedding

        Returns:
            ID of the added item
        """
        # Ensure initialized
        await self.ensure_initialized()

        try:
            # Generate ID if not provided
            if not item.id:
                item.id = str(uuid.uuid4())

            # Add item to database
            self.cursor.execute(
                'INSERT OR REPLACE INTO items (id, content, metadata) VALUES (?, ?, ?)',
                (item.id, item.content, json.dumps(item.metadata))
            )

            # Add embedding to database
            embedding_bytes = self._embedding_to_bytes(embedding)
            self.cursor.execute(
                'INSERT OR REPLACE INTO embeddings (id, embedding) VALUES (?, ?)',
                (item.id, embedding_bytes)
            )

            # Commit changes
            self.conn.commit()

            # Invalidate query cache
            self.query_cache.clear()

            return item.id
        except Exception as e:
            print(f"Error adding item with embedding: {e}")
            self.conn.rollback()
            return item.id

    async def add_batch_with_embeddings(
        self, items: List[Item], embeddings: List[np.ndarray]
    ) -> List[str]:
        """Add multiple items with pre-computed embeddings.

        Args:
            items: Items to add
            embeddings: Pre-computed embeddings

        Returns:
            List of IDs of the added items
        """
        if not items:
            return []

        # Ensure initialized
        await self.ensure_initialized()

        try:
            # Generate IDs if not provided
            for item in items:
                if not item.id:
                    item.id = str(uuid.uuid4())

            # Begin transaction
            self.conn.execute('BEGIN TRANSACTION')

            # Add items to database
            self.cursor.executemany(
                'INSERT OR REPLACE INTO items (id, content, metadata) VALUES (?, ?, ?)',
                [(item.id, item.content, json.dumps(item.metadata)) for item in items]
            )

            # Add embeddings to database
            self.cursor.executemany(
                'INSERT OR REPLACE INTO embeddings (id, embedding) VALUES (?, ?)',
                [(item.id, self._embedding_to_bytes(embedding))
                 for item, embedding in zip(items, embeddings)]
            )

            # Commit changes
            self.conn.commit()

            # Invalidate query cache
            self.query_cache.clear()

            return [item.id for item in items]
        except Exception as e:
            print(f"Error adding items with embeddings: {e}")
            self.conn.rollback()

            # Try individual adds
            results = []
            for item, embedding in zip(items, embeddings):
                item_id = await self.add_with_embedding(item, embedding)
                results.append(item_id)
            return results

    async def get(self, item_id: str) -> Optional[Item]:
        """Get an item by ID.

        Args:
            item_id: ID of the item to get

        Returns:
            Item if found, None otherwise
        """
        # Ensure initialized
        await self.ensure_initialized()

        try:
            # Get item from database
            self.cursor.execute(
                'SELECT id, content, metadata FROM items WHERE id = ?',
                (item_id,)
            )

            row = self.cursor.fetchone()

            if row:
                return self._item_from_row(row)

            return None
        except Exception as e:
            print(f"Error getting item: {e}")
            return None

    async def get_batch(self, item_ids: List[str]) -> List[Optional[Item]]:
        """Get multiple items by ID.

        Args:
            item_ids: IDs of the items to get

        Returns:
            List of items (None for items not found)
        """
        if not item_ids:
            return []

        # Ensure initialized
        await self.ensure_initialized()

        try:
            # Get items from database
            placeholders = ', '.join(['?'] * len(item_ids))
            self.cursor.execute(
                f'SELECT id, content, metadata FROM items WHERE id IN ({placeholders})',
                item_ids
            )

            rows = self.cursor.fetchall()

            # Create a dictionary of id -> item
            items_dict = {row['id']: self._item_from_row(row) for row in rows}

            # Return items in the same order as item_ids
            return [items_dict.get(item_id) for item_id in item_ids]
        except Exception as e:
            print(f"Error getting items: {e}")

            # Try individual gets
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
        # Ensure initialized
        await self.ensure_initialized()

        try:
            # Get embedding from database
            self.cursor.execute(
                'SELECT embedding FROM embeddings WHERE id = ?',
                (item_id,)
            )

            row = self.cursor.fetchone()

            if row:
                return self._bytes_to_embedding(row['embedding'])

            return None
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return None

    async def get_batch_embeddings(
        self, item_ids: List[str]
    ) -> List[Optional[np.ndarray]]:
        """Get embeddings for multiple items.

        Args:
            item_ids: IDs of the items

        Returns:
            List of embeddings (None for items not found)
        """
        if not item_ids:
            return []

        # Ensure initialized
        await self.ensure_initialized()

        try:
            # Get embeddings from database
            placeholders = ', '.join(['?'] * len(item_ids))
            self.cursor.execute(
                f'SELECT id, embedding FROM embeddings WHERE id IN ({placeholders})',
                item_ids
            )

            rows = self.cursor.fetchall()

            # Create a dictionary of id -> embedding
            embeddings_dict = {
                row['id']: self._bytes_to_embedding(row['embedding']) for row in rows
            }

            # Return embeddings in the same order as item_ids
            return [embeddings_dict.get(item_id) for item_id in item_ids]
        except Exception as e:
            print(f"Error getting embeddings: {e}")

            # Try individual gets
            results = []
            for item_id in item_ids:
                embedding = await self.get_embedding(item_id)
                results.append(embedding)
            return results

    async def update_with_embedding(
        self, item_id: str, item: Item, embedding: np.ndarray
    ) -> bool:
        """Update an item with a pre-computed embedding.

        Args:
            item_id: ID of the item to update
            item: New item data
            embedding: Pre-computed embedding

        Returns:
            True if successful, False otherwise
        """
        # Ensure initialized
        await self.ensure_initialized()

        try:
            # Check if item exists
            self.cursor.execute(
                'SELECT id FROM items WHERE id = ?',
                (item_id,)
            )

            if not self.cursor.fetchone():
                return False

            # Begin transaction
            self.conn.execute('BEGIN TRANSACTION')

            # Update item
            self.cursor.execute(
                'UPDATE items SET content = ?, metadata = ? WHERE id = ?',
                (item.content, json.dumps(item.metadata), item_id)
            )

            # Update embedding
            embedding_bytes = self._embedding_to_bytes(embedding)
            self.cursor.execute(
                'UPDATE embeddings SET embedding = ? WHERE id = ?',
                (embedding_bytes, item_id)
            )

            # Commit changes
            self.conn.commit()

            # Invalidate query cache
            self.query_cache.clear()

            return True
        except Exception as e:
            print(f"Error updating item with embedding: {e}")
            self.conn.rollback()
            return False

    async def update_batch_with_embeddings(
        self,
        item_ids: List[str],
        items: List[Item],
        embeddings: List[np.ndarray]
    ) -> List[bool]:
        """Update multiple items with pre-computed embeddings.

        Args:
            item_ids: IDs of the items to update
            items: New item data
            embeddings: Pre-computed embeddings

        Returns:
            List of success flags
        """
        if not item_ids or not items or len(item_ids) != len(items):
            return [False] * len(item_ids)

        # Ensure initialized
        await self.ensure_initialized()

        try:
            # Begin transaction
            self.conn.execute('BEGIN TRANSACTION')

            # Update items
            self.cursor.executemany(
                'UPDATE items SET content = ?, metadata = ? WHERE id = ?',
                [(item.content, json.dumps(item.metadata), item_id)
                 for item, item_id in zip(items, item_ids)]
            )

            # Update embeddings
            self.cursor.executemany(
                'UPDATE embeddings SET embedding = ? WHERE id = ?',
                [(self._embedding_to_bytes(embedding), item_id)
                 for embedding, item_id in zip(embeddings, item_ids)]
            )

            # Commit changes
            self.conn.commit()

            # Invalidate query cache
            self.query_cache.clear()

            # Check which updates were successful
            self.cursor.execute(
                f'SELECT id FROM items WHERE id IN ({", ".join(["?"] * len(item_ids))})',
                item_ids
            )

            updated_ids = {row['id'] for row in self.cursor.fetchall()}

            return [item_id in updated_ids for item_id in item_ids]
        except Exception as e:
            print(f"Error updating items with embeddings: {e}")
            self.conn.rollback()

            # Try individual updates
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
        # Ensure initialized
        await self.ensure_initialized()

        try:
            # Check if item exists
            self.cursor.execute(
                'SELECT id FROM items WHERE id = ?',
                (item_id,)
            )

            if not self.cursor.fetchone():
                return False

            # Begin transaction
            self.conn.execute('BEGIN TRANSACTION')

            # Delete item (will cascade to embeddings)
            self.cursor.execute(
                'DELETE FROM items WHERE id = ?',
                (item_id,)
            )

            # Commit changes
            self.conn.commit()

            # Invalidate query cache
            self.query_cache.clear()

            return True
        except Exception as e:
            print(f"Error deleting item: {e}")
            self.conn.rollback()
            return False

    async def delete_batch(self, item_ids: List[str]) -> List[bool]:
        """Delete multiple items.

        Args:
            item_ids: IDs of the items to delete

        Returns:
            List of success flags
        """
        if not item_ids:
            return []

        # Ensure initialized
        await self.ensure_initialized()

        try:
            # Begin transaction
            self.conn.execute('BEGIN TRANSACTION')

            # Delete items (will cascade to embeddings)
            placeholders = ', '.join(['?'] * len(item_ids))
            self.cursor.execute(
                f'DELETE FROM items WHERE id IN ({placeholders})',
                item_ids
            )

            # Commit changes
            self.conn.commit()

            # Invalidate query cache
            self.query_cache.clear()

            # All deletions are considered successful
            return [True] * len(item_ids)
        except Exception as e:
            print(f"Error deleting items: {e}")
            self.conn.rollback()

            # Try individual deletes
            results = []
            for item_id in item_ids:
                result = await self.delete(item_id)
                results.append(result)
            return results

    async def query_by_embedding(
        self, embedding: np.ndarray, top_k: Optional[int] = None
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

        # Ensure initialized
        await self.ensure_initialized()

        try:
            # Get all embeddings
            self.cursor.execute('SELECT id, embedding FROM embeddings')
            rows = self.cursor.fetchall()

            # Get similarity threshold from configuration
            similarity_threshold = get_similarity_threshold()

            # Calculate cosine similarity for each embedding
            similarities = []
            for row in rows:
                item_id = row['id']
                item_embedding = self._bytes_to_embedding(row['embedding'])

                # Calculate cosine similarity
                similarity = self._cosine_similarity(embedding, item_embedding)

                if similarity >= similarity_threshold:
                    similarities.append((item_id, similarity))

            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)

            # Get top-k results
            top_similarities = similarities[:top_k]

            # Get items for top similarities
            results = []
            for item_id, similarity in top_similarities:
                # Get item
                self.cursor.execute(
                    'SELECT id, content, metadata FROM items WHERE id = ?',
                    (item_id,)
                )

                row = self.cursor.fetchone()

                if row:
                    item = self._item_from_row(row)

                    # Create QueryResult
                    results.append(
                        QueryResult(
                            id=item.id,
                            content=item.content,
                            score=similarity,
                            metadata=item.metadata,
                            store_type=self.store_type,
                        )
                    )

            return results
        except Exception as e:
            print(f"Error querying by embedding: {e}")
            return []

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            a: First vector
            b: Second vector

        Returns:
            Cosine similarity
        """
        # Normalize vectors
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)

        # Handle zero vectors
        if a_norm == 0 or b_norm == 0:
            return 0.0

        # Calculate cosine similarity
        return np.dot(a, b) / (a_norm * b_norm)

    async def get_nearest_neighbors(
        self, item_id: str, top_k: Optional[int] = None
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

        # Filter out the item itself
        return [result for result in results if result.id != item_id][:top_k]

    async def clear(self) -> bool:
        """Clear the store.

        Returns:
            True if successful, False otherwise
        """
        # Ensure initialized
        await self.ensure_initialized()

        try:
            # Begin transaction
            self.conn.execute('BEGIN TRANSACTION')

            # Delete all items (will cascade to embeddings)
            self.cursor.execute('DELETE FROM items')

            # Commit changes
            self.conn.commit()

            # Invalidate query cache
            self.query_cache.clear()

            return True
        except Exception as e:
            print(f"Error clearing store: {e}")
            self.conn.rollback()
            return False

    async def add(self, item: Item) -> str:
        """Add an item to the store.

        Args:
            item: Item to add

        Returns:
            ID of the added item
        """
        # Generate embedding
        embedding = await self._generate_embedding(item.content)

        # Add with embedding
        return await self.add_with_embedding(item, embedding)

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

        # Add with embeddings
        return await self.add_batch_with_embeddings(items, embeddings)

    async def update(self, item_id: str, item: Item) -> bool:
        """Update an item.

        Args:
            item_id: ID of the item to update
            item: New item data

        Returns:
            True if successful, False otherwise
        """
        # Generate embedding
        embedding = await self._generate_embedding(item.content)

        # Update with embedding
        return await self.update_with_embedding(item_id, item, embedding)

    async def update_batch(self, item_ids: List[str], items: List[Item]) -> List[bool]:
        """Update multiple items.

        Args:
            item_ids: IDs of the items to update
            items: New item data

        Returns:
            List of success flags
        """
        if not item_ids or not items or len(item_ids) != len(items):
            return [False] * len(item_ids)

        # Generate embeddings
        contents = [item.content for item in items]
        embeddings = await self._generate_embeddings(contents)

        # Update with embeddings
        return await self.update_batch_with_embeddings(item_ids, items, embeddings)

    async def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        # Use the embedding service to generate embedding
        return await self.embedding_service.get_embedding(text)

    async def _generate_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts.

        Args:
            texts: Texts to embed

        Returns:
            List of embedding vectors
        """
        # Use the embedding service to generate embeddings
        return await self.embedding_service.get_embeddings(texts)