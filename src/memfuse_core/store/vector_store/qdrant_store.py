"""Qdrant vector store implementation for MemFuse server."""

from loguru import logger
from typing import List, Optional, Dict, Any
import os
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from ...models.core import Item, Query, QueryResult, StoreType
from ...utils.path_manager import PathManager
from .base import VectorStore
from ...rag.encode.base import EncoderBase
from ...rag.chunk.base import ChunkData


class QdrantVectorStore(VectorStore):
    """Qdrant vector store implementation.

    This implementation uses Qdrant for storing and retrieving embeddings.
    It provides high-performance vector operations with advanced filtering capabilities.
    """

    def __init__(
        self,
        data_dir: str,
        collection_name: str = "memfuse",
        encoder: Optional[EncoderBase] = None,
        embedding_dim: int = 384,
        **kwargs
    ):
        """Initialize the vector store.

        Args:
            data_dir: Directory to store data
            collection_name: Name of the collection
            encoder: Encoder to use
            embedding_dim: Dimension of the embeddings
            **kwargs: Additional arguments
        """
        super().__init__(data_dir, encoder=encoder, **kwargs)
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim

        # Initialize client
        self.client = None
        self.qdrant_path = os.path.join(data_dir, "qdrant")

    async def initialize(self) -> bool:
        """Initialize the vector store.

        Returns:
            True if successful, False otherwise
        """
        await super().initialize()

        try:
            logger.info(
                f"Initializing Qdrant vector store at {self.qdrant_path}")
            # Create Qdrant directory if it doesn't exist
            PathManager.ensure_directory(self.qdrant_path)
            logger.debug(
                f"Qdrant directory exists: {os.path.exists(self.qdrant_path)}")

            # Close any existing client first to release locks
            if hasattr(self, 'client') and self.client is not None:
                try:
                    logger.debug("Closing existing Qdrant client")
                    self.client.close()
                    # Set client to None to ensure it's properly garbage collected
                    self.client = None
                    logger.debug("Existing Qdrant client closed successfully")
                except Exception as e:
                    logger.warning(f"Error closing existing client: {e}")

            # Create client with minimal timeout to fail fast if there's a problem
            try:
                self.client = QdrantClient(path=self.qdrant_path, timeout=5.0)
                logger.info("Qdrant client created successfully")
            except Exception as e:
                logger.error(f"Failed to create Qdrant client: {e}")
                # Don't attempt to handle lock issues, just propagate the error
                raise

            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]

            if self.collection_name in collection_names:
                # Collection exists, check if dimensions match
                collection_info = self.client.get_collection(
                    self.collection_name)
                current_dim = collection_info.config.params.vectors.size

                if current_dim != self.embedding_dim:
                    logger.warning(
                        f"Collection dimension mismatch: {current_dim} != {self.embedding_dim}. "
                        f"Recreating collection with dimension {self.embedding_dim}"
                    )

                    # Delete the collection
                    self.client.delete_collection(
                        collection_name=self.collection_name)

                    # Recreate the collection
                    self._create_collection()
            else:
                # Collection doesn't exist, create it
                self._create_collection()

            logger.info(
                f"Initialized Qdrant vector store at {self.qdrant_path} with collection {self.collection_name}")
            return True
        except Exception as e:
            logger.error(
                f"Error initializing Qdrant vector store: {e}", exc_info=True)
            return False

    # Removed _handle_qdrant_locks method to avoid any latency-inducing operations

    def _create_collection(self):
        """Create a new collection in Qdrant."""
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=rest.VectorParams(
                    size=self.embedding_dim,
                    distance=rest.Distance.COSINE
                )
            )
            logger.info(
                f"Created collection {self.collection_name} with dimension {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise

    async def add_with_embedding(self, item: Item, embedding: np.ndarray) -> str:
        """Add an item with a pre-computed embedding.

        Args:
            item: Item to add
            embedding: Pre-computed embedding

        Returns:
            ID of the added item
        """
        await self.ensure_initialized()

        try:
            # Add to Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    rest.PointStruct(
                        id=item.id,
                        vector=embedding.tolist(),
                        payload={
                            "content": item.content,
                            "metadata": item.metadata
                        }
                    )
                ]
            )
            return item.id
        except Exception as e:
            logger.error(f"Error adding item to Qdrant: {e}")
            raise

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

        await self.ensure_initialized()

        try:
            # Add to Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    rest.PointStruct(
                        id=item.id,
                        vector=embedding.tolist(),
                        payload={
                            "content": item.content,
                            "metadata": item.metadata
                        }
                    )
                    for item, embedding in zip(items, embeddings)
                ]
            )
            return [item.id for item in items]
        except Exception as e:
            logger.error(f"Error adding batch to Qdrant: {e}")
            raise

    async def get(self, item_id: str) -> Optional[Item]:
        """Get an item by ID.

        Args:
            item_id: ID of the item

        Returns:
            Item if found, None otherwise
        """
        await self.ensure_initialized()

        try:
            # Get from Qdrant
            points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[item_id]
            )

            if not points:
                return None

            point = points[0]

            return Item(
                id=point.id,
                content=point.payload["content"],
                metadata=point.payload["metadata"]
            )
        except Exception as e:
            logger.error(f"Error getting item from Qdrant: {e}")
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

        await self.ensure_initialized()

        try:
            # Get from Qdrant
            points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=item_ids
            )

            # Create a map of ID to point
            point_map = {point.id: point for point in points}

            # Return items in the same order as item_ids
            return [
                Item(
                    id=point_map[item_id].id,
                    content=point_map[item_id].payload["content"],
                    metadata=point_map[item_id].payload["metadata"]
                )
                if item_id in point_map else None
                for item_id in item_ids
            ]
        except Exception as e:
            logger.error(f"Error getting batch from Qdrant: {e}")
            return [None] * len(item_ids)

    async def get_embedding(self, item_id: str) -> Optional[np.ndarray]:
        """Get the embedding for an item.

        Args:
            item_id: ID of the item

        Returns:
            Embedding if found, None otherwise
        """
        await self.ensure_initialized()

        try:
            # Get from Qdrant
            points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[item_id],
                with_payload=False,
                with_vectors=True
            )

            if not points:
                return None

            point = points[0]

            return np.array(point.vector)
        except Exception as e:
            logger.error(f"Error getting embedding from Qdrant: {e}")
            return None

    async def update_with_embedding(self, item_id: str, item: Item, embedding: np.ndarray) -> bool:
        """Update an item with a pre-computed embedding.

        Args:
            item_id: ID of the item to update
            item: New item data
            embedding: Pre-computed embedding

        Returns:
            True if successful, False otherwise
        """
        await self.ensure_initialized()

        try:
            # Update in Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    rest.PointStruct(
                        id=item_id,
                        vector=embedding.tolist(),
                        payload={
                            "content": item.content,
                            "metadata": item.metadata
                        }
                    )
                ]
            )
            return True
        except Exception as e:
            logger.error(f"Error updating item in Qdrant: {e}")
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
        if not items:
            return []

        await self.ensure_initialized()

        try:
            # Update in Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    rest.PointStruct(
                        id=item_id,
                        vector=embedding.tolist(),
                        payload={
                            "content": item.content,
                            "metadata": item.metadata
                        }
                    )
                    for item_id, item, embedding in zip(item_ids, items, embeddings)
                ]
            )
            return [True] * len(item_ids)
        except Exception as e:
            logger.error(f"Error updating batch in Qdrant: {e}")
            return [False] * len(item_ids)



    async def query_by_embedding(self, embedding: np.ndarray, top_k: int = 5, query: Optional[Query] = None) -> List[QueryResult]:
        """Query the store by embedding.

        Args:
            embedding: Query embedding
            top_k: Number of results to return
            query: Original query object for filtering (optional)

        Returns:
            List of query results
        """
        await self.ensure_initialized()

        try:
            # Prepare filter if user_id is provided
            filter_condition = None
            if query and query.metadata and "user_id" in query.metadata:
                user_id = query.metadata["user_id"]
                filter_condition = rest.Filter(
                    must=[
                        rest.FieldCondition(
                            key="metadata.user_id",
                            match=rest.MatchValue(value=user_id)
                        )
                    ]
                )
                logger.debug(f"Applying Qdrant filter for user_id: {user_id}")

            # Search in Qdrant with filter
            if filter_condition:
                search_results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=embedding.tolist(),
                    limit=top_k,
                    query_filter=filter_condition
                )
            else:
                search_results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=embedding.tolist(),
                    limit=top_k
                )

            # Convert to QueryResult objects and ensure user_id filter is applied
            results = []
            for result in search_results:
                # Double-check user_id filter
                if query and query.metadata and "user_id" in query.metadata:
                    user_id = query.metadata["user_id"]
                    result_user_id = result.payload["metadata"].get("user_id")
                    if result_user_id != user_id:
                        # Skip results that don't match the user_id
                        logger.debug(
                            f"Filtering out result with user_id={result_user_id}, expected {user_id}")
                        continue

                results.append(
                    QueryResult(
                        id=result.id,
                        content=result.payload["content"],
                        metadata=result.payload["metadata"],
                        # Qdrant already returns similarity scores (higher is better)
                        score=result.score,
                        store_type=StoreType.VECTOR
                    )
                )

            return results
        except Exception as e:
            logger.error(f"Error querying Qdrant by embedding: {e}")
            return []

    async def get_nearest_neighbors(self, item_id: str, top_k: int = 5) -> List[QueryResult]:
        """Get the nearest neighbors of an item.

        Args:
            item_id: ID of the item
            top_k: Number of results to return

        Returns:
            List of query results
        """
        await self.ensure_initialized()

        try:
            # Get embedding
            embedding = await self.get_embedding(item_id)

            if embedding is None:
                return []

            # Search in Qdrant
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=embedding.tolist(),
                limit=top_k + 1  # Add 1 to account for the item itself
            )

            # Filter out the item itself
            search_results = [
                result for result in search_results if result.id != item_id]

            # Convert to QueryResult objects
            return [
                QueryResult(
                    id=result.id,
                    content=result.payload["content"],
                    metadata=result.payload["metadata"],
                    # Qdrant already returns similarity scores (higher is better)
                    score=result.score,
                    store_type=StoreType.VECTOR
                )
                for result in search_results[:top_k]
            ]
        except Exception as e:
            logger.error(f"Error getting nearest neighbors from Qdrant: {e}")
            return []

    async def clear(self) -> bool:
        """Clear the store.

        Returns:
            True if successful, False otherwise
        """
        await self.ensure_initialized()

        try:
            # Delete collection
            self.client.delete_collection(collection_name=self.collection_name)

            # Recreate collection
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=rest.VectorParams(
                    size=self.embedding_dim,
                    distance=rest.Distance.COSINE
                )
            )
            return True
        except Exception as e:
            logger.error(f"Error clearing Qdrant store: {e}")
            return False

    async def close(self) -> None:
        """Close the store.

        This method should be called when the store is no longer needed.
        It will flush any pending operations and release resources.
        """
        await super().close()

        try:
            # Close Qdrant client
            if hasattr(self, 'client') and self.client is not None:
                self.client.close()
        except Exception as e:
            logger.error(f"Error closing Qdrant client: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store.

        Returns:
            Dictionary of statistics
        """
        stats = super().get_metrics()

        try:
            if self.client is not None:
                # Get collection info
                collection_info = self.client.get_collection(
                    self.collection_name)
                stats["vector_count"] = collection_info.vectors_count
                stats["embedding_dim"] = self.embedding_dim
                stats["collection_name"] = self.collection_name
        except Exception as e:
            logger.error(f"Error getting Qdrant stats: {e}")

        return stats

    # New ChunkStoreInterface methods
    async def add_with_embeddings(self, chunks: List[ChunkData], embeddings: List[np.ndarray]) -> List[str]:
        """Add chunks with pre-computed embeddings.

        Args:
            chunks: Chunks to add
            embeddings: Pre-computed embeddings

        Returns:
            List of IDs of the added chunks
        """
        if not chunks:
            return []

        await self.ensure_initialized()

        try:
            # Add to Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    rest.PointStruct(
                        id=chunk.chunk_id,
                        vector=embedding.tolist(),
                        payload={
                            "content": chunk.content,
                            "metadata": chunk.metadata
                        }
                    )
                    for chunk, embedding in zip(chunks, embeddings)
                ]
            )
            return [chunk.chunk_id for chunk in chunks]
        except Exception as e:
            logger.error(f"Error adding chunks to Qdrant: {e}")
            raise

    async def get_chunks_by_ids(self, chunk_ids: List[str]) -> List[Optional[ChunkData]]:
        """Get chunks by their IDs.

        Args:
            chunk_ids: List of chunk IDs to retrieve

        Returns:
            List of ChunkData objects, None for chunks not found
        """
        if not chunk_ids:
            return []

        await self.ensure_initialized()

        try:
            # Get from Qdrant
            points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=chunk_ids
            )

            # Create a map of ID to point
            point_map = {point.id: point for point in points}

            # Return chunks in the same order as chunk_ids
            return [
                ChunkData(
                    content=point_map[chunk_id].payload["content"],
                    chunk_id=chunk_id,
                    metadata=point_map[chunk_id].payload["metadata"]
                )
                if chunk_id in point_map else None
                for chunk_id in chunk_ids
            ]
        except Exception as e:
            logger.error(f"Error getting chunks from Qdrant: {e}")
            return [None] * len(chunk_ids)

    async def update_chunk_with_embedding(self, chunk_id: str, chunk: ChunkData, embedding: np.ndarray) -> bool:
        """Update a chunk with pre-computed embedding.

        Args:
            chunk_id: ID of the chunk to update
            chunk: New chunk data
            embedding: Pre-computed embedding

        Returns:
            True if successful, False if chunk not found
        """
        await self.ensure_initialized()

        try:
            # Update in Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    rest.PointStruct(
                        id=chunk_id,
                        vector=embedding.tolist(),
                        payload={
                            "content": chunk.content,
                            "metadata": chunk.metadata
                        }
                    )
                ]
            )
            return True
        except Exception as e:
            logger.error(f"Error updating chunk in Qdrant: {e}")
            return False

    async def delete_chunks_by_ids(self, chunk_ids: List[str]) -> List[bool]:
        """Delete chunks by their IDs.

        Args:
            chunk_ids: List of chunk IDs to delete

        Returns:
            List of deletion success flags
        """
        if not chunk_ids:
            return []

        await self.ensure_initialized()

        try:
            # Delete from Qdrant
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=rest.PointIdsList(
                    points=chunk_ids
                )
            )
            return [True] * len(chunk_ids)
        except Exception as e:
            logger.error(f"Error deleting chunks from Qdrant: {e}")
            return [False] * len(chunk_ids)

    async def query_by_embedding_chunks(self, embedding: np.ndarray, top_k: int = 5, query: Optional[Query] = None) -> List[ChunkData]:
        """Query the store by embedding and return chunks.

        Args:
            embedding: Query embedding
            top_k: Number of results to return
            query: Original query object for filtering (optional)

        Returns:
            List of ChunkData objects
        """
        await self.ensure_initialized()

        try:
            # Prepare filter if user_id is provided
            filter_condition = None
            if query and query.metadata and "user_id" in query.metadata:
                user_id = query.metadata["user_id"]
                filter_condition = rest.Filter(
                    must=[
                        rest.FieldCondition(
                            key="metadata.user_id",
                            match=rest.MatchValue(value=user_id)
                        )
                    ]
                )
                logger.debug(f"Applying Qdrant filter for user_id: {user_id}")

            # Search in Qdrant with filter
            if filter_condition:
                search_results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=embedding.tolist(),
                    limit=top_k,
                    query_filter=filter_condition
                )
            else:
                search_results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=embedding.tolist(),
                    limit=top_k
                )

            # Convert to ChunkData objects
            results = []
            for result in search_results:
                # Double-check user_id filter
                if query and query.metadata and "user_id" in query.metadata:
                    user_id = query.metadata["user_id"]
                    result_user_id = result.payload["metadata"].get("user_id")
                    if result_user_id != user_id:
                        # Skip results that don't match the user_id
                        logger.debug(
                            f"Filtering out result with user_id={result_user_id}, expected {user_id}")
                        continue

                # Add score to metadata for ranking
                chunk_metadata = result.payload["metadata"].copy()
                chunk_metadata["score"] = result.score

                results.append(
                    ChunkData(
                        content=result.payload["content"],
                        chunk_id=result.id,
                        metadata=chunk_metadata
                    )
                )

            return results
        except Exception as e:
            logger.error(f"Error querying Qdrant by embedding: {e}")
            return []

    async def get_chunk_count(self) -> int:
        """Get the total number of chunks in the store.

        Returns:
            Total number of chunks stored
        """
        await self.ensure_initialized()

        try:
            collection_info = self.client.get_collection(self.collection_name)
            return collection_info.vectors_count or 0
        except Exception as e:
            logger.error(f"Error getting chunk count from Qdrant: {e}")
            return 0

    async def clear_all_chunks(self) -> bool:
        """Clear all chunks from the store.

        Returns:
            True if successful, False otherwise
        """
        await self.ensure_initialized()

        try:
            # Delete the collection and recreate it
            self.client.delete_collection(collection_name=self.collection_name)
            self._create_collection()
            return True
        except Exception as e:
            logger.error(f"Error clearing Qdrant store: {e}")
            return False

    # Business Query Operations
    async def get_chunks_by_session(self, session_id: str) -> List[ChunkData]:
        """Get all chunks for a specific session.

        Args:
            session_id: Session ID to filter chunks

        Returns:
            List of ChunkData objects for the session
        """
        await self.ensure_initialized()

        try:
            filter_condition = rest.Filter(
                must=[
                    rest.FieldCondition(
                        key="metadata.session_id",
                        match=rest.MatchValue(value=session_id)
                    )
                ]
            )

            # Use search with a dummy vector to get all matching chunks
            dummy_vector = [0.0] * self.embedding_dim
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=dummy_vector,
                limit=10000,  # Large limit to get all results
                query_filter=filter_condition,
                with_payload=True,
                with_vectors=False
            )

            results = []
            for result in search_results:
                results.append(
                    ChunkData(
                        content=result.payload["content"],
                        chunk_id=str(result.id),
                        metadata=result.payload["metadata"]
                    )
                )

            logger.info(f"Retrieved {len(results)} chunks for session {session_id}")
            return results

        except Exception as e:
            logger.error(f"Error getting chunks by session from Qdrant: {e}")
            return []

    async def get_chunks_by_round(self, round_id: str) -> List[ChunkData]:
        """Get all chunks for a specific round.

        Args:
            round_id: Round ID to filter chunks

        Returns:
            List of ChunkData objects for the round
        """
        await self.ensure_initialized()

        try:
            filter_condition = rest.Filter(
                must=[
                    rest.FieldCondition(
                        key="metadata.round_id",
                        match=rest.MatchValue(value=round_id)
                    )
                ]
            )

            # Use search with a dummy vector to get all matching chunks
            dummy_vector = [0.0] * self.embedding_dim
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=dummy_vector,
                limit=10000,  # Large limit to get all results
                query_filter=filter_condition,
                with_payload=True,
                with_vectors=False
            )

            results = []
            for result in search_results:
                results.append(
                    ChunkData(
                        content=result.payload["content"],
                        chunk_id=str(result.id),
                        metadata=result.payload["metadata"]
                    )
                )

            logger.info(f"Retrieved {len(results)} chunks for round {round_id}")
            return results

        except Exception as e:
            logger.error(f"Error getting chunks by round from Qdrant: {e}")
            return []

    async def get_chunks_by_user(self, user_id: str) -> List[ChunkData]:
        """Get all chunks for a specific user.

        Args:
            user_id: User ID to filter chunks

        Returns:
            List of ChunkData objects for the user
        """
        await self.ensure_initialized()

        try:
            filter_condition = rest.Filter(
                must=[
                    rest.FieldCondition(
                        key="metadata.user_id",
                        match=rest.MatchValue(value=user_id)
                    )
                ]
            )

            # Use search with a dummy vector to get all matching chunks
            # This is a workaround since scroll might not be available in all Qdrant versions
            dummy_vector = [0.0] * self.embedding_dim
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=dummy_vector,
                limit=10000,  # Large limit to get all results
                query_filter=filter_condition,
                with_payload=True,
                with_vectors=False
            )

            results = []
            for result in search_results:
                results.append(
                    ChunkData(
                        content=result.payload["content"],
                        chunk_id=str(result.id),
                        metadata=result.payload["metadata"]
                    )
                )

            logger.info(f"Retrieved {len(results)} chunks for user {user_id}")
            return results

        except Exception as e:
            logger.error(f"Error getting chunks by user from Qdrant: {e}")
            return []

    async def get_chunks_by_strategy(self, strategy_type: str) -> List[ChunkData]:
        """Get all chunks created by a specific strategy.

        Args:
            strategy_type: Strategy type to filter chunks

        Returns:
            List of ChunkData objects for the strategy
        """
        await self.ensure_initialized()

        try:
            filter_condition = rest.Filter(
                must=[
                    rest.FieldCondition(
                        key="metadata.strategy",
                        match=rest.MatchValue(value=strategy_type)
                    )
                ]
            )

            # Use search with a dummy vector to get all matching chunks
            dummy_vector = [0.0] * self.embedding_dim
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=dummy_vector,
                limit=10000,  # Large limit to get all results
                query_filter=filter_condition,
                with_payload=True,
                with_vectors=False
            )

            results = []
            for result in search_results:
                results.append(
                    ChunkData(
                        content=result.payload["content"],
                        chunk_id=str(result.id),
                        metadata=result.payload["metadata"]
                    )
                )

            logger.info(f"Retrieved {len(results)} chunks for strategy {strategy_type}")
            return results

        except Exception as e:
            logger.error(f"Error getting chunks by strategy from Qdrant: {e}")
            return []

    async def get_chunks_stats(self, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get statistics about chunks in the store.

        Args:
            filters: Optional filters to apply (e.g., session_id, user_id)

        Returns:
            Dictionary containing statistics
        """
        await self.ensure_initialized()

        try:
            # Get total count
            total_chunks = await self.get_chunk_count()

            # Basic stats
            stats = {
                "total_chunks": total_chunks,
                "store_type": "vector",
                "by_session": {},
                "by_strategy": {},
                "by_user": {},
                "storage_size": "N/A"  # Qdrant doesn't provide easy storage size info
            }

            # If we have a reasonable number of chunks, get detailed stats
            if total_chunks > 0 and total_chunks < 10000:
                # Get all chunks to compute detailed stats
                dummy_vector = [0.0] * self.embedding_dim
                all_results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=dummy_vector,
                    limit=total_chunks,
                    with_payload=True,
                    with_vectors=False
                )

                # Count by different dimensions
                session_counts = {}
                strategy_counts = {}
                user_counts = {}

                for result in all_results:
                    metadata = result.payload.get("metadata", {})

                    # Count by session
                    session_id = metadata.get("session_id")
                    if session_id:
                        session_counts[session_id] = session_counts.get(session_id, 0) + 1

                    # Count by strategy
                    strategy = metadata.get("strategy")
                    if strategy:
                        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

                    # Count by user
                    user_id = metadata.get("user_id")
                    if user_id:
                        user_counts[user_id] = user_counts.get(user_id, 0) + 1

                stats["by_session"] = session_counts
                stats["by_strategy"] = strategy_counts
                stats["by_user"] = user_counts

            return stats

        except Exception as e:
            logger.error(f"Error getting chunks stats from Qdrant: {e}")
            return {
                "total_chunks": 0,
                "store_type": "vector",
                "by_session": {},
                "by_strategy": {},
                "by_user": {},
                "storage_size": "N/A",
                "error": str(e)
            }
