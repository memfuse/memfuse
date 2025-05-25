"""Base keyword store module for MemFuse server."""

from loguru import logger
from abc import abstractmethod
from typing import List, Optional

from ..base import StoreBase
from ...models.core import Item, Query, QueryResult
from ...models.core import StoreType


class KeywordStore(StoreBase):
    """Base class for keyword store implementations."""

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

    @abstractmethod
    async def add_document(self, item: Item) -> str:
        """Add a document to the store.

        Args:
            item: Item to add

        Returns:
            ID of the added document
        """
        pass

    @abstractmethod
    async def add_documents(self, items: List[Item]) -> List[str]:
        """Add multiple documents to the store.

        Args:
            items: Items to add

        Returns:
            List of IDs of the added documents
        """
        pass

    @abstractmethod
    async def get_document(self, item_id: str) -> Optional[Item]:
        """Get a document by ID.

        Args:
            item_id: ID of the document to get

        Returns:
            Document if found, None otherwise
        """
        pass

    @abstractmethod
    async def get_documents(self, item_ids: List[str]) -> List[Optional[Item]]:
        """Get multiple documents by ID.

        Args:
            item_ids: IDs of the documents to get

        Returns:
            List of documents (None for documents not found)
        """
        pass

    @abstractmethod
    async def update_document(self, item_id: str, item: Item) -> bool:
        """Update a document.

        Args:
            item_id: ID of the document to update
            item: New document data

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    async def update_documents(self, item_ids: List[str], items: List[Item]) -> List[bool]:
        """Update multiple documents.

        Args:
            item_ids: IDs of the documents to update
            items: New document data

        Returns:
            List of success flags
        """
        pass

    @abstractmethod
    async def delete_document(self, item_id: str) -> bool:
        """Delete a document.

        Args:
            item_id: ID of the document to delete

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    async def delete_documents(self, item_ids: List[str]) -> List[bool]:
        """Delete multiple documents.

        Args:
            item_ids: IDs of the documents to delete

        Returns:
            List of success flags
        """
        pass

    @abstractmethod
    async def search(self, query_text: str, top_k: int = 5, user_id: Optional[str] = None) -> List[QueryResult]:
        """Search the store.

        Args:
            query_text: Query text
            top_k: Number of results to return
            user_id: User ID to filter by (optional)

        Returns:
            List of query results
        """
        pass

    async def add(self, item: Item) -> str:
        """Add an item to the store.

        Args:
            item: Item to add

        Returns:
            ID of the added item
        """
        return await self.add_document(item)

    async def add_batch(self, items: List[Item]) -> List[str]:
        """Add multiple items to the store.

        Args:
            items: Items to add

        Returns:
            List of IDs of the added items
        """
        return await self.add_documents(items)

    async def get(self, item_id: str) -> Optional[Item]:
        """Get an item by ID.

        Args:
            item_id: ID of the item to get

        Returns:
            Item if found, None otherwise
        """
        return await self.get_document(item_id)

    async def get_batch(self, item_ids: List[str]) -> List[Optional[Item]]:
        """Get multiple items by ID.

        Args:
            item_ids: IDs of the items to get

        Returns:
            List of items (None for items not found)
        """
        return await self.get_documents(item_ids)

    async def update(self, item_id: str, item: Item) -> bool:
        """Update an item.

        Args:
            item_id: ID of the item to update
            item: New item data

        Returns:
            True if successful, False otherwise
        """
        return await self.update_document(item_id, item)

    async def update_batch(self, item_ids: List[str], items: List[Item]) -> List[bool]:
        """Update multiple items.

        Args:
            item_ids: IDs of the items to update
            items: New item data

        Returns:
            List of success flags
        """
        return await self.update_documents(item_ids, items)

    async def delete(self, item_id: str) -> bool:
        """Delete an item.

        Args:
            item_id: ID of the item to delete

        Returns:
            True if successful, False otherwise
        """
        return await self.delete_document(item_id)

    async def delete_batch(self, item_ids: List[str]) -> List[bool]:
        """Delete multiple items.

        Args:
            item_ids: IDs of the items to delete

        Returns:
            List of success flags
        """
        return await self.delete_documents(item_ids)

    async def query(self, query: Query, top_k: int = 5) -> List[QueryResult]:
        """Query the store.

        Args:
            query: Query to execute
            top_k: Number of results to return

        Returns:
            List of query results
        """
        # Add user_id to cache key if present
        cache_key = f"{query.text}:{top_k}"
        user_id = None
        if query.metadata and "user_id" in query.metadata:
            user_id = query.metadata["user_id"]
            cache_key += f":{user_id}"

        if cache_key in self.query_cache:
            return self.query_cache[cache_key]

        # Search for documents with user_id filter applied at the database level
        # Pass the user_id to the search method for database-level filtering
        results = await self.search(query.text, top_k, user_id=user_id)

        # Log the filtering
        if user_id:
            logger.debug(
                f"Applied user_id filter: {user_id} at database level for keyword query")

        # Apply user_id filter as a post-processing step
        if user_id:
            filtered_results = []
            for result in results:
                result_user_id = result.metadata.get("user_id")
                if result_user_id == user_id:
                    filtered_results.append(result)
                else:
                    logger.debug(
                        f"Post-filtering: Removing result with user_id={result_user_id}, expected {user_id}")

            results = filtered_results

        # Filter results based on metadata
        if query.metadata:
            include_messages = query.metadata.get("include_messages", True)
            include_knowledge = query.metadata.get("include_knowledge", True)

            filtered_results = []
            for result in results:
                item_type = result.metadata.get("type")
                if ((item_type == "message" and include_messages)
                        or (item_type == "knowledge" and include_knowledge)):
                    filtered_results.append(result)

            results = filtered_results[:top_k]

        # Cache results
        self.query_cache[cache_key] = results

        return results
