"""SQLite keyword store adapter for the refactored architecture."""

from typing import List, Optional

from .base import KeywordStore
from ...models.core import Item, Query, QueryResult
from .sqlite_store import SQLiteKeywordStore as OldSQLiteKeywordStore

class SQLiteKeywordStore(KeywordStore):
    """SQLite keyword store adapter for the refactored architecture.

    This class adapts the existing SQLiteKeywordStore implementation to the new
    KeywordStore interface.
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
        super().__init__(data_dir, cache_size=cache_size, **kwargs)

        # Create the old implementation
        self.store = OldSQLiteKeywordStore(
            data_dir=data_dir,
            cache_size=cache_size,
            **kwargs
        )

    async def initialize(self) -> bool:
        """Initialize the keyword store.

        Returns:
            True if successful, False otherwise
        """
        return await self.store.initialize()

    async def add_document(self, item: Item) -> str:
        """Add a document to the store.

        Args:
            item: Item to add as a document

        Returns:
            ID of the added document
        """
        return await self.store.add_document(item)

    async def add_documents(self, items: List[Item]) -> List[str]:
        """Add multiple documents to the store.

        Args:
            items: Items to add as documents

        Returns:
            List of IDs of the added documents
        """
        return await self.store.add_documents(items)

    async def get_document(self, document_id: str) -> Optional[Item]:
        """Get a document by ID.

        Args:
            document_id: ID of the document

        Returns:
            Document if found, None otherwise
        """
        return await self.store.get_document(document_id)

    async def get_documents(self, document_ids: List[str]) -> List[Optional[Item]]:
        """Get multiple documents by ID.

        Args:
            document_ids: IDs of the documents

        Returns:
            List of documents (None for documents not found)
        """
        return await self.store.get_documents(document_ids)

    async def update_document(self, document_id: str, item: Item) -> bool:
        """Update a document.

        Args:
            document_id: ID of the document to update
            item: New document data

        Returns:
            True if successful, False otherwise
        """
        return await self.store.update_document(document_id, item)

    async def update_documents(self, document_ids: List[str], items: List[Item]) -> List[bool]:
        """Update multiple documents.

        Args:
            document_ids: IDs of the documents to update
            items: New document data

        Returns:
            List of success flags
        """
        return await self.store.update_documents(document_ids, items)

    async def delete_document(self, document_id: str) -> bool:
        """Delete a document.

        Args:
            document_id: ID of the document to delete

        Returns:
            True if successful, False otherwise
        """
        return await self.store.delete_document(document_id)

    async def delete_documents(self, document_ids: List[str]) -> List[bool]:
        """Delete multiple documents.

        Args:
            document_ids: IDs of the documents to delete

        Returns:
            List of success flags
        """
        return await self.store.delete_documents(document_ids)

    async def search(self, query_text: str, top_k: int = 5) -> List[QueryResult]:
        """Search for documents matching the query text.

        Args:
            query_text: Query text
            top_k: Number of results to return

        Returns:
            List of query results
        """
        return await self.store.search(query_text, top_k)

    async def query(self, query: Query, top_k: int = 5) -> List[QueryResult]:
        """Query the store.

        Args:
            query: Query to execute
            top_k: Number of results to return

        Returns:
            List of query results
        """
        return await self.store.query(query, top_k)

    async def clear(self) -> bool:
        """Clear the store.

        Returns:
            True if successful, False otherwise
        """
        return await self.store.clear()
