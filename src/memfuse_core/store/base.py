"""Base store module for MemFuse server."""

from abc import ABC, abstractmethod
from typing import List, Optional
import asyncio

from ..models import Item, Query, QueryResult, StoreType
from ..utils.cache import Cache
from ..utils.path_manager import PathManager
from ..interfaces import StoreInterface

# ModelService will be imported locally to avoid circular imports


class StoreBase(StoreInterface, ABC):
    """Base class for all store implementations.

    This class implements the StoreInterface and provides common functionality
    for all store implementations. It removes the registry pattern which is
    now handled by the factory.
    """

    def __init__(
        self,
        data_dir: str,
        model_name: str = "all-MiniLM-L6-v2",
        cache_size: int = 100,
        buffer_size: int = 10,
        **kwargs
    ):
        """Initialize the storage.

        Args:
            data_dir: Directory to store data
            model_name: Name of the embedding model
            cache_size: Size of the query cache
            buffer_size: Size of the write buffer
            **kwargs: Additional arguments
        """
        self.data_dir = data_dir
        self.model_name = model_name
        self.initialized = False
        self._lock = asyncio.Lock()

        # Initialize cache
        self.query_cache = Cache(max_size=cache_size)

        # Use model service for embeddings
        # (using local import to avoid circular imports)
        from ..services.model_service import get_model_service
        self.model_service = get_model_service()

        # Initialize buffer
        from ..buffer.base import BufferBase
        self.buffer = BufferBase(max_size=buffer_size)

    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the storage.

        Returns:
            True if successful, False otherwise
        """
        # Create data directory if it doesn't exist
        PathManager.ensure_directory(self.data_dir)
        self.initialized = True
        return True

    async def ensure_initialized(self) -> bool:
        """Ensure the store is initialized.

        Returns:
            True if successful, False otherwise
        """
        if self.initialized:
            return True

        async with self._lock:
            if not self.initialized:
                return await self.initialize()
            return True

    @abstractmethod
    async def add(self, item: Item) -> str:
        """Add an item to the store.

        Args:
            item: Item to add

        Returns:
            ID of the added item
        """
        pass

    @abstractmethod
    async def add_batch(self, items: List[Item]) -> List[str]:
        """Add multiple items to the store.

        Args:
            items: Items to add

        Returns:
            List of IDs of the added items
        """
        pass

    @abstractmethod
    async def get(self, item_id: str) -> Optional[Item]:
        """Get an item by ID.

        Args:
            item_id: ID of the item to get

        Returns:
            Item if found, None otherwise
        """
        pass

    @abstractmethod
    async def get_batch(self, item_ids: List[str]) -> List[Optional[Item]]:
        """Get multiple items by ID.

        Args:
            item_ids: IDs of the items to get

        Returns:
            List of items (None for items not found)
        """
        pass

    @abstractmethod
    async def update(self, item_id: str, item: Item) -> bool:
        """Update an item.

        Args:
            item_id: ID of the item to update
            item: New item data

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    async def update_batch(self, item_ids: List[str], items: List[Item]) -> List[bool]:
        """Update multiple items.

        Args:
            item_ids: IDs of the items to update
            items: New item data

        Returns:
            List of success flags
        """
        pass

    @abstractmethod
    async def delete(self, item_id: str) -> bool:
        """Delete an item.

        Args:
            item_id: ID of the item to delete

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    async def delete_batch(self, item_ids: List[str]) -> List[bool]:
        """Delete multiple items.

        Args:
            item_ids: IDs of the items to delete

        Returns:
            List of success flags
        """
        pass

    @abstractmethod
    async def query(self, query: Query, top_k: int = 5) -> List[QueryResult]:
        """Query the store.

        Args:
            query: Query to execute
            top_k: Number of results to return

        Returns:
            List of query results
        """
        pass

    @abstractmethod
    async def clear(self) -> bool:
        """Clear the store.

        Returns:
            True if successful, False otherwise
        """
        pass

    @property
    @abstractmethod
    def store_type(self) -> StoreType:
        """Get the store type.

        Returns:
            Store type
        """
        pass

    async def close(self) -> None:
        """Close the store.

        This method should be called when the store is no longer needed.
        It will flush any pending operations and release resources.
        """
        # Flush buffer
        if hasattr(self, 'buffer'):
            await self.buffer.flush()
            # Check if buffer has stop method
            if hasattr(self.buffer, 'stop'):
                await self.buffer.stop()

    def __del__(self):
        """Destructor."""
        # Try to close the store
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.close())
            else:
                loop.run_until_complete(self.close())
        except Exception:
            pass
