"""Base store module for MemFuse server."""

from abc import ABC, abstractmethod
import asyncio

from ..models.core import StoreType


class StoreBase(ABC):
    """Base class for all store implementations.

    This class provides common functionality for all store types.
    Subclasses should implement the ChunkStoreInterface.
    """

    def __init__(
        self,
        data_dir: str,
        buffer_size: int = 10,
        **kwargs
    ):
        """Initialize the store.

        Args:
            data_dir: Directory to store data
            buffer_size: Size of the buffer
            **kwargs: Additional arguments
        """
        self.data_dir = data_dir
        self.buffer_size = buffer_size
        self.initialized = False
        self._lock = asyncio.Lock()

        # Store additional arguments
        self.kwargs = kwargs

        # Initialize metrics
        self.metrics = {
            "add_count": 0,
            "read_count": 0,
            "update_count": 0,
            "delete_count": 0,
            "query_count": 0,
            "add_time": 0,
            "read_time": 0,
            "update_time": 0,
            "delete_time": 0,
            "query_time": 0,
        }

        # Initialize buffer
        from ..buffer.base import BufferBase
        self.buffer = BufferBase(max_size=buffer_size)

    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the storage.

        Returns:
            True if successful, False otherwise
        """
        pass

    async def ensure_initialized(self) -> bool:
        """Ensure the store is initialized.

        Returns:
            True if successful, False otherwise
        """
        async with self._lock:
            if not self.initialized:
                return await self.initialize()
            return True

    @property
    @abstractmethod
    def store_type(self) -> StoreType:
        """Get the store type.

        Returns:
            Store type
        """
        pass

    def get_metrics(self):
        """Get store metrics.

        Returns:
            Dictionary of metrics
        """
        return self.metrics.copy()

    async def close(self) -> None:
        """Close the store.

        This method should be called when the store is no longer needed.
        It will flush any pending operations and release resources.
        """
        # Flush buffer
        if hasattr(self, 'buffer'):
            await self.buffer.flush()
