"""Buffer component interface for MemFuse."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BufferComponentInterface(ABC):
    """Interface for individual buffer components.

    This interface defines the methods that must be implemented by any
    buffer component (WriteBuffer, SpeculativeBuffer, QueryBuffer).
    """

    @property
    @abstractmethod
    def items(self) -> List[Any]:
        """Get all items in the buffer.

        Returns:
            List of items in the buffer
        """
        pass

    @property
    @abstractmethod
    def max_size(self) -> int:
        """Get the maximum size of the buffer.

        Returns:
            Maximum number of items in the buffer
        """
        pass

    @abstractmethod
    async def get_items(self) -> List[Any]:
        """Get all items in the buffer (async version).

        Returns:
            List of items in the buffer
        """
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear all items from the buffer.

        Returns:
            None
        """
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the buffer.

        Returns:
            Dictionary with buffer statistics
        """
        pass
