"""Buffer interface for MemFuse."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BufferInterface(ABC):
    """Interface for buffer components.

    This interface defines the methods that must be implemented by any
    buffer component, including the BufferManager.
    """

    @abstractmethod
    async def add_item(self, item: Any) -> bool:
        """Add an item to the buffer.

        Args:
            item: The item to add

        Returns:
            Whether a batch write was triggered
        """
        pass

    @abstractmethod
    async def query(
        self,
        query: str,
        limit: int = 5,
        store_type: Optional[str] = None,
        session_id: Optional[str] = None,
        scope: str = "all",
    ) -> Dict[str, Any]:
        """Query the buffer for relevant items.

        Args:
            query: Query string
            limit: Maximum number of results to return
            store_type: Type of store to query (vector, graph, keyword, or None for all)
            session_id: Session ID to filter results (optional)
            scope: Scope of the query (all, session, or user)

        Returns:
            Dictionary with query results
        """
        pass
