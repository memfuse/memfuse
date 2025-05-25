"""Memory interface for MemFuse."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class MemoryInterface(ABC):
    """Interface for memory services.

    This interface defines the methods that must be implemented by any
    memory service, including the MemoryService and BufferService.
    """

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the memory service."""
        pass

    @abstractmethod
    async def add(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Add messages to memory.

        Args:
            messages: List of message dictionaries with role and content

        Returns:
            Dictionary with status, code, and message IDs
        """
        pass

    @abstractmethod
    async def add_batch(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Add a batch of messages to memory.

        Args:
            messages: List of message dictionaries with role and content

        Returns:
            Dictionary with status, code, and message IDs
        """
        pass

    @abstractmethod
    async def query(
        self,
        query: str,
        top_k: int = 5,
        store_type: Optional[str] = None,
        session_id: Optional[str] = None,
        scope: str = "all",
    ) -> Dict[str, Any]:
        """Query memory for relevant messages.

        Args:
            query: Query string
            top_k: Maximum number of results to return
            store_type: Type of store to query (vector, graph, keyword, or None for all)
            session_id: Session ID to filter results (optional)
            scope: Scope of the query (all, session, or user)

        Returns:
            Dictionary with status, code, and query results
        """
        pass

    @abstractmethod
    async def read(self, message_ids: List[str]) -> Dict[str, Any]:
        """Read messages from memory.

        Args:
            message_ids: List of message IDs

        Returns:
            Dictionary with status, code, and messages
        """
        pass

    @abstractmethod
    async def update(self, message_ids: List[str], new_messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Update messages in memory.

        Args:
            message_ids: List of message IDs
            new_messages: List of new message dictionaries with role and content

        Returns:
            Dictionary with status, code, and updated message IDs
        """
        pass

    @abstractmethod
    async def delete(self, message_ids: List[str]) -> Dict[str, Any]:
        """Delete messages from memory.

        Args:
            message_ids: List of message IDs

        Returns:
            Dictionary with status, code, and deleted message IDs
        """
        pass

    @abstractmethod
    async def add_knowledge(self, knowledge: List[str]) -> Dict[str, Any]:
        """Add knowledge to memory.

        Args:
            knowledge: List of knowledge strings

        Returns:
            Dictionary with status, code, and knowledge IDs
        """
        pass

    @abstractmethod
    async def read_knowledge(self, knowledge_ids: List[str]) -> Dict[str, Any]:
        """Read knowledge from memory.

        Args:
            knowledge_ids: List of knowledge IDs

        Returns:
            Dictionary with status, code, and knowledge items
        """
        pass

    @abstractmethod
    async def update_knowledge(self, knowledge_ids: List[str], new_knowledge: List[str]) -> Dict[str, Any]:
        """Update knowledge in memory.

        Args:
            knowledge_ids: List of knowledge IDs
            new_knowledge: List of new knowledge strings

        Returns:
            Dictionary with status, code, and updated knowledge IDs
        """
        pass

    @abstractmethod
    async def delete_knowledge(self, knowledge_ids: List[str]) -> Dict[str, Any]:
        """Delete knowledge from memory.

        Args:
            knowledge_ids: List of knowledge IDs

        Returns:
            Dictionary with status, code, and deleted knowledge IDs
        """
        pass
