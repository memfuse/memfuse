"""Interface definitions for MemFuse components.

This module defines the abstract interfaces that various components of the MemFuse system
must implement, ensuring consistent behavior across different implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypeVar, Generic

# Type variable for generic interfaces
T = TypeVar('T')


class MemoryInterface(ABC):
    """Interface for memory operations.
    
    Defines the core operations that any memory implementation must support.
    """

    @abstractmethod
    async def add(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Add messages to memory.
        
        Args:
            messages: List of message dictionaries with role and content
            
        Returns:
            Response with status, code, and message IDs
        """

    @abstractmethod
    async def add_batch(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Add multiple messages to memory in a single operation.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Response with status, code, and message IDs
        """

    @abstractmethod
    async def query(
        self,
        query: str,
        top_k: int = 5,
        store_type: Optional[str] = None,
        include_messages: bool = True,
        include_knowledge: bool = True,
    ) -> Dict[str, Any]:
        """Query memory for relevant information.
        
        Args:
            query: Query string
            top_k: Number of results to return
            store_type: Type of store to query
            include_messages: Whether to include messages
            include_knowledge: Whether to include knowledge
            
        Returns:
            Response with query results
        """

    @abstractmethod
    async def read(self, message_ids: List[str]) -> Dict[str, Any]:
        """Read messages from memory.
        
        Args:
            message_ids: List of message IDs
            
        Returns:
            Response with messages
        """

    @abstractmethod
    async def update(self, message_ids: List[str], new_messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Update messages in memory.
        
        Args:
            message_ids: List of message IDs
            new_messages: List of new message dictionaries
            
        Returns:
            Response with updated message IDs
        """

    @abstractmethod
    async def delete(self, message_ids: List[str]) -> Dict[str, Any]:
        """Delete messages from memory.
        
        Args:
            message_ids: List of message IDs
            
        Returns:
            Response with deleted message IDs
        """

    @abstractmethod
    async def clear(self) -> Dict[str, Any]:
        """Clear all messages from memory.
        
        Returns:
            Response with operation status
        """

    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics.
        
        Returns:
            Response with memory statistics
        """

    @abstractmethod
    async def add_knowledge(self, knowledge_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Add knowledge items to memory.
        
        Args:
            knowledge_items: List of knowledge item dictionaries
            
        Returns:
            Response with knowledge item IDs
        """

    @abstractmethod
    async def get_knowledge(self, knowledge_ids: List[str]) -> Dict[str, Any]:
        """Get knowledge items from memory.
        
        Args:
            knowledge_ids: List of knowledge item IDs
            
        Returns:
            Response with knowledge items
        """

    @abstractmethod
    async def delete_knowledge(self, knowledge_ids: List[str]) -> Dict[str, Any]:
        """Delete knowledge items from memory.
        
        Args:
            knowledge_ids: List of knowledge item IDs
            
        Returns:
            Response with deleted knowledge item IDs
        """


class BufferInterface(Generic[T], ABC):
    """Interface for buffer operations.
    
    Defines operations for any buffer implementation.
    """

    @abstractmethod
    def add(self, item: T) -> bool:
        """Add an item to the buffer.
        
        Args:
            item: The item to add
            
        Returns:
            Whether a batch write was triggered
        """

    @abstractmethod
    def query(self, query: str, top_k: int = 5) -> List[T]:
        """Query the buffer for relevant items.
        
        Args:
            query: Query string
            top_k: Number of results to return
            
        Returns:
            List of matching items
        """

    @abstractmethod
    def shutdown(self) -> None:
        """Shut down the buffer system.
        
        Performs cleanup operations when buffer is no longer needed.
        """
