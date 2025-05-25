"""Store interface for MemFuse stores."""

from abc import ABC, abstractmethod
from typing import List, Optional, Any
from ..models.core import Item, Query, QueryResult


class StoreInterface(ABC):
    """Base interface for all MemFuse stores."""
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the store.
        
        Returns:
            True if initialization was successful, False otherwise
        """
        pass
    
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
    async def query(self, query: Query) -> QueryResult:
        """Query the store.
        
        Args:
            query: Query to execute
            
        Returns:
            Query result
        """
        pass
    
    @abstractmethod
    async def delete(self, item_id: str) -> bool:
        """Delete an item from the store.
        
        Args:
            item_id: ID of the item to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        pass
