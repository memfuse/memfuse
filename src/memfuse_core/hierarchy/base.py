"""Base classes for the hierarchical memory system."""

from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional


class MemoryLayer(ABC):
    """Base class for memory layers in the hierarchical memory system."""
    
    @abstractmethod
    async def add(self, item: Any) -> str:
        """Add an item to the memory layer.
        
        Args:
            item: Item to add
            
        Returns:
            ID of the added item
        """
        pass
    
    @abstractmethod
    async def get(self, item_id: str) -> Optional[Any]:
        """Get an item from the memory layer.
        
        Args:
            item_id: ID of the item to get
            
        Returns:
            Item if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def search(self, query: str, top_k: int = 5) -> List[Any]:
        """Search for items in the memory layer.
        
        Args:
            query: Query string
            top_k: Number of results to return
            
        Returns:
            List of matching items
        """
        pass
    
    @abstractmethod
    async def update(self, item_id: str, item: Any) -> bool:
        """Update an item in the memory layer.
        
        Args:
            item_id: ID of the item to update
            item: Updated item
            
        Returns:
            True if the item was updated, False otherwise
        """
        pass
    
    @abstractmethod
    async def delete(self, item_id: str) -> bool:
        """Delete an item from the memory layer.
        
        Args:
            item_id: ID of the item to delete
            
        Returns:
            True if the item was deleted, False otherwise
        """
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """Clear all items from the memory layer.
        
        Returns:
            True if the memory layer was cleared, False otherwise
        """
        pass
    
    @abstractmethod
    async def count(self) -> int:
        """Count the number of items in the memory layer.
        
        Returns:
            Number of items
        """
        pass


class MemoryItem:
    """Base class for items in the hierarchical memory system."""
    
    def __init__(
        self,
        id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        source_ids: Optional[List[str]] = None,
    ):
        """Initialize a memory item.
        
        Args:
            id: Item ID
            content: Item content
            metadata: Item metadata
            source_ids: IDs of source items
        """
        self.id = id
        self.content = content
        self.metadata = metadata or {}
        self.source_ids = source_ids or []
        self.score = 0.0  # Default score for search results


class Fact(MemoryItem):
    """Fact extracted from raw data."""
    
    def __init__(
        self,
        id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        source_ids: Optional[List[str]] = None,
        confidence: float = 1.0,
    ):
        """Initialize a fact.
        
        Args:
            id: Fact ID
            content: Fact content
            metadata: Fact metadata
            source_ids: IDs of source items
            confidence: Confidence score for the fact
        """
        super().__init__(id, content, metadata, source_ids)
        self.confidence = confidence
        self.metadata["type"] = "fact"
        self.metadata["confidence"] = confidence


class Entity(MemoryItem):
    """Entity in the knowledge graph."""
    
    def __init__(
        self,
        id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        source_ids: Optional[List[str]] = None,
        entity_type: str = "generic",
    ):
        """Initialize an entity.
        
        Args:
            id: Entity ID
            content: Entity content
            metadata: Entity metadata
            source_ids: IDs of source items
            entity_type: Type of entity
        """
        super().__init__(id, content, metadata, source_ids)
        self.entity_type = entity_type
        self.metadata["type"] = "entity"
        self.metadata["entity_type"] = entity_type


class Relationship(MemoryItem):
    """Relationship between entities in the knowledge graph."""
    
    def __init__(
        self,
        id: str,
        content: str,
        source_id: str,
        target_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        source_ids: Optional[List[str]] = None,
        relationship_type: str = "generic",
    ):
        """Initialize a relationship.
        
        Args:
            id: Relationship ID
            content: Relationship content
            source_id: ID of the source entity
            target_id: ID of the target entity
            metadata: Relationship metadata
            source_ids: IDs of source items
            relationship_type: Type of relationship
        """
        super().__init__(id, content, metadata, source_ids)
        self.source_id = source_id
        self.target_id = target_id
        self.relationship_type = relationship_type
        self.metadata["type"] = "relationship"
        self.metadata["relationship_type"] = relationship_type
        self.metadata["source_id"] = source_id
        self.metadata["target_id"] = target_id
