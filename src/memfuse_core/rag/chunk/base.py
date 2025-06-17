"""Base classes and data structures for chunking functionality."""

import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from dataclasses import dataclass


@dataclass
class ChunkData:
    """Chunk data structure containing text content.
    
    This is the fundamental unit for retrieval and reranking.
    Contains only string content to support splitting of long messages.
    """
    content: str  # The text content of the chunk
    chunk_id: str  # Unique identifier for the chunk
    metadata: Dict[str, Any]  # Additional metadata (source info, timestamps, etc.)
    
    def __init__(self, content: str, chunk_id: str = None, metadata: Dict[str, Any] = None):
        """Initialize ChunkData.
        
        Args:
            content: The text content of the chunk
            chunk_id: Unique identifier, auto-generated if not provided
            metadata: Additional metadata dictionary
        """
        self.content = content
        self.chunk_id = chunk_id or self._generate_chunk_id()
        self.metadata = metadata or {}
    
    def _generate_chunk_id(self) -> str:
        """Generate a unique chunk ID as a proper UUID."""
        return str(uuid.uuid4())
    
    def __len__(self) -> int:
        """Return the length of the content."""
        return len(self.content)
    
    def __str__(self) -> str:
        """String representation of the chunk."""
        return f"ChunkData(id={self.chunk_id}, length={len(self.content)})"


class ChunkStrategy(ABC):
    """Abstract base class for chunking strategies.
    
    Defines the interface for converting MessageBatchList to List[ChunkData].
    """
    
    @abstractmethod
    async def create_chunks(self, message_batch_list: List[List[Dict[str, Any]]]) -> List[ChunkData]:
        """Create chunks from a batch of message lists.
        
        Args:
            message_batch_list: List of lists of messages (MessageBatchList)
            
        Returns:
            List of ChunkData objects
        """
        pass
    
    def _extract_message_content(self, message: Dict[str, Any]) -> str:
        """Extract content from a message with role prefix.
        
        Args:
            message: Message dictionary with role and content
            
        Returns:
            Formatted message content with role prefix
        """
        role = message.get("role", "unknown")
        content = message.get("content", "")
        return f"[{role}]: {content}"
    
    def _combine_messages(self, messages: List[Dict[str, Any]]) -> str:
        """Combine multiple messages into a single content string.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Combined content string
        """
        content_parts = []
        for message in messages:
            content_parts.append(self._extract_message_content(message))
        return "\n".join(content_parts)
