"""Contextual chunking strategy implementation."""

from typing import Any, Dict, List
from .base import ChunkData, ChunkStrategy


class ContextualChunkStrategy(ChunkStrategy):
    """Contextual chunking strategy that intelligently splits content based on context.

    This strategy combines multiple MessageLists and splits content based on semantic
    boundaries and length limits. It maintains conversational context while ensuring
    optimal chunk sizes for retrieval.
    """
    
    def __init__(self, max_chunk_length: int = 1000):
        """Initialize the contextual chunking strategy.

        Args:
            max_chunk_length: Maximum character length per chunk
        """
        self.max_chunk_length = max_chunk_length
    
    async def create_chunks(self, message_batch_list: List[List[Dict[str, Any]]]) -> List[ChunkData]:
        """Create chunks from message batch list with length-based splitting.
        
        Combines MessageLists and splits based on content length.
        Long messages are split into multiple chunks if needed.
        
        Args:
            message_batch_list: List of lists of messages (MessageBatchList)
            
        Returns:
            List of ChunkData objects with length constraints
        """
        chunks = []
        current_content_parts = []
        current_length = 0
        
        for batch_index, message_list in enumerate(message_batch_list):
            if not message_list:
                continue
                
            for message_index, message in enumerate(message_list):
                message_text = self._extract_message_content(message)
                
                # Check if adding this message would exceed the limit
                if current_length + len(message_text) > self.max_chunk_length and current_content_parts:
                    # Create chunk with current content
                    chunk = self._create_chunk_from_parts(current_content_parts, current_length)
                    chunks.append(chunk)
                    current_content_parts = []
                    current_length = 0
                
                # Handle very long messages that exceed max_chunk_length
                if len(message_text) > self.max_chunk_length:
                    # Split the long message into multiple chunks
                    split_chunks = self._split_long_message(
                        message_text, 
                        batch_index, 
                        message_index
                    )
                    chunks.extend(split_chunks)
                else:
                    # Add message to current chunk
                    current_content_parts.append(message_text)
                    current_length += len(message_text) + 1  # +1 for newline
        
        # Handle remaining content
        if current_content_parts:
            chunk = self._create_chunk_from_parts(current_content_parts, current_length)
            chunks.append(chunk)
        
        return chunks
    
    def _create_chunk_from_parts(self, content_parts: List[str], content_length: int) -> ChunkData:
        """Create a ChunkData from content parts.
        
        Args:
            content_parts: List of content strings to combine
            content_length: Total length of content
            
        Returns:
            ChunkData object
        """
        chunk_content = "\n".join(content_parts)
        metadata = {
            "strategy": "contextual",
            "content_length": content_length,
            "part_count": len(content_parts)
        }
        
        return ChunkData(
            content=chunk_content,
            metadata=metadata
        )
    
    def _split_long_message(self, message_text: str, batch_index: int, message_index: int) -> List[ChunkData]:
        """Split a long message into multiple chunks.
        
        Args:
            message_text: The long message text to split
            batch_index: Index of the batch containing this message
            message_index: Index of the message within its batch
            
        Returns:
            List of ChunkData objects from the split message
        """
        chunks = []
        start = 0
        part_index = 0
        
        while start < len(message_text):
            end = start + self.max_chunk_length
            chunk_content = message_text[start:end]
            
            metadata = {
                "strategy": "contextual_split",
                "is_split": True,
                "content_length": len(chunk_content),
                "batch_index": batch_index,
                "message_index": message_index,
                "part_index": part_index,
                "total_parts": (len(message_text) + self.max_chunk_length - 1) // self.max_chunk_length
            }
            
            chunk = ChunkData(
                content=chunk_content,
                metadata=metadata
            )
            chunks.append(chunk)
            
            start = end
            part_index += 1
        
        return chunks
