"""Character-based chunking strategy implementation."""

from typing import Any, Dict, List
from .base import ChunkData, ChunkStrategy


class CharacterChunkStrategy(ChunkStrategy):
    """Character-based chunking strategy that splits content by character count.
    
    This strategy splits content into chunks based on a fixed character limit,
    regardless of message boundaries. It's useful for handling very long content
    that needs to be split into uniform-sized chunks.
    """
    
    def __init__(self, max_chunk_length: int = 1000, overlap_length: int = 100):
        """Initialize the character chunking strategy.
        
        Args:
            max_chunk_length: Maximum character length per chunk
            overlap_length: Number of characters to overlap between chunks
        """
        self.max_chunk_length = max_chunk_length
        self.overlap_length = overlap_length
    
    async def create_chunks(self, message_batch_list: List[List[Dict[str, Any]]]) -> List[ChunkData]:
        """Create chunks from message batch list with character-based splitting.
        
        Combines all messages into a single text and splits by character count.
        
        Args:
            message_batch_list: List of lists of messages (MessageBatchList)
            
        Returns:
            List of ChunkData objects split by character count
        """
        chunks = []
        
        # Combine all messages from all MessageLists into one continuous text
        all_content = []
        message_metadata = []
        
        for batch_index, message_list in enumerate(message_batch_list):
            for message_index, message in enumerate(message_list):
                content = self._extract_message_content(message)
                all_content.append(content)
                message_metadata.append({
                    "batch_index": batch_index,
                    "message_index": message_index,
                    "role": message.get("role", "unknown"),
                    "original_metadata": message.get("metadata", {})
                })
        
        # Join all content with newlines
        full_text = "\n".join(all_content)
        
        # Split by character count with overlap
        start = 0
        chunk_index = 0
        
        while start < len(full_text):
            end = start + self.max_chunk_length
            chunk_content = full_text[start:end]
            
            # Try to break at word boundary if possible
            if end < len(full_text) and not full_text[end].isspace():
                # Look for the last space within the chunk
                last_space = chunk_content.rfind(' ')
                if last_space > start + (self.max_chunk_length * 0.8):  # Don't break too early
                    end = start + last_space
                    chunk_content = full_text[start:end]
            
            # Create metadata for the chunk
            metadata = {
                "strategy": "character",
                "chunk_index": chunk_index,
                "start_position": start,
                "end_position": end,
                "content_length": len(chunk_content),
                "has_overlap": chunk_index > 0,
                "overlap_length": self.overlap_length if chunk_index > 0 else 0,
                "source": "character_split",
                "total_messages": len(all_content),
                "message_metadata": message_metadata
            }
            
            # Create the chunk
            chunk = ChunkData(
                content=chunk_content,
                metadata=metadata
            )
            chunks.append(chunk)
            
            # Move start position with overlap
            if end >= len(full_text):
                break
            start = max(start + 1, end - self.overlap_length)
            chunk_index += 1
        
        return chunks
    
    def _find_optimal_break_point(self, text: str, max_length: int) -> int:
        """Find the optimal break point for splitting text.
        
        Tries to break at sentence boundaries, then word boundaries.
        
        Args:
            text: Text to find break point in
            max_length: Maximum length to consider
            
        Returns:
            Optimal break point position
        """
        if len(text) <= max_length:
            return len(text)
        
        # Try to break at sentence boundary (. ! ?)
        for i in range(max_length - 1, max_length // 2, -1):
            if text[i] in '.!?':
                return i + 1
        
        # Try to break at word boundary
        for i in range(max_length - 1, max_length // 2, -1):
            if text[i].isspace():
                return i
        
        # If no good break point found, use max_length
        return max_length
