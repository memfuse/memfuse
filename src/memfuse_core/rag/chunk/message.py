"""Message-based chunking strategy implementation."""

from typing import Any, Dict, List
from .base import ChunkData, ChunkStrategy


class MessageChunkStrategy(ChunkStrategy):
    """Message-based chunking strategy that creates one chunk per MessageList.

    This strategy converts each MessageList (List of Messages) into a single ChunkData.
    All messages in a MessageList are combined into one chunk content.
    This is the most straightforward chunking approach.
    """
    
    def __init__(self):
        """Initialize the message chunking strategy."""
        pass
    
    async def create_chunks(self, message_batch_list: List[List[Dict[str, Any]]]) -> List[ChunkData]:
        """Create chunks from message batch list.
        
        Each MessageList becomes one ChunkData with all messages combined.
        
        Args:
            message_batch_list: List of lists of messages (MessageBatchList)
            
        Returns:
            List of ChunkData objects, one per MessageList
        """
        chunks = []
        
        for i, message_list in enumerate(message_batch_list):
            if not message_list:
                continue
                
            # Combine all messages in the list into one chunk content
            chunk_content = self._combine_messages(message_list)
            
            # Create metadata for the chunk
            metadata = {
                "strategy": "message",
                "message_count": len(message_list),
                "source": "message_list",
                "batch_index": i,
                "roles": [msg.get("role", "unknown") for msg in message_list]
            }
            
            # Create the chunk
            chunk = ChunkData(
                content=chunk_content,
                metadata=metadata
            )
            
            chunks.append(chunk)
        
        return chunks
