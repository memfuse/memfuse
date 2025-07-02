"""WriteBuffer implementation for MemFuse Buffer.

The WriteBuffer serves as a unified entry point for RoundBuffer and HybridBuffer,
providing a clean abstraction layer and maintaining the original architecture design.
"""

import asyncio
from typing import Any, Dict, List, Optional, Callable
from loguru import logger

from ..interfaces import MessageList, MessageBatchList
from .round_buffer import RoundBuffer
from .hybrid_buffer import HybridBuffer


class WriteBuffer:
    """Unified write buffer integrating RoundBuffer and HybridBuffer.
    
    This buffer serves as the main entry point for all write operations,
    internally managing the coordination between RoundBuffer and HybridBuffer
    according to the PRD architecture design.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        sqlite_handler: Optional[Callable] = None,
        qdrant_handler: Optional[Callable] = None
    ):
        """Initialize the WriteBuffer with integrated components.
        
        Args:
            config: Configuration dictionary containing buffer settings
            sqlite_handler: Handler for SQLite storage operations
            qdrant_handler: Handler for Qdrant storage operations
        """
        self.config = config
        
        # Extract configuration for sub-components
        round_config = config.get('round_buffer', {})
        hybrid_config = config.get('hybrid_buffer', {})
        
        # Initialize RoundBuffer
        self.round_buffer = RoundBuffer(
            max_tokens=round_config.get('max_tokens', 800),
            max_size=round_config.get('max_size', 5),
            token_model=round_config.get('token_model', 'gpt-4o-mini')
        )
        
        # Initialize HybridBuffer
        self.hybrid_buffer = HybridBuffer(
            max_size=hybrid_config.get('max_size', 5),
            chunk_strategy=hybrid_config.get('chunk_strategy', 'message'),
            embedding_model=hybrid_config.get('embedding_model', 'all-MiniLM-L6-v2')
        )
        
        # Set up component connections
        self.round_buffer.set_transfer_handler(self.hybrid_buffer.add_from_rounds)
        
        # Statistics
        self.total_writes = 0
        self.total_transfers = 0
        
        logger.info("WriteBuffer: Initialized with RoundBuffer and HybridBuffer integration")
    
    async def add(self, messages: MessageList, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Add a single list of messages to the buffer.
        
        Args:
            messages: List of message dictionaries
            session_id: Session ID for context
            
        Returns:
            Dictionary with operation status and metadata
        """
        self.total_writes += 1
        
        # Delegate to RoundBuffer (which may trigger transfer to HybridBuffer)
        result = await self.round_buffer.add(messages, session_id)
        
        if result:  # Transfer was triggered
            self.total_transfers += 1
            logger.debug(f"WriteBuffer: Transfer triggered, total transfers: {self.total_transfers}")
        
        return {
            "status": "success",
            "transfer_triggered": result,
            "total_writes": self.total_writes,
            "total_transfers": self.total_transfers
        }
    
    async def add_batch(self, message_batch_list: MessageBatchList, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Add a batch of message lists to the buffer.
        
        Args:
            message_batch_list: List of lists of messages
            session_id: Session ID for context
            
        Returns:
            Dictionary with batch operation status and metadata
        """
        if not message_batch_list:
            return {"status": "success", "message": "No message lists to add"}
        
        results = []
        total_transfers = 0
        
        for i, messages in enumerate(message_batch_list):
            if not messages:
                continue
                
            result = await self.add(messages, session_id)
            results.append(result)
            
            if result.get("transfer_triggered"):
                total_transfers += 1
        
        return {
            "status": "success",
            "batch_size": len(message_batch_list),
            "processed": len(results),
            "total_transfers": total_transfers,
            "results": results
        }
    
    def get_round_buffer(self) -> RoundBuffer:
        """Get the RoundBuffer instance for Read API operations.
        
        Returns:
            RoundBuffer instance
        """
        return self.round_buffer
    
    def get_hybrid_buffer(self) -> HybridBuffer:
        """Get the HybridBuffer instance for Query API operations.
        
        Returns:
            HybridBuffer instance
        """
        return self.hybrid_buffer
    
    async def flush_all(self) -> Dict[str, Any]:
        """Force flush all buffers to persistent storage.
        
        Returns:
            Dictionary with flush operation status
        """
        try:
            # Force transfer from RoundBuffer to HybridBuffer
            if self.round_buffer.rounds:
                await self.round_buffer._transfer_and_clear("manual_flush")
            
            # Force flush HybridBuffer to storage
            await self.hybrid_buffer.flush_to_storage()
            
            return {"status": "success", "message": "All buffers flushed successfully"}
        except Exception as e:
            logger.error(f"WriteBuffer: Error during flush_all: {e}")
            return {"status": "error", "message": f"Flush failed: {str(e)}"}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics for the WriteBuffer system.
        
        Returns:
            Dictionary with detailed statistics
        """
        round_stats = self.round_buffer.get_stats()
        hybrid_stats = self.hybrid_buffer.get_stats()
        
        return {
            "write_buffer": {
                "total_writes": self.total_writes,
                "total_transfers": self.total_transfers,
                "round_buffer": round_stats,
                "hybrid_buffer": hybrid_stats
            }
        }
    
    def is_empty(self) -> bool:
        """Check if both buffers are empty.
        
        Returns:
            True if both RoundBuffer and HybridBuffer are empty
        """
        return (len(self.round_buffer.rounds) == 0 and 
                len(self.hybrid_buffer.chunks) == 0)
    
    async def clear_all(self) -> Dict[str, Any]:
        """Clear all buffers (for testing purposes).
        
        Returns:
            Dictionary with clear operation status
        """
        try:
            # Clear RoundBuffer
            self.round_buffer.rounds.clear()
            self.round_buffer.current_tokens = 0
            self.round_buffer.current_session_id = None
            
            # Clear HybridBuffer
            self.hybrid_buffer.chunks.clear()
            self.hybrid_buffer.embeddings.clear()
            self.hybrid_buffer.original_rounds.clear()
            
            # Reset statistics
            self.total_writes = 0
            self.total_transfers = 0
            
            logger.info("WriteBuffer: All buffers cleared")
            return {"status": "success", "message": "All buffers cleared"}
        except Exception as e:
            logger.error(f"WriteBuffer: Error during clear_all: {e}")
            return {"status": "error", "message": f"Clear failed: {str(e)}"}
