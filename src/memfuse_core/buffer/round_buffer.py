"""RoundBuffer implementation for MemFuse Buffer.

The RoundBuffer is a token-based FIFO queue that manages short-term message storage.
It automatically transfers data to HybridBuffer when token limits are exceeded or
when a new session starts.
"""

import asyncio
from typing import Any, Dict, List, Optional, Callable
from loguru import logger

from ..interfaces import MessageList
from ..utils.token_counter import get_token_counter


class RoundBuffer:
    """Token-based FIFO buffer for short-term message storage.
    
    This buffer maintains messages until either:
    1. Token count exceeds max_tokens (default: 800)
    2. A new session starts
    
    When triggered, it transfers all data to HybridBuffer and clears itself.
    """
    
    def __init__(
        self,
        max_tokens: int = 800,
        max_size: int = 5,
        token_model: str = "gpt-4o-mini"
    ):
        """Initialize the RoundBuffer.
        
        Args:
            max_tokens: Maximum token count before transfer (default: 800)
            max_size: Maximum number of rounds before forced transfer
            token_model: Model name for token counting
        """
        self.max_tokens = max_tokens
        self.max_size = max_size
        self.token_model = token_model
        
        # Buffer state
        self.rounds: List[MessageList] = []
        self.current_tokens = 0
        self.current_session_id: Optional[str] = None
        
        # Token counter
        self.token_counter = get_token_counter(token_model)
        
        # Transfer handler (set by HybridBuffer)
        self.transfer_handler: Optional[Callable] = None
        
        # Async lock for thread safety
        self._lock = asyncio.Lock()
        
        # Statistics
        self.total_rounds_added = 0
        self.total_transfers = 0
        self.total_session_changes = 0
        
        logger.info(f"RoundBuffer: Initialized with max_tokens={max_tokens}, max_size={max_size}")
    
    def set_transfer_handler(self, handler: Callable) -> None:
        """Set the transfer handler for moving data to HybridBuffer.
        
        Args:
            handler: Async function to handle data transfer
        """
        self.transfer_handler = handler
        logger.debug("RoundBuffer: Transfer handler set")
    
    async def add(self, messages: MessageList, session_id: Optional[str] = None) -> bool:
        """Add a MessageList to the buffer.
        
        Args:
            messages: List of message dictionaries
            session_id: Session ID for the messages
            
        Returns:
            True if transfer was triggered, False otherwise
        """
        if not messages:
            return False
        
        async with self._lock:
            # Extract session_id from messages if not provided
            if session_id is None:
                session_id = self._extract_session_id(messages)
            
            # Check for session change
            if self.current_session_id is not None and session_id != self.current_session_id:
                logger.info(f"RoundBuffer: Session change detected ({self.current_session_id} -> {session_id})")
                await self._transfer_and_clear("session_change")
                self.total_session_changes += 1
            
            # Update current session
            self.current_session_id = session_id
            
            # Calculate tokens for new messages
            new_tokens = self.token_counter.count_message_tokens(messages)
            
            # Check token limit
            if self.current_tokens + new_tokens > self.max_tokens:
                logger.info(f"RoundBuffer: Token limit exceeded ({self.current_tokens + new_tokens} > {self.max_tokens})")
                await self._transfer_and_clear("token_limit")
            
            # Check size limit
            elif len(self.rounds) >= self.max_size:
                logger.info(f"RoundBuffer: Size limit exceeded ({len(self.rounds)} >= {self.max_size})")
                await self._transfer_and_clear("size_limit")
            
            # Add messages to buffer
            self.rounds.append(messages)
            self.current_tokens += new_tokens
            self.total_rounds_added += 1
            
            logger.debug(f"RoundBuffer: Added round with {new_tokens} tokens, total: {self.current_tokens}")
            return False
    
    async def _transfer_and_clear(self, reason: str) -> None:
        """Transfer all data to HybridBuffer and clear the buffer.
        
        Args:
            reason: Reason for the transfer (for logging)
        """
        if not self.rounds:
            logger.debug(f"RoundBuffer: No data to transfer (reason: {reason})")
            return
        
        if self.transfer_handler:
            try:
                logger.info(f"RoundBuffer: Transferring {len(self.rounds)} rounds to HybridBuffer (reason: {reason})")
                await self.transfer_handler(self.rounds.copy())
                self.total_transfers += 1
            except Exception as e:
                logger.error(f"RoundBuffer: Transfer failed: {e}")
                # Don't clear on transfer failure to avoid data loss
                return
        else:
            logger.warning("RoundBuffer: No transfer handler set, data will be lost")
        
        # Clear buffer
        self.rounds.clear()
        self.current_tokens = 0
        logger.debug("RoundBuffer: Buffer cleared after transfer")
    
    def _extract_session_id(self, messages: MessageList) -> Optional[str]:
        """Extract session_id from messages metadata.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Session ID if found, None otherwise
        """
        for message in messages:
            metadata = message.get("metadata", {})
            session_id = metadata.get("session_id")
            if session_id:
                return session_id
        return None
    
    async def get_all_messages_for_read_api(
        self,
        limit: Optional[int] = None,
        sort_by: str = "timestamp",
        order: str = "desc"
    ) -> List[Dict[str, Any]]:
        """Get all messages in buffer for Read API.
        
        Args:
            limit: Maximum number of messages to return
            sort_by: Field to sort by ('timestamp' or 'id')
            order: Sort order ('asc' or 'desc')
            
        Returns:
            List of message dictionaries formatted for API response
        """
        async with self._lock:
            all_messages = []
            
            for round_messages in self.rounds:
                for message in round_messages:
                    # Convert to API format
                    api_message = {
                        "id": message.get("id", ""),
                        "role": message.get("role", "user"),
                        "content": message.get("content", ""),
                        "created_at": message.get("created_at", ""),
                        "updated_at": message.get("updated_at", ""),
                        "metadata": message.get("metadata", {}).copy()
                    }
                    # Add buffer source metadata
                    api_message["metadata"]["source"] = "round_buffer"
                    all_messages.append(api_message)
            
            # Sort messages
            if sort_by == "timestamp":
                all_messages.sort(
                    key=lambda x: x.get("created_at", ""),
                    reverse=(order == "desc")
                )
            elif sort_by == "id":
                all_messages.sort(
                    key=lambda x: x.get("id", ""),
                    reverse=(order == "desc")
                )
            
            # Apply limit
            if limit is not None and limit > 0:
                all_messages = all_messages[:limit]
            
            return all_messages
    
    async def get_buffer_info(self) -> Dict[str, Any]:
        """Get buffer information for Query API metadata.
        
        Returns:
            Dictionary with buffer status information
        """
        async with self._lock:
            return {
                "messages_available": len(self.rounds) > 0,
                "messages_count": sum(len(round_msgs) for round_msgs in self.rounds),
                "rounds_count": len(self.rounds),
                "current_tokens": self.current_tokens,
                "max_tokens": self.max_tokens,
                "current_session_id": self.current_session_id,
                "buffer_type": "round_buffer"
            }
    
    async def force_transfer(self) -> bool:
        """Force transfer of all data to HybridBuffer.
        
        Returns:
            True if transfer was successful, False otherwise
        """
        async with self._lock:
            if not self.rounds:
                return True
            
            try:
                await self._transfer_and_clear("force_transfer")
                return True
            except Exception as e:
                logger.error(f"RoundBuffer: Force transfer failed: {e}")
                return False
    
    async def clear(self) -> None:
        """Clear all data from buffer without transfer.
        
        Warning: This will cause data loss if transfer handler is not called.
        """
        async with self._lock:
            self.rounds.clear()
            self.current_tokens = 0
            self.current_session_id = None
            logger.warning("RoundBuffer: Buffer cleared without transfer")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics.
        
        Returns:
            Dictionary with buffer statistics
        """
        return {
            "rounds_count": len(self.rounds),
            "current_tokens": self.current_tokens,
            "max_tokens": self.max_tokens,
            "max_size": self.max_size,
            "current_session_id": self.current_session_id,
            "total_rounds_added": self.total_rounds_added,
            "total_transfers": self.total_transfers,
            "total_session_changes": self.total_session_changes,
            "has_transfer_handler": self.transfer_handler is not None,
            "token_model": self.token_model
        }
