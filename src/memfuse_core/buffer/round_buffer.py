"""RoundBuffer implementation for MemFuse Buffer.

The RoundBuffer is a token-based FIFO queue that manages short-term message storage.
It automatically transfers data to HybridBuffer when token limits are exceeded or
when a new session starts.
"""

import asyncio
import uuid
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
    
    async def add(self, messages: MessageList, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Add a MessageList to the buffer.

        Args:
            messages: List of message dictionaries
            session_id: Session ID for the messages

        Returns:
            Dictionary with transfer status and message IDs
        """
        if not messages:
            return {"transfer_triggered": False, "message_ids": []}
        
        async with self._lock:
            # Extract session_id from messages if not provided
            if session_id is None:
                session_id = self._extract_session_id(messages)

            # Pre-generate message IDs for all messages
            message_ids = []
            for message in messages:
                if 'id' not in message or not message['id']:
                    message['id'] = str(uuid.uuid4())
                message_ids.append(message['id'])

            # Check for session change
            if self.current_session_id is not None and session_id != self.current_session_id:
                logger.info(f"RoundBuffer: Session change detected ({self.current_session_id} -> {session_id})")
                await self._transfer_and_clear("session_change")
                self.total_session_changes += 1

            # Update current session
            self.current_session_id = session_id
            
            # Calculate tokens for new messages
            new_tokens = self.token_counter.count_message_tokens(messages)
            
            # Check if new message is too large for RoundBuffer
            if new_tokens > self.max_tokens:
                logger.info(f"RoundBuffer: Single message too large ({new_tokens} > {self.max_tokens}), transferring directly to HybridBuffer")
                # Transfer existing data first if any
                if self.rounds:
                    logger.info(f"RoundBuffer: Transferring existing {len(self.rounds)} rounds first")
                    await self._transfer_and_clear("large_message_existing_data")

                # Transfer the large message directly to HybridBuffer
                logger.info(f"RoundBuffer: Transferring large message directly to HybridBuffer")
                if self.transfer_handler:
                    await self.transfer_handler([messages])
                    self.total_transfers += 1

                return {"transfer_triggered": True, "message_ids": message_ids}

            # Check if adding new messages would exceed token limit
            transfer_triggered = False

            if self.current_tokens + new_tokens > self.max_tokens:
                logger.info(f"RoundBuffer: Token limit would be exceeded ({self.current_tokens + new_tokens} > {self.max_tokens})")
                logger.info(f"RoundBuffer: Transferring existing {len(self.rounds)} rounds before adding new data")
                await self._transfer_and_clear("token_limit")
                transfer_triggered = True

            # Check size limit
            elif len(self.rounds) >= self.max_size:
                logger.info(f"RoundBuffer: Size limit exceeded ({len(self.rounds)} >= {self.max_size})")
                logger.info(f"RoundBuffer: Transferring existing {len(self.rounds)} rounds before adding new data")
                await self._transfer_and_clear("size_limit")
                transfer_triggered = True

            # Add new messages to buffer (now empty if transfer occurred)
            self.rounds.append(messages)
            if transfer_triggered:
                self.current_tokens = new_tokens  # Reset to just the new tokens
            else:
                self.current_tokens += new_tokens  # Add to existing tokens
            self.total_rounds_added += 1

            if transfer_triggered:
                logger.info(f"RoundBuffer: Added new round after transfer with {new_tokens} tokens")
            else:
                logger.debug(f"RoundBuffer: Added round with {new_tokens} tokens, total: {self.current_tokens}")

            return {"transfer_triggered": transfer_triggered, "message_ids": message_ids}
    
    async def _transfer_and_clear(self, reason: str) -> None:
        """Transfer all data to HybridBuffer and clear the buffer.

        Args:
            reason: Reason for the transfer (for logging)
        """
        logger.info(f"RoundBuffer: _transfer_and_clear called with reason '{reason}', rounds count: {len(self.rounds)}")

        if not self.rounds:
            logger.info(f"RoundBuffer: No data to transfer (reason: {reason})")
            return

        logger.info(f"RoundBuffer: Transfer handler exists: {self.transfer_handler is not None}")

        if self.transfer_handler:
            try:
                logger.info(f"RoundBuffer: Transferring {len(self.rounds)} rounds to HybridBuffer (reason: {reason})")
                await self.transfer_handler(self.rounds.copy())
                self.total_transfers += 1
                logger.info(f"RoundBuffer: Transfer completed successfully")
            except Exception as e:
                logger.error(f"RoundBuffer: Transfer failed: {e}")
                # Don't clear on transfer failure to avoid data loss
                return
        else:
            logger.warning("RoundBuffer: No transfer handler set, data will be lost")

        # Clear buffer
        self.rounds.clear()
        self.current_tokens = 0
        logger.info("RoundBuffer: Buffer cleared after transfer")
    
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

    async def get_messages_by_session(
        self,
        session_id: str,
        limit: Optional[int] = None,
        sort_by: str = "created_at",
        order: str = "desc"
    ) -> List[Dict[str, Any]]:
        """Get messages from buffer filtered by session_id.

        Args:
            session_id: Session ID to filter by
            limit: Maximum number of messages to return
            sort_by: Field to sort by (created_at, updated_at, etc.)
            order: Sort order (asc or desc)

        Returns:
            List of message dictionaries for the specified session
        """
        async with self._lock:
            session_messages = []

            for round_messages in self.rounds:
                for message in round_messages:
                    # Check if message belongs to the requested session
                    message_session_id = None

                    # Try to get session_id from metadata first
                    metadata = message.get("metadata", {})
                    message_session_id = metadata.get("session_id")

                    # If not in metadata, try to get from message directly
                    if not message_session_id:
                        message_session_id = message.get("session_id")

                    # If still not found, check if this buffer's current session matches
                    if not message_session_id and self.current_session_id == session_id:
                        message_session_id = session_id

                    # Include message if session matches
                    if message_session_id == session_id:
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
                        session_messages.append(api_message)

            # Sort messages
            reverse_order = (order.lower() == "desc")

            try:
                if sort_by == "created_at":
                    session_messages.sort(
                        key=lambda x: x.get("created_at", ""),
                        reverse=reverse_order
                    )
                elif sort_by == "updated_at":
                    session_messages.sort(
                        key=lambda x: x.get("updated_at", ""),
                        reverse=reverse_order
                    )
                elif sort_by == "timestamp":  # Backward compatibility
                    session_messages.sort(
                        key=lambda x: x.get("created_at", ""),
                        reverse=reverse_order
                    )
                elif sort_by == "id":
                    session_messages.sort(
                        key=lambda x: x.get("id", ""),
                        reverse=reverse_order
                    )
            except Exception as e:
                logger.warning(f"Error sorting messages by {sort_by}: {e}")

            # Apply limit
            if limit is not None and limit > 0:
                session_messages = session_messages[:limit]

            logger.debug(f"RoundBuffer: Found {len(session_messages)} messages for session {session_id}")
            return session_messages

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
