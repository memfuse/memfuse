"""M0 data processor for raw message handling."""

import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
from loguru import logger

from ..rag.chunk.base import ChunkData


class M0Processor:
    """Processor for M0 raw message data."""
    
    def __init__(self):
        self.processed_count = 0
        
    async def process_messages(
        self,
        messages: List[Dict[str, Any]],
        session_id: str,
        user_id: Optional[str] = None,
        round_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Process raw messages for M0 storage.

        Args:
            messages: List of message dictionaries with 'content' and 'role'
            session_id: Session identifier
            user_id: User identifier (optional, will generate if not provided)
            round_id: Round identifier (optional)
            metadata: Optional metadata

        Returns:
            List of processed M0 records ready for database insertion
        """
        processed_messages = []

        # Generate user_id if not provided
        if user_id is None:
            user_id = str(uuid.uuid4())

        for i, message in enumerate(messages):
            # Generate unique message ID
            message_id = str(uuid.uuid4())

            # Calculate token count (simple approximation)
            token_count = self._estimate_token_count(message.get('content', ''))

            # Create M0 record
            m0_record = {
                'message_id': message_id,
                'content': message['content'],
                'role': message['role'],
                'user_id': user_id,
                'session_id': session_id,
                'conversation_id': session_id,  # Use session_id as conversation_id for compatibility
                'round_id': round_id,
                'sequence_number': i + 1,
                'token_count': token_count,
                'created_at': datetime.now(),
                'updated_at': datetime.now(),
                'processing_status': 'pending',
                'chunk_assignments': []
            }
            
            # Add metadata if provided
            if metadata:
                m0_record['metadata'] = metadata
            
            processed_messages.append(m0_record)
            
        self.processed_count += len(processed_messages)
        logger.debug(f"M0Processor: Processed {len(processed_messages)} messages")
        
        return processed_messages
    
    def convert_to_chunks(
        self,
        messages: List[Dict[str, Any]],
        session_id: str
    ) -> List[ChunkData]:
        """Convert messages to ChunkData objects for M1 processing.
        
        Args:
            messages: List of message dictionaries
            session_id: Session identifier
            
        Returns:
            List of ChunkData objects
        """
        chunks = []
        
        for message in messages:
            chunk = ChunkData(
                content=message['content'],
                metadata={
                    'role': message['role'],
                    'session_id': session_id,
                    'message_id': message.get('message_id', str(uuid.uuid4())),
                    'token_count': message.get('token_count', 0),
                    'source': 'M0_raw'
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _estimate_token_count(self, text: str) -> int:
        """Estimate token count for text.
        
        Simple approximation: ~4 characters per token for English text.
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        if not text:
            return 0
        
        # Simple approximation
        return max(1, len(text) // 4)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics."""
        return {
            'processed_count': self.processed_count,
            'processor_type': 'M0Processor'
        }
