"""
Chunking processor for PgVectorScale store.

This module integrates existing chunking strategies from the RAG module
to provide intelligent chunking for the M0 -> M1 processing pipeline.
"""

import asyncio
from typing import List, Dict, Any, Optional
from loguru import logger

from .base import ChunkData, ChunkStrategy
from .contextual import ContextualChunkStrategy
from .message import MessageChunkStrategy
from .character import CharacterChunkStrategy


class IntegratedChunkingProcessor:
    """
    Integrated chunking processor that uses existing MemFuse chunking strategies.
    
    This processor converts raw messages into intelligent chunks using the
    existing chunking infrastructure, supporting multiple strategies:
    - Token-based chunking (contextual strategy)
    - Message-based chunking
    - Character-based chunking
    """
    
    def __init__(
        self,
        strategy_name: str = "contextual",
        max_words_per_chunk: int = 200,
        enable_contextual: bool = False,  # Disable LLM enhancement for simplicity
        **kwargs
    ):
        """Initialize the chunking processor.
        
        Args:
            strategy_name: Name of the chunking strategy to use
            max_words_per_chunk: Maximum words per chunk
            enable_contextual: Whether to enable contextual enhancement
            **kwargs: Additional strategy-specific parameters
        """
        self.strategy_name = strategy_name
        self.max_words_per_chunk = max_words_per_chunk
        self.enable_contextual = enable_contextual
        self.strategy_kwargs = kwargs
        
        self.chunk_strategy: Optional[ChunkStrategy] = None
        
        logger.info(f"IntegratedChunkingProcessor: Initialized with strategy '{strategy_name}'")
    
    async def initialize(self) -> None:
        """Initialize the chunking strategy."""
        try:
            if self.strategy_name == "contextual":
                self.chunk_strategy = ContextualChunkStrategy(
                    max_words_per_chunk=self.max_words_per_chunk,
                    enable_contextual=self.enable_contextual,
                    **self.strategy_kwargs
                )
            elif self.strategy_name == "message":
                self.chunk_strategy = MessageChunkStrategy()
            elif self.strategy_name == "character":
                max_chars = self.strategy_kwargs.get('max_chunk_length', 1000)
                overlap = self.strategy_kwargs.get('overlap_length', 100)
                self.chunk_strategy = CharacterChunkStrategy(
                    max_chunk_length=max_chars,
                    overlap_length=overlap
                )
            else:
                raise ValueError(f"Unknown chunking strategy: {self.strategy_name}")
            
            logger.info(f"IntegratedChunkingProcessor: Strategy '{self.strategy_name}' initialized")
            
        except Exception as e:
            logger.error(f"IntegratedChunkingProcessor: Strategy initialization failed: {e}")
            raise
    
    async def process_messages_to_chunks(
        self,
        messages: List[Dict[str, Any]],
        session_id: Optional[str] = None
    ) -> List[ChunkData]:
        """Process raw messages into intelligent chunks.
        
        Args:
            messages: List of message dictionaries with 'content', 'role', etc.
            session_id: Optional session ID for contextual processing
            
        Returns:
            List of ChunkData objects
        """
        if not self.chunk_strategy:
            await self.initialize()
        
        try:
            # Convert messages to the format expected by chunking strategies
            message_batch_list = [messages]  # Wrap in a batch
            
            # Apply chunking strategy
            chunks = await self.chunk_strategy.create_chunks(message_batch_list)
            
            # Enhance chunks with additional metadata
            enhanced_chunks = []
            for i, chunk in enumerate(chunks):
                # Add processing metadata
                chunk.metadata.update({
                    'chunking_processor': 'integrated',
                    'original_strategy': self.strategy_name,
                    'session_id': session_id,
                    'chunk_index': i,
                    'processing_timestamp': asyncio.get_event_loop().time()
                })
                
                enhanced_chunks.append(chunk)
            
            logger.info(f"IntegratedChunkingProcessor: Processed {len(messages)} messages -> {len(enhanced_chunks)} chunks")
            
            return enhanced_chunks
            
        except Exception as e:
            logger.error(f"IntegratedChunkingProcessor: Message processing failed: {e}")
            raise
    
    async def create_chunks_from_content(
        self,
        content_list: List[str],
        metadata_list: Optional[List[Dict[str, Any]]] = None
    ) -> List[ChunkData]:
        """Create chunks directly from content strings.
        
        Args:
            content_list: List of content strings
            metadata_list: Optional list of metadata dictionaries
            
        Returns:
            List of ChunkData objects
        """
        if not content_list:
            return []
        
        # Convert content to message format
        messages = []
        for i, content in enumerate(content_list):
            metadata = metadata_list[i] if metadata_list and i < len(metadata_list) else {}
            
            message = {
                'content': content,
                'role': metadata.get('role', 'user'),
                'sequence_number': i + 1,
                **metadata
            }
            messages.append(message)
        
        return await self.process_messages_to_chunks(messages)
    
    def estimate_token_count(self, text: str) -> int:
        """Estimate token count for text (rough approximation)."""
        # Simple approximation: 1 token ≈ 4 characters
        return max(1, len(text) // 4)
    
    def estimate_word_count(self, text: str) -> int:
        """Estimate word count for text."""
        return len(text.split())
    
    async def validate_chunks(self, chunks: List[ChunkData]) -> List[ChunkData]:
        """Validate and filter chunks based on quality criteria.
        
        Args:
            chunks: List of chunks to validate
            
        Returns:
            List of validated chunks
        """
        validated_chunks = []
        
        for chunk in chunks:
            # Basic validation criteria
            if not chunk.content or not chunk.content.strip():
                logger.warning("IntegratedChunkingProcessor: Skipping empty chunk")
                continue
            
            # Check minimum content length
            if len(chunk.content.strip()) < 10:
                logger.warning("IntegratedChunkingProcessor: Skipping too short chunk")
                continue
            
            # Check maximum content length (prevent extremely large chunks)
            if len(chunk.content) > 10000:
                logger.warning("IntegratedChunkingProcessor: Truncating oversized chunk")
                chunk.content = chunk.content[:10000] + "..."
            
            # Add quality metrics to metadata
            chunk.metadata.update({
                'estimated_tokens': self.estimate_token_count(chunk.content),
                'estimated_words': self.estimate_word_count(chunk.content),
                'content_length': len(chunk.content),
                'quality_validated': True
            })
            
            validated_chunks.append(chunk)
        
        logger.info(f"IntegratedChunkingProcessor: Validated {len(validated_chunks)}/{len(chunks)} chunks")
        
        return validated_chunks
    
    async def get_strategy_info(self) -> Dict[str, Any]:
        """Get information about the current chunking strategy."""
        return {
            'strategy_name': self.strategy_name,
            'max_words_per_chunk': self.max_words_per_chunk,
            'enable_contextual': self.enable_contextual,
            'strategy_kwargs': self.strategy_kwargs,
            'initialized': self.chunk_strategy is not None
        }


class TokenBasedChunkingProcessor(IntegratedChunkingProcessor):
    """Specialized processor for token-based chunking (simplified version)."""
    
    def __init__(self, max_tokens_per_chunk: int = 200, **kwargs):
        """Initialize token-based chunking processor.
        
        Args:
            max_tokens_per_chunk: Maximum tokens per chunk
            **kwargs: Additional parameters
        """
        # Convert tokens to approximate words (1 token ≈ 0.75 words)
        max_words = int(max_tokens_per_chunk * 0.75)
        
        super().__init__(
            strategy_name="contextual",
            max_words_per_chunk=max_words,
            enable_contextual=False,  # Disable LLM for performance
            **kwargs
        )
        
        self.max_tokens_per_chunk = max_tokens_per_chunk
        
        logger.info(f"TokenBasedChunkingProcessor: Initialized with max_tokens={max_tokens_per_chunk}")
    
    async def create_token_based_chunks(
        self,
        messages: List[Dict[str, Any]],
        session_id: Optional[str] = None
    ) -> List[ChunkData]:
        """Create chunks using token-based strategy similar to the demo.
        
        This method mimics the chunking logic from the pgvectorscale demo
        while using the existing chunking infrastructure.
        """
        if not messages:
            return []
        
        # Group messages by token count
        current_chunk_messages = []
        current_token_count = 0
        all_chunks = []
        
        for message in messages:
            content = message.get('content', '')
            token_count = self.estimate_token_count(content)
            
            # Check if adding this message would exceed token limit
            if current_token_count + token_count > self.max_tokens_per_chunk and current_chunk_messages:
                # Process current chunk
                chunk_data = await self._create_chunk_from_messages(current_chunk_messages, session_id)
                if chunk_data:
                    all_chunks.append(chunk_data)
                
                # Start new chunk
                current_chunk_messages = [message]
                current_token_count = token_count
            else:
                # Add message to current chunk
                current_chunk_messages.append(message)
                current_token_count += token_count
        
        # Handle remaining messages
        if current_chunk_messages:
            chunk_data = await self._create_chunk_from_messages(current_chunk_messages, session_id)
            if chunk_data:
                all_chunks.append(chunk_data)
        
        logger.info(f"TokenBasedChunkingProcessor: Created {len(all_chunks)} token-based chunks from {len(messages)} messages")
        
        return all_chunks
    
    async def _create_chunk_from_messages(
        self,
        messages: List[Dict[str, Any]],
        session_id: Optional[str] = None
    ) -> Optional[ChunkData]:
        """Create a single chunk from a list of messages."""
        if not messages:
            return None
        
        # Combine message contents
        combined_content = " ".join([msg.get('content', '') for msg in messages])
        
        if not combined_content.strip():
            return None
        
        # Calculate metadata
        total_tokens = sum(self.estimate_token_count(msg.get('content', '')) for msg in messages)
        
        metadata = {
            'chunking_strategy': 'token_based',
            'source_message_count': len(messages),
            'estimated_tokens': total_tokens,
            'session_id': session_id,
            'message_roles': [msg.get('role', 'unknown') for msg in messages]
        }
        
        return ChunkData(
            content=combined_content,
            metadata=metadata
        )
