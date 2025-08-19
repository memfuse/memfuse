"""M1 data processor for episodic memory chunks."""

import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
from loguru import logger

from ..rag.chunk.base import ChunkData
from ..rag.chunk import MessageChunkStrategy
from ..rag.encode.MiniLM import MiniLMEncoder


class M1Processor:
    """Processor for M1 episodic memory chunks."""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embedding_model = embedding_model
        self.encoder = None
        self.chunking_strategy = None
        self.processed_count = 0
        
    async def initialize(self):
        """Initialize the M1 processor."""
        try:
            # Initialize embedding encoder (MiniLMEncoder doesn't have async initialize)
            self.encoder = MiniLMEncoder(model_name=self.embedding_model)

            # Initialize chunking strategy
            self.chunking_strategy = MessageChunkStrategy()

            logger.info(f"M1Processor: Initialized with model {self.embedding_model}")

        except Exception as e:
            logger.error(f"M1Processor: Initialization failed: {e}")
            raise
    
    async def process_chunks(
        self,
        chunks: List[ChunkData],
        m0_message_ids: List[str],
        session_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Process chunks for M1 storage with embeddings.
        
        Args:
            chunks: List of ChunkData objects
            m0_message_ids: List of M0 message IDs for lineage tracking
            session_id: Session identifier
            metadata: Optional metadata
            
        Returns:
            List of processed M1 records ready for database insertion
        """
        if not self.encoder:
            await self.initialize()
        
        processed_chunks = []
        
        try:
            # Step 1: Apply intelligent chunking strategy
            optimized_chunks = await self._apply_chunking_strategy(chunks, session_id)
            
            # Step 2: Generate embeddings for all chunks
            embeddings = await self._generate_embeddings(optimized_chunks)
            
            # Step 3: Create M1 records
            for i, (chunk, embedding) in enumerate(zip(optimized_chunks, embeddings)):
                chunk_id = str(uuid.uuid4())
                
                # Calculate token count
                token_count = self._estimate_token_count(chunk.content)
                
                # Determine chunking strategy
                chunking_strategy = chunk.metadata.get('chunking_strategy', 'token_based')
                
                # Create M1 record
                m1_record = {
                    'chunk_id': chunk_id,
                    'content': chunk.content,
                    'chunking_strategy': chunking_strategy,
                    'token_count': token_count,
                    'embedding': embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding),
                    'needs_embedding': False,  # Embedding is generated synchronously
                    'm0_message_ids': m0_message_ids,
                    'session_id': session_id,
                    'created_at': datetime.now(),
                    'updated_at': datetime.now(),
                    'embedding_generated_at': datetime.now(),
                    'embedding_model': self.embedding_model,
                    'chunk_quality_score': self._calculate_quality_score(chunk)
                }
                
                # Add metadata if provided
                if metadata:
                    m1_record['metadata'] = metadata
                
                processed_chunks.append(m1_record)
            
            self.processed_count += len(processed_chunks)
            logger.debug(f"M1Processor: Processed {len(processed_chunks)} chunks with embeddings")
            
            return processed_chunks
            
        except Exception as e:
            logger.error(f"M1Processor: Processing failed: {e}")
            raise
    
    async def _apply_chunking_strategy(
        self,
        chunks: List[ChunkData],
        session_id: str
    ) -> List[ChunkData]:
        """Apply intelligent chunking strategy to optimize chunks.

        Args:
            chunks: Input chunks
            session_id: Session identifier

        Returns:
            Optimized chunks
        """
        if not self.chunking_strategy:
            # Simple pass-through if no strategy available
            return chunks

        try:
            # Convert chunks to message batch list format expected by create_chunks
            message_batch_list = []
            for chunk in chunks:
                message = {
                    'content': chunk.content,
                    'role': chunk.metadata.get('role', 'user'),
                    'metadata': chunk.metadata
                }
                # Each chunk becomes a single-message batch
                message_batch_list.append([message])

            # Apply chunking strategy using the correct method
            optimized_chunks = await self.chunking_strategy.create_chunks(message_batch_list)

            return optimized_chunks

        except Exception as e:
            logger.warning(f"M1Processor: Chunking strategy failed, using original chunks: {e}")
            return chunks
    
    async def _generate_embeddings(self, chunks: List[ChunkData]) -> List[Any]:
        """Generate embeddings for chunks.
        
        Args:
            chunks: List of chunks
            
        Returns:
            List of embedding vectors
        """
        if not chunks:
            return []
        
        try:
            # Extract content for embedding
            contents = [chunk.content for chunk in chunks]
            
            # Generate embeddings using the correct method
            embeddings = await self.encoder.encode_texts(contents)
            
            logger.debug(f"M1Processor: Generated embeddings for {len(chunks)} chunks")
            return embeddings
            
        except Exception as e:
            logger.error(f"M1Processor: Embedding generation failed: {e}")
            raise
    
    def _estimate_token_count(self, text: str) -> int:
        """Estimate token count for text.
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        if not text:
            return 0
        
        # Simple approximation: ~4 characters per token for English text
        return max(1, len(text) // 4)
    
    def _calculate_quality_score(self, chunk: ChunkData) -> float:
        """Calculate quality score for a chunk.
        
        Args:
            chunk: Chunk to evaluate
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        if not chunk.content:
            return 0.0
        
        # Simple quality metrics
        content_length = len(chunk.content)
        
        # Prefer chunks with reasonable length (not too short, not too long)
        if content_length < 10:
            length_score = 0.3
        elif content_length > 2000:
            length_score = 0.7
        else:
            length_score = 1.0
        
        # Check for meaningful content (not just whitespace)
        meaningful_chars = len(chunk.content.strip())
        if meaningful_chars == 0:
            content_score = 0.0
        else:
            content_score = min(1.0, meaningful_chars / content_length)
        
        # Combine scores
        quality_score = (length_score + content_score) / 2.0
        
        return round(quality_score, 3)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics."""
        return {
            'processed_count': self.processed_count,
            'processor_type': 'M1Processor',
            'embedding_model': self.embedding_model,
            'encoder_initialized': self.encoder is not None
        }
