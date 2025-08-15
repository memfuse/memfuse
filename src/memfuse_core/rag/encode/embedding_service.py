"""Unified embedding service for MemFuse.

This service provides a unified interface for embedding generation,
building on the existing MiniLM implementation.
"""

import asyncio
from typing import List, Optional, Any, Dict
from loguru import logger

from .MiniLM import MiniLMEncoder


class EmbeddingService:
    """Unified embedding service."""
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_size: int = 10000,
        batch_size: int = 32
    ):
        """Initialize the embedding service.
        
        Args:
            model_name: Name of the embedding model
            cache_size: Size of the embedding cache
            batch_size: Batch size for embedding generation
        """
        self.model_name = model_name
        self.cache_size = cache_size
        self.batch_size = batch_size
        self.encoder = None
        self.initialized = False
        
    async def initialize(self) -> bool:
        """Initialize the embedding service.

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            self.encoder = MiniLMEncoder(
                model_name=self.model_name,
                cache_size=self.cache_size
            )

            # MiniLMEncoder doesn't have async initialize, it initializes in __init__
            self.initialized = True
            logger.info(f"EmbeddingService: Initialized with model {self.model_name}")

            return True

        except Exception as e:
            logger.error(f"EmbeddingService: Initialization failed: {e}")
            return False
    
    async def encode(self, texts: List[str]) -> List[Any]:
        """Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        if not self.initialized:
            success = await self.initialize()
            if not success:
                raise RuntimeError("EmbeddingService: Failed to initialize")
        
        if not texts:
            return []
        
        try:
            # Use the encoder to generate embeddings
            embeddings = await self.encoder.encode(texts)
            
            logger.debug(f"EmbeddingService: Generated embeddings for {len(texts)} texts")
            return embeddings
            
        except Exception as e:
            logger.error(f"EmbeddingService: Encoding failed: {e}")
            raise
    
    async def encode_single(self, text: str) -> Any:
        """Generate embedding for a single text.
        
        Args:
            text: Text string
            
        Returns:
            Embedding vector
        """
        embeddings = await self.encode([text])
        return embeddings[0] if embeddings else None
    
    async def encode_batch(self, texts: List[str]) -> List[Any]:
        """Generate embeddings for texts in batches.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = await self.encode(batch)
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the embedding model.
        
        Returns:
            Dictionary with model information
        """
        info = {
            'model_name': self.model_name,
            'cache_size': self.cache_size,
            'batch_size': self.batch_size,
            'initialized': self.initialized
        }
        
        if self.encoder:
            # Get additional info from encoder
            encoder_info = getattr(self.encoder, 'get_model_info', lambda: {})()
            info.update(encoder_info)
        
        return info
    
    def get_stats(self) -> Dict[str, Any]:
        """Get embedding service statistics.
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            'service_type': 'EmbeddingService',
            'model_name': self.model_name,
            'initialized': self.initialized
        }
        
        if self.encoder:
            # Get stats from encoder
            encoder_stats = getattr(self.encoder, 'get_stats', lambda: {})()
            stats.update(encoder_stats)
        
        return stats
    
    async def close(self):
        """Close the embedding service and clean up resources."""
        if self.encoder:
            # Close encoder if it has a close method
            close_method = getattr(self.encoder, 'close', None)
            if close_method:
                if asyncio.iscoroutinefunction(close_method):
                    await close_method()
                else:
                    close_method()
        
        self.initialized = False
        logger.info("EmbeddingService: Closed")


# Global embedding service instance
_global_embedding_service: Optional[EmbeddingService] = None


async def get_embedding_service(
    model_name: str = "all-MiniLM-L6-v2",
    cache_size: int = 10000,
    batch_size: int = 32
) -> EmbeddingService:
    """Get or create the global embedding service instance.
    
    Args:
        model_name: Name of the embedding model
        cache_size: Size of the embedding cache
        batch_size: Batch size for embedding generation
        
    Returns:
        EmbeddingService instance
    """
    global _global_embedding_service
    
    if _global_embedding_service is None:
        _global_embedding_service = EmbeddingService(
            model_name=model_name,
            cache_size=cache_size,
            batch_size=batch_size
        )
        await _global_embedding_service.initialize()
    
    return _global_embedding_service


async def close_global_embedding_service():
    """Close the global embedding service."""
    global _global_embedding_service
    
    if _global_embedding_service:
        await _global_embedding_service.close()
        _global_embedding_service = None
