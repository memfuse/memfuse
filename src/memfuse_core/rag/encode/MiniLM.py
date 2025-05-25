"""MiniLM encoder implementation for MemFuse server.

This module implements the EncoderBase interface using MiniLM models
from the sentence-transformers library.
"""

from typing import List, Optional, Any, Dict
import numpy as np
from loguru import logger
import asyncio
from sentence_transformers import SentenceTransformer

from .base import EncoderBase, EncoderRegistry
from ...utils.cache import Cache


@EncoderRegistry.register("minilm")
class MiniLMEncoder(EncoderBase):
    """MiniLM encoder implementation.
    
    This class implements the EncoderBase interface using MiniLM models
    from the sentence-transformers library. It supports various MiniLM models
    such as all-MiniLM-L6-v2, all-MiniLM-L12-v2, etc.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        cache_size: int = 10000,
        existing_model: Any = None,
        **kwargs
    ):
        """Initialize the encoder.

        Args:
            model_name: Name of the model to use (e.g., 'all-MiniLM-L6-v2')
            cache_size: Size of the embedding cache
            existing_model: An existing SentenceTransformer model instance to reuse
            **kwargs: Additional arguments
        """
        # Get configuration
        from ...utils.config import config_manager
        from omegaconf import OmegaConf

        config_dict = config_manager.get_config()
        cfg = OmegaConf.create(config_dict)

        # Use model from config if not specified
        if model_name is None:
            model_name = cfg.embedding.model

        # Validate that this is a MiniLM model
        if not self._is_minilm_model(model_name):
            logger.warning(f"Model {model_name} may not be a MiniLM model, but will try to use it anyway")

        self.model_name = model_name

        # Use existing model if provided
        if existing_model is not None:
            # Don't log here to avoid duplicate logs
            self.model = existing_model
        else:
            # Load the model
            try:
                logger.info(f"Loading MiniLM embedding model: {model_name}")
                self.model = SentenceTransformer(
                    model_name, trust_remote_code=False)
            except Exception as e:
                logger.error(f"Error loading model {model_name}: {e}")
                # Use hardcoded default as last resort
                logger.warning("Using hardcoded default: all-MiniLM-L6-v2")
                self.model_name = "all-MiniLM-L6-v2"
                self.model = SentenceTransformer(
                    "all-MiniLM-L6-v2", trust_remote_code=False)

        # Set up caching
        self.cache = Cache[str, np.ndarray](max_size=cache_size)
        self._lock = asyncio.Lock()
        
        # Apply FP16 optimization if configured
        if hasattr(cfg, 'embedding') and hasattr(cfg.embedding, 'use_fp16') and cfg.embedding.use_fp16:
            logger.info("Using FP16 precision for embedding model inference")
            if hasattr(self.model, "half"):
                self.model.half()
                logger.info("Successfully converted embedding model to FP16 precision")

    def _is_minilm_model(self, model_name: str) -> bool:
        """Check if the model is a MiniLM model.
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            True if the model is a MiniLM model, False otherwise
        """
        return "minilm" in model_name.lower()

    async def encode_text(self, text: str) -> np.ndarray:
        """Encode a single text string.

        Args:
            text: Text to encode

        Returns:
            Embedding vector
        """
        # Add task-specific prefix for better results
        prefixed_text = f"search_document: {text}"
        
        # Check cache
        cache_key = f"{self.model_name}:{prefixed_text}"
        cached_embedding = self.cache.get(cache_key)
        if cached_embedding is not None:
            return cached_embedding

        # Generate embedding
        embedding = await self._generate_embedding(prefixed_text)

        # Cache embedding
        self.cache.set(cache_key, embedding)
        
        # Ensure we have a numpy array
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)
            
        return embedding

    async def encode_texts(self, texts: List[str]) -> List[np.ndarray]:
        """Encode multiple text strings.

        Args:
            texts: Texts to encode

        Returns:
            List of embedding vectors
        """
        # Add task-specific prefix for better results
        prefixed_texts = [f"search_document: {text}" for text in texts]
        
        # Check cache for each text
        embeddings = []
        texts_to_embed = []
        indices_to_embed = []

        for i, text in enumerate(prefixed_texts):
            cache_key = f"{self.model_name}:{text}"
            cached_embedding = self.cache.get(cache_key)
            if cached_embedding is not None:
                embeddings.append(cached_embedding)
            else:
                texts_to_embed.append(text)
                indices_to_embed.append(i)

        # Generate embeddings for texts not in cache
        if texts_to_embed:
            new_embeddings = await self._generate_embeddings(texts_to_embed)

            # Cache new embeddings
            for text, embedding in zip(texts_to_embed, new_embeddings):
                cache_key = f"{self.model_name}:{text}"
                self.cache.set(cache_key, embedding)

            # Insert new embeddings at the correct positions
            result = [None] * len(prefixed_texts)
            for i, embedding in enumerate(embeddings):
                result[i] = embedding
            for i, idx in enumerate(indices_to_embed):
                result[idx] = new_embeddings[i]

            embeddings = result
        
        # Ensure we have numpy arrays
        results = []
        for embedding in embeddings:
            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding)
            results.append(embedding)
            
        return results
        
    async def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        # Generate embedding
        embedding = await asyncio.to_thread(self.model.encode, text, convert_to_numpy=True)
        return embedding

    async def _generate_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts.

        Args:
            texts: Texts to embed

        Returns:
            List of embedding vectors
        """
        # Generate embeddings
        embeddings = await asyncio.to_thread(self.model.encode, texts, convert_to_numpy=True)
        return embeddings
        
    def get_stats(self) -> Dict[str, Any]:
        """Get encoder statistics.

        Returns:
            Dictionary of encoder statistics
        """
        return {
            "model_name": self.model_name,
            "cache_stats": self.cache.get_stats(),
            "model_loaded": self.model is not None,
        }

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self.cache.clear()
        
    def set_model(self, model: Any) -> None:
        """Set the model instance.
        
        Args:
            model: Model instance to use
        """
        self.model = model
