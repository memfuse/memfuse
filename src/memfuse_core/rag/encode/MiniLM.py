"""MiniLM encoder implementation for MemFuse server.

This module implements the EncoderBase interface using MiniLM models
from the sentence-transformers library.
"""

from typing import List, Optional, Any, Dict
import numpy as np
import os
from loguru import logger
import asyncio

# Set offline mode before importing sentence_transformers
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

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
        logger.info(f"MiniLMEncoder: Initializing with model_name={model_name}, cache_size={cache_size}, existing_model={existing_model is not None}")
        
        # Get configuration
        from ...utils.config import config_manager
        from omegaconf import OmegaConf

        config_dict = config_manager.get_config()
        cfg = OmegaConf.create(config_dict)

        # Use model from config if not specified
        if model_name is None:
            if hasattr(cfg, 'embedding') and hasattr(cfg.embedding, 'model'):
                model_name = cfg.embedding.model
            else:
                model_name = "all-MiniLM-L6-v2"  # Default model

        # Validate that this is a MiniLM model
        if not self._is_minilm_model(model_name):
            logger.warning(f"Model {model_name} may not be a MiniLM model, but will try to use it anyway")

        self.model_name = model_name

        # Use existing model if provided
        if existing_model is not None:
            # Don't log here to avoid duplicate logs
            self.model = existing_model
            logger.info(f"MiniLMEncoder: Using existing model: {type(existing_model)}")
        else:
            # Try to get global model instance first
            try:
                from ...services.global_model_manager import get_global_model_manager
                global_model_manager = get_global_model_manager()
                global_model = global_model_manager.get_embedding_model()
                
                if global_model is not None:
                    # Check if the global model is another MiniLMEncoder
                    if hasattr(global_model, 'model') and hasattr(global_model.model, 'encode'):
                        # Use the underlying SentenceTransformer model
                        self.model = global_model.model
                        logger.info(f"MiniLMEncoder: Using underlying model from global MiniLMEncoder: {model_name}")
                    else:
                        # Use the global model directly (should be SentenceTransformer)
                        self.model = global_model
                        logger.info(f"MiniLMEncoder: Using global embedding model: {model_name}")
                else:
                    # Load the model locally
                    logger.info(f"MiniLMEncoder: Loading MiniLM embedding model: {model_name}")
                    # Ensure correct model name format (remove sentence-transformers/ prefix if present)
                    clean_model_name = model_name.replace("sentence-transformers/", "")

                    self.model = SentenceTransformer(clean_model_name, trust_remote_code=False)
                    self.model_name = clean_model_name
                    logger.info(f"MiniLMEncoder: Successfully loaded model {clean_model_name}")
            except Exception as e:
                logger.warning(f"MiniLMEncoder: Could not get global model: {e}, loading locally")
                # Load the model locally as fallback
                logger.info(f"MiniLMEncoder: Loading MiniLM embedding model: {model_name}")
                # Ensure correct model name format (remove sentence-transformers/ prefix if present)
                clean_model_name = model_name.replace("sentence-transformers/", "")

                self.model = SentenceTransformer(clean_model_name, trust_remote_code=False)
                self.model_name = clean_model_name
                logger.info(f"MiniLMEncoder: Successfully loaded model {clean_model_name}")

        # Set up caching
        self.cache = Cache[str, np.ndarray](max_size=cache_size)
        self._lock = asyncio.Lock()
        
        # Apply FP16 optimization if configured
        if hasattr(cfg, 'embedding') and hasattr(cfg.embedding, 'use_fp16') and cfg.embedding.use_fp16:
            logger.info("Using FP16 precision for embedding model inference")
            if hasattr(self.model, "half"):
                self.model.half()
                logger.info("Successfully converted embedding model to FP16 precision")
        
        # Verify that the encoder is properly initialized
        logger.info(f"MiniLMEncoder: Initialization complete. Model type: {type(self.model)}")
        logger.info(f"MiniLMEncoder: Has encode_text method: {hasattr(self, 'encode_text')}")
        logger.info(f"MiniLMEncoder: Has encode method: {hasattr(self.model, 'encode')}")

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

    async def encode(self, text: str, **kwargs) -> np.ndarray:
        """Fallback encode method for compatibility with BufferRetrieval.
        
        This method provides backward compatibility for code that expects
        an 'encode' method instead of 'encode_text'. It simply delegates
        to the encode_text method to maintain consistent behavior.
        
        Args:
            text: Text to encode
            **kwargs: Additional keyword arguments (ignored for compatibility)
            
        Returns:
            Embedding vector
        """
        return await self.encode_text(text)
