"""Model service for MemFuse server.

This module provides a centralized service for managing models across the MemFuse framework.
It dynamically loads models based on configuration and provides a unified interface for model access.
"""

from typing import Dict, Any, Optional, Type, List, Union
from loguru import logger
from omegaconf import DictConfig
import importlib
import inspect
import time
import numpy as np

# Import config manager
from ..utils.config import config_manager

# Import base classes for type checking
from ..rag.encode.base import EncoderBase, EncoderRegistry
from ..rag.rerank.base import RerankerBase, RerankerRegistry

# Import core interfaces
from ..interfaces.model_provider import ModelProviderInterface, ModelRegistry
from .base_service import BaseService



class ModelService(BaseService, ModelProviderInterface):
    """Central service for models in the MemFuse server.

    This class provides a unified interface for model loading and access.
    It dynamically loads models based on configuration and caches them for reuse.
    """

    _instance = None

    @classmethod
    def get_instance(cls):
        """Get the singleton instance of ModelService."""
        if cls._instance is None:
            cls._instance = ModelService()
        return cls._instance

    def __init__(self):
        """Initialize the model service."""
        # Initialize base service
        super().__init__("model")

        # Cache for loaded models
        self._encoders: Dict[str, EncoderBase] = {}
        self._rerankers: Dict[str, RerankerBase] = {}

        # Default models
        self._default_encoder: Optional[EncoderBase] = None
        self._default_reranker: Optional[RerankerBase] = None

        # Ensure all model implementations are imported and registered
        self._discover_implementations()

        # Register this service as the global model provider
        ModelRegistry.set_provider(self)

    def _discover_implementations(self) -> None:
        """Discover and import all model implementations.

        This ensures that all model implementations are registered with their
        respective registries before we try to create instances of them.
        """
        try:
            # Import encoder implementations
            importlib.import_module("memfuse_core.rag.encode.MiniLM")

            # Import reranker implementations
            importlib.import_module("memfuse_core.rag.rerank.MiniLM")

            # Log available implementations
            logger.info(f"Available encoder implementations: {EncoderRegistry.list_available()}")
            logger.info(f"Available reranker implementations: {RerankerRegistry.list_available()}")
        except Exception as e:
            logger.error(f"Error discovering model implementations: {e}")

    async def initialize(self, cfg: DictConfig) -> bool:
        """Initialize all models based on configuration.

        Args:
            cfg: Configuration from Hydra

        Returns:
            True if initialization was successful, False otherwise
        """
        try:
            # Preload embedding model
            await self._preload_embedding_model(cfg)

            # Preload reranking model
            await self._preload_rerank_model(cfg)

            self._mark_initialized()
            return True
        except Exception as e:
            logger.error(f"Failed to initialize model service: {e}")
            return False

    async def shutdown(self) -> bool:
        """Shutdown the model service.

        Returns:
            True if shutdown was successful, False otherwise
        """
        try:
            # Clear model caches
            self._encoders.clear()
            self._rerankers.clear()
            self._default_encoder = None
            self._default_reranker = None

            self._mark_shutdown()
            return True
        except Exception as e:
            logger.error(f"Failed to shutdown model service: {e}")
            return False

    async def _preload_embedding_model(self, cfg: DictConfig) -> None:
        """Pre-load the embedding model based on configuration.

        Args:
            cfg: Configuration from Hydra
        """
        try:
            # Check if embedding model is configured
            if not hasattr(cfg, 'embedding') or not hasattr(cfg.embedding, 'model'):
                logger.warning("No embedding model configured, skipping preload")
                return

            model_name = cfg.embedding.model
            implementation = cfg.embedding.get('implementation', 'minilm')

            logger.info(f"Pre-loading embedding model: {model_name} using {implementation} implementation")

            # Start timing the model loading
            start_time = time.time()

            # Create the encoder instance
            encoder = EncoderRegistry.create(
                implementation,
                model_name=model_name,
                cache_size=cfg.embedding.get('cache_size', 10000)
            )

            # Store the encoder
            self._encoders[model_name] = encoder
            self._default_encoder = encoder

            # Log model loading time
            load_time = time.time() - start_time
            logger.info(f"Embedding model loaded in {load_time:.2f} seconds")

            # Apply FP16 optimization if configured
            if hasattr(cfg.embedding, 'use_fp16') and cfg.embedding.use_fp16:
                logger.info("Using FP16 precision for embedding model inference")
                # Note: FP16 conversion is handled by the encoder implementation

            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Error pre-loading embedding model: {e}")

    async def _preload_rerank_model(self, cfg: DictConfig) -> None:
        """Pre-load the rerank model based on configuration.

        Args:
            cfg: Configuration from Hydra
        """
        try:
            # Check if rerank model preloading is enabled
            preload_rerank = True  # Default to True
            if hasattr(cfg, 'retrieval') and hasattr(cfg.retrieval, 'preload_model'):
                preload_rerank = cfg.retrieval.preload_model

            if not preload_rerank:
                logger.info("Rerank model preloading is disabled in configuration")
                return

            # Get rerank model configuration
            model_name = "cross-encoder/ms-marco-MiniLM-L6-v2"  # Default model
            implementation = "minilm"  # Default implementation

            if hasattr(cfg, 'retrieval'):
                if hasattr(cfg.retrieval, 'rerank_model'):
                    model_name = cfg.retrieval.rerank_model
                if hasattr(cfg.retrieval, 'rerank_implementation'):
                    implementation = cfg.retrieval.rerank_implementation

            logger.info(f"Pre-loading rerank model: {model_name} using {implementation} implementation")

            # Start timing the model loading
            start_time = time.time()

            # Create the reranker instance
            reranker = RerankerRegistry.create(
                implementation,
                model_name=model_name
            )

            # Store the reranker
            self._rerankers[model_name] = reranker
            self._default_reranker = reranker

            # Log model loading time
            load_time = time.time() - start_time
            logger.info(f"Rerank model loaded in {load_time:.2f} seconds")

            # Apply FP16 optimization if configured
            if hasattr(cfg, 'retrieval') and hasattr(cfg.retrieval, 'use_fp16') and cfg.retrieval.use_fp16:
                logger.info("Using FP16 precision for rerank model inference")
                # Note: FP16 conversion is handled by the reranker implementation

            logger.info("Rerank model loaded successfully")
        except Exception as e:
            logger.error(f"Error pre-loading rerank model: {e}")

    def get_encoder(self, name: Optional[str] = None) -> Optional[EncoderBase]:
        """Get an encoder by name.

        Args:
            name: Name of the encoder to get, or None for the default encoder

        Returns:
            Encoder instance or None if not available
        """
        if name is None:
            return self._default_encoder
        return self._encoders.get(name)

    def get_reranker(self, name: Optional[str] = None) -> Optional[RerankerBase]:
        """Get a reranker by name.

        Args:
            name: Name of the reranker to get, or None for the default reranker

        Returns:
            Reranker instance or None if not available
        """
        if name is None:
            return self._default_reranker
        return self._rerankers.get(name)

    def get_embedding_model(self) -> Optional[Any]:
        """Get the pre-loaded embedding model.

        Returns:
            Pre-loaded embedding model or None if not available
        """
        encoder = self.get_encoder()
        if encoder and hasattr(encoder, 'model'):
            return encoder.model
        return None

    def get_rerank_model(self) -> Optional[Any]:
        """Get the pre-loaded cross encoder rerank model.

        Returns:
            Pre-loaded cross encoder rerank model or None if not available
        """
        reranker = self.get_reranker()
        if reranker and hasattr(reranker, 'model'):
            return reranker.model
        return None

    async def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a text using the default encoder.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        encoder = self.get_encoder()
        if encoder is None:
            raise ValueError("No encoder available")
        return await encoder.encode_text(text)

    async def get_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Get embeddings for multiple texts using the default encoder.

        Args:
            texts: Texts to embed

        Returns:
            List of embedding vectors
        """
        encoder = self.get_encoder()
        if encoder is None:
            raise ValueError("No encoder available")
        return await encoder.encode_texts(texts)





# Add methods to ModelService for embedding functionality
def get_model_service() -> ModelService:
    """Get the global model service instance.

    Returns:
        ModelService instance
    """
    return ModelService.get_instance()
