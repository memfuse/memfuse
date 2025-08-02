"""Global Model Manager for MemFuse Performance Optimization.

This module implements a high-performance global model manager that:
1. Loads expensive models (embedding, reranking, LLM) once at startup
2. Shares model instances across all users and services
3. Provides model health checking with TTL-based caching
4. Eliminates repeated model loading overhead
"""

import asyncio
import time
from typing import Dict, Any, Optional, Union, Callable
from threading import Lock
from loguru import logger
from dataclasses import dataclass
from enum import Enum


class ModelType(Enum):
    """Enumeration of supported model types."""
    EMBEDDING = "embedding"
    RERANKING = "reranking"
    LLM = "llm"


@dataclass
class ModelInfo:
    """Information about a loaded model."""
    model_type: ModelType
    model_name: str
    model_instance: Any
    load_time: float
    last_health_check: Optional[float] = None
    health_status: bool = True
    usage_count: int = 0


class GlobalModelManager:
    """High-performance global model manager with singleton pattern.
    
    This class implements the global singleton pattern for model management,
    providing one-time model loading and sharing across all users and services.
    
    Features:
    - One-time model loading at startup
    - Model sharing across all users and services
    - Model health checking with TTL-based caching
    - Performance monitoring and metrics
    - Thread-safe operations
    """
    
    _instance: Optional['GlobalModelManager'] = None
    _lock = Lock()
    _initialized = False
    
    def __init__(self):
        """Initialize the global model manager."""
        if GlobalModelManager._initialized:
            return
            
        # Model storage
        self._models: Dict[str, ModelInfo] = {}
        self._model_factories: Dict[ModelType, Callable] = {}
        
        # Performance tracking
        self._total_load_time = 0.0
        self._initialization_time: Optional[float] = None
        
        # Health checking
        self._health_check_ttl = 300.0  # 5 minutes
        self._health_check_lock = Lock()
        
        # Thread safety
        self._models_lock = Lock()
        
        GlobalModelManager._initialized = True
        logger.info("GlobalModelManager: Singleton instance created")
    
    def __new__(cls) -> 'GlobalModelManager':
        """Ensure singleton pattern with thread safety."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def get_instance(cls) -> 'GlobalModelManager':
        """Get the singleton instance."""
        return cls()
    
    def register_model_factory(
        self, 
        model_type: ModelType, 
        factory_func: Callable[..., Any]
    ) -> None:
        """Register a factory function for a model type.
        
        Args:
            model_type: Type of model
            factory_func: Function that creates the model instance
        """
        self._model_factories[model_type] = factory_func
        logger.info(f"GlobalModelManager: Registered factory for {model_type.value}")
    
    async def initialize_models(self, config: Dict[str, Any]) -> None:
        """Initialize all models based on configuration.
        
        Args:
            config: Configuration dictionary containing model settings
        """
        start_time = time.time()
        
        logger.info("GlobalModelManager: Starting model initialization...")
        
        # Initialize embedding model
        await self._initialize_embedding_model(config)
        
        # Initialize reranking model
        await self._initialize_reranking_model(config)
        
        # Initialize LLM models
        await self._initialize_llm_models(config)
        
        self._initialization_time = time.time() - start_time
        self._total_load_time = sum(model.load_time for model in self._models.values())
        
        logger.info(f"GlobalModelManager: Initialized {len(self._models)} models in {self._initialization_time:.3f}s")
        logger.info(f"GlobalModelManager: Total model load time: {self._total_load_time:.3f}s")
    
    async def _initialize_embedding_model(self, config: Dict[str, Any]) -> None:
        """Initialize embedding model."""
        embedding_config = config.get("embedding", {})
        if not embedding_config.get("enabled", True):
            logger.info("GlobalModelManager: Embedding model disabled in config")
            return
        
        model_name = embedding_config.get("model", "all-MiniLM-L6-v2")
        
        try:
            start_time = time.time()
            
            # Check if we have a registered factory
            if ModelType.EMBEDDING in self._model_factories:
                factory = self._model_factories[ModelType.EMBEDDING]
                model_instance = await self._safe_model_creation(factory, embedding_config)
            else:
                # Fallback to default embedding model creation
                model_instance = await self._create_default_embedding_model(embedding_config)
            
            load_time = time.time() - start_time
            
            model_info = ModelInfo(
                model_type=ModelType.EMBEDDING,
                model_name=model_name,
                model_instance=model_instance,
                load_time=load_time
            )
            
            with self._models_lock:
                self._models["embedding"] = model_info
            
            logger.info(f"GlobalModelManager: Loaded embedding model '{model_name}' in {load_time:.3f}s")
            
        except Exception as e:
            logger.error(f"GlobalModelManager: Failed to load embedding model: {e}")
    
    async def _initialize_reranking_model(self, config: Dict[str, Any]) -> None:
        """Initialize reranking model."""
        retrieval_config = config.get("retrieval", {})
        if not retrieval_config.get("use_rerank", True):
            logger.info("GlobalModelManager: Reranking disabled in config")
            return
        
        rerank_config = retrieval_config.get("rerank", {})
        model_name = rerank_config.get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        
        try:
            start_time = time.time()
            
            # Check if we have a registered factory
            if ModelType.RERANKING in self._model_factories:
                factory = self._model_factories[ModelType.RERANKING]
                model_instance = await self._safe_model_creation(factory, rerank_config)
            else:
                # Fallback to default reranking model creation
                model_instance = await self._create_default_reranking_model(rerank_config)
            
            load_time = time.time() - start_time
            
            model_info = ModelInfo(
                model_type=ModelType.RERANKING,
                model_name=model_name,
                model_instance=model_instance,
                load_time=load_time
            )
            
            with self._models_lock:
                self._models["reranking"] = model_info
            
            logger.info(f"GlobalModelManager: Loaded reranking model '{model_name}' in {load_time:.3f}s")
            
        except Exception as e:
            logger.error(f"GlobalModelManager: Failed to load reranking model: {e}")
    
    async def _initialize_llm_models(self, config: Dict[str, Any]) -> None:
        """Initialize LLM models."""
        memory_config = config.get("memory", {})
        layers_config = memory_config.get("layers", {})
        
        # Initialize LLM models for M1 and M2 layers if enabled
        for layer_name in ["m1", "m2"]:
            layer_config = layers_config.get(layer_name, {})
            if layer_config.get("enabled", False):
                await self._initialize_layer_llm_model(layer_name, layer_config)
    
    async def _initialize_layer_llm_model(self, layer_name: str, layer_config: Dict[str, Any]) -> None:
        """Initialize LLM model for a specific memory layer."""
        model_name = layer_config.get("llm_model", "grok-3-mini")
        
        try:
            start_time = time.time()
            
            # Check if we have a registered factory
            if ModelType.LLM in self._model_factories:
                factory = self._model_factories[ModelType.LLM]
                model_instance = await self._safe_model_creation(factory, layer_config)
            else:
                # For now, we'll skip LLM initialization if no factory is registered
                logger.info(f"GlobalModelManager: No LLM factory registered, skipping {layer_name}")
                return
            
            load_time = time.time() - start_time
            
            model_info = ModelInfo(
                model_type=ModelType.LLM,
                model_name=model_name,
                model_instance=model_instance,
                load_time=load_time
            )
            
            with self._models_lock:
                self._models[f"llm_{layer_name}"] = model_info
            
            logger.info(f"GlobalModelManager: Loaded LLM model '{model_name}' for {layer_name} in {load_time:.3f}s")
            
        except Exception as e:
            logger.error(f"GlobalModelManager: Failed to load LLM model for {layer_name}: {e}")
    
    async def _safe_model_creation(self, factory: Callable, config: Dict[str, Any]) -> Any:
        """Safely create a model using a factory function."""
        try:
            if asyncio.iscoroutinefunction(factory):
                return await factory(config)
            else:
                return factory(config)
        except Exception as e:
            logger.error(f"GlobalModelManager: Model factory failed: {e}")
            raise
    
    async def _create_default_embedding_model(self, config: Dict[str, Any]) -> Any:
        """Create default embedding model."""
        # Import here to avoid circular imports
        try:
            from sentence_transformers import SentenceTransformer
            model_name = config.get("model", "all-MiniLM-L6-v2")
            return SentenceTransformer(model_name)
        except ImportError:
            logger.warning("sentence-transformers not available, using mock embedding model")
            return MockEmbeddingModel(config.get("model", "all-MiniLM-L6-v2"))
    
    async def _create_default_reranking_model(self, config: Dict[str, Any]) -> Any:
        """Create default reranking model."""
        # Import here to avoid circular imports
        try:
            from ..rag.rerank import MiniLMReranker
            return MiniLMReranker()
        except ImportError:
            logger.warning("Reranker not available, using mock reranking model")
            return MockRerankingModel(config.get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
    
    def get_model(self, model_key: str) -> Optional[Any]:
        """Get a model instance by key.
        
        Args:
            model_key: Model key (e.g., 'embedding', 'reranking', 'llm_m1')
            
        Returns:
            Model instance or None if not found
        """
        with self._models_lock:
            model_info = self._models.get(model_key)
            if model_info:
                model_info.usage_count += 1
                return model_info.model_instance
        return None
    
    def get_embedding_model(self) -> Optional[Any]:
        """Get the global embedding model instance."""
        return self.get_model("embedding")
    
    def get_reranking_model(self) -> Optional[Any]:
        """Get the global reranking model instance."""
        return self.get_model("reranking")
    
    def get_llm_model(self, layer: str = "m1") -> Optional[Any]:
        """Get the LLM model for a specific layer."""
        return self.get_model(f"llm_{layer}")
    
    async def check_model_health(self, model_key: str, force: bool = False) -> bool:
        """Check model health with TTL-based caching.
        
        Args:
            model_key: Model key to check
            force: Force health check even if within TTL
            
        Returns:
            True if model is healthy, False otherwise
        """
        with self._models_lock:
            model_info = self._models.get(model_key)
            if not model_info:
                return False
            
            # Check if we need to perform health check
            current_time = time.time()
            if (not force and 
                model_info.last_health_check and 
                current_time - model_info.last_health_check < self._health_check_ttl):
                return model_info.health_status
            
            # Perform health check
            try:
                # Basic health check - ensure model instance exists and is callable
                if model_info.model_instance is None:
                    model_info.health_status = False
                else:
                    # For now, just check if the model instance exists
                    # More sophisticated health checks can be added per model type
                    model_info.health_status = True
                
                model_info.last_health_check = current_time
                return model_info.health_status
                
            except Exception as e:
                logger.error(f"GlobalModelManager: Health check failed for {model_key}: {e}")
                model_info.health_status = False
                model_info.last_health_check = current_time
                return False
    
    def get_model_info(self, model_key: str) -> Optional[ModelInfo]:
        """Get detailed information about a model.
        
        Args:
            model_key: Model key
            
        Returns:
            ModelInfo instance or None if not found
        """
        with self._models_lock:
            return self._models.get(model_key)
    
    def list_models(self) -> Dict[str, ModelInfo]:
        """List all loaded models.
        
        Returns:
            Dictionary of model key to ModelInfo
        """
        with self._models_lock:
            return dict(self._models)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics.
        
        Returns:
            Dictionary containing performance metrics
        """
        with self._models_lock:
            total_usage = sum(model.usage_count for model in self._models.values())
            
            return {
                "initialized": len(self._models) > 0,
                "total_models": len(self._models),
                "initialization_time_seconds": self._initialization_time,
                "total_load_time_seconds": self._total_load_time,
                "total_model_usage": total_usage,
                "models": {
                    key: {
                        "type": model.model_type.value,
                        "name": model.model_name,
                        "load_time": model.load_time,
                        "usage_count": model.usage_count,
                        "health_status": model.health_status,
                        "last_health_check": model.last_health_check
                    }
                    for key, model in self._models.items()
                }
            }
    
    async def cleanup(self) -> None:
        """Cleanup all models and resources."""
        logger.info("GlobalModelManager: Starting cleanup...")
        
        with self._models_lock:
            for model_key, model_info in self._models.items():
                try:
                    # Call cleanup method if available
                    if hasattr(model_info.model_instance, 'cleanup'):
                        await model_info.model_instance.cleanup()
                    elif hasattr(model_info.model_instance, 'close'):
                        model_info.model_instance.close()
                    
                    logger.debug(f"GlobalModelManager: Cleaned up model {model_key}")
                except Exception as e:
                    logger.error(f"GlobalModelManager: Error cleaning up model {model_key}: {e}")
            
            self._models.clear()
        
        logger.info("GlobalModelManager: Cleanup completed")


class MockEmbeddingModel:
    """Mock embedding model for testing."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    def encode(self, texts, **kwargs):
        """Mock encode method."""
        import numpy as np
        if isinstance(texts, str):
            texts = [texts]
        return np.random.random((len(texts), 384))


class MockRerankingModel:
    """Mock reranking model for testing."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    def rerank(self, query: str, documents: list, **kwargs):
        """Mock rerank method."""
        import random
        scores = [random.random() for _ in documents]
        return list(zip(documents, scores))


# Global singleton instance
_global_model_manager: Optional[GlobalModelManager] = None


def get_global_model_manager() -> GlobalModelManager:
    """Get the global model manager singleton instance."""
    global _global_model_manager
    if _global_model_manager is None:
        _global_model_manager = GlobalModelManager.get_instance()
    return _global_model_manager


# Convenience functions for model access
def get_embedding_model() -> Optional[Any]:
    """Get the global embedding model instance."""
    return get_global_model_manager().get_embedding_model()


def get_reranking_model() -> Optional[Any]:
    """Get the global reranking model instance."""
    return get_global_model_manager().get_reranking_model()


def get_llm_model(layer: str = "m1") -> Optional[Any]:
    """Get the LLM model for a specific layer."""
    return get_global_model_manager().get_llm_model(layer)