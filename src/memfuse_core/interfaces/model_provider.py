"""Model provider interface for dependency injection.

This module provides a clean interface for model access without circular dependencies.
It follows the Dependency Inversion Principle to decouple high-level modules from
low-level modules.
"""

from abc import ABC, abstractmethod
from typing import Optional, Any, Protocol, List
from loguru import logger


class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""

    def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode texts to embeddings."""
        ...

    def encode_single(self, text: str) -> List[float]:
        """Encode a single text to embedding."""
        ...


class RerankProvider(Protocol):
    """Protocol for rerank providers."""

    def rerank(self, query: str, documents: List[str]) -> List[float]:
        """Rerank documents based on query."""
        ...


class ModelProvider(Protocol):
    """Protocol for model providers."""

    def get_embedding_model(self) -> Optional[Any]:
        """Get the embedding model."""
        ...

    def get_rerank_model(self) -> Optional[Any]:
        """Get the rerank model."""
        ...


class ModelProviderInterface(ABC):
    """Abstract interface for model providers."""

    @abstractmethod
    def get_embedding_model(self) -> Optional[Any]:
        """Get the embedding model.

        Returns:
            Embedding model instance or None if not available
        """
        pass

    @abstractmethod
    def get_rerank_model(self) -> Optional[Any]:
        """Get the rerank model.

        Returns:
            Rerank model instance or None if not available
        """
        pass


class NullModelProvider(ModelProviderInterface):
    """Null object pattern implementation for model provider."""

    def get_embedding_model(self) -> Optional[Any]:
        """Return None for embedding model."""
        return None

    def get_rerank_model(self) -> Optional[Any]:
        """Return None for rerank model."""
        return None


class ModelRegistry:
    """Global registry for model providers."""

    _provider: ModelProviderInterface = NullModelProvider()

    @classmethod
    def set_provider(cls, provider: ModelProviderInterface) -> None:
        """Set the global model provider.

        Args:
            provider: Model provider to set
        """
        cls._provider = provider
        logger.debug("Model provider updated in registry")

    @classmethod
    def get_provider(cls) -> ModelProviderInterface:
        """Get the global model provider.

        Returns:
            Current model provider
        """
        return cls._provider

    @classmethod
    def get_embedding_model(cls) -> Optional[Any]:
        """Get embedding model from the current provider.

        Returns:
            Embedding model or None
        """
        return cls._provider.get_embedding_model()

    @classmethod
    def get_rerank_model(cls) -> Optional[Any]:
        """Get rerank model from the current provider.

        Returns:
            Rerank model or None
        """
        return cls._provider.get_rerank_model()


# Convenience functions for easy access
def get_embedding_model() -> Optional[Any]:
    """Get embedding model from the global registry.

    Returns:
        Embedding model or None
    """
    return ModelRegistry.get_embedding_model()


def get_rerank_model() -> Optional[Any]:
    """Get rerank model from the global registry.

    Returns:
        Rerank model or None
    """
    return ModelRegistry.get_rerank_model()


def set_model_provider(provider: ModelProviderInterface) -> None:
    """Set the global model provider.

    Args:
        provider: Model provider to set
    """
    ModelRegistry.set_provider(provider)
