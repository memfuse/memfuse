"""Model adapter for store layer.

This module provides adapters that allow store implementations to use models
without directly depending on the services layer.
"""

from typing import List, Optional, Any
from loguru import logger

from ...interfaces.model_provider import EmbeddingProvider, RerankProvider, ModelRegistry


class StoreEmbeddingAdapter:
    """Adapter for embedding functionality in stores."""

    def __init__(self, provider: Optional[EmbeddingProvider] = None):
        """Initialize the embedding adapter.

        Args:
            provider: Optional embedding provider. If None, will use ModelRegistry.
        """
        self._provider = provider

    def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode texts to embeddings.

        Args:
            texts: List of texts to encode

        Returns:
            List of embeddings
        """
        provider = self._get_provider()
        if provider is None:
            logger.warning("No embedding provider available")
            return []

        try:
            if hasattr(provider, 'encode'):
                return provider.encode(texts)
            elif hasattr(provider, 'encode_batch'):
                return provider.encode_batch(texts)
            else:
                # Fallback to single encoding
                return [self.encode_single(text) for text in texts]
        except Exception as e:
            logger.error(f"Error encoding texts: {e}")
            return []

    def encode_single(self, text: str) -> List[float]:
        """Encode a single text to embedding.

        Args:
            text: Text to encode

        Returns:
            Embedding vector
        """
        provider = self._get_provider()
        if provider is None:
            logger.warning("No embedding provider available")
            return []

        try:
            if hasattr(provider, 'encode_single'):
                return provider.encode_single(text)
            elif hasattr(provider, 'encode'):
                result = provider.encode([text])
                return result[0] if result else []
            else:
                logger.warning("Provider does not support encoding")
                return []
        except Exception as e:
            logger.error(f"Error encoding text: {e}")
            return []

    def _get_provider(self) -> Optional[Any]:
        """Get the embedding provider.

        Returns:
            Embedding provider or None
        """
        if self._provider is not None:
            return self._provider

        # Try to get from ModelRegistry
        model = ModelRegistry.get_embedding_model()
        if model is not None:
            return model

        return None


class StoreRerankAdapter:
    """Adapter for reranking functionality in stores."""

    def __init__(self, provider: Optional[RerankProvider] = None):
        """Initialize the rerank adapter.

        Args:
            provider: Optional rerank provider. If None, will use ModelRegistry.
        """
        self._provider = provider

    def rerank(self, query: str, documents: List[str], top_k: Optional[int] = None) -> List[float]:
        """Rerank documents based on query.

        Args:
            query: Query text
            documents: List of documents to rerank
            top_k: Optional number of top results to return

        Returns:
            List of relevance scores
        """
        provider = self._get_provider()
        if provider is None:
            logger.warning("No rerank provider available")
            return [0.0] * len(documents)

        try:
            if hasattr(provider, 'rerank'):
                scores = provider.rerank(query, documents)
                if top_k is not None and len(scores) > top_k:
                    return scores[:top_k]
                return scores
            else:
                logger.warning("Provider does not support reranking")
                return [0.0] * len(documents)
        except Exception as e:
            logger.error(f"Error reranking documents: {e}")
            return [0.0] * len(documents)

    def _get_provider(self) -> Optional[Any]:
        """Get the rerank provider.

        Returns:
            Rerank provider or None
        """
        if self._provider is not None:
            return self._provider

        # Try to get from ModelRegistry
        model = ModelRegistry.get_rerank_model()
        if model is not None:
            return model

        return None


class ModelAdapterFactory:
    """Factory for creating model adapters."""

    @staticmethod
    def create_embedding_adapter(provider: Optional[EmbeddingProvider] = None) -> StoreEmbeddingAdapter:
        """Create an embedding adapter.

        Args:
            provider: Optional embedding provider

        Returns:
            Embedding adapter
        """
        return StoreEmbeddingAdapter(provider)

    @staticmethod
    def create_rerank_adapter(provider: Optional[RerankProvider] = None) -> StoreRerankAdapter:
        """Create a rerank adapter.

        Args:
            provider: Optional rerank provider

        Returns:
            Rerank adapter
        """
        return StoreRerankAdapter(provider)
