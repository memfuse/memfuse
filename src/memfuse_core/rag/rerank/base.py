"""Base reranker interface for MemFuse server.

This module defines the base interface for all reranking implementations.
All reranker implementations should inherit from RerankerBase and register with RerankerRegistry.
"""

from abc import ABC, abstractmethod
from typing import List, Any, Dict, Optional, Union, Tuple, Type

from ...models.core import Item, Query


class RerankerBase(ABC):
    """Base class for all reranker implementations.
    
    This abstract class defines the interface that all rerankers must implement.
    It provides methods for reranking items based on their relevance to a query.
    """

    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the reranker.
        
        Returns:
            True if initialization was successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def rerank(
        self,
        query: str,
        items: List[Any],
        top_k: int = 5,
        source: str = "default"
    ) -> List[Any]:
        """Rerank items based on their relevance to the query.
        
        Args:
            query: The query string
            items: List of items to rerank
            top_k: Number of top items to return
            source: Source of the items
            
        Returns:
            List of reranked items
        """
        pass
    
    @abstractmethod
    async def score(
        self,
        query: str,
        texts: List[str]
    ) -> List[float]:
        """Score texts based on their relevance to the query.
        
        Args:
            query: The query string
            texts: List of texts to score
            
        Returns:
            List of relevance scores
        """
        pass
    
    @abstractmethod
    async def score_pairs(
        self,
        pairs: List[Tuple[str, str]]
    ) -> List[float]:
        """Score text pairs based on their relevance.
        
        Args:
            pairs: List of (query, text) pairs to score
            
        Returns:
            List of relevance scores
        """
        pass


class RerankerRegistry:
    """Registry for reranker implementations.
    
    This class provides a registry for reranker implementations and
    a factory method for creating reranker instances.
    """
    
    _registry: Dict[str, Type[RerankerBase]] = {}
    
    @classmethod
    def register(cls, name):
        """Register a reranker implementation.
        
        Args:
            name: Name of the reranker implementation
            
        Returns:
            Decorator function
        """
        def decorator(reranker_class):
            cls._registry[name] = reranker_class
            return reranker_class
        return decorator
    
    @classmethod
    def create(cls, name, **kwargs):
        """Create a reranker instance.
        
        Args:
            name: Name of the reranker implementation
            **kwargs: Additional arguments to pass to the reranker constructor
            
        Returns:
            Reranker instance
            
        Raises:
            ValueError: If the reranker implementation is not registered
        """
        if name not in cls._registry:
            raise ValueError(f"Reranker implementation '{name}' not registered")
        return cls._registry[name](**kwargs)
    
    @classmethod
    def list_available(cls):
        """List available reranker implementations.
        
        Returns:
            List of available reranker implementations
        """
        return list(cls._registry.keys())
