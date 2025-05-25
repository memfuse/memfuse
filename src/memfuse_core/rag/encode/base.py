"""Base encoder interface for MemFuse server.

This module defines the base interface for all encoding implementations.
All encoder implementations should inherit from EncoderBase and register with EncoderRegistry.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Type
import numpy as np

from ...models.core import Item, Query


class EncoderBase(ABC):
    """Base class for all encoder implementations.
    
    This abstract class defines the interface that all encoders must implement.
    It provides methods for encoding text and items into vector representations.
    """

    @abstractmethod
    async def encode_text(self, text: str) -> np.ndarray:
        """Encode a single text string.

        Args:
            text: Text to encode

        Returns:
            Embedding vector
        """
        pass

    @abstractmethod
    async def encode_texts(self, texts: List[str]) -> List[np.ndarray]:
        """Encode multiple text strings.

        Args:
            texts: Texts to encode

        Returns:
            List of embedding vectors
        """
        pass

    async def encode_item(self, item: Item) -> np.ndarray:
        """Encode an item.

        Args:
            item: Item to encode

        Returns:
            Embedding vector
        """
        return await self.encode_text(item.content)

    async def encode_items(self, items: List[Item]) -> List[np.ndarray]:
        """Encode multiple items.

        Args:
            items: Items to encode

        Returns:
            List of embedding vectors
        """
        return await self.encode_texts([item.content for item in items])

    async def encode_query(self, query: Query) -> Query:
        """Encode a query.

        Args:
            query: Query to encode

        Returns:
            Encoded query
        """
        query.embedding = await self.encode_text(query.text)
        return query


class EncoderRegistry:
    """Registry for encoder implementations.
    
    This class provides a registry for encoder implementations and
    a factory method for creating encoder instances.
    """
    
    _registry: Dict[str, Type[EncoderBase]] = {}
    
    @classmethod
    def register(cls, name):
        """Register an encoder implementation.
        
        Args:
            name: Name of the encoder implementation
            
        Returns:
            Decorator function
        """
        def decorator(encoder_class):
            cls._registry[name] = encoder_class
            return encoder_class
        return decorator
    
    @classmethod
    def create(cls, name, **kwargs):
        """Create an encoder instance.
        
        Args:
            name: Name of the encoder implementation
            **kwargs: Additional arguments to pass to the encoder constructor
            
        Returns:
            Encoder instance
            
        Raises:
            ValueError: If the encoder implementation is not registered
        """
        if name not in cls._registry:
            raise ValueError(f"Encoder implementation '{name}' not registered")
        return cls._registry[name](**kwargs)
    
    @classmethod
    def list_available(cls):
        """List available encoder implementations.
        
        Returns:
            List of available encoder implementations
        """
        return list(cls._registry.keys())
