"""Adapters for store layer.

This module provides adapters that decouple the store layer from other layers.
"""

from .model_adapter import (
    StoreEmbeddingAdapter,
    StoreRerankAdapter,
    ModelAdapterFactory
)

__all__ = [
    "StoreEmbeddingAdapter",
    "StoreRerankAdapter", 
    "ModelAdapterFactory"
]
