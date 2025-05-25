"""Interfaces module for MemFuse.

This module contains core interfaces and abstractions that are used
throughout the MemFuse framework without creating circular dependencies.
"""

# Model provider interfaces
from .model_provider import (
    ModelProvider,
    ModelProviderInterface,
    EmbeddingProvider,
    RerankProvider,
    NullModelProvider,
    ModelRegistry,
    get_embedding_model,
    get_rerank_model,
    set_model_provider
)

# Service interfaces
from .service_interface import ServiceInterface

# Store interfaces
from .store_interface import StoreInterface

# Memory interfaces
from .memory_interface import MemoryInterface

# Buffer interfaces
from .buffer_interface import BufferInterface
from .buffer_component_interface import BufferComponentInterface

# Storage handler
from .storage_handler import MemoryServiceStorageHandler

__all__ = [
    # Model provider interfaces
    "ModelProvider",
    "ModelProviderInterface",
    "EmbeddingProvider",
    "RerankProvider",
    "NullModelProvider",
    "ModelRegistry",
    "get_embedding_model",
    "get_rerank_model",
    "set_model_provider",

    # Service interfaces
    "ServiceInterface",

    # Store interfaces
    "StoreInterface",

    # Memory interfaces
    "MemoryInterface",

    # Buffer interfaces
    "BufferInterface",
    "BufferComponentInterface",

    # Storage handler
    "MemoryServiceStorageHandler",
]
