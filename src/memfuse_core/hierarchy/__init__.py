"""
Optimized memory hierarchy system for MemFuse.

This module provides a clean, unified architecture for the three-tier memory system:

1. M0 (Raw Data): Stores original data in its unprocessed form
   - Vector Store: For semantic similarity search
   - Keyword Store: For keyword-based search
   - SQL Store: For structured metadata

2. M1 (Episodic Memory): Stores event-centered experiences and contexts
   - Episode formation from raw data
   - Contextual and temporal metadata preservation
   - Episode storage and indexing

3. M2 (Semantic Memory): Extracts and stores facts and concepts
   - LLM-based fact extraction from episodes
   - Fact storage and indexing
   - Semantic search over facts

The optimized architecture features:
- Unified interfaces across all layers
- Event-driven inter-layer communication
- Centralized storage management
- Comprehensive error handling and statistics
"""

# Core interfaces and data structures
from .core import (
    MemoryLayer, StorageManager, StorageBackend,
    LayerType, ProcessingMode, StorageType,
    ProcessingResult, QueryResult, LayerStats, LayerConfig
)



# Storage management
from .storage import UnifiedStorageManager, StoreBackendAdapter

# Memory layer implementations
from .layers import M0RawDataLayer, M1EpisodicLayer, M2SemanticLayer, M3ProceduralLayer

# Main memory manager
from .manager import MemoryHierarchyManager, create_memory_manager



__all__ = [
    # Core interfaces and types
    "MemoryLayer",
    "StorageManager",
    "StorageBackend",
    "LayerType",
    "ProcessingMode",
    "StorageType",
    "ProcessingResult",
    "QueryResult",
    "LayerStats",
    "LayerConfig",

    # Storage management
    "UnifiedStorageManager",
    "StoreBackendAdapter",

    # Memory layers
    "M0RawDataLayer",
    "M1EpisodicLayer",
    "M2SemanticLayer",
    "M3ProceduralLayer",

    # Main manager
    "MemoryHierarchyManager",
    "create_memory_manager",

    # Legacy components
    "AdvancedLLMService",
    "AdvancedFactsDatabase",
    "ConflictDetectionEngine",
]
