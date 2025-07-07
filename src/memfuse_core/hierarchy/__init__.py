"""
Optimized memory hierarchy system for MemFuse.

This module provides a clean, unified architecture for the three-tier memory system:

1. M0 (Episodic Memory): Stores raw data in its original form
   - Vector Store: For semantic similarity search
   - Keyword Store: For keyword-based search
   - SQL Store: For structured metadata

2. M1 (Semantic Memory): Extracts and stores facts from raw data
   - LLM-based fact extraction
   - Fact storage and indexing
   - Semantic search over facts

3. M2 (Relational Memory): Constructs a knowledge graph from facts
   - Entity extraction from facts
   - Relationship identification
   - Graph construction and updates
   - Graph-based querying

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
from .layers import M0EpisodicLayer, M1SemanticLayer, M2RelationalLayer

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
    "M0EpisodicLayer",
    "M1SemanticLayer",
    "M2RelationalLayer",

    # Main manager
    "MemoryHierarchyManager",
    "create_memory_manager",

    # Legacy components
    "AdvancedLLMService",
    "AdvancedFactsDatabase",
    "ConflictDetectionEngine",
]
