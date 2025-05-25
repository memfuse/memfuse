"""Memory hierarchy system for MemFuse.

The memory hierarchy system organizes memory into three layers:

1. L0 (Raw Data): Stores raw data in its original form
   - Vector Store: For semantic similarity search
   - Graph Store: For relationship-based search
   - Keyword Store: For keyword-based search

2. L1 (Facts): Extracts and stores facts from raw data
   - Fact Extraction: Extracts facts from raw data
   - Fact Storage: Stores facts with links to source data
   - Fact Retrieval: Retrieves facts based on relevance

3. L2 (Knowledge Graph): Constructs a knowledge graph from facts
   - Entity Extraction: Identifies entities from facts
   - Relationship Extraction: Identifies relationships between entities
   - Graph Construction: Builds a knowledge graph
   - Graph Traversal: Navigates the knowledge graph

The memory hierarchy system works with the buffer system to optimize
data flow and provide efficient memory operations.
"""

from .base import MemoryLayer, MemoryItem, Fact, Entity, Relationship
from .l0 import L0Manager
from .l1 import L1Manager, FactsDatabase, LLMService
from .l2 import L2Manager, GraphDatabase
from .manager import HierarchyMemoryManager

__all__ = [
    "MemoryLayer",
    "MemoryItem",
    "Fact",
    "Entity",
    "Relationship",
    "L0Manager",
    "L1Manager",
    "FactsDatabase",
    "LLMService",
    "L2Manager",
    "GraphDatabase",
    "HierarchyMemoryManager",
]
