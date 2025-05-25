"""Buffer system for MemFuse.

The buffer system optimizes data flow between the client and storage layers,
improving performance and enabling advanced features:

1. WriteBuffer (Write Combining Buffer):
   Queues incoming data for batch processing, optimizing write operations
2. SpeculativeBuffer (Speculative Prefetch Buffer):
   Proactively retrieves data that might be needed soon, reducing latency
3. QueryBuffer (Heterogeneous Query Cache):
   Manages retrieval and ranking of data from multiple sources
4. EvictionBuffer (Cache Replacement Buffer):
   Manages memory usage by selectively removing less important items
5. LocalityBuffer (Temporal and Semantic Locality Buffer):
   Optimizes access to related items
6. CoherencyController (Cache Coherence Controller):
   Ensures data consistency across multiple buffers

The buffer system serves as a foundation for the hierarchical memory system (L0/L1/L2).
"""

from .write_buffer import WriteBuffer
from .speculative_buffer import SpeculativeBuffer
from .query_buffer import QueryBuffer

__all__ = [
    "WriteBuffer",
    "SpeculativeBuffer",
    "QueryBuffer",
]
