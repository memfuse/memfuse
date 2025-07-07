"""
Memory Layer Interface for MemFuse.

This interface provides an abstraction for all memory operations,
decoupling MemoryService from specific memory layer implementations (M0/M1/M2).
The memory layer handles parallel processing across all memory layers internally.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from enum import Enum

from .message_interface import MessageBatchList


class LayerStatus(Enum):
    """Status of individual memory layers."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    INITIALIZING = "initializing"


class WriteResult:
    """Result of a write operation across memory layers."""
    
    def __init__(
        self,
        success: bool,
        message: str = "",
        layer_results: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.success = success
        self.message = message
        self.layer_results = layer_results or {}
        self.metadata = metadata or {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "success": self.success,
            "message": self.message,
            "layer_results": self.layer_results,
            "metadata": self.metadata
        }


class QueryResult:
    """Result of a query across all memory layers."""
    
    def __init__(
        self,
        results: List[Dict[str, Any]],
        layer_sources: Dict[str, List[Dict[str, Any]]],
        total_count: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.results = results
        self.layer_sources = layer_sources  # Results grouped by layer (M0, M1, M2)
        self.total_count = total_count
        self.metadata = metadata or {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "results": self.results,
            "layer_sources": self.layer_sources,
            "total_count": self.total_count,
            "metadata": self.metadata
        }


class MemoryLayer(ABC):
    """
    Memory Layer Interface.

    This interface provides a single point of interaction for all memory operations,
    abstracting away the complexity of M0/M1/M2 parallel processing from MemoryService.

    Key Design Principles:
    1. Complete decoupling from MemoryService
    2. Parallel processing of M0/M1/M2 layers
    3. Result aggregation
    4. Configuration-driven layer activation
    5. Graceful error handling and fallback
    """

    @abstractmethod
    async def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Initialize the memory layer.

        Args:
            config: Optional configuration dictionary

        Returns:
            True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def write_parallel(
        self,
        message_batch_list: MessageBatchList,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> WriteResult:
        """
        Write data to all active memory layers in parallel.
        
        This method handles:
        - M0: Immediate episodic storage (vector/keyword/graph)
        - M1: Semantic fact extraction and storage (parallel, not triggered)
        - M2: Relational knowledge graph construction (parallel, not triggered)
        
        Args:
            message_batch_list: Batch of message lists to process
            session_id: Optional session identifier
            metadata: Optional metadata for the operation
            
        Returns:
            WriteResult with success status and layer-specific results
        """
        pass
    
    @abstractmethod
    async def query(
        self,
        query: str,
        top_k: int = 15,
        store_type: Optional[str] = None,
        include_messages: bool = True,
        include_knowledge: bool = True,
        include_chunks: bool = True,
        use_rerank: bool = True,
        session_id: Optional[str] = None,
        scope: str = "all"
    ) -> QueryResult:
        """
        Query all active memory layers and return results.

        This method:
        1. Queries M0, M1, M2 layers in parallel
        2. Aggregates and ranks results
        3. Applies reranking if enabled
        4. Returns result set

        Args:
            query: Query string
            top_k: Maximum number of results to return
            store_type: Type of store to query (vector, graph, keyword, or None for all)
            include_messages: Whether to include message results
            include_knowledge: Whether to include knowledge results
            include_chunks: Whether to include chunk results
            use_rerank: Whether to apply reranking
            session_id: Session ID to filter results
            scope: Scope of the query (all, session, or user)

        Returns:
            QueryResult with aggregated results from all layers
        """
        pass
    
    @abstractmethod
    async def get_layer_status(self) -> Dict[str, LayerStatus]:
        """
        Get the current status of all memory layers.
        
        Returns:
            Dictionary mapping layer names (M0, M1, M2) to their status
        """
        pass
    
    @abstractmethod
    async def get_layer_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all memory layers.
        
        Returns:
            Dictionary with statistics for each layer (counts, performance metrics, etc.)
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on all memory layers.
        
        Returns:
            Dictionary with health status and diagnostic information
        """
        pass
    
    @abstractmethod
    async def cleanup(self) -> bool:
        """
        Clean up resources and shut down all memory layers.
        
        Returns:
            True if cleanup successful, False otherwise
        """
        pass


class MemoryLayerConfig:
    """Configuration for MemoryLayer."""
    
    def __init__(
        self,
        m0_enabled: bool = True,
        m1_enabled: bool = True,
        m2_enabled: bool = True,
        parallel_strategy: str = "parallel",
        fallback_strategy: str = "sequential",
        enable_fallback: bool = True,
        timeout_per_layer: float = 30.0,
        total_timeout: float = 120.0,
        max_retries: int = 3,
        enable_monitoring: bool = True
    ):
        self.m0_enabled = m0_enabled
        self.m1_enabled = m1_enabled
        self.m2_enabled = m2_enabled
        self.parallel_strategy = parallel_strategy
        self.fallback_strategy = fallback_strategy
        self.enable_fallback = enable_fallback
        self.timeout_per_layer = timeout_per_layer
        self.total_timeout = total_timeout
        self.max_retries = max_retries
        self.enable_monitoring = enable_monitoring
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "m0_enabled": self.m0_enabled,
            "m1_enabled": self.m1_enabled,
            "m2_enabled": self.m2_enabled,
            "parallel_strategy": self.parallel_strategy,
            "fallback_strategy": self.fallback_strategy,
            "enable_fallback": self.enable_fallback,
            "timeout_per_layer": self.timeout_per_layer,
            "total_timeout": self.total_timeout,
            "max_retries": self.max_retries,
            "enable_monitoring": self.enable_monitoring
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MemoryLayerConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
