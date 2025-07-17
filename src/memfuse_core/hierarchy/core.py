"""
Core interfaces and data structures for the MemFuse memory hierarchy system.

This module defines the fundamental abstractions that all memory layers implement,
providing a unified interface for data processing, querying, and management.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime


class LayerType(Enum):
    """Memory layer types in the hierarchy."""
    M0 = "m0"  # Raw Data: Original data storage
    M1 = "m1"  # Episodic Memory: Event-centered experiences
    M2 = "m2"  # Semantic Memory: Facts and concepts
    M3 = "m3"  # Procedural Memory: Learned patterns and procedures


class ProcessingMode(Enum):
    """Processing modes for memory layers."""
    SYNC = "sync"      # Synchronous processing
    ASYNC = "async"    # Asynchronous processing
    BATCH = "batch"    # Batch processing





class StorageType(Enum):
    """Storage backend types."""
    VECTOR = "vector"
    GRAPH = "graph"
    KEYWORD = "keyword"
    SQL = "sql"


@dataclass
class ProcessingResult:
    """Result of data processing operation."""
    success: bool
    layer_type: LayerType
    processed_items: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0
    
    def __post_init__(self):
        """Validate the result after initialization."""
        if not isinstance(self.layer_type, LayerType):
            raise ValueError(f"layer_type must be LayerType, got {type(self.layer_type)}")


@dataclass
class QueryResult:
    """Result of query operation."""
    results: List[Any]
    total_count: int
    layer_type: LayerType
    metadata: Dict[str, Any] = field(default_factory=dict)
    query_time: float = 0.0


@dataclass
class LayerStats:
    """Statistics for a memory layer."""
    layer_type: LayerType
    total_items_processed: int = 0
    total_queries: int = 0
    total_errors: int = 0
    average_processing_time: float = 0.0
    last_activity: Optional[datetime] = None
    background_task_stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LayerConfig:
    """Configuration for a memory layer."""
    enabled: bool = True
    processing_mode: ProcessingMode = ProcessingMode.ASYNC
    batch_size: int = 10
    trigger_delay: float = 0.0
    background_tasks: Dict[str, Any] = field(default_factory=dict)
    storage_backends: List[str] = field(default_factory=list)
    custom_config: Dict[str, Any] = field(default_factory=dict)





class StorageBackend(ABC):
    """Abstract storage backend interface."""
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the storage backend."""
        pass
    
    @abstractmethod
    async def write(self, data: Any, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Write data and return ID."""
        pass
    
    @abstractmethod
    async def read(self, query: str, **kwargs) -> List[Any]:
        """Read data based on query."""
        pass
    
    @abstractmethod
    async def update(self, item_id: str, data: Any) -> bool:
        """Update existing data."""
        pass
    
    @abstractmethod
    async def delete(self, item_id: str) -> bool:
        """Delete data by ID."""
        pass
    
    @abstractmethod
    async def batch_write(self, items: List[Any]) -> List[str]:
        """Write multiple items in batch."""
        pass


class MemoryLayer(ABC):
    """Abstract base class for all memory layers."""
    
    def __init__(
        self,
        layer_type: LayerType,
        config: LayerConfig,
        user_id: str,
        storage_manager: Optional['StorageManager'] = None
    ):
        self.layer_type = layer_type
        self.config = config
        self.user_id = user_id
        self.storage_manager = storage_manager
        
        # Statistics
        self.total_items_processed = 0
        self.total_queries = 0
        self.total_errors = 0
        self.processing_times: List[float] = []
        self.last_activity: Optional[datetime] = None
        self.background_task_stats: Dict[str, Any] = {}
        
        # State
        self.initialized = False
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the memory layer."""
        pass
    
    @abstractmethod
    async def process_data(self, data: Any, metadata: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """Process data through this layer."""
        pass
    
    @abstractmethod
    async def query(self, query: str, **kwargs) -> List[Any]:
        """Query data from this layer."""
        pass
    
    async def get_stats(self) -> LayerStats:
        """Get layer statistics."""
        return LayerStats(
            layer_type=self.layer_type,
            total_items_processed=self.total_items_processed,
            total_queries=self.total_queries,
            total_errors=self.total_errors,
            average_processing_time=self._get_average_processing_time(),
            last_activity=self.last_activity,
            background_task_stats=self.background_task_stats.copy()
        )
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the layer."""
        self.initialized = False
    
    def _get_average_processing_time(self) -> float:
        """Calculate average processing time."""
        if not self.processing_times:
            return 0.0
        return sum(self.processing_times) / len(self.processing_times)
    
    def _update_stats(self, processing_time: float, success: bool = True) -> None:
        """Update layer statistics."""
        if success:
            self.total_items_processed += 1
        else:
            self.total_errors += 1
        
        self.processing_times.append(processing_time)
        # Keep only last 1000 processing times to prevent memory issues
        if len(self.processing_times) > 1000:
            self.processing_times = self.processing_times[-1000:]
        
        self.last_activity = datetime.utcnow()
    



class StorageManager(ABC):
    """Abstract storage manager for unified storage access."""
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize all storage backends."""
        pass
    
    @abstractmethod
    async def get_backend(self, storage_type: StorageType) -> Optional[StorageBackend]:
        """Get a storage backend by type."""
        pass
    
    @abstractmethod
    async def write_to_all(self, data: Any, metadata: Optional[Dict[str, Any]] = None) -> Dict[StorageType, Optional[str]]:
        """Write data to all available storage backends."""
        pass
    
    @abstractmethod
    async def write_to_backend(self, storage_type: StorageType, data: Any, metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Write data to a specific storage backend."""
        pass
    
    @abstractmethod
    async def read_from_backend(self, storage_type: StorageType, query: str, **kwargs) -> List[Any]:
        """Read data from a specific storage backend."""
        pass
    
    @abstractmethod
    def get_available_backends(self) -> List[StorageType]:
        """Get list of available storage backends."""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown all storage backends."""
        pass
