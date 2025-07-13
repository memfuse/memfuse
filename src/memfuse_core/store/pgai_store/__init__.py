"""
PgAI Store Package

This package contains all PostgreSQL + pgai related store implementations
for MemFuse's M0 episodic memory layer.

Key Components:
- PgaiStore: Base pgai store implementation
- SimplifiedEventDrivenPgaiStore: Event-driven store with immediate triggers
- ImmediateTriggerComponents: Components for immediate trigger system
- EmbeddingMonitor: Performance monitoring
- PgaiStoreFactory: Store factory for automatic selection
"""

from .pgai_store import PgaiStore
from .simplified_event_driven_store import SimplifiedEventDrivenPgaiStore, EventDrivenPgaiStore
from .store_factory import PgaiStoreFactory, create_pgai_store
from .monitoring import EmbeddingMonitor
from .immediate_trigger_components import (
    TriggerManager,
    RetryProcessor, 
    WorkerPool,
    ImmediateTriggerCoordinator
)

__all__ = [
    # Core stores
    "PgaiStore",
    "SimplifiedEventDrivenPgaiStore", 
    "EventDrivenPgaiStore",  # Backward compatibility alias
    
    # Factory
    "PgaiStoreFactory",
    "create_pgai_store",
    
    # Monitoring
    "EmbeddingMonitor",
    
    # Immediate trigger components
    "TriggerManager",
    "RetryProcessor",
    "WorkerPool", 
    "ImmediateTriggerCoordinator",
]
