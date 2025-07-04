"""
Unified memory manager for the MemFuse hierarchy system.

This module provides the main coordination layer that manages all memory
layers, storage backends, and event-driven processing.
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
from loguru import logger

from .core import LayerType, LayerConfig, ProcessingResult, StorageManager
from .storage import UnifiedStorageManager
from .layers import L0EpisodicLayer, L1SemanticLayer, L2RelationalLayer


class MemoryHierarchyManager:
    """
    Unified memory hierarchy manager.
    
    This manager coordinates all memory layers, storage backends, and
    event-driven processing to provide a cohesive memory system.
    
    Features:
    - Unified layer management
    - Event-driven inter-layer communication
    - Centralized storage management
    - Comprehensive statistics and monitoring
    - Graceful initialization and shutdown
    """
    
    def __init__(self, user_id: str, config: Dict[str, Any]):
        """
        Initialize the memory hierarchy manager.
        
        Args:
            user_id: User ID for user-specific memory
            config: Complete memory configuration
        """
        self.user_id = user_id
        self.config = config
        self.initialized = False
        
        # Core components
        self.storage_manager: Optional[StorageManager] = None
        self.layers: Dict[LayerType, Any] = {}
        
        # Statistics
        self.total_operations = 0
        self.total_errors = 0
        self.start_time = datetime.utcnow()
        
        logger.info(f"MemoryHierarchyManager: Created for user {user_id}")
    
    async def initialize(self) -> bool:
        """
        Initialize the complete memory hierarchy.
        
        Returns:
            True if initialization successful
        """
        try:
            logger.info(f"MemoryHierarchyManager: Initializing for user {self.user_id}")
            
            # Initialize storage manager
            storage_config = self.config.get("storage", {})
            self.storage_manager = UnifiedStorageManager(storage_config, self.user_id)
            
            if not await self.storage_manager.initialize():
                logger.error("MemoryHierarchyManager: Storage manager initialization failed")
                return False
            
            # Initialize memory layers
            layers_config = self.config.get("layers", {})
            
            # L0 Layer
            if layers_config.get("l0", {}).get("enabled", True):
                l0_config = self._create_layer_config(layers_config.get("l0", {}))
                self.layers[LayerType.L0] = L0EpisodicLayer(
                    LayerType.L0, l0_config, self.user_id,
                    self.storage_manager
                )
                await self.layers[LayerType.L0].initialize()

            # L1 Layer
            if layers_config.get("l1", {}).get("enabled", True):
                l1_config = self._create_layer_config(layers_config.get("l1", {}))
                self.layers[LayerType.L1] = L1SemanticLayer(
                    LayerType.L1, l1_config, self.user_id,
                    self.storage_manager
                )
                await self.layers[LayerType.L1].initialize()

            # L2 Layer
            if layers_config.get("l2", {}).get("enabled", True):
                l2_config = self._create_layer_config(layers_config.get("l2", {}))
                self.layers[LayerType.L2] = L2RelationalLayer(
                    LayerType.L2, l2_config, self.user_id,
                    self.storage_manager
                )
                await self.layers[LayerType.L2].initialize()
            
            self.initialized = True
            logger.info(f"MemoryHierarchyManager: Initialized {len(self.layers)} layers successfully")
            return True
            
        except Exception as e:
            logger.error(f"MemoryHierarchyManager: Initialization failed: {e}")
            return False
    
    async def process_data(self, data: Any, metadata: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """
        Process data through the memory hierarchy.
        
        Data flows: Input -> L0 -> (event) -> L1 -> (event) -> L2
        
        Args:
            data: Data to process
            metadata: Optional metadata
            
        Returns:
            ProcessingResult from L0 layer (downstream processing is async)
        """
        try:
            if not self.initialized:
                await self.initialize()
            
            # Process through L0 layer (entry point)
            l0_layer = self.layers.get(LayerType.L0)
            if not l0_layer:
                raise ValueError("L0 layer not available")
            
            result = await l0_layer.process_data(data, metadata)
            
            # Update statistics
            self.total_operations += 1
            if not result.success:
                self.total_errors += 1
            
            logger.debug(f"MemoryHierarchyManager: Processed data through L0, success={result.success}")
            return result
            
        except Exception as e:
            self.total_errors += 1
            logger.error(f"MemoryHierarchyManager: Data processing failed: {e}")
            
            return ProcessingResult(
                success=False,
                layer_type=LayerType.L0,
                errors=[str(e)]
            )
    
    async def query(self, query: str, layers: Optional[List[LayerType]] = None, **kwargs) -> Dict[LayerType, List[Any]]:
        """
        Query data from memory layers.
        
        Args:
            query: Query string
            layers: Specific layers to query (default: all available)
            **kwargs: Additional query parameters
            
        Returns:
            Dictionary mapping layer types to results
        """
        try:
            if not self.initialized:
                await self.initialize()
            
            # Default to all available layers
            if layers is None:
                layers = list(self.layers.keys())
            
            results = {}
            
            # Query each requested layer
            for layer_type in layers:
                layer = self.layers.get(layer_type)
                if layer:
                    layer_results = await layer.query(query, **kwargs)
                    results[layer_type] = layer_results
                else:
                    results[layer_type] = []
            
            logger.debug(f"MemoryHierarchyManager: Query returned results from {len(results)} layers")
            return results
            
        except Exception as e:
            logger.error(f"MemoryHierarchyManager: Query failed: {e}")
            return {}
    
    async def get_layer_stats(self, layer_type: LayerType) -> Optional[Any]:
        """Get statistics for a specific layer."""
        layer = self.layers.get(layer_type)
        if layer:
            return await layer.get_stats()
        return None
    
    def get_available_layers(self) -> List[LayerType]:
        """Get list of available memory layers."""
        return list(self.layers.keys())
    
    async def get_all_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics for the entire memory hierarchy."""
        try:
            # Collect layer stats
            layer_stats = {}
            for layer_type, layer in self.layers.items():
                stats = await layer.get_stats()
                layer_stats[layer_type.value] = {
                    "total_items_processed": stats.total_items_processed,
                    "total_queries": stats.total_queries,
                    "total_errors": stats.total_errors,
                    "average_processing_time": stats.average_processing_time,
                    "last_activity": stats.last_activity
                }
            
            # Collect storage stats
            storage_stats = {}
            if self.storage_manager:
                storage_stats = self.storage_manager.get_stats()
            
            # Collect event bus stats
            event_bus_stats = {}
            if self.event_bus:
                event_bus_stats = self.event_bus.get_stats()
            
            return {
                "user_id": self.user_id,
                "initialized": self.initialized,
                "available_layers": [lt.value for lt in self.layers.keys()],
                "total_operations": self.total_operations,
                "total_errors": self.total_errors,
                "uptime_seconds": (datetime.utcnow() - self.start_time).total_seconds(),
                "layer_stats": layer_stats,
                "storage_stats": storage_stats,
                "event_bus_stats": event_bus_stats
            }
            
        except Exception as e:
            logger.error(f"MemoryHierarchyManager: Error collecting stats: {e}")
            return {"error": str(e)}
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the memory hierarchy."""
        try:
            logger.info(f"MemoryHierarchyManager: Shutting down for user {self.user_id}")
            
            # Shutdown layers
            for layer_type, layer in self.layers.items():
                try:
                    await layer.shutdown()
                    logger.info(f"MemoryHierarchyManager: Shutdown {layer_type.value} layer")
                except Exception as e:
                    logger.error(f"MemoryHierarchyManager: Error shutting down {layer_type.value}: {e}")
            
            # Shutdown storage manager
            if self.storage_manager:
                await self.storage_manager.shutdown()
            
            # Clear state
            self.layers.clear()
            self.initialized = False
            
            logger.info(f"MemoryHierarchyManager: Shutdown complete for user {self.user_id}")
            
        except Exception as e:
            logger.error(f"MemoryHierarchyManager: Shutdown error: {e}")
    
    def _create_layer_config(self, config_dict: Dict[str, Any]) -> LayerConfig:
        """Create LayerConfig from dictionary."""
        from .core import ProcessingMode
        
        return LayerConfig(
            enabled=config_dict.get("enabled", True),
            processing_mode=ProcessingMode(config_dict.get("processing_mode", "async")),
            batch_size=config_dict.get("batch_size", 10),
            trigger_delay=config_dict.get("trigger_delay", 0.0),
            background_tasks=config_dict.get("background_tasks", {}),
            storage_backends=config_dict.get("storage_backends", []),
            custom_config=config_dict
        )


# Convenience function for creating and initializing a memory manager
async def create_memory_manager(user_id: str, config: Dict[str, Any]) -> MemoryHierarchyManager:
    """
    Create and initialize a memory hierarchy manager.
    
    Args:
        user_id: User ID for user-specific memory
        config: Complete memory configuration
        
    Returns:
        Initialized MemoryHierarchyManager
    """
    manager = MemoryHierarchyManager(user_id, config)
    
    if not await manager.initialize():
        raise RuntimeError(f"Failed to initialize memory manager for user {user_id}")
    
    return manager
