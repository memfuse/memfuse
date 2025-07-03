"""
Unified Memory Layer Implementation for MemFuse.

This implementation provides the concrete implementation of UnifiedMemoryLayer,
coordinating parallel processing across L0/L1/L2 memory layers.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from ..interfaces.unified_memory_layer import (
    UnifiedMemoryLayer,
    UnifiedMemoryLayerConfig,
    WriteResult,
    UnifiedResult,
    LayerStatus
)
from ..interfaces.message_interface import MessageBatchList
from .parallel_manager import ParallelMemoryLayerManager
from .types import WriteStrategy, ParallelWriteResult
from ..utils.config import ConfigManager


logger = logging.getLogger(__name__)


class UnifiedMemoryLayerImpl(UnifiedMemoryLayer):
    """
    Concrete implementation of UnifiedMemoryLayer.
    
    This class coordinates parallel processing across L0/L1/L2 memory layers,
    providing a unified interface for MemoryService while handling all the
    complexity of parallel processing internally.
    """
    
    def __init__(
        self,
        user_id: str,
        config_manager: Optional[ConfigManager] = None,
        config: Optional[UnifiedMemoryLayerConfig] = None
    ):
        self.user_id = user_id
        self.config_manager = config_manager or ConfigManager()
        self.config = config or UnifiedMemoryLayerConfig()
        
        # Core components
        self.hierarchy_manager: Optional[Any] = None  # MemoryHierarchyManager
        self.parallel_manager: Optional[ParallelMemoryLayerManager] = None
        
        # State tracking
        self.initialized = False
        self.layer_status = {
            "L0": LayerStatus.INACTIVE,
            "L1": LayerStatus.INACTIVE,
            "L2": LayerStatus.INACTIVE
        }
        
        # Statistics
        self.total_operations = 0
        self.successful_operations = 0
        self.failed_operations = 0
        self.last_operation_time = None
        
        logger.info(f"UnifiedMemoryLayerImpl: Created for user {user_id}")
    
    async def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """Initialize the unified memory layer."""
        try:
            if self.initialized:
                logger.info("UnifiedMemoryLayerImpl: Already initialized")
                return True
            
            logger.info("UnifiedMemoryLayerImpl: Starting initialization...")
            
            # Update configuration if provided
            if config:
                memory_service_config = config.get("memory_service", {})
                self.config = UnifiedMemoryLayerConfig(
                    l0_enabled=config.get("l0_enabled", True),
                    l1_enabled=config.get("l1_enabled", True),
                    l2_enabled=config.get("l2_enabled", True),
                    parallel_strategy=memory_service_config.get("parallel_strategy", "parallel"),
                    enable_fallback=memory_service_config.get("enable_fallback", True),
                    timeout_per_layer=memory_service_config.get("timeout_per_layer", 30.0),
                    max_retries=memory_service_config.get("max_retries", 3)
                )
            
            # Initialize hierarchy manager first
            from .manager import MemoryHierarchyManager

            # Get storage configuration from global config
            global_config = self.config_manager.get_config()
            memory_config = global_config.get("memory", {})
            storage_config = memory_config.get("storage", {})

            # Debug logging
            logger.info(f"UnifiedMemoryLayerImpl: Global config keys: {list(global_config.keys())}")
            logger.info(f"UnifiedMemoryLayerImpl: Memory config keys: {list(memory_config.keys())}")
            logger.info(f"UnifiedMemoryLayerImpl: Storage config: {storage_config}")

            hierarchy_config = {
                "layers": {
                    "l0": {"enabled": self.config.l0_enabled},
                    "l1": {"enabled": self.config.l1_enabled},
                    "l2": {"enabled": self.config.l2_enabled}
                },
                "storage": storage_config
            }

            self.hierarchy_manager = MemoryHierarchyManager(
                user_id=self.user_id,
                config=hierarchy_config
            )

            if not await self.hierarchy_manager.initialize():
                logger.error("UnifiedMemoryLayerImpl: Failed to initialize hierarchy manager")
                return False

            # Initialize parallel manager with hierarchy manager
            self.parallel_manager = ParallelMemoryLayerManager(
                hierarchy_manager=self.hierarchy_manager,
                config_manager=self.config_manager
            )
            
            # Initialize parallel manager
            if not await self.parallel_manager.initialize():
                logger.error("UnifiedMemoryLayerImpl: Failed to initialize parallel manager")
                return False
            
            # Update layer status based on configuration
            self.layer_status["L0"] = LayerStatus.ACTIVE if self.config.l0_enabled else LayerStatus.INACTIVE
            self.layer_status["L1"] = LayerStatus.ACTIVE if self.config.l1_enabled else LayerStatus.INACTIVE
            self.layer_status["L2"] = LayerStatus.ACTIVE if self.config.l2_enabled else LayerStatus.INACTIVE
            
            self.initialized = True
            logger.info("UnifiedMemoryLayerImpl: Initialization successful")
            
            # Log active layers
            active_layers = [layer for layer, status in self.layer_status.items() if status == LayerStatus.ACTIVE]
            logger.info(f"UnifiedMemoryLayerImpl: Active layers: {active_layers}")
            
            return True
            
        except Exception as e:
            logger.error(f"UnifiedMemoryLayerImpl: Initialization failed: {e}")
            return False
    
    async def write_parallel(
        self,
        message_batch_list: MessageBatchList,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> WriteResult:
        """Write data to all active memory layers in parallel."""
        try:
            if not self.initialized:
                await self.initialize()
            
            if not self.parallel_manager:
                raise RuntimeError("Parallel manager not initialized")
            
            self.total_operations += 1
            start_time = time.time()
            
            # Prepare metadata
            operation_metadata = {
                "user_id": self.user_id,
                "session_id": session_id,
                "operation_type": "parallel_write",
                "batch_size": len(message_batch_list),
                "timestamp": start_time,
                **(metadata or {})
            }
            
            logger.info(f"UnifiedMemoryLayerImpl: Starting parallel write for {len(message_batch_list)} message batches")
            
            # Convert MessageBatchList to format suitable for parallel processing
            processed_data = self._prepare_data_for_layers(message_batch_list, operation_metadata)
            
            # Execute parallel write using the configured strategy
            strategy = WriteStrategy.PARALLEL if self.config.parallel_strategy == "parallel" else WriteStrategy.SEQUENTIAL
            
            result: ParallelWriteResult = await self.parallel_manager.write_data(
                data=processed_data,
                metadata=operation_metadata,
                strategy=strategy
            )
            
            # Process results
            operation_time = time.time() - start_time
            self.last_operation_time = operation_time
            
            if result.success:
                self.successful_operations += 1
                logger.info(f"UnifiedMemoryLayerImpl: Parallel write successful in {operation_time:.2f}s")
                
                return WriteResult(
                    success=True,
                    message=f"Successfully processed {len(message_batch_list)} message batches",
                    layer_results=result.layer_results,
                    metadata={
                        "operation_time": operation_time,
                        "layers_processed": list(result.layer_results.keys()),
                        "total_chunks": result.total_processed,
                        **operation_metadata
                    }
                )
            else:
                self.failed_operations += 1
                logger.error(f"UnifiedMemoryLayerImpl: Parallel write failed: {result.error_message}")

                return WriteResult(
                    success=False,
                    message=f"Failed to process message batches: {result.error_message}",
                    layer_results=result.layer_results,
                    metadata={
                        "operation_time": operation_time,
                        "error_message": result.error_message,
                        **operation_metadata
                    }
                )
                
        except Exception as e:
            self.failed_operations += 1
            logger.error(f"UnifiedMemoryLayerImpl: Write operation failed: {e}")
            
            return WriteResult(
                success=False,
                message=f"Write operation failed: {str(e)}",
                layer_results={},
                metadata=metadata or {}
            )
    
    def _prepare_data_for_layers(
        self,
        message_batch_list: MessageBatchList,
        metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Prepare message batch data for parallel layer processing.
        
        This method converts MessageBatchList into a format that can be
        processed by the parallel memory layers.
        """
        processed_data = []
        
        for batch_idx, message_list in enumerate(message_batch_list):
            for msg_idx, message in enumerate(message_list):
                processed_item = {
                    "content": message.get("content", ""),
                    "role": message.get("role", "user"),
                    "message_id": message.get("id"),
                    "batch_index": batch_idx,
                    "message_index": msg_idx,
                    "metadata": {
                        **metadata,
                        "original_message": message
                    }
                }
                processed_data.append(processed_item)
        
        return processed_data
    
    async def query_unified(
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
    ) -> UnifiedResult:
        """Query all active memory layers and return unified results."""
        try:
            if not self.initialized:
                await self.initialize()
            
            if not self.parallel_manager:
                raise RuntimeError("Parallel manager not initialized")
            
            logger.info(f"UnifiedMemoryLayerImpl: Starting unified query: '{query[:50]}...'")
            
            # For now, delegate to parallel manager's query capabilities
            # This is a placeholder - the actual implementation would need
            # to be enhanced to support unified querying across layers
            
            # TODO: Implement unified querying across L0/L1/L2 layers
            # This would involve:
            # 1. Querying each active layer in parallel
            # 2. Aggregating results
            # 3. Applying reranking if enabled
            # 4. Returning unified result set
            
            return UnifiedResult(
                results=[],
                layer_sources={"L0": [], "L1": [], "L2": []},
                total_count=0,
                metadata={
                    "query": query,
                    "top_k": top_k,
                    "active_layers": [layer for layer, status in self.layer_status.items() if status == LayerStatus.ACTIVE]
                }
            )
            
        except Exception as e:
            logger.error(f"UnifiedMemoryLayerImpl: Query operation failed: {e}")
            
            return UnifiedResult(
                results=[],
                layer_sources={"L0": [], "L1": [], "L2": []},
                total_count=0,
                metadata={"error": str(e)}
            )
    
    async def get_layer_status(self) -> Dict[str, LayerStatus]:
        """Get the current status of all memory layers."""
        return self.layer_status.copy()
    
    async def get_layer_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all memory layers."""
        return {
            "unified_layer": {
                "total_operations": self.total_operations,
                "successful_operations": self.successful_operations,
                "failed_operations": self.failed_operations,
                "success_rate": self.successful_operations / max(self.total_operations, 1),
                "last_operation_time": self.last_operation_time,
                "initialized": self.initialized
            },
            "layer_status": {layer: status.value for layer, status in self.layer_status.items()}
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all memory layers."""
        health_status = {
            "overall_status": "healthy" if self.initialized else "unhealthy",
            "initialized": self.initialized,
            "layer_status": {layer: status.value for layer, status in self.layer_status.items()},
            "statistics": await self.get_layer_statistics(),
            "timestamp": time.time()
        }
        
        # Check parallel manager health if available
        if self.parallel_manager:
            try:
                # Add parallel manager specific health checks here
                health_status["parallel_manager"] = "available"
            except Exception as e:
                health_status["parallel_manager"] = f"error: {e}"
        else:
            health_status["parallel_manager"] = "not_initialized"
        
        return health_status
    
    async def cleanup(self) -> bool:
        """Clean up resources and shut down all memory layers."""
        try:
            logger.info("UnifiedMemoryLayerImpl: Starting cleanup...")
            
            if self.parallel_manager:
                # Cleanup parallel manager if it has cleanup method
                # await self.parallel_manager.cleanup()
                pass
            
            self.initialized = False
            self.layer_status = {layer: LayerStatus.INACTIVE for layer in self.layer_status}
            
            logger.info("UnifiedMemoryLayerImpl: Cleanup completed")
            return True
            
        except Exception as e:
            logger.error(f"UnifiedMemoryLayerImpl: Cleanup failed: {e}")
            return False
