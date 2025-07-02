"""
Parallel Memory Adapter for MemoryService integration.

This adapter integrates the ParallelMemoryLayerManager with the existing
MemoryService, providing backward compatibility while enabling parallel
processing capabilities.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from ..hierarchy.parallel_manager import ParallelMemoryLayerManager
from ..hierarchy.types import WriteStrategy, ParallelWriteResult, RetryConfig
from ..hierarchy.manager import MemoryHierarchyManager
from ..hierarchy.config import ConfigManager
from ..hierarchy.config_loader import ConfigLoader
from ..interfaces import MessageBatchList

logger = logging.getLogger(__name__)


class ParallelMemoryAdapter:
    """
    Adapter that integrates parallel memory processing with MemoryService.
    
    This adapter provides a bridge between the existing MemoryService interface
    and the new parallel processing capabilities, allowing for gradual migration
    and performance optimization.
    """
    
    def __init__(
        self,
        user_id: str,
        config: Optional[Dict[str, Any]] = None,
        write_strategy: WriteStrategy = WriteStrategy.PARALLEL,
        config_manager: Optional[ConfigManager] = None
    ):
        """Initialize the parallel memory adapter.

        Args:
            user_id: User identifier
            config: Configuration dictionary (legacy support)
            write_strategy: Default write strategy to use
            config_manager: Configuration manager (preferred over config dict)
        """
        self.user_id = user_id
        self.write_strategy = write_strategy

        # Initialize configuration
        if config_manager:
            self.config_manager = config_manager
            self.config = config_manager.get_config().to_dict()
        else:
            # Legacy configuration support
            self.config = self._validate_and_normalize_config(config or {})
            # Create ConfigManager from legacy config
            config_loader = ConfigLoader()
            self.config_manager = config_loader.from_dict(self.config)

        # Initialize managers
        self.hierarchy_manager: Optional[MemoryHierarchyManager] = None
        self.parallel_manager: Optional[ParallelMemoryLayerManager] = None

        # State tracking
        self.initialized = False
        self.total_operations = 0
        self.successful_operations = 0
        self.failed_operations = 0
        self.total_processing_time = 0.0

        # Error tracking
        self.recent_errors: List[Dict[str, Any]] = []
        self.max_recent_errors = 100

        logger.info(f"ParallelMemoryAdapter: Initialized for user {user_id} with strategy {write_strategy.value}")

    def _validate_and_normalize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize configuration."""
        normalized = config.copy()

        # Set defaults for parallel processing
        if "parallel" not in normalized:
            normalized["parallel"] = {}

        parallel_config = normalized["parallel"]
        parallel_config.setdefault("max_concurrent_layers", 3)
        parallel_config.setdefault("timeout_per_layer", 30.0)
        parallel_config.setdefault("enable_retry", True)

        # Set defaults for retry configuration
        if "retry" not in normalized:
            normalized["retry"] = {}

        retry_config = normalized["retry"]
        retry_config.setdefault("max_retries", 3)
        retry_config.setdefault("base_delay", 1.0)
        retry_config.setdefault("max_delay", 10.0)
        retry_config.setdefault("exponential_backoff", True)
        retry_config.setdefault("retry_on_timeout", True)
        retry_config.setdefault("retry_on_error", True)

        # Set defaults for hierarchy
        if "hierarchy" not in normalized:
            normalized["hierarchy"] = {}

        return normalized
    
    async def initialize(self) -> bool:
        """Initialize the parallel memory adapter."""
        try:
            if self.initialized:
                return True
            
            # Initialize hierarchy manager
            hierarchy_config = self.config.get("hierarchy", {})
            self.hierarchy_manager = MemoryHierarchyManager(
                user_id=self.user_id,
                config=hierarchy_config
            )
            
            if not await self.hierarchy_manager.initialize():
                logger.error("ParallelMemoryAdapter: Failed to initialize hierarchy manager")
                return False
            
            # Initialize parallel manager with ConfigManager
            self.parallel_manager = ParallelMemoryLayerManager(
                hierarchy_manager=self.hierarchy_manager,
                config_manager=self.config_manager,
                default_strategy=self.write_strategy
            )
            
            if not await self.parallel_manager.initialize():
                logger.error("ParallelMemoryAdapter: Failed to initialize parallel manager")
                return False
            
            self.initialized = True
            logger.info("ParallelMemoryAdapter: Initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"ParallelMemoryAdapter: Initialization failed: {e}")
            return False
    
    async def process_message_batch(
        self,
        message_batch_list: MessageBatchList,
        session_id: Optional[str] = None,
        strategy: Optional[WriteStrategy] = None
    ) -> Dict[str, Any]:
        """
        Process a batch of messages using parallel memory layers.
        
        Args:
            message_batch_list: List of message lists to process
            session_id: Optional session identifier
            strategy: Write strategy to use (overrides default)
            
        Returns:
            Dictionary with processing results and statistics
        """
        try:
            if not self.initialized:
                await self.initialize()
            
            if not self.parallel_manager:
                raise RuntimeError("Parallel manager not initialized")
            
            self.total_operations += 1
            
            # Prepare metadata
            metadata = {
                "user_id": self.user_id,
                "session_id": session_id,
                "operation_type": "message_batch_processing",
                "batch_size": len(message_batch_list)
            }
            
            # Convert MessageBatchList to format suitable for parallel processing
            processed_data = self._prepare_data_for_layers(message_batch_list, metadata)
            
            # Execute parallel write
            result = await self.parallel_manager.write_data(
                data=processed_data,
                metadata=metadata,
                strategy=strategy
            )
            
            # Update statistics
            if result.success:
                self.successful_operations += 1
            
            # Convert result to MemoryService-compatible format
            return self._convert_result_to_memory_service_format(result, message_batch_list)
            
        except Exception as e:
            self.failed_operations += 1
            self._record_error("process_message_batch", str(e), {
                "batch_size": len(message_batch_list),
                "session_id": session_id,
                "strategy": (strategy or self.write_strategy).value
            })

            logger.error(f"ParallelMemoryAdapter: Message batch processing failed: {e}")
            return {
                "status": "error",
                "message": f"Parallel processing failed: {str(e)}",
                "data": [],
                "chunk_count": 0,
                "processing_time": 0.0,
                "strategy_used": (strategy or self.write_strategy).value,
                "error_details": {
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            }

    def _record_error(self, operation: str, error_message: str, context: Dict[str, Any]) -> None:
        """Record error for tracking and analysis."""
        error_record = {
            "timestamp": time.time(),
            "operation": operation,
            "error_message": error_message,
            "context": context,
            "user_id": self.user_id
        }

        self.recent_errors.append(error_record)

        # Keep only recent errors
        if len(self.recent_errors) > self.max_recent_errors:
            self.recent_errors = self.recent_errors[-self.max_recent_errors:]
    
    def _prepare_data_for_layers(
        self,
        message_batch_list: MessageBatchList,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare message batch data for parallel layer processing."""
        return {
            "message_batches": message_batch_list,
            "metadata": metadata,
            "processing_type": "incremental_write"
        }
    
    def _convert_result_to_memory_service_format(
        self,
        result: ParallelWriteResult,
        original_batch: MessageBatchList
    ) -> Dict[str, Any]:
        """Convert parallel processing result to MemoryService format."""
        
        # Aggregate processed items from all layers
        all_processed_items = []
        for layer_result in result.layer_results.values():
            all_processed_items.extend(layer_result.processed_items)
        
        # Calculate chunk count (estimate based on message count)
        estimated_chunk_count = sum(len(batch) for batch in original_batch)
        
        # Prepare detailed layer information
        layer_info = {}
        for layer_type, layer_result in result.layer_results.items():
            layer_info[layer_type.value] = {
                "success": layer_result.success,
                "processing_time": layer_result.processing_time,
                "processed_items": len(layer_result.processed_items),
                "errors": len(layer_result.errors) if hasattr(layer_result, 'errors') else 0,
                "error_messages": layer_result.errors if hasattr(layer_result, 'errors') else []
            }
        
        return {
            "status": "success" if result.success else "partial_success",
            "message": f"Processed {len(original_batch)} message batches using {result.strategy_used.value} strategy",
            "data": all_processed_items,
            "chunk_count": estimated_chunk_count,
            "processing_time": result.total_processing_time,
            "strategy_used": result.strategy_used.value,
            "layer_results": layer_info,
            "total_processed_items": result.total_processed_items,
            "total_errors": result.total_errors,
            "parallel_processing": True
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get adapter statistics."""
        base_stats = {
            "total_operations": self.total_operations,
            "successful_operations": self.successful_operations,
            "success_rate": self.successful_operations / max(self.total_operations, 1),
            "write_strategy": self.write_strategy.value,
            "initialized": self.initialized
        }
        
        # Add parallel manager statistics if available
        if self.parallel_manager:
            parallel_stats = self.parallel_manager.get_statistics()
            base_stats.update({"parallel_manager": parallel_stats})
        
        return base_stats
    
    async def shutdown(self) -> None:
        """Shutdown the adapter and all managers."""
        try:
            logger.info("ParallelMemoryAdapter: Shutting down...")
            
            if self.parallel_manager:
                await self.parallel_manager.shutdown()
            
            self.initialized = False
            logger.info("ParallelMemoryAdapter: Shutdown completed")
            
        except Exception as e:
            logger.error(f"ParallelMemoryAdapter: Shutdown failed: {e}")
            raise
