"""
Parallel Memory Layer Manager for MemFuse.

This module implements a unified interface for parallel writing to L0/L1/L2 layers,
providing improved performance through concurrent processing while maintaining
data consistency and error handling.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager

from .core import LayerType, ProcessingResult
from .manager import MemoryHierarchyManager
from .types import WriteStrategy, RetryConfig, LayerWriteResult, ParallelWriteResult
from ..utils.config import ConfigManager

logger = logging.getLogger(__name__)


class ParallelMemoryLayerManager:
    """
    Unified manager for parallel memory layer operations.
    
    This manager coordinates writes across L0/L1/L2 layers with different
    strategies to optimize performance while maintaining consistency.
    """
    
    def __init__(
        self,
        hierarchy_manager: Optional[MemoryHierarchyManager] = None,
        config_manager: Optional[ConfigManager] = None,
        default_strategy: Optional[WriteStrategy] = None,
        max_concurrent_layers: Optional[int] = None,
        timeout_per_layer: Optional[float] = None,
        retry_config: Optional[RetryConfig] = None
    ):
        """Initialize the parallel manager.

        Args:
            hierarchy_manager: Underlying hierarchy manager
            config_manager: Configuration manager
            default_strategy: Default write strategy
            max_concurrent_layers: Maximum concurrent layer operations
            timeout_per_layer: Timeout for each layer operation
            retry_config: Retry configuration
        """
        self.hierarchy_manager = hierarchy_manager
        self.config_manager = config_manager

        # Configuration with defaults
        self.default_strategy = default_strategy or WriteStrategy.PARALLEL
        self.max_concurrent_layers = max_concurrent_layers or 3
        self.timeout_per_layer = timeout_per_layer or 30.0
        self.retry_config = retry_config or RetryConfig()

        # Concurrency control
        self.layer_semaphore = asyncio.Semaphore(self.max_concurrent_layers)

        # Connection pool management
        self._connection_pools: Dict[str, Any] = {}
        self._pool_locks: Dict[str, asyncio.Lock] = {}

        # Statistics
        self.total_operations = 0
        self.successful_operations = 0
        self.failed_operations = 0
        self.total_processing_time = 0.0
        self.total_retries = 0

        logger.info(f"ParallelMemoryLayerManager: Initialized with strategy={self.default_strategy.value}, "
                   f"max_retries={self.retry_config.max_retries}")
    
    async def initialize(self) -> bool:
        """Initialize the parallel manager."""
        try:
            # Initialize underlying hierarchy manager
            if self.hierarchy_manager and not await self.hierarchy_manager.initialize():
                logger.error("ParallelMemoryLayerManager: Failed to initialize hierarchy manager")
                return False
            
            logger.info("ParallelMemoryLayerManager: Initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"ParallelMemoryLayerManager: Initialization failed: {e}")
            return False
    
    async def write_data(
        self,
        data: Any,
        metadata: Optional[Dict[str, Any]] = None,
        strategy: Optional[WriteStrategy] = None
    ) -> ParallelWriteResult:
        """
        Write data to memory layers using specified strategy.
        
        Args:
            data: Data to write to memory layers
            metadata: Optional metadata for the operation
            strategy: Write strategy to use (defaults to configured strategy)
            
        Returns:
            ParallelWriteResult with aggregated results from all layers
        """
        start_time = time.time()
        strategy = strategy or self.default_strategy
        
        try:
            self.total_operations += 1
            logger.info(f"ParallelMemoryLayerManager: Writing data using {strategy.value} strategy")
            
            # Execute write based on strategy
            if strategy == WriteStrategy.PARALLEL:
                result = await self._write_parallel(data, metadata)
            elif strategy == WriteStrategy.SEQUENTIAL:
                result = await self._write_sequential(data, metadata)
            elif strategy == WriteStrategy.HYBRID:
                result = await self._write_hybrid(data, metadata)
            else:
                raise ValueError(f"Unknown write strategy: {strategy}")
            
            # Update statistics
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            
            if result.success:
                self.successful_operations += 1
            else:
                self.failed_operations += 1
            
            # Update result with actual processing time
            result.total_processing_time = processing_time
            result.strategy_used = strategy
            
            logger.info(f"ParallelMemoryLayerManager: Write completed in {processing_time:.3f}s, "
                       f"success={result.success}")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.failed_operations += 1
            self.total_processing_time += processing_time
            
            logger.error(f"ParallelMemoryLayerManager: Write failed: {e}")
            
            return ParallelWriteResult(
                success=False,
                layer_results={},
                total_processing_time=processing_time,
                strategy_used=strategy,
                error_message=str(e)
            )
    
    async def _write_parallel(
        self,
        data: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ParallelWriteResult:
        """Write to all layers in parallel."""
        logger.debug("ParallelMemoryLayerManager: Executing parallel write strategy")
        
        if not self.hierarchy_manager:
            return ParallelWriteResult(
                success=False,
                layer_results={},
                total_processing_time=0.0,
                strategy_used=WriteStrategy.PARALLEL,
                error_message="No hierarchy manager available"
            )
        
        # Create tasks for all available layers
        tasks = []
        layer_types = []
        
        for layer_type, layer in self.hierarchy_manager.layers.items():
            if layer and layer.initialized:
                task = asyncio.create_task(
                    self._write_to_layer(layer_type, layer, data, metadata)
                )
                tasks.append(task)
                layer_types.append(layer_type)
        
        if not tasks:
            logger.warning("ParallelMemoryLayerManager: No initialized layers available")
            return ParallelWriteResult(
                success=False,
                layer_results={},
                total_processing_time=0.0,
                strategy_used=WriteStrategy.PARALLEL,
                error_message="No initialized layers available"
            )
        
        # Execute all tasks concurrently
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            layer_results = {}
            overall_success = True
            
            for i, result in enumerate(results):
                layer_type = layer_types[i]

                if isinstance(result, Exception):
                    logger.error(f"ParallelMemoryLayerManager: Layer {layer_type.value} failed: {result}")
                    layer_results[layer_type] = LayerWriteResult(
                        success=False,
                        result=None,
                        processed_items=[],
                        processing_time=0.0,
                        retry_count=0,
                        error_message=str(result)
                    )
                    overall_success = False
                else:
                    layer_results[layer_type] = result
                    if not result.success:
                        # Get error message based on result type
                        if hasattr(result, 'errors') and result.errors:
                            # ProcessingResult has errors list
                            error_msg = "; ".join(result.errors)
                        elif hasattr(result, 'error_message') and result.error_message:
                            # LayerWriteResult has error_message string
                            error_msg = result.error_message
                        else:
                            error_msg = "Processing failed"
                        logger.error(f"ParallelMemoryLayerManager: Layer {layer_type.value} processing failed: {error_msg}")
                        overall_success = False
                    else:
                        logger.info(f"ParallelMemoryLayerManager: Layer {layer_type.value} processing succeeded: {len(result.processed_items)} items")
            
            return ParallelWriteResult(
                success=overall_success,
                layer_results=layer_results,
                total_processing_time=0.0,  # Will be set by caller
                strategy_used=WriteStrategy.PARALLEL,
                error_message=None if overall_success else "Some layers failed"
            )
            
        except Exception as e:
            logger.error(f"ParallelMemoryLayerManager: Parallel write failed: {e}")
            raise
    
    async def _write_sequential(
        self,
        data: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ParallelWriteResult:
        """Write to layers sequentially (L0 -> L1 -> L2)."""
        logger.debug("ParallelMemoryLayerManager: Executing sequential write strategy")
        
        if not self.hierarchy_manager:
            return ParallelWriteResult(
                success=False,
                layer_results={},
                total_processing_time=0.0,
                strategy_used=WriteStrategy.SEQUENTIAL,
                error_message="No hierarchy manager available"
            )
        
        layer_results = {}
        overall_success = True
        
        # Process layers in order: L0, L1, L2
        for layer_type in [LayerType.L0, LayerType.L1, LayerType.L2]:
            layer = self.hierarchy_manager.layers.get(layer_type)
            if layer and layer.initialized:
                try:
                    result = await self._write_to_layer(layer_type, layer, data, metadata)
                    layer_results[layer_type] = result
                    
                    if not result.success:
                        overall_success = False
                        logger.warning(f"ParallelMemoryLayerManager: Layer {layer_type.value} failed, "
                                     f"continuing with remaining layers")
                    
                except Exception as e:
                    logger.error(f"ParallelMemoryLayerManager: Layer {layer_type.value} failed: {e}")
                    layer_results[layer_type] = LayerWriteResult(
                        success=False,
                        result=None,
                        processed_items=[],
                        processing_time=0.0,
                        retry_count=0,
                        error_message=str(e)
                    )
                    overall_success = False
        
        return ParallelWriteResult(
            success=overall_success,
            layer_results=layer_results,
            total_processing_time=0.0,  # Will be set by caller
            strategy_used=WriteStrategy.SEQUENTIAL,
            error_message=None if overall_success else "Some layers failed"
        )
    
    async def _write_hybrid(
        self,
        data: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ParallelWriteResult:
        """Write L0 first, then L1/L2 in parallel."""
        logger.debug("ParallelMemoryLayerManager: Executing hybrid write strategy")
        
        if not self.hierarchy_manager:
            return ParallelWriteResult(
                success=False,
                layer_results={},
                total_processing_time=0.0,
                strategy_used=WriteStrategy.HYBRID,
                error_message="No hierarchy manager available"
            )
        
        layer_results = {}
        overall_success = True
        
        # Step 1: Write to L0 first
        l0_layer = self.hierarchy_manager.layers.get(LayerType.L0)
        if l0_layer and l0_layer.initialized:
            try:
                l0_result = await self._write_to_layer(LayerType.L0, l0_layer, data, metadata)
                layer_results[LayerType.L0] = l0_result
                
                if not l0_result.success:
                    overall_success = False
                    logger.warning("ParallelMemoryLayerManager: L0 write failed in hybrid strategy")
                
            except Exception as e:
                logger.error(f"ParallelMemoryLayerManager: L0 write failed: {e}")
                layer_results[LayerType.L0] = LayerWriteResult(
                    success=False,
                    result=None,
                    processed_items=[],
                    processing_time=0.0,
                    retry_count=0,
                    error_message=str(e)
                )
                overall_success = False
        
        # Step 2: Write to L1 and L2 in parallel
        l1_l2_tasks = []
        l1_l2_types = []
        
        for layer_type in [LayerType.L1, LayerType.L2]:
            layer = self.hierarchy_manager.layers.get(layer_type)
            if layer and layer.initialized:
                task = asyncio.create_task(
                    self._write_to_layer(layer_type, layer, data, metadata)
                )
                l1_l2_tasks.append(task)
                l1_l2_types.append(layer_type)
        
        if l1_l2_tasks:
            try:
                l1_l2_results = await asyncio.gather(*l1_l2_tasks, return_exceptions=True)
                
                for i, result in enumerate(l1_l2_results):
                    layer_type = l1_l2_types[i]
                    
                    if isinstance(result, Exception):
                        logger.error(f"ParallelMemoryLayerManager: Layer {layer_type.value} failed: {result}")
                        layer_results[layer_type] = LayerWriteResult(
                            success=False,
                            result=None,
                            processed_items=[],
                            processing_time=0.0,
                            retry_count=0,
                            error_message=str(result)
                        )
                        overall_success = False
                    else:
                        layer_results[layer_type] = result
                        if not result.success:
                            overall_success = False
                
            except Exception as e:
                logger.error(f"ParallelMemoryLayerManager: L1/L2 parallel write failed: {e}")
                overall_success = False
        
        return ParallelWriteResult(
            success=overall_success,
            layer_results=layer_results,
            total_processing_time=0.0,  # Will be set by caller
            strategy_used=WriteStrategy.HYBRID,
            error_message=None if overall_success else "Some layers failed"
        )

    async def _write_to_layer(
        self,
        layer_type: LayerType,
        layer: Any,
        data: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> LayerWriteResult:
        """Write data to a specific layer with timeout, retry, and error handling."""
        start_time = time.time()
        retry_count = 0
        last_error = None

        async with self.layer_semaphore:  # Limit concurrent layer operations
            while retry_count <= self.retry_config.max_retries:
                try:
                    logger.debug(f"ParallelMemoryLayerManager: Writing to {layer_type.value} layer "
                               f"(attempt {retry_count + 1}/{self.retry_config.max_retries + 1})")

                    # Execute layer processing with timeout
                    result = await asyncio.wait_for(
                        layer.process_data(data, metadata),
                        timeout=self.timeout_per_layer
                    )

                    processing_time = time.time() - start_time

                    # Convert ProcessingResult to LayerWriteResult
                    if hasattr(result, 'success'):
                        # It's a ProcessingResult object
                        return LayerWriteResult(
                            success=result.success,
                            result=str(result.result) if hasattr(result, 'result') else None,
                            processed_items=result.processed_items if hasattr(result, 'processed_items') else [],
                            processing_time=processing_time,
                            retry_count=retry_count,
                            error_message=None
                        )
                    else:
                        # It's a simple return value (string, etc.)
                        return LayerWriteResult(
                            success=True,
                            result=str(result),
                            processed_items=[],
                            processing_time=processing_time,
                            retry_count=retry_count,
                            error_message=None
                        )

                except asyncio.TimeoutError as e:
                    last_error = e
                    if not self.retry_config.retry_on_timeout or retry_count >= self.retry_config.max_retries:
                        break

                    logger.warning(f"ParallelMemoryLayerManager: Layer {layer_type.value} timed out, "
                                 f"retrying in {self._calculate_retry_delay(retry_count)}s")

                except Exception as e:
                    last_error = e
                    if not self.retry_config.retry_on_error or retry_count >= self.retry_config.max_retries:
                        break

                    logger.warning(f"ParallelMemoryLayerManager: Layer {layer_type.value} failed: {e}, "
                                 f"retrying in {self._calculate_retry_delay(retry_count)}s")

                # Wait before retry
                if retry_count < self.retry_config.max_retries:
                    delay = self._calculate_retry_delay(retry_count)
                    await asyncio.sleep(delay)
                    retry_count += 1
                    self.total_retries += 1

            # All retries exhausted
            processing_time = time.time() - start_time

            if isinstance(last_error, asyncio.TimeoutError):
                error_msg = f"Layer {layer_type.value} write timed out after {retry_count + 1} attempts"
            else:
                error_msg = f"Layer {layer_type.value} write failed after {retry_count + 1} attempts: {str(last_error)}"

            logger.error(f"ParallelMemoryLayerManager: {error_msg}")

            return LayerWriteResult(
                success=False,
                result=None,
                processed_items=[],
                processing_time=processing_time,
                retry_count=retry_count,
                error_message=error_msg
            )

    def _calculate_retry_delay(self, retry_count: int) -> float:
        """Calculate delay for retry based on configuration."""
        if not self.retry_config.exponential_backoff:
            return self.retry_config.base_delay

        delay = self.retry_config.base_delay * (2 ** retry_count)
        return min(delay, self.retry_config.max_delay)

    def get_statistics(self) -> Dict[str, Any]:
        """Get manager statistics."""
        return {
            "total_operations": self.total_operations,
            "successful_operations": self.successful_operations,
            "failed_operations": self.failed_operations,
            "success_rate": self.successful_operations / max(self.total_operations, 1),
            "average_processing_time": self.total_processing_time / max(self.total_operations, 1),
            "total_processing_time": self.total_processing_time,
            "total_retries": self.total_retries,
            "default_strategy": self.default_strategy.value,
            "max_concurrent_layers": self.max_concurrent_layers,
            "timeout_per_layer": self.timeout_per_layer
        }

    async def shutdown(self) -> None:
        """Shutdown the parallel manager."""
        try:
            logger.info("ParallelMemoryLayerManager: Shutting down...")

            # Shutdown hierarchy manager
            if self.hierarchy_manager and hasattr(self.hierarchy_manager, 'shutdown'):
                await self.hierarchy_manager.shutdown()

            logger.info("ParallelMemoryLayerManager: Shutdown completed")

        except Exception as e:
            logger.error(f"ParallelMemoryLayerManager: Shutdown failed: {e}")
            raise
