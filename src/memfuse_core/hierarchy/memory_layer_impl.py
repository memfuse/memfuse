"""
Memory Layer Implementation for MemFuse.

This implementation provides the concrete implementation of MemoryLayer,
coordinating parallel processing across M0/M1/M2 memory layers.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional
from loguru import logger

from ..interfaces.memory_layer import (
    MemoryLayer,
    MemoryLayerConfig,
    WriteResult,
    QueryResult,
    LayerStatus
)
from ..interfaces.message_interface import MessageBatchList
from .parallel_manager import ParallelMemoryLayerManager
from .types import WriteStrategy, ParallelWriteResult, LayerWriteResult
from .core import LayerType
from ..utils.config import ConfigManager


class MemoryLayerImpl(MemoryLayer):
    """
    Concrete implementation of MemoryLayer.

    This class coordinates parallel processing across M0/M1/M2 memory layers,
    providing an interface for MemoryService while handling all the
    complexity of parallel processing internally.
    """

    def __init__(
        self,
        user_id: str,
        config_manager: Optional[ConfigManager] = None,
        config: Optional[MemoryLayerConfig] = None
    ):
        self.user_id = user_id
        self.config_manager = config_manager or ConfigManager()
        self.config = config or MemoryLayerConfig()
        
        # Core components
        self.hierarchy_manager: Optional[Any] = None  # MemoryHierarchyManager
        self.parallel_manager: Optional[ParallelMemoryLayerManager] = None
        
        # State tracking
        self.initialized = False
        self.layer_status = {
            "M0": LayerStatus.INACTIVE,
            "M1": LayerStatus.INACTIVE,
            "M2": LayerStatus.INACTIVE
        }
        
        # Statistics
        self.total_operations = 0
        self.successful_operations = 0
        self.failed_operations = 0
        self.last_operation_time = None
        
        logger.info(f"MemoryLayerImpl: Created for user {user_id}")
    
    async def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """Initialize the unified memory layer."""
        try:
            if self.initialized:
                logger.info("MemoryLayerImpl: Already initialized")
                return True
            
            logger.info("MemoryLayerImpl: Starting initialization...")

            # Update configuration if provided
            if config:
                memory_service_config = config.get("memory_service", {})
                self.config = MemoryLayerConfig(
                    m0_enabled=config.get("m0_enabled", True),
                    m1_enabled=config.get("m1_enabled", True),
                    m2_enabled=config.get("m2_enabled", True),
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
            storage_config = global_config.get("storage", {})

            # Debug logging
            logger.info(f"MemoryLayerImpl: Global config keys: {list(global_config.keys())}")
            logger.info(f"MemoryLayerImpl: Memory config keys: {list(memory_config.keys())}")
            logger.info(f"MemoryLayerImpl: Storage config: {storage_config}")

            hierarchy_config = {
                "layers": {
                    "m0": {"enabled": self.config.m0_enabled},
                    "m1": {"enabled": self.config.m1_enabled},
                    "m2": {"enabled": self.config.m2_enabled}
                },
                "storage": storage_config
            }

            self.hierarchy_manager = MemoryHierarchyManager(
                user_id=self.user_id,
                config=hierarchy_config
            )

            if not await self.hierarchy_manager.initialize():
                logger.error("MemoryLayerImpl: Failed to initialize hierarchy manager")
                return False

            # Initialize parallel manager with hierarchy manager
            self.parallel_manager = ParallelMemoryLayerManager(
                hierarchy_manager=self.hierarchy_manager,
                config_manager=self.config_manager
            )
            
            # Initialize parallel manager
            if not await self.parallel_manager.initialize():
                logger.error("MemoryLayerImpl: Failed to initialize parallel manager")
                return False

            # Update layer status based on configuration
            self.layer_status["M0"] = LayerStatus.ACTIVE if self.config.m0_enabled else LayerStatus.INACTIVE
            self.layer_status["M1"] = LayerStatus.ACTIVE if self.config.m1_enabled else LayerStatus.INACTIVE
            self.layer_status["M2"] = LayerStatus.ACTIVE if self.config.m2_enabled else LayerStatus.INACTIVE

            self.initialized = True
            logger.info("MemoryLayerImpl: Initialization successful")

            # Log active layers
            active_layers = [layer for layer, status in self.layer_status.items() if status == LayerStatus.ACTIVE]
            logger.info(f"MemoryLayerImpl: Active layers: {active_layers}")
            
            return True
            
        except Exception as e:
            logger.error(f"MemoryLayerImpl: Initialization failed: {e}")
            return False
    
    async def write_m0_priority(
        self,
        message_batch_list: MessageBatchList,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> WriteResult:
        """Write data with M0 priority + async M1/M2 trigger strategy.

        This method implements the optimized architecture:
        1. Write to M0 immediately (synchronous, fast response)
        2. Return success to user immediately
        3. Trigger M1/M2 processing asynchronously in background
        """
        try:
            if not self.initialized:
                await self.initialize()

            self.total_operations += 1
            start_time = time.time()

            # Prepare metadata
            operation_metadata = {
                "user_id": self.user_id,
                "session_id": session_id,
                "operation_type": "m0_priority_write",
                "batch_size": len(message_batch_list),
                "timestamp": start_time,
                **(metadata or {})
            }

            logger.info(f"MemoryLayerImpl: Starting M0-priority write for {len(message_batch_list)} message batches")

            # Step 1: Write to M0 immediately (synchronous)
            m0_result = await self._write_to_m0_immediate(message_batch_list, operation_metadata)

            # Step 2: Return success immediately if M0 write succeeded
            if m0_result.success:
                operation_time = time.time() - start_time
                self.last_operation_time = operation_time
                self.successful_operations += 1

                logger.info(f"MemoryLayerImpl: M0 write successful in {operation_time:.2f}s, triggering async M1/M2")

                # Step 3: Trigger M1/M2 processing asynchronously (fire-and-forget)
                asyncio.create_task(self._trigger_async_m1_m2_processing(
                    message_batch_list, operation_metadata, m0_result
                ))

                return WriteResult(
                    success=True,
                    message=f"Successfully processed {len(message_batch_list)} message batches to M0, M1/M2 processing triggered",
                    layer_results={"M0": m0_result.layer_results.get("M0", {})},
                    metadata={
                        "operation_time": operation_time,
                        "m0_immediate": True,
                        "m1_m2_async": True,
                        **operation_metadata
                    }
                )
            else:
                # M0 write failed
                operation_time = time.time() - start_time
                self.last_operation_time = operation_time
                self.failed_operations += 1

                logger.error(f"MemoryLayerImpl: M0 write failed in {operation_time:.2f}s")

                return WriteResult(
                    success=False,
                    message=f"Failed to write to M0: {m0_result.error_message}",
                    layer_results=m0_result.layer_results,
                    metadata={
                        "operation_time": operation_time,
                        "error": m0_result.error_message,
                        **operation_metadata
                    }
                )

        except Exception as e:
            operation_time = time.time() - start_time
            self.last_operation_time = operation_time
            self.failed_operations += 1

            logger.error(f"MemoryLayerImpl: M0-priority write failed: {e}")

            return WriteResult(
                success=False,
                message=f"Failed to process message batches: {str(e)}",
                layer_results={},
                metadata={
                    "operation_time": operation_time,
                    "error": str(e),
                    **(metadata or {})
                }
            )

    async def _write_to_m0_immediate(
        self,
        message_batch_list: MessageBatchList,
        metadata: Dict[str, Any]
    ) -> ParallelWriteResult:
        """Write data to M0 layer immediately for fast response."""
        try:
            if not self.hierarchy_manager:
                raise RuntimeError("Hierarchy manager not initialized")

            # Convert MessageBatchList to format suitable for M0 processing
            processed_data = self._prepare_data_for_layers(message_batch_list, metadata)

            # All data goes to M0 in this method (no filtering needed)
            m0_data = processed_data

            logger.debug(f"MemoryLayerImpl: Writing {len(m0_data)} items to M0 layer")

            m0_result = await self.hierarchy_manager.write_to_layer(
                layer_name="M0",
                data=m0_data,
                metadata=metadata
            )

            # Create a proper LayerWriteResult for M0
            # m0_result is a ProcessingResult object, not a dict
            m0_layer_result = LayerWriteResult(
                success=m0_result.success,
                result=str(m0_result.processed_items) if m0_result.processed_items else None,
                processed_items=m0_result.processed_items,
                processing_time=m0_result.processing_time,
                retry_count=0,
                error_message=m0_result.errors[0] if m0_result.errors else None
            )

            return ParallelWriteResult(
                success=m0_result.success,
                layer_results={"M0": m0_layer_result},
                total_processing_time=m0_result.processing_time,
                strategy_used=WriteStrategy.HYBRID,
                error_message=m0_result.errors[0] if m0_result.errors else None
            )

        except Exception as e:
            logger.error(f"Error writing to M0 layer immediately: {e}")
            return ParallelWriteResult(
                success=False,
                layer_results={},
                total_processing_time=0.0,
                strategy_used=WriteStrategy.HYBRID,
                error_message=str(e)
            )

    async def _trigger_async_m1_m2_processing(
        self,
        message_batch_list: MessageBatchList,
        metadata: Dict[str, Any],
        m0_result: ParallelWriteResult
    ):
        """Trigger M1/M2 processing asynchronously in background.

        This method runs in background and doesn't affect user response time.
        """
        try:
            logger.info("MemoryLayerImpl: Starting async M1/M2 processing")

            # Add async processing metadata
            async_metadata = metadata.copy()
            async_metadata.update({
                "operation_type": "async_m1_m2_processing",
                "m0_success": m0_result.success,
                "m0_processed_count": m0_result.total_processed
            })

            # Process M1 and M2 layers if enabled
            async_results = {}

            if self.config.m1_enabled and self.hierarchy_manager:
                try:
                    # FIXED: Pass original message content to M1, not just M0 IDs
                    # M1 needs actual message content for chunking and embedding
                    logger.debug(f"MemoryLayerImpl: Processing original message content for M1 layer")

                    # Prepare original message data for M1 processing (chunking + embedding)
                    m1_input_data = self._prepare_original_data_for_m1(message_batch_list, async_metadata)

                    if m1_input_data:
                        m1_result = await self.hierarchy_manager.write_to_layer(
                            layer_name="M1",
                            data=m1_input_data,
                            metadata=async_metadata
                        )
                        async_results["M1"] = {"success": m1_result.success if hasattr(m1_result, 'success') else False}
                        logger.info(f"MemoryLayerImpl: M1 async processing completed - success: {m1_result.success if hasattr(m1_result, 'success') else False}")
                    else:
                        logger.warning("MemoryLayerImpl: No M1 input data prepared, skipping M1 processing")

                except Exception as e:
                    logger.error(f"MemoryLayerImpl: M1 async processing failed: {e}")
                    async_results["M1"] = {"success": False, "error": str(e)}

            if self.config.m2_enabled and self.hierarchy_manager:
                try:
                    # Convert data for M2 processing
                    processed_data = self._prepare_data_for_layers(message_batch_list, async_metadata)
                    m2_data = [item for item in processed_data if item.get("target_layer") == "M2"]

                    if m2_data:
                        logger.debug(f"MemoryLayerImpl: Processing {len(m2_data)} items for M2 layer")
                        m2_result = await self.hierarchy_manager.write_to_layer(
                            layer_name="M2",
                            data=m2_data,
                            metadata=async_metadata
                        )
                        async_results["M2"] = m2_result
                        logger.info(f"MemoryLayerImpl: M2 async processing completed - success: {m2_result.get('success', False)}")

                except Exception as e:
                    logger.error(f"MemoryLayerImpl: M2 async processing failed: {e}")
                    async_results["M2"] = {"success": False, "error": str(e)}

            # Log overall async processing results
            successful_layers = [layer for layer, result in async_results.items() if result.get("success", False)]
            logger.info(f"MemoryLayerImpl: Async M1/M2 processing completed - successful layers: {successful_layers}")

        except Exception as e:
            logger.error(f"MemoryLayerImpl: Async M1/M2 processing failed: {e}")

    def _prepare_original_data_for_m1(self, message_batch_list: MessageBatchList, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prepare original message data for M1 layer processing.

        Args:
            message_batch_list: Original message batches with actual content
            metadata: Metadata from the original operation

        Returns:
            List of data items formatted for M1 processing with actual content
        """
        try:
            m1_data = []

            # Process each message batch
            for batch_idx, message_batch in enumerate(message_batch_list):
                for msg_idx, message in enumerate(message_batch):
                    # Create M1 item with actual message content
                    m1_item = {
                        "content": message.get("content", ""),
                        "role": message.get("role", "user"),
                        "source_layer": "M0",
                        "target_layer": "M1",
                        "operation_type": "m0_to_m1_processing",
                        "batch_index": batch_idx,
                        "message_index": msg_idx,
                        "message_id": message.get("message_id", f"msg_{batch_idx}_{msg_idx}"),
                        "session_id": metadata.get("session_id"),
                        "user_id": metadata.get("user_id"),
                        "metadata": {
                            **metadata,
                            "original_message": message,
                            "processing_timestamp": time.time()
                        }
                    }
                    m1_data.append(m1_item)

            logger.debug(f"MemoryLayerImpl: Prepared {len(m1_data)} items for M1 processing from original messages")
            return m1_data

        except Exception as e:
            logger.error(f"MemoryLayerImpl: Failed to prepare original data for M1: {e}")
            return []

    def _prepare_m0_data_for_m1(self, m0_processed_items: List[str], metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prepare M0 processed items for M1 layer processing.

        DEPRECATED: This method only passes M0 IDs, not actual content.
        Use _prepare_original_data_for_m1 instead for better M1 processing.

        Args:
            m0_processed_items: List of M0 record IDs that were successfully stored
            metadata: Metadata from the original operation

        Returns:
            List of data items formatted for M1 processing
        """
        try:
            # For now, we'll use the M0 IDs as references for M1 to fetch and process
            # In a full implementation, M1 would query M0 storage to get the actual content
            m1_data = []

            for item_id in m0_processed_items:
                m1_item = {
                    "source_layer": "M0",
                    "source_id": item_id,
                    "target_layer": "M1",
                    "operation_type": "m0_to_m1_processing",
                    "metadata": metadata
                }
                m1_data.append(m1_item)

            logger.debug(f"MemoryLayerImpl: Prepared {len(m1_data)} items for M1 processing from M0")
            return m1_data

        except Exception as e:
            logger.error(f"MemoryLayerImpl: Failed to prepare M0 data for M1: {e}")
            return []

    async def write_parallel(
        self,
        message_batch_list: MessageBatchList,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> WriteResult:
        """Legacy parallel write method - now delegates to M0-priority approach."""
        logger.info("MemoryLayerImpl: Using M0-priority write strategy (legacy parallel method)")
        return await self.write_m0_priority(message_batch_list, session_id, metadata)
    
    async def write_m0_first(
        self,
        message_batch_list: MessageBatchList,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> WriteResult:
        """Write data using M0-first strategy for improved performance and reliability.
        
        This method implements M0-first writing strategy:
        1. Synchronously write to M0 (critical data path)
        2. Asynchronously trigger M1/M2 processing
        3. Return M0 result immediately for faster response
        4. Continue background processing for higher layers
        
        Args:
            message_batch_list: Message batches to write
            session_id: Optional session ID
            metadata: Optional metadata dictionary
            
        Returns:
            WriteResult with M0 completion status and background processing info
        """
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
                "operation_type": "m0_first_write",
                "batch_size": len(message_batch_list),
                "timestamp": start_time,
                **(metadata or {})
            }
            
            logger.info(f"MemoryLayerImpl: Starting M0-first write for {len(message_batch_list)} message batches")
            
            # Convert MessageBatchList to format suitable for processing
            processed_data = self._prepare_data_for_layers(message_batch_list, operation_metadata)
            
            # Step 1: Synchronously write to M0 layer
            m0_result = await self._write_to_m0_layer(processed_data, operation_metadata)
            
            if not m0_result.success:
                self.failed_operations += 1
                operation_time = time.time() - start_time
                self.last_operation_time = operation_time
                
                return WriteResult(
                    success=False,
                    message=f"M0 write failed: {m0_result.error_message}",
                    layer_results={"M0": m0_result},
                    metadata={
                        "operation_time": operation_time,
                        "m0_success": False,
                        "background_processing": "not_started",
                        **operation_metadata
                    }
                )
            
            # Step 2: Asynchronously trigger M1/M2 processing
            logger.debug("Starting background M1/M2 processing...")
            asyncio.create_task(self._async_m1_m2_processing(processed_data, operation_metadata))
            
            # Step 3: Return M0 result immediately
            operation_time = time.time() - start_time
            self.last_operation_time = operation_time
            self.successful_operations += 1
            
            logger.info(f"MemoryLayerImpl: M0-first write completed in {operation_time:.2f}s (background processing continues)")
            
            return WriteResult(
                success=True,
                message=f"Successfully wrote {len(message_batch_list)} message batches to M0, M1/M2 processing in background",
                layer_results={"M0": m0_result},
                metadata={
                    "operation_time": operation_time,
                    "m0_success": True,
                    "background_processing": "started",
                    "m0_chunks_processed": m0_result.total_processed,
                    **operation_metadata
                }
            )
            
        except Exception as e:
            self.failed_operations += 1
            logger.error(f"MemoryLayerImpl: M0-first write operation failed: {e}")
            
            return WriteResult(
                success=False,
                message=f"M0-first write operation failed: {str(e)}",
                layer_results={},
                metadata=metadata or {}
            )
    
    async def _write_to_m0_layer(
        self, 
        processed_data: List[Dict[str, Any]], 
        metadata: Dict[str, Any]
    ) -> ParallelWriteResult:
        """Write data to M0 layer only."""
        try:
            if not self.hierarchy_manager:
                raise RuntimeError("Hierarchy manager not initialized")
            
            # Filter data for M0 layer only
            m0_data = [item.copy() for item in processed_data]
            for item in m0_data:
                item["target_layer"] = "M0"
            
            # Use hierarchy manager to write to M0
            # This assumes hierarchy manager has a method to write to specific layers
            m0_result = await self.hierarchy_manager.write_to_layer(
                layer_name="M0",
                data=m0_data,
                metadata=metadata
            )
            
            # Create a proper LayerWriteResult for M0
            # m0_result is a ProcessingResult object, not a dict
            m0_layer_result = LayerWriteResult(
                success=m0_result.success,
                result=str(m0_result.processed_items) if m0_result.processed_items else None,
                processed_items=m0_result.processed_items,
                processing_time=m0_result.processing_time,
                retry_count=0,
                error_message=m0_result.errors[0] if m0_result.errors else None
            )

            return ParallelWriteResult(
                success=m0_result.success,
                layer_results={"M0": m0_layer_result},
                total_processing_time=m0_result.processing_time,
                strategy_used=WriteStrategy.HYBRID,
                error_message=m0_result.errors[0] if m0_result.errors else None
            )

        except Exception as e:
            logger.error(f"Error writing to M0 layer: {e}")
            return ParallelWriteResult(
                success=False,
                layer_results={},
                total_processing_time=0.0,
                strategy_used=WriteStrategy.HYBRID,
                error_message=str(e)
            )
    
    async def _async_m1_m2_processing(
        self, 
        processed_data: List[Dict[str, Any]], 
        metadata: Dict[str, Any]
    ) -> None:
        """Asynchronously process M1 and M2 layers in the background."""
        try:
            logger.debug("Starting background M1/M2 processing")
            
            if not self.hierarchy_manager:
                logger.error("Hierarchy manager not initialized, skipping M1/M2 processing")
                return
            
            # Process M1 layer
            if self.config.m1_enabled:
                try:
                    m1_data = [item.copy() for item in processed_data]
                    for item in m1_data:
                        item["target_layer"] = "M1"
                    
                    m1_result = await self.hierarchy_manager.write_to_layer(
                        layer_name="M1",
                        data=m1_data,
                        metadata={**metadata, "async_processing": True}
                    )
                    
                    logger.debug(f"M1 processing completed: {m1_result.get('success', False)}")
                    
                except Exception as e:
                    logger.error(f"Error in M1 background processing: {e}")
            
            # Process M2 layer
            if self.config.m2_enabled:
                try:
                    m2_data = [item.copy() for item in processed_data]
                    for item in m2_data:
                        item["target_layer"] = "M2"
                    
                    m2_result = await self.hierarchy_manager.write_to_layer(
                        layer_name="M2",
                        data=m2_data,
                        metadata={**metadata, "async_processing": True}
                    )
                    
                    logger.debug(f"M2 processing completed: {m2_result.get('success', False)}")
                    
                except Exception as e:
                    logger.error(f"Error in M2 background processing: {e}")
            
            logger.debug("Background M1/M2 processing completed")
            
        except Exception as e:
            logger.error(f"Error in background M1/M2 processing: {e}")
    
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
        """Query all active memory layers and return results."""
        try:
            if not self.initialized:
                await self.initialize()

            if not self.parallel_manager:
                raise RuntimeError("Parallel manager not initialized")

            logger.info(f"MemoryLayerImpl: Starting query: '{query[:50]}...'")

            # For now, delegate to parallel manager's query capabilities
            # This is a placeholder - the actual implementation would need
            # to be enhanced to support querying across layers

            # Implement querying with M1 as primary vector search layer
            # Architecture: M0 (raw data, no vectors) -> M1 (chunks + embeddings) -> M2 (semantic facts)

            all_results = []
            layer_sources = {"M0": [], "M1": [], "M2": []}

            if self.hierarchy_manager:
                # Primary vector search from M1 layer (has embeddings)
                if self.config.m1_enabled and self.layer_status.get("M1") == LayerStatus.ACTIVE:
                    try:
                        # Prepare query parameters for M1 layer
                        query_params = {
                            "store_type": store_type,
                            "include_messages": include_messages,
                            "include_knowledge": include_knowledge,
                            "include_chunks": include_chunks,
                            "session_id": session_id,
                            "scope": scope
                        }
                        m1_results = await self.hierarchy_manager.query_layers(
                            query=query, layers=[LayerType.M1], top_k=top_k, **query_params
                        )
                        if LayerType.M1 in m1_results:
                            layer_sources["M1"] = m1_results[LayerType.M1]
                            all_results.extend(m1_results[LayerType.M1])
                            logger.debug(f"MemoryLayerImpl: M1 vector search returned {len(m1_results[LayerType.M1])} results")
                    except Exception as e:
                        logger.error(f"MemoryLayerImpl: M1 query failed: {e}")

                # Fallback to M0 for keyword/SQL search if M1 has no results
                if not all_results and self.config.m0_enabled and self.layer_status.get("M0") == LayerStatus.ACTIVE:
                    try:
                        m0_results = await self.hierarchy_manager.query_layers(
                            query=query, layers=[LayerType.M0], top_k=top_k, **query_params
                        )
                        if LayerType.M0 in m0_results:
                            layer_sources["M0"] = m0_results[LayerType.M0]
                            all_results.extend(m0_results[LayerType.M0])
                            logger.debug(f"MemoryLayerImpl: M0 fallback search returned {len(m0_results[LayerType.M0])} results")
                    except Exception as e:
                        logger.error(f"MemoryLayerImpl: M0 query failed: {e}")

                # Optional M2 semantic search
                if self.config.m2_enabled and self.layer_status.get("M2") == LayerStatus.ACTIVE:
                    try:
                        m2_results = await self.hierarchy_manager.query_layers(
                            query=query, layers=[LayerType.M2], top_k=top_k // 2, **query_params  # Fewer M2 results
                        )
                        if LayerType.M2 in m2_results:
                            layer_sources["M2"] = m2_results[LayerType.M2]
                            all_results.extend(m2_results[LayerType.M2])
                            logger.debug(f"MemoryLayerImpl: M2 semantic search returned {len(m2_results[LayerType.M2])} results")
                    except Exception as e:
                        logger.error(f"MemoryLayerImpl: M2 query failed: {e}")

            return QueryResult(
                results=all_results[:top_k],  # Limit to requested top_k
                layer_sources=layer_sources,
                total_count=len(all_results),
                metadata={
                    "query": query,
                    "top_k": top_k,
                    "active_layers": [layer for layer, status in self.layer_status.items() if status == LayerStatus.ACTIVE],
                    "primary_search_layer": "M1",
                    "search_strategy": "M1_primary_vector_search"
                }
            )
            
        except Exception as e:
            logger.error(f"MemoryLayerImpl: Query operation failed: {e}")

            return QueryResult(
                results=[],
                layer_sources={"M0": [], "M1": [], "M2": []},
                total_count=0,
                metadata={"error": str(e)}
            )
    
    async def get_layer_status(self) -> Dict[str, LayerStatus]:
        """Get the current status of all memory layers."""
        return self.layer_status.copy()

    async def get_statistics(self) -> Dict[str, Any]:
        """Get performance and usage statistics for all memory layers."""
        return {
            "total_operations": self.total_operations,
            "successful_operations": self.successful_operations,
            "failed_operations": self.failed_operations,
            "success_rate": self.successful_operations / max(self.total_operations, 1),
            "last_operation_time": self.last_operation_time,
            "layer_status": {k: v.value for k, v in self.layer_status.items()},
            "initialized": self.initialized
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform a health check on all memory layers."""
        health_status = {
            "overall_status": "healthy" if self.initialized else "unhealthy",
            "layers": {},
            "components": {
                "hierarchy_manager": self.hierarchy_manager is not None,
                "parallel_manager": self.parallel_manager is not None
            }
        }

        # Check each layer status
        for layer_name, status in self.layer_status.items():
            health_status["layers"][layer_name] = {
                "status": status.value,
                "healthy": status in [LayerStatus.ACTIVE, LayerStatus.INACTIVE]
            }

        return health_status

    async def reset_layer(self, layer_name: str) -> bool:
        """Reset a specific memory layer."""
        try:
            if layer_name not in self.layer_status:
                logger.error(f"MemoryLayerImpl: Unknown layer {layer_name}")
                return False

            # Set layer to initializing status
            self.layer_status[layer_name] = LayerStatus.INITIALIZING

            # Reset layer through hierarchy manager if available
            if self.hierarchy_manager and hasattr(self.hierarchy_manager, 'reset_layer'):
                success = await self.hierarchy_manager.reset_layer(layer_name)
                if success:
                    self.layer_status[layer_name] = LayerStatus.ACTIVE
                    logger.info(f"MemoryLayerImpl: Successfully reset layer {layer_name}")
                else:
                    self.layer_status[layer_name] = LayerStatus.ERROR
                    logger.error(f"MemoryLayerImpl: Failed to reset layer {layer_name}")
                return success
            else:
                logger.warning(f"MemoryLayerImpl: Reset not supported for layer {layer_name}")
                self.layer_status[layer_name] = LayerStatus.INACTIVE
                return False

        except Exception as e:
            logger.error(f"MemoryLayerImpl: Error resetting layer {layer_name}: {e}")
            self.layer_status[layer_name] = LayerStatus.ERROR
            return False

    async def cleanup(self) -> bool:
        """Clean up resources and shut down all memory layers."""
        try:
            logger.info("MemoryLayerImpl: Starting cleanup...")

            # Cleanup parallel manager
            if self.parallel_manager and hasattr(self.parallel_manager, 'cleanup'):
                await self.parallel_manager.cleanup()

            # Cleanup hierarchy manager
            if self.hierarchy_manager and hasattr(self.hierarchy_manager, 'cleanup'):
                await self.hierarchy_manager.cleanup()

            # Reset state
            self.initialized = False
            for layer_name in self.layer_status:
                self.layer_status[layer_name] = LayerStatus.INACTIVE

            logger.info("MemoryLayerImpl: Cleanup completed successfully")
            return True

        except Exception as e:
            logger.error(f"MemoryLayerImpl: Error during cleanup: {e}")
            return False
    
    async def get_layer_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all memory layers."""
        return {
            "memory_layer": {
                "total_operations": self.total_operations,
                "successful_operations": self.successful_operations,
                "failed_operations": self.failed_operations,
                "success_rate": self.successful_operations / max(self.total_operations, 1),
                "last_operation_time": self.last_operation_time,
                "initialized": self.initialized
            },
            "layer_status": {layer: status.value for layer, status in self.layer_status.items()}
        }
    
    async def get_m0_first_status(self) -> Dict[str, Any]:
        """Get status of M0-first writing strategy and background processing."""
        # This would ideally track actual background tasks, but for now provides config status
        return {
            "strategy_enabled": self.config.m0_enabled,  # Could be extended with specific M0-first config
            "active_layers": {
                "M0": self.layer_status["M0"].value,
                "M1": self.layer_status["M1"].value,
                "M2": self.layer_status["M2"].value
            },
            "background_processing_capable": self.config.m1_enabled or self.config.m2_enabled,
            "parallel_manager_available": self.parallel_manager is not None,
            "hierarchy_manager_available": self.hierarchy_manager is not None,
            "statistics": {
                "total_operations": self.total_operations,
                "success_rate": self.successful_operations / max(self.total_operations, 1),
                "last_operation_time": self.last_operation_time
            }
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
            logger.info("MemoryLayerImpl: Starting cleanup...")

            if self.parallel_manager:
                # Shutdown parallel manager
                if hasattr(self.parallel_manager, 'shutdown'):
                    await self.parallel_manager.shutdown()
                elif hasattr(self.parallel_manager, 'cleanup'):
                    await self.parallel_manager.cleanup()

            if self.hierarchy_manager:
                # Shutdown hierarchy manager
                if hasattr(self.hierarchy_manager, 'shutdown'):
                    await self.hierarchy_manager.shutdown()
                elif hasattr(self.hierarchy_manager, 'cleanup'):
                    await self.hierarchy_manager.cleanup()

            self.initialized = False
            self.layer_status = {layer: LayerStatus.INACTIVE for layer in self.layer_status}

            logger.info("MemoryLayerImpl: Cleanup completed")
            return True

        except Exception as e:
            logger.error(f"MemoryLayerImpl: Cleanup failed: {e}")
            return False
