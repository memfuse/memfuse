"""WriteBuffer implementation for MemFuse Buffer.

The WriteBuffer serves as a unified entry point for RoundBuffer and HybridBuffer,
providing a clean abstraction layer and maintaining the original architecture design.
"""

import asyncio
import time
import uuid
from typing import Any, Dict, List, Optional, Callable
from loguru import logger

from ..interfaces import MessageList, MessageBatchList
from .round_buffer import RoundBuffer
from .hybrid_buffer import HybridBuffer
from .flush_manager import FlushManager


class WriteBuffer:
    """Unified write buffer integrating RoundBuffer and HybridBuffer.
    
    This buffer serves as the main entry point for all write operations,
    internally managing the coordination between RoundBuffer and HybridBuffer
    according to the PRD architecture design.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        memory_service_handler: Optional[Callable] = None,
        qdrant_handler: Optional[Callable] = None
    ):
        """Initialize the WriteBuffer with integrated components.

        Args:
            config: Configuration dictionary containing buffer settings
            memory_service_handler: Handler for MemoryService operations
            qdrant_handler: Handler for Qdrant storage operations
        """
        self.config = config

        # Extract configuration for sub-components
        round_config = config.get('round_buffer', {})
        hybrid_config = config.get('hybrid_buffer', {})
        flush_config = config.get('flush_manager', {})

        # Initialize FlushManager first
        self.flush_manager = FlushManager(
            max_workers=flush_config.get('max_workers', 2),
            max_queue_size=flush_config.get('max_queue_size', 100),
            default_timeout=flush_config.get('default_timeout', 30.0),
            flush_interval=flush_config.get('flush_interval', 5.0),
            enable_auto_flush=flush_config.get('enable_auto_flush', True),
            memory_service_handler=memory_service_handler
        )

        # Initialize RoundBuffer
        self.round_buffer = RoundBuffer(
            max_tokens=round_config.get('max_tokens', 800),
            max_size=round_config.get('max_size', 5),
            token_model=round_config.get('token_model', 'gpt-4o-mini')
        )

        # Initialize HybridBuffer with FlushManager
        self.hybrid_buffer = HybridBuffer(
            max_size=hybrid_config.get('max_size', 5),
            chunk_strategy=hybrid_config.get('chunk_strategy', 'message'),
            embedding_model=hybrid_config.get('embedding_model', 'all-MiniLM-L6-v2'),
            flush_manager=self.flush_manager
        )

        # Set up component connections
        self.round_buffer.set_transfer_handler(self.hybrid_buffer.add_from_rounds)

        # Statistics
        self.total_writes = 0
        self.total_transfers = 0

        logger.info("WriteBuffer: Initialized with RoundBuffer, HybridBuffer, and FlushManager integration")

    async def initialize(self) -> bool:
        """Initialize WriteBuffer and start FlushManager workers.

        Returns:
            True if initialization was successful
        """
        try:
            # Initialize FlushManager workers
            if not await self.flush_manager.initialize():
                logger.error("WriteBuffer: Failed to initialize FlushManager")
                return False

            logger.info("WriteBuffer: Initialization completed successfully")
            return True

        except Exception as e:
            logger.error(f"WriteBuffer: Initialization failed: {e}")
            return False

    async def add(self, messages: MessageList, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Add a single list of messages to the buffer.
        
        Args:
            messages: List of message dictionaries
            session_id: Session ID for context
            
        Returns:
            Dictionary with operation status and metadata
        """
        self.total_writes += 1
        
        # Delegate to RoundBuffer (which may trigger transfer to HybridBuffer)
        result = await self.round_buffer.add(messages, session_id)

        # Extract transfer status and message IDs from result
        transfer_triggered = result.get("transfer_triggered", False)
        message_ids = result.get("message_ids", [])

        if transfer_triggered:
            self.total_transfers += 1
            logger.debug(f"WriteBuffer: Transfer triggered, total transfers: {self.total_transfers}")

        return {
            "status": "success",
            "transfer_triggered": transfer_triggered,
            "message_ids": message_ids,
            "total_writes": self.total_writes,
            "total_transfers": self.total_transfers
        }
    
    async def add_batch(self, message_batch_list: MessageBatchList, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Optimized batch processing with intelligent coordination.

        P2 OPTIMIZATION: Enhanced with pipeline processing and parallel operations.

        Args:
            message_batch_list: List of lists of messages
            session_id: Session ID for context

        Returns:
            Dictionary with batch operation status and metadata
        """
        if not message_batch_list:
            return {"status": "success", "message": "No message lists to add"}

        start_time = time.time()

        # P2 OPTIMIZATION: Pipeline processing with parallel operations
        # Run preprocessing and token calculation in parallel
        preprocessing_task = asyncio.create_task(self._preprocess_batch(message_batch_list, session_id))

        # Start session analysis early (can run in parallel with preprocessing)
        session_changes = self._detect_session_changes(message_batch_list)

        # Wait for preprocessing to complete
        processed_batch = await preprocessing_task

        # Run token calculation and transfer planning in parallel
        token_calculation_task = asyncio.create_task(self._calculate_batch_tokens(processed_batch))
        transfer_strategy = self._plan_transfer_strategy_async(session_changes)

        # Wait for token calculation
        total_tokens = await token_calculation_task

        # Update transfer strategy with token information
        transfer_strategy = self._finalize_transfer_strategy(transfer_strategy, total_tokens)

        # Execute optimized batch strategy
        execution_result = await self._execute_batch_strategy(processed_batch, transfer_strategy)

        # 5. Update statistics
        processing_time = time.time() - start_time
        self.total_writes += 1  # Update write count
        self._update_batch_stats(execution_result, processing_time)

        return {
            "status": "success",
            "batch_size": len(message_batch_list),
            "total_messages": sum(len(ml) for ml in message_batch_list if ml),
            "total_tokens": total_tokens,
            "transfers_triggered": execution_result.get("transfers", 0),
            "processing_time": processing_time,
            "strategy_used": transfer_strategy["type"],
            "message_ids": execution_result.get("message_ids", [])
        }
    
    def get_round_buffer(self) -> RoundBuffer:
        """Get the RoundBuffer instance for Read API operations.
        
        Returns:
            RoundBuffer instance
        """
        return self.round_buffer
    
    def get_hybrid_buffer(self) -> HybridBuffer:
        """Get the HybridBuffer instance for Query API operations.
        
        Returns:
            HybridBuffer instance
        """
        return self.hybrid_buffer

    def get_flush_manager(self) -> FlushManager:
        """Get FlushManager instance for direct operations."""
        return self.flush_manager

    async def flush_all(self) -> Dict[str, Any]:
        """Force flush all buffers to persistent storage.
        
        Returns:
            Dictionary with flush operation status
        """
        try:
            # Force transfer from RoundBuffer to HybridBuffer
            if self.round_buffer.rounds:
                await self.round_buffer._transfer_and_clear("manual_flush")
            
            # Force flush HybridBuffer to storage
            await self.hybrid_buffer.flush_to_storage()
            
            return {"status": "success", "message": "All buffers flushed successfully"}
        except Exception as e:
            logger.error(f"WriteBuffer: Error during flush_all: {e}")
            return {"status": "error", "message": f"Flush failed: {str(e)}"}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics for the WriteBuffer system.
        
        Returns:
            Dictionary with detailed statistics
        """
        round_stats = self.round_buffer.get_stats()
        hybrid_stats = self.hybrid_buffer.get_stats()
        
        return {
            "write_buffer": {
                "total_writes": self.total_writes,
                "total_transfers": self.total_transfers,
                "round_buffer": round_stats,
                "hybrid_buffer": hybrid_stats
            }
        }
    
    def is_empty(self) -> bool:
        """Check if both buffers are empty.
        
        Returns:
            True if both RoundBuffer and HybridBuffer are empty
        """
        return (len(self.round_buffer.rounds) == 0 and 
                len(self.hybrid_buffer.chunks) == 0)
    


    async def shutdown(self) -> None:
        """Gracefully shutdown WriteBuffer and all its components.
        
        Optimized shutdown sequence:
        1. FlushManager (critical for async task cleanup)
        2. HybridBuffer (may have pending operations)
        3. RoundBuffer (simple memory cleanup)
        """
        logger.info("WriteBuffer: Shutdown initiated")
        
        shutdown_tasks = []
        
        try:
            # Critical: Shutdown FlushManager first to clean up async tasks
            if hasattr(self.flush_manager, 'shutdown'):
                shutdown_tasks.append(("FlushManager", self.flush_manager.shutdown()))
                
            # Then shutdown HybridBuffer if it has async cleanup
            if hasattr(self.hybrid_buffer, 'shutdown'):
                shutdown_tasks.append(("HybridBuffer", self.hybrid_buffer.shutdown()))
            
            # Execute shutdowns concurrently where possible
            if shutdown_tasks:
                logger.info(f"WriteBuffer: Executing {len(shutdown_tasks)} concurrent shutdowns")
                results = await asyncio.gather(
                    *[task for _, task in shutdown_tasks],
                    return_exceptions=True
                )
                
                # Check results and log any errors
                for i, (component_name, _) in enumerate(shutdown_tasks):
                    if isinstance(results[i], Exception):
                        logger.error(f"WriteBuffer: Error shutting down {component_name}: {results[i]}")
                    else:
                        logger.info(f"WriteBuffer: {component_name} shutdown completed")
            
            # Finally clear RoundBuffer (synchronous operation)
            if hasattr(self.round_buffer, 'clear'):
                await self.round_buffer.clear()
                logger.info("WriteBuffer: RoundBuffer cleared")
                
            logger.info("WriteBuffer: All components shutdown successfully")
            
        except Exception as e:
            logger.error(f"WriteBuffer: Critical error during shutdown: {e}")
            # Don't re-raise to prevent blocking the shutdown process
            # The error has been logged for debugging

    async def clear_all(self) -> Dict[str, Any]:
        """Clear all buffers (for testing purposes).
        
        Returns:
            Dictionary with clear operation status
        """
        try:
            # Clear RoundBuffer
            self.round_buffer.rounds.clear()
            self.round_buffer.current_tokens = 0
            self.round_buffer.current_session_id = None
            
            # Clear HybridBuffer
            self.hybrid_buffer.chunks.clear()
            self.hybrid_buffer.embeddings.clear()
            self.hybrid_buffer.original_rounds.clear()
            
            # Reset statistics
            self.total_writes = 0
            self.total_transfers = 0
            
            logger.info("WriteBuffer: All buffers cleared")
            return {"status": "success", "message": "All buffers cleared"}
        except Exception as e:
            logger.error(f"WriteBuffer: Error during clear_all: {e}")
            return {"status": "error", "message": f"Clear failed: {str(e)}"}

    # Batch processing optimization methods

    async def _preprocess_batch(self, message_batch_list: MessageBatchList, session_id: Optional[str]) -> MessageBatchList:
        """Batch preprocessing with field validation.

        Args:
            message_batch_list: List of lists of messages
            session_id: Session ID for context

        Returns:
            Preprocessed message batch list
        """
        processed_batch = []

        for message_list in message_batch_list:
            if not message_list:
                continue

            processed_list = []
            for message in message_list:
                # Ensure required fields
                self._ensure_message_fields(message)
                processed_list.append(message)

            if processed_list:  # Only add non-empty lists
                processed_batch.append(processed_list)

        return processed_batch

    def _ensure_message_fields(self, message: Dict[str, Any]) -> None:
        """Ensure message has required fields.

        Args:
            message: Message dictionary to validate
        """
        if not isinstance(message, dict):
            return

        # Ensure ID
        if 'id' not in message or not message['id']:
            message['id'] = str(uuid.uuid4())

        # Ensure timestamps
        current_time = time.time()
        if 'created_at' not in message or not message['created_at']:
            message['created_at'] = current_time

        if 'updated_at' not in message or not message['updated_at']:
            message['updated_at'] = message['created_at']

        # Ensure metadata
        if 'metadata' not in message:
            message['metadata'] = {}

    async def _calculate_batch_tokens(self, message_batch_list: MessageBatchList) -> int:
        """Optimized batch token calculation.

        Args:
            message_batch_list: List of lists of messages

        Returns:
            Total token count for all messages
        """
        if not message_batch_list:
            return 0

        # Flatten all messages for batch calculation
        all_messages = []
        for message_list in message_batch_list:
            all_messages.extend(message_list)

        if not all_messages:
            return 0

        # Use RoundBuffer's token counter for consistency
        return self.round_buffer.token_counter.count_message_tokens(all_messages)

    def _detect_session_changes(self, message_batch_list: MessageBatchList) -> List[str]:
        """Detect session changes for optimized transfer strategy.

        Args:
            message_batch_list: List of lists of messages

        Returns:
            List of session transition identifiers
        """
        session_sequence = []

        for message_list in message_batch_list:
            # Extract session from first message in list
            if message_list and isinstance(message_list[0], dict):
                session = message_list[0].get('metadata', {}).get('session_id', 'default')
                session_sequence.append(session)

        # Identify transition points
        transitions = []
        current_session = None

        for i, session in enumerate(session_sequence):
            if session != current_session:
                transitions.append(f"transition_{i}_{session}")
                current_session = session

        return transitions

    def _plan_transfer_strategy(self, total_tokens: int, session_changes: List[str]) -> Dict[str, Any]:
        """Plan optimal transfer strategy based on batch characteristics.

        Args:
            total_tokens: Total token count for the batch
            session_changes: List of session transitions

        Returns:
            Dictionary describing the optimal strategy
        """
        if total_tokens > self.round_buffer.max_tokens * 2:
            return {"type": "bulk_transfer", "reason": "high_token_count"}
        elif len(session_changes) > 3:
            return {"type": "session_grouped", "groups": session_changes, "reason": "multiple_sessions"}
        else:
            return {"type": "sequential", "reason": "standard_processing"}

    async def _execute_batch_strategy(self, processed_batch: MessageBatchList, strategy: Dict) -> Dict[str, Any]:
        """Execute batch processing according to strategy.

        Args:
            processed_batch: Preprocessed message batch list
            strategy: Strategy dictionary from _plan_transfer_strategy

        Returns:
            Dictionary with execution results
        """
        if strategy["type"] == "bulk_transfer":
            return await self._bulk_transfer_strategy(processed_batch)
        elif strategy["type"] == "session_grouped":
            return await self._session_grouped_strategy(processed_batch, strategy["groups"])
        else:
            return await self._sequential_strategy(processed_batch)

    async def _bulk_transfer_strategy(self, processed_batch: MessageBatchList) -> Dict[str, Any]:
        """Process entire batch as single bulk operation.

        Args:
            processed_batch: Preprocessed message batch list

        Returns:
            Dictionary with execution results
        """
        transfers = 0
        all_message_ids = []

        # Add all to RoundBuffer first
        for message_list in processed_batch:
            result = await self.round_buffer.add(message_list)
            if isinstance(result, dict):
                if result.get("transfer_triggered", False):
                    transfers += 1
                all_message_ids.extend(result.get("message_ids", []))
            else:
                # Backward compatibility for old boolean return
                if result:
                    transfers += 1

        # Force transfer if needed
        if self.round_buffer.rounds:
            await self.round_buffer._transfer_and_clear("bulk_strategy")
            transfers += 1

        return {"transfers": transfers, "strategy": "bulk", "message_ids": all_message_ids}

    async def _session_grouped_strategy(self, processed_batch: MessageBatchList, groups: List[str]) -> Dict[str, Any]:
        """Process batch grouped by session for optimal performance.

        Args:
            processed_batch: Preprocessed message batch list
            groups: Session group identifiers

        Returns:
            Dictionary with execution results
        """
        transfers = 0
        all_message_ids = []

        # Group messages by session
        session_groups = self._group_by_session(processed_batch)

        # Process each session group
        for session_id, message_lists in session_groups.items():
            for message_list in message_lists:
                result = await self.round_buffer.add(message_list, session_id)
                if isinstance(result, dict):
                    if result.get("transfer_triggered", False):
                        transfers += 1
                    all_message_ids.extend(result.get("message_ids", []))
                else:
                    # Backward compatibility for old boolean return
                    if result:
                        transfers += 1

        return {"transfers": transfers, "strategy": "session_grouped", "groups": len(session_groups), "message_ids": all_message_ids}

    async def _sequential_strategy(self, processed_batch: MessageBatchList) -> Dict[str, Any]:
        """Standard sequential processing with optimizations.

        Args:
            processed_batch: Preprocessed message batch list

        Returns:
            Dictionary with execution results
        """
        transfers = 0
        all_message_ids = []

        for message_list in processed_batch:
            result = await self.round_buffer.add(message_list)
            if isinstance(result, dict):
                if result.get("transfer_triggered", False):
                    transfers += 1
                all_message_ids.extend(result.get("message_ids", []))
            else:
                # Backward compatibility for old boolean return
                if result:
                    transfers += 1

        return {"transfers": transfers, "strategy": "sequential", "message_ids": all_message_ids}

    def _group_by_session(self, message_batch_list: MessageBatchList) -> Dict[str, List[MessageList]]:
        """Group message lists by session ID.

        Args:
            message_batch_list: List of lists of messages

        Returns:
            Dictionary mapping session IDs to message lists
        """
        session_groups = {}

        for message_list in message_batch_list:
            if not message_list:
                continue

            # Extract session from first message
            session_id = 'default'
            if isinstance(message_list[0], dict):
                session_id = message_list[0].get('metadata', {}).get('session_id', 'default')

            if session_id not in session_groups:
                session_groups[session_id] = []
            session_groups[session_id].append(message_list)

        return session_groups

    def _update_batch_stats(self, execution_result: Dict[str, Any], processing_time: float) -> None:
        """Update batch processing statistics.

        Args:
            execution_result: Result from batch execution
            processing_time: Time taken for processing
        """
        # Update transfer count
        self.total_transfers += execution_result.get("transfers", 0)

        # Log performance metrics
        logger.info(f"WriteBuffer: Batch processing completed in {processing_time:.3f}s, "
                   f"strategy: {execution_result.get('strategy', 'unknown')}, "
                   f"transfers: {execution_result.get('transfers', 0)}")

    # P2 OPTIMIZATION: Pipeline processing helper methods

    def _plan_transfer_strategy_async(self, session_changes: List) -> Dict[str, Any]:
        """Asynchronous version of transfer strategy planning.

        This method can run in parallel with token calculation.

        Args:
            session_changes: List of session change information

        Returns:
            Preliminary transfer strategy (will be finalized with token info)
        """
        if len(session_changes) > 1:
            return {
                "type": "session_grouped",
                "groups": session_changes,
                "reason": "multiple_sessions",
                "finalized": False
            }
        else:
            return {
                "type": "sequential",
                "reason": "standard_processing",
                "finalized": False
            }

    def _finalize_transfer_strategy(self, preliminary_strategy: Dict[str, Any], total_tokens: int) -> Dict[str, Any]:
        """Finalize transfer strategy with token information.

        Args:
            preliminary_strategy: Strategy from _plan_transfer_strategy_async
            total_tokens: Total tokens calculated for the batch

        Returns:
            Finalized transfer strategy
        """
        strategy = preliminary_strategy.copy()
        strategy["finalized"] = True

        # Apply token-based optimizations
        if total_tokens > self.round_buffer.max_tokens * 2:
            # Large batch: use bulk transfer
            strategy["type"] = "bulk_transfer"
            strategy["reason"] = f"large_batch_tokens ({total_tokens} > {self.round_buffer.max_tokens * 2})"

        return strategy
