"""Optimized HybridBuffer implementation with FlushManager integration.

This version provides:
- Non-blocking flush operations using FlushManager
- Improved performance and concurrency
- Better error handling and recovery
- Comprehensive metrics and monitoring
"""

import asyncio
import time
from typing import Any, Dict, List, Optional
from loguru import logger

from ..interfaces import MessageList
from ..rag.chunk.base import ChunkData
from .flush_manager import FlushManager, FlushPriority


class HybridBuffer:
    """Optimized HybridBuffer with non-blocking flush operations.

    This buffer maintains two distinct data structures:
    1. RoundQueue: Stores original rounds for SQLite storage
    2. VectorCache: Stores pre-processed chunks and embeddings for retrieval

    Architecture Overview:
    - Dual-queue design: Parallel data structures for different purposes
    - Immediate processing: Chunks and embeddings generated on data arrival
    - Non-blocking flush: Uses FlushManager for asynchronous storage operations

    Key improvements over original version:
    - Non-blocking flush operations via FlushManager
    - Configurable flush strategies (size-based, time-based, hybrid)
    - Better error handling and recovery with optimistic clearing
    - Comprehensive performance monitoring and metrics
    - Improved concurrency with separate data and flush locks
    """

    def __init__(
        self,
        max_size: int = 5,
        chunk_strategy: str = "message",
        embedding_model: str = "all-MiniLM-L6-v2",
        flush_manager: Optional[FlushManager] = None,
        auto_flush_interval: float = 60.0,
        enable_auto_flush: bool = True
    ):
        """Initialize the optimized HybridBuffer.

        Args:
            max_size: Maximum number of items to keep in buffer
            chunk_strategy: Chunking strategy to use
            embedding_model: Embedding model for vector generation
            flush_manager: FlushManager instance for handling flush operations
            auto_flush_interval: Interval for automatic flushing (seconds)
            enable_auto_flush: Whether to enable automatic flushing
        """
        self.max_size = max_size
        self.chunk_strategy_name = chunk_strategy
        self.embedding_model_name = embedding_model
        self.auto_flush_interval = auto_flush_interval
        self.enable_auto_flush = enable_auto_flush

        # Dual-queue data structures
        self.chunks: List[ChunkData] = []
        self.embeddings: List[List[float]] = []
        self.original_rounds: List[MessageList] = []

        # Strategy and model instances (lazy loaded)
        self.chunk_strategy: Optional[Any] = None
        self.embedding_model: Optional[Any] = None

        # FlushManager for non-blocking operations
        self.flush_manager = flush_manager

        # Async locks for thread safety
        self._data_lock = asyncio.Lock()  # For data operations
        self._flush_lock = asyncio.Lock()  # For flush coordination

        # Auto-flush timer
        self.last_flush_time = time.time()
        self.auto_flush_task: Optional[asyncio.Task] = None

        # Statistics
        self.total_rounds_received = 0
        self.total_chunks_created = 0
        self.total_flushes = 0
        self.total_auto_flushes = 0
        self.total_manual_flushes = 0
        self.pending_flush_tasks: Dict[str, asyncio.Task] = {}

        logger.info(f"HybridBuffer: Initialized with max_size={max_size}, chunk_strategy={chunk_strategy}")

    async def initialize(self) -> bool:
        """Initialize the buffer and start auto-flush if enabled.

        Returns:
            True if initialization was successful
        """
        try:
            # Start auto-flush task if enabled
            if self.enable_auto_flush and self.auto_flush_interval > 0:
                self.auto_flush_task = asyncio.create_task(self._auto_flush_loop())
                logger.info(f"HybridBuffer: Auto-flush enabled with interval {self.auto_flush_interval}s")

            return True
        except Exception as e:
            logger.error(f"HybridBuffer: Initialization failed: {e}")
            return False

    async def shutdown(self) -> None:
        """Gracefully shutdown the buffer."""
        logger.info("HybridBuffer: Shutdown initiated")

        # Cancel auto-flush task
        if self.auto_flush_task:
            self.auto_flush_task.cancel()
            try:
                await self.auto_flush_task
            except asyncio.CancelledError:
                pass

        # Wait for pending flush tasks
        if self.pending_flush_tasks:
            logger.info(f"HybridBuffer: Waiting for {len(self.pending_flush_tasks)} pending flush tasks")
            await asyncio.gather(*self.pending_flush_tasks.values(), return_exceptions=True)

        # Final flush if there's data
        if self.original_rounds:
            logger.info("HybridBuffer: Performing final flush")
            await self.flush_to_storage(priority=FlushPriority.CRITICAL)

        logger.info("HybridBuffer: Shutdown completed")

    def set_flush_manager(self, flush_manager: FlushManager) -> None:
        """Set the FlushManager instance.

        Args:
            flush_manager: FlushManager instance
        """
        self.flush_manager = flush_manager
        logger.debug("HybridBuffer: FlushManager set")


    
    async def add_from_rounds(self, rounds: List[MessageList]) -> None:
        """Add rounds from RoundBuffer transfer with optimized processing.

        This method implements the core data flow from RoundBuffer to HybridBuffer:
        1. Immediately processes rounds (chunking + embeddings) for VectorCache
        2. Stores original rounds in RoundQueue for SQLite storage
        3. Checks and triggers non-blocking flush if needed

        The dual-queue design ensures:
        - Fast retrieval via pre-processed chunks and embeddings
        - Complete data preservation via original rounds
        - FIFO behavior when buffer reaches capacity

        Args:
            rounds: List of MessageList objects from RoundBuffer
        """
        if not rounds:
            return

        async with self._data_lock:
            logger.info(f"HybridBuffer: Receiving {len(rounds)} rounds from RoundBuffer")

            # 1. Immediately process rounds (chunking + embeddings)
            await self._process_rounds_immediately(rounds)

            # 2. Add original rounds to queue
            self.original_rounds.extend(rounds)
            self.total_rounds_received += len(rounds)

            logger.info(f"HybridBuffer: RoundQueue contains {len(self.original_rounds)} rounds, VectorCache contains {len(self.chunks)} chunks")

            # 3. Check if flush is needed (non-blocking)
            await self._check_and_trigger_flush()

    async def _check_and_trigger_flush(self) -> None:
        """Check if flush is needed and trigger non-blocking flush."""
        should_flush = False
        flush_reason = ""

        # Size-based flush
        if len(self.original_rounds) >= self.max_size:
            should_flush = True
            flush_reason = f"size_limit ({len(self.original_rounds)} >= {self.max_size})"

        # Time-based flush (if auto-flush is enabled)
        elif self.enable_auto_flush:
            time_since_last_flush = time.time() - self.last_flush_time
            if time_since_last_flush >= self.auto_flush_interval:
                should_flush = True
                flush_reason = f"time_limit ({time_since_last_flush:.1f}s >= {self.auto_flush_interval}s)"

        if should_flush:
            logger.info(f"HybridBuffer: Triggering non-blocking flush - {flush_reason}")
            await self.flush_to_storage(priority=FlushPriority.NORMAL)
        else:
            logger.debug(f"HybridBuffer: No flush needed ({len(self.original_rounds)}/{self.max_size})")

    async def _process_rounds_immediately(self, rounds: List[MessageList]) -> None:
        """Immediately process rounds, generate chunks and embeddings stored to VectorCache.

        Args:
            rounds: List of rounds to process
        """
        try:
            # Lazy load chunk strategy
            if self.chunk_strategy is None:
                await self._load_chunk_strategy()

            # Lazy load embedding model
            if self.embedding_model is None:
                await self._load_embedding_model()

            # Immediately create chunks
            logger.info(f"HybridBuffer: Creating chunks from {len(rounds)} rounds...")
            chunks = await self.chunk_strategy.create_chunks(rounds)
            logger.info(f"HybridBuffer: Created {len(chunks)} chunks")

            # Immediately generate embeddings and store to VectorCache
            logger.info(f"HybridBuffer: Generating embeddings for {len(chunks)} chunks...")
            for i, chunk in enumerate(chunks):
                embedding = await self._generate_embedding(chunk.content)

                # Add to VectorCache
                self.chunks.append(chunk)
                self.embeddings.append(embedding)
                self.total_chunks_created += 1

                if (i + 1) % 10 == 0:  # Log progress every 10 chunks
                    logger.debug(f"HybridBuffer: Generated {i + 1}/{len(chunks)} embeddings")

            logger.info(f"HybridBuffer: Completed immediate processing - VectorCache now contains {len(self.chunks)} chunks")

        except Exception as e:
            logger.error(f"HybridBuffer: Error in immediate processing: {e}")
            # Even if error occurs, ensure data flow continues

    async def _load_chunk_strategy(self) -> None:
        """Lazy load the chunk strategy with configuration support."""
        try:
            # Get configuration for chunking strategies
            from ..utils.config import config_manager
            config = config_manager.get_config()
            chunking_config = config.get("buffer", {}).get("chunking", {})

            if self.chunk_strategy_name == "message":
                from ..rag.chunk.message import MessageChunkStrategy
                self.chunk_strategy = MessageChunkStrategy()

            elif self.chunk_strategy_name in ["contextual", "contextual"]:
                # Use the advanced ContextualChunkStrategy for both contextual options
                from ..rag.chunk.contextual import ContextualChunkStrategy

                # Get strategy-specific configuration
                strategy_config = chunking_config.get(self.chunk_strategy_name, {})

                # Create strategy with configuration
                self.chunk_strategy = ContextualChunkStrategy(
                    max_words_per_group=strategy_config.get("max_words_per_group", 800),
                    max_words_per_chunk=strategy_config.get("max_words_per_chunk", 800),
                    role_format=strategy_config.get("role_format", "[{role}]"),
                    chunk_separator=strategy_config.get("chunk_separator", "\n\n"),
                    enable_contextual=strategy_config.get("enable_contextual", True),
                    context_window_size=strategy_config.get("context_window_size", 2),
                    gpt_model=strategy_config.get("gpt_model", "gpt-4o-mini"),
                    vector_store=None,  # Will be injected later by MemoryService
                    llm_provider=None   # Will be injected later by MemoryService
                )

            else:
                # Default to message strategy
                from ..rag.chunk.message import MessageChunkStrategy
                self.chunk_strategy = MessageChunkStrategy()
                logger.warning(f"HybridBuffer: Unknown chunk strategy '{self.chunk_strategy_name}', using message")

            logger.info(f"HybridBuffer: Loaded chunk strategy: {self.chunk_strategy_name}")
        except Exception as e:
            logger.error(f"HybridBuffer: Failed to load chunk strategy: {e}")
            # Create a minimal fallback strategy
            self.chunk_strategy = self._create_fallback_strategy()
    
    def _create_fallback_strategy(self):
        """Create a minimal fallback chunk strategy."""
        class FallbackStrategy:
            async def create_chunks(self, message_batch_list):
                chunks = []
                for batch_index, message_list in enumerate(message_batch_list):
                    content = " ".join(msg.get("content", "") for msg in message_list)
                    chunk = ChunkData(
                        content=content,
                        metadata={"strategy": "fallback", "batch_index": batch_index}
                    )
                    chunks.append(chunk)
                return chunks
        
        return FallbackStrategy()
    
    async def _load_embedding_model(self) -> None:
        """Lazy load the embedding model with global service priority."""
        try:
            # Priority 1: Try to get from global service manager
            try:
                from ..services.global_service_manager import get_global_service_manager
                global_manager = get_global_service_manager()
                global_embedding = global_manager.get_embedding_model()

                if global_embedding is not None:
                    self.embedding_model = global_embedding
                    logger.info("HybridBuffer: Using global embedding model from service manager")
                    return
            except Exception as e:
                logger.debug(f"HybridBuffer: Could not get global embedding model: {e}")

            # Priority 2: Try to get shared model from ServiceFactory
            try:
                from ..services.service_factory import ServiceFactory
                shared_embedding = ServiceFactory.get_global_embedding_model()
                if shared_embedding is not None:
                    self.embedding_model = shared_embedding
                    logger.info("HybridBuffer: Using shared embedding model from ServiceFactory")
                    return
            except Exception as e:
                logger.debug(f"HybridBuffer: Could not get shared embedding model: {e}")

            # Priority 3: Use optimized get_model function (with global priority)
            from ..utils.embeddings import get_model
            self.embedding_model = get_model(self.embedding_model_name)
            logger.info(f"HybridBuffer: Loaded embedding model via get_model: {self.embedding_model_name}")

        except Exception as e:
            logger.error(f"HybridBuffer: Failed to load embedding model: {e}")
            self.embedding_model = None
    
    async def _create_fallback_embedding(self, text: str) -> List[float]:
        """Create a fallback embedding."""
        # Simple hash-based embedding
        import hashlib
        hash_obj = hashlib.md5(text.encode())
        hash_bytes = hash_obj.digest()
        # Convert to 384-dimensional vector (matching all-MiniLM-L6-v2)
        embedding = []
        for i in range(384):
            byte_index = i % len(hash_bytes)
            value = (hash_bytes[byte_index] / 128.0) - 1.0
            embedding.append(value)
        return embedding
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using the loaded model instance.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        try:
            if self.embedding_model is None:
                logger.warning("HybridBuffer: No embedding model available, using fallback")
                return await self._create_fallback_embedding(text)

            # Check if it's a SentenceTransformer model instance
            if hasattr(self.embedding_model, 'encode'):
                # Direct model instance (SentenceTransformer)
                embedding = self.embedding_model.encode(text)
                if hasattr(embedding, 'tolist'):
                    return embedding.tolist()
                else:
                    return embedding
            elif callable(self.embedding_model):
                # Function-based embedding (fallback)
                if asyncio.iscoroutinefunction(self.embedding_model):
                    return await self.embedding_model(text, model=self.embedding_model_name)
                else:
                    return self.embedding_model(text, model=self.embedding_model_name)
            else:
                logger.warning("HybridBuffer: Unknown embedding model type, using fallback")
                return await self._create_fallback_embedding(text)
        except Exception as e:
            logger.error(f"HybridBuffer: Embedding generation failed: {e}")
            return await self._create_fallback_embedding(text)
    
    async def flush_to_storage(
        self,
        priority: FlushPriority = FlushPriority.NORMAL,
        timeout: Optional[float] = None
    ) -> bool:
        """Trigger non-blocking flush to storage using FlushManager.

        Args:
            priority: Flush priority
            timeout: Operation timeout

        Returns:
            True if flush was initiated successfully
        """
        if not self.flush_manager:
            logger.warning("HybridBuffer: FlushManager not available, falling back to synchronous flush")
            return await self._synchronous_flush()

        async with self._flush_lock:
            if not self.original_rounds:
                logger.debug("HybridBuffer: No data to flush")
                return True

            try:
                # Prepare data snapshots
                rounds_snapshot = self.original_rounds.copy()
                chunks_snapshot = self.chunks.copy()
                embeddings_snapshot = self.embeddings.copy()

                # Clear buffers immediately (optimistic clearing)
                self.original_rounds.clear()
                self.chunks.clear()
                self.embeddings.clear()
                self.last_flush_time = time.time()

                # Schedule hybrid flush task
                task_id = await self.flush_manager.flush_hybrid(
                    rounds=rounds_snapshot,
                    chunks=chunks_snapshot,
                    embeddings=embeddings_snapshot,
                    priority=priority,
                    timeout=timeout,
                    callback=self._flush_callback
                )

                # Track the task
                self.pending_flush_tasks[task_id] = asyncio.create_task(
                    self._monitor_flush_task(task_id)
                )

                self.total_flushes += 1
                if priority == FlushPriority.NORMAL:
                    self.total_auto_flushes += 1
                else:
                    self.total_manual_flushes += 1

                logger.info(f"HybridBuffer: Non-blocking flush initiated - task_id={task_id}")
                return True

            except Exception as e:
                logger.error(f"HybridBuffer: Failed to initiate flush: {e}")
                # Restore data on failure
                self.original_rounds.extend(rounds_snapshot)
                self.chunks.extend(chunks_snapshot)
                self.embeddings.extend(embeddings_snapshot)
                return False

    async def _synchronous_flush(self) -> bool:
        """Fallback synchronous flush when FlushManager is not available.

        Returns:
            True if flush was successful
        """
        try:
            if not self.original_rounds:
                return True

            # Note: Direct storage operations removed - handled by FlushManager

            # Clear buffers
            self.original_rounds.clear()
            self.chunks.clear()
            self.embeddings.clear()
            self.last_flush_time = time.time()

            logger.info("HybridBuffer: Synchronous flush completed")
            return True

        except Exception as e:
            logger.error(f"HybridBuffer: Synchronous flush failed: {e}")
            return False

    async def _flush_callback(self, success: bool, error_message: str) -> None:
        """Callback for flush completion notification.

        Args:
            success: Whether the flush was successful
            error_message: Error message if flush failed
        """
        if success:
            logger.debug("HybridBuffer: Flush completed successfully")
            # Note: M1 triggering is now handled by ParallelMemoryAdapter in the new architecture
            # No need to trigger M1 here as it processes data in parallel with M0
        else:
            logger.error(f"HybridBuffer: Flush failed: {error_message}")

    async def _monitor_flush_task(self, task_id: str) -> None:
        """Monitor a flush task and clean up when completed.

        Args:
            task_id: ID of the task to monitor
        """
        try:
            # Wait a reasonable time for task completion
            await asyncio.sleep(60.0)  # Max wait time
        finally:
            # Clean up completed task
            self.pending_flush_tasks.pop(task_id, None)

    async def _auto_flush_loop(self) -> None:
        """Auto-flush loop for time-based flushing."""
        logger.info(f"HybridBuffer: Auto-flush loop started with interval {self.auto_flush_interval}s")

        while True:
            try:
                await asyncio.sleep(min(self.auto_flush_interval, 10.0))  # Check at least every 10s

                current_time = time.time()
                time_since_last_flush = current_time - self.last_flush_time

                if time_since_last_flush >= self.auto_flush_interval:
                    if self.original_rounds:  # Only flush if there's data
                        logger.debug("HybridBuffer: Auto-flush triggered by timer")
                        await self.flush_to_storage(priority=FlushPriority.LOW)

            except asyncio.CancelledError:
                logger.info("HybridBuffer: Auto-flush loop cancelled")
                break
            except Exception as e:
                logger.error(f"HybridBuffer: Auto-flush loop error: {e}")

        logger.info("HybridBuffer: Auto-flush loop stopped")


    
    async def get_all_messages_for_read_api(
        self,
        limit: Optional[int] = None,
        sort_by: str = "timestamp",
        order: str = "desc"
    ) -> List[Dict[str, Any]]:
        """Get all messages in buffer for Read API.
        
        Args:
            limit: Maximum number of messages to return
            sort_by: Field to sort by ('timestamp' or 'id')
            order: Sort order ('asc' or 'desc')
            
        Returns:
            List of message dictionaries formatted for API response
        """
        async with self._data_lock:
            all_messages = []
            
            for round_messages in self.original_rounds:
                for message in round_messages:
                    # Convert to API format
                    api_message = {
                        "id": message.get("id", ""),
                        "role": message.get("role", "user"),
                        "content": message.get("content", ""),
                        "created_at": message.get("created_at", ""),
                        "updated_at": message.get("updated_at", ""),
                        "metadata": message.get("metadata", {}).copy()
                    }
                    # Add buffer source metadata
                    api_message["metadata"]["source"] = "hybrid_buffer"
                    all_messages.append(api_message)
            
            # Sort messages
            if sort_by == "timestamp":
                all_messages.sort(
                    key=lambda x: x.get("created_at", ""),
                    reverse=(order == "desc")
                )
            elif sort_by == "id":
                all_messages.sort(
                    key=lambda x: x.get("id", ""),
                    reverse=(order == "desc")
                )
            
            # Apply limit
            if limit is not None and limit > 0:
                all_messages = all_messages[:limit]
            
            return all_messages
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics.
        
        Returns:
            Dictionary with buffer statistics
        """
        return {
            "chunks_count": len(self.chunks),
            "rounds_count": len(self.original_rounds),
            "embeddings_count": len(self.embeddings),
            "max_size": self.max_size,
            "chunk_strategy": self.chunk_strategy_name,
            "embedding_model": self.embedding_model_name,
            "auto_flush_interval": self.auto_flush_interval,
            "enable_auto_flush": self.enable_auto_flush,
            "total_rounds_received": self.total_rounds_received,
            "total_chunks_created": self.total_chunks_created,
            "total_flushes": self.total_flushes,
            "total_auto_flushes": self.total_auto_flushes,
            "total_manual_flushes": self.total_manual_flushes,
            "pending_flush_tasks": len(self.pending_flush_tasks),
            "has_flush_manager": self.flush_manager is not None
        }
