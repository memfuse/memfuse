"""Unified Flush Manager for MemFuse Buffer System.

This module provides a centralized, asynchronous, and non-blocking flush mechanism
that handles all buffer-to-storage operations with proper timeout, retry, and
error handling capabilities.
"""

import asyncio
import time
from typing import Any, List, Optional, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger

from ..interfaces import MessageList
from ..rag.chunk.base import ChunkData


class FlushStrategy(Enum):
    """Flush strategy enumeration."""
    SIZE_BASED = "size_based"
    TIME_BASED = "time_based"
    HYBRID = "hybrid"
    IMMEDIATE = "immediate"


class FlushPriority(Enum):
    """Flush priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class FlushTask:
    """Represents a single flush task."""
    task_id: str
    priority: FlushPriority
    data_type: str  # "sqlite", "qdrant", "hybrid"
    data: Any
    callback: Optional[Callable[[bool, str], Awaitable[None]]] = None
    created_at: float = field(default_factory=time.time)
    timeout: float = 30.0
    max_retries: int = 3
    retry_count: int = 0

    def __lt__(self, other):
        """Enable comparison for PriorityQueue when priorities are equal."""
        if not isinstance(other, FlushTask):
            return NotImplemented
        # Compare by creation time if priorities are equal
        return self.created_at < other.created_at


@dataclass
class FlushMetrics:
    """Flush operation metrics."""
    total_flushes: int = 0
    successful_flushes: int = 0
    failed_flushes: int = 0
    timeout_flushes: int = 0
    avg_flush_time: float = 0.0
    max_flush_time: float = 0.0
    min_flush_time: float = float('inf')
    total_flush_time: float = 0.0
    queue_size: int = 0
    active_workers: int = 0


class FlushManager:
    """Unified flush manager for all buffer operations.
    
    This manager provides:
    - Asynchronous, non-blocking flush operations
    - Priority-based task queue
    - Configurable timeout and retry mechanisms
    - Comprehensive error handling and recovery
    - Performance monitoring and metrics
    """
    
    def __init__(
        self,
        max_workers: int = 3,
        max_queue_size: int = 100,
        default_timeout: float = 30.0,
        flush_interval: float = 60.0,
        enable_auto_flush: bool = True,
        memory_service_handler: Optional[Callable] = None,
        qdrant_handler: Optional[Callable] = None
    ):
        """Initialize the flush manager.

        Args:
            max_workers: Maximum number of concurrent flush workers
            max_queue_size: Maximum size of the flush queue
            default_timeout: Default timeout for flush operations
            flush_interval: Interval for auto-flush operations
            enable_auto_flush: Whether to enable automatic flushing
            memory_service_handler: Handler for MemoryService operations
            qdrant_handler: Handler for Qdrant operations (legacy, will be removed)
        """
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        self.default_timeout = default_timeout
        self.flush_interval = flush_interval
        self.enable_auto_flush = enable_auto_flush

        # Storage handlers
        self.memory_service_handler = memory_service_handler
        self.qdrant_handler = qdrant_handler  # Legacy support
        
        # Task queue and workers
        self.task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue(maxsize=max_queue_size)
        self.workers: List[asyncio.Task] = []
        self.worker_semaphore = asyncio.Semaphore(max_workers)
        self._task_counter = 0  # Unique sequence number for queue ordering
        
        # Auto-flush timer
        self.auto_flush_task: Optional[asyncio.Task] = None
        self.last_auto_flush = time.time()
        
        # Metrics and monitoring
        self.metrics = FlushMetrics()
        self._lock = asyncio.Lock()
        self._shutdown = False
        self._task_counter = 0
        
        logger.info(f"FlushManager initialized: workers={max_workers}, queue_size={max_queue_size}")
    
    async def initialize(self) -> bool:
        """Initialize the flush manager and start workers.
        
        Returns:
            True if initialization was successful
        """
        try:
            # Start worker tasks
            for i in range(self.max_workers):
                worker = asyncio.create_task(self._worker_loop(f"worker-{i}"))
                self.workers.append(worker)
            
            # Start auto-flush task if enabled
            if self.enable_auto_flush and self.flush_interval > 0:
                self.auto_flush_task = asyncio.create_task(self._auto_flush_loop())
            
            logger.info(f"FlushManager started with {len(self.workers)} workers")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize FlushManager: {e}")
            return False
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the flush manager."""
        logger.info("FlushManager shutdown initiated")
        self._shutdown = True
        
        # Cancel auto-flush task
        if self.auto_flush_task:
            self.auto_flush_task.cancel()
            try:
                await self.auto_flush_task
            except asyncio.CancelledError:
                pass
        
        # Wait for queue to empty (with timeout)
        try:
            await asyncio.wait_for(self.task_queue.join(), timeout=30.0)
        except asyncio.TimeoutError:
            logger.warning("FlushManager shutdown: queue did not empty within timeout")
        
        # Cancel all workers
        for worker in self.workers:
            worker.cancel()
        
        # Wait for workers to finish
        if self.workers:
            await asyncio.gather(*self.workers, return_exceptions=True)
        
        logger.info("FlushManager shutdown completed")
    
    def set_handlers(
        self,
        memory_service_handler: Optional[Callable] = None,
        qdrant_handler: Optional[Callable] = None
    ) -> None:
        """Set or update storage handlers.

        Args:
            memory_service_handler: Handler for MemoryService operations
            qdrant_handler: Handler for Qdrant operations (legacy)
        """
        if memory_service_handler:
            self.memory_service_handler = memory_service_handler
            logger.debug("FlushManager: MemoryService handler updated")

        if qdrant_handler:
            self.qdrant_handler = qdrant_handler
            logger.debug("FlushManager: Qdrant handler updated")
    
    async def flush_messages(
        self,
        rounds: List[MessageList],
        priority: FlushPriority = FlushPriority.NORMAL,
        timeout: Optional[float] = None,
        callback: Optional[Callable[[bool, str], Awaitable[None]]] = None
    ) -> str:
        """Schedule message flush operation to persistent storage.

        Args:
            rounds: List of message rounds to flush
            priority: Task priority
            timeout: Operation timeout
            callback: Optional callback for completion notification

        Returns:
            Task ID for tracking
        """
        if not self.memory_service_handler:
            raise ValueError("MemoryService handler not configured")

        task_id = self._generate_task_id("messages")
        task = FlushTask(
            task_id=task_id,
            priority=priority,
            data_type="messages",
            data=rounds,
            callback=callback,
            timeout=timeout or self.default_timeout
        )

        await self._enqueue_task(task)
        logger.debug(f"Messages flush task queued: {task_id}, rounds={len(rounds)}")
        return task_id
    
    async def flush_qdrant(
        self,
        chunks: List[ChunkData],
        embeddings: List[List[float]],
        priority: FlushPriority = FlushPriority.NORMAL,
        timeout: Optional[float] = None,
        callback: Optional[Callable[[bool, str], Awaitable[None]]] = None
    ) -> str:
        """Schedule Qdrant flush operation.
        
        Args:
            chunks: List of chunks to flush
            embeddings: List of embeddings
            priority: Task priority
            timeout: Operation timeout
            callback: Optional callback for completion notification
            
        Returns:
            Task ID for tracking
        """
        if not self.qdrant_handler:
            raise ValueError("Qdrant handler not configured")
        
        task_id = self._generate_task_id("qdrant")
        task = FlushTask(
            task_id=task_id,
            priority=priority,
            data_type="qdrant",
            data={"chunks": chunks, "embeddings": embeddings},
            callback=callback,
            timeout=timeout or self.default_timeout
        )
        
        await self._enqueue_task(task)
        logger.debug(f"Qdrant flush task queued: {task_id}, chunks={len(chunks)}")
        return task_id

    async def flush_buffer_data(
        self,
        rounds: List[MessageList],
        priority: FlushPriority = FlushPriority.NORMAL,
        timeout: Optional[float] = None,
        callback: Optional[Callable[[bool, str], Awaitable[None]]] = None
    ) -> str:
        """Schedule buffer data flush operation to MemoryService.

        This method only flushes the rounds data to MemoryService, which will handle
        the routing to appropriate storage layers (PostgreSQL rounds/messages tables).
        VectorCache data (chunks/embeddings) is not persisted - it's only used for
        immediate querying and will be regenerated by Memory Layer processing.

        Args:
            rounds: List of message rounds to flush
            priority: Task priority
            timeout: Operation timeout
            callback: Optional callback for completion notification

        Returns:
            Task ID for tracking
        """
        if not self.memory_service_handler:
            raise ValueError("MemoryService handler must be configured for buffer flush")

        logger.debug("FlushManager: Using MemoryService handler for buffer data flush")

        task_id = self._generate_task_id("buffer_data")
        task = FlushTask(
            task_id=task_id,
            priority=priority,
            data_type="buffer_data",
            data={"rounds": rounds},
            callback=callback,
            timeout=timeout or self.default_timeout
        )

        await self._enqueue_task(task)
        logger.debug(f"Buffer data flush task queued: {task_id}, rounds={len(rounds)}")
        return task_id

    async def get_metrics(self) -> FlushMetrics:
        """Get current flush metrics.

        Returns:
            Current metrics snapshot
        """
        async with self._lock:
            # Update queue size
            self.metrics.queue_size = self.task_queue.qsize()
            self.metrics.active_workers = sum(1 for w in self.workers if not w.done())
            return FlushMetrics(**self.metrics.__dict__)

    def _generate_task_id(self, prefix: str) -> str:
        """Generate unique task ID.

        Args:
            prefix: Task type prefix

        Returns:
            Unique task ID
        """
        self._task_counter += 1
        return f"{prefix}-{int(time.time())}-{self._task_counter}"

    async def _enqueue_task(self, task: FlushTask) -> None:
        """Enqueue a flush task.

        Args:
            task: Flush task to enqueue

        Raises:
            asyncio.QueueFull: If queue is full
        """
        try:
            # Priority queue uses tuple (priority, sequence, task)
            # Lower priority value = higher priority
            # Sequence number ensures deterministic ordering for same priority
            self._task_counter += 1
            await self.task_queue.put((task.priority.value, self._task_counter, task))

            async with self._lock:
                self.metrics.queue_size = self.task_queue.qsize()

        except asyncio.QueueFull:
            logger.error(f"Flush queue full, dropping task: {task.task_id}")
            raise

    async def _worker_loop(self, worker_name: str) -> None:
        """Main worker loop for processing flush tasks.

        Args:
            worker_name: Name of the worker for logging
        """
        logger.info(f"FlushManager worker started: {worker_name}")

        while not self._shutdown:
            try:
                # Get task from queue with timeout
                try:
                    _, _, task = await asyncio.wait_for(
                        self.task_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                # Process the task
                await self._process_task(task, worker_name)

                # Mark task as done
                self.task_queue.task_done()

            except asyncio.CancelledError:
                logger.info(f"FlushManager worker cancelled: {worker_name}")
                break
            except Exception as e:
                logger.error(f"FlushManager worker error in {worker_name}: {e}")
                # Continue processing other tasks

        logger.info(f"FlushManager worker stopped: {worker_name}")

    async def _process_task(self, task: FlushTask, worker_name: str) -> None:
        """Process a single flush task.

        Args:
            task: Task to process
            worker_name: Name of the worker processing the task
        """
        start_time = time.time()
        success = False
        error_message = ""

        try:
            logger.debug(f"Processing flush task: {task.task_id} by {worker_name}")

            # Execute the task with timeout
            success = await asyncio.wait_for(
                self._execute_task(task),
                timeout=task.timeout
            )

            if success:
                logger.debug(f"Flush task completed successfully: {task.task_id}")
            else:
                error_message = "Task execution returned False"
                logger.warning(f"Flush task failed: {task.task_id}")

        except asyncio.TimeoutError:
            error_message = f"Task timed out after {task.timeout}s"
            logger.error(f"Flush task timeout: {task.task_id} - {error_message}")

            async with self._lock:
                self.metrics.timeout_flushes += 1

        except Exception as e:
            error_message = str(e)
            logger.error(f"Flush task error: {task.task_id} - {error_message}")

        # Update metrics
        duration = time.time() - start_time
        await self._update_metrics(success, duration)

        # Handle retry logic
        if not success and task.retry_count < task.max_retries:
            task.retry_count += 1
            retry_delay = min(2 ** task.retry_count, 30)  # Exponential backoff, max 30s

            logger.info(f"Retrying flush task: {task.task_id}, attempt {task.retry_count}/{task.max_retries} in {retry_delay}s")

            # Schedule retry
            asyncio.create_task(self._schedule_retry(task, retry_delay))
        else:
            # Call callback if provided
            if task.callback:
                try:
                    await task.callback(success, error_message)
                except Exception as e:
                    logger.error(f"Callback error for task {task.task_id}: {e}")

    async def _execute_task(self, task: FlushTask) -> bool:
        """Execute a specific flush task.

        Args:
            task: Task to execute

        Returns:
            True if successful, False otherwise
        """
        try:
            if task.data_type == "messages":
                return await self._execute_messages_flush(task.data)
            elif task.data_type == "qdrant":
                return await self._execute_qdrant_flush(task.data["chunks"], task.data["embeddings"])
            elif task.data_type == "buffer_data":
                return await self._execute_buffer_data_flush(task.data["rounds"])
            else:
                logger.error(f"Unknown task data type: {task.data_type}")
                return False

        except Exception as e:
            logger.error(f"Task execution error: {e}")
            return False

    async def _execute_messages_flush(self, rounds: List[MessageList]) -> bool:
        """Execute messages flush operation through MemoryService.

        Args:
            rounds: List of message rounds to flush

        Returns:
            True if successful, False otherwise
        """
        max_retries = 3
        retry_delay = 0.5

        for attempt in range(max_retries):
            try:
                if self.memory_service_handler:
                    await self.memory_service_handler(rounds)
                    logger.debug(f"Messages flush completed: {len(rounds)} rounds")
                    return True
                else:
                    logger.error("MemoryService handler not available")
                    return False

            except Exception as e:
                error_msg = str(e).lower()

                # Check for connection pool related errors
                if any(keyword in error_msg for keyword in ["connection", "pool", "timeout", "server closed"]):
                    if attempt < max_retries - 1:
                        logger.warning(f"FlushManager: Connection issue on attempt {attempt + 1}, retrying in {retry_delay}s: {e}")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 1.5  # Moderate backoff
                        continue

                logger.error(f"Messages flush error after {attempt + 1} attempts: {e}")
                return False

        return False

    async def _execute_qdrant_flush(self, chunks: List[ChunkData], embeddings: List[List[float]]) -> bool:
        """Execute Qdrant flush operation.

        Args:
            chunks: List of chunks to flush
            embeddings: List of embeddings

        Returns:
            True if successful, False otherwise
        """
        try:
            if self.qdrant_handler:
                await self.qdrant_handler(chunks, embeddings)
                logger.debug(f"Qdrant flush completed: {len(chunks)} chunks")
                return True
            else:
                logger.error("Qdrant handler not available")
                return False
        except Exception as e:
            logger.error(f"Qdrant flush error: {e}")
            return False

    async def _execute_hybrid_flush(
        self,
        rounds: List[MessageList],
        chunks: List[ChunkData],
        embeddings: List[List[float]]
    ) -> bool:
        """Execute hybrid flush operation using unified MemoryService handler.

        Since we now use a unified MemoryService handler instead of separate SQLite/Qdrant handlers,
        this method only needs to call the SQLite handler (which is actually the MemoryService handler).
        The MemoryService will handle routing to M0/M1/M2 layers and all storage operations.

        Args:
            rounds: List of message rounds to flush
            chunks: List of chunks to flush (processed by MemoryService)
            embeddings: List of embeddings (processed by MemoryService)

        Returns:
            True if MemoryService processing successful, False otherwise
        """
        # Only call the unified MemoryService handler (mapped to memory_service_handler)
        # The chunks and embeddings are already processed and stored in the rounds data
        success = await self._execute_messages_flush(rounds)

        if success:
            logger.info(f"Unified MemoryService flush completed: {len(rounds)} rounds processed through M0/M1/M2 hierarchy")
        else:
            logger.error(f"Unified MemoryService flush failed for {len(rounds)} rounds")

        return success

    async def _execute_buffer_data_flush(self, rounds: List[MessageList]) -> bool:
        """Execute buffer data flush operation through MemoryService.

        This method only flushes the rounds data to MemoryService, which will handle
        the routing to appropriate storage layers (PostgreSQL rounds/messages tables).
        VectorCache data (chunks/embeddings) is not persisted - it's only used for
        immediate querying and will be regenerated by Memory Layer processing.

        Args:
            rounds: List of message rounds to flush

        Returns:
            True if successful, False otherwise
        """
        # Delegate to the common messages flush implementation
        return await self._execute_messages_flush(rounds)

    async def _update_metrics(self, success: bool, duration: float) -> None:
        """Update flush metrics.

        Args:
            success: Whether the operation was successful
            duration: Operation duration in seconds
        """
        async with self._lock:
            self.metrics.total_flushes += 1

            if success:
                self.metrics.successful_flushes += 1
            else:
                self.metrics.failed_flushes += 1

            # Update timing metrics
            self.metrics.total_flush_time += duration
            self.metrics.avg_flush_time = self.metrics.total_flush_time / self.metrics.total_flushes
            self.metrics.max_flush_time = max(self.metrics.max_flush_time, duration)
            self.metrics.min_flush_time = min(self.metrics.min_flush_time, duration)

    async def _schedule_retry(self, task: FlushTask, delay: float) -> None:
        """Schedule a task retry after delay.

        Args:
            task: Task to retry
            delay: Delay in seconds before retry
        """
        await asyncio.sleep(delay)
        try:
            await self._enqueue_task(task)
            logger.debug(f"Task retry scheduled: {task.task_id}")
        except asyncio.QueueFull:
            logger.error(f"Failed to schedule retry for task {task.task_id}: queue full")

    async def _auto_flush_loop(self) -> None:
        """Auto-flush loop that triggers periodic flushes."""
        logger.info(f"Auto-flush loop started: interval={self.flush_interval}s")

        while not self._shutdown:
            try:
                await asyncio.sleep(min(self.flush_interval, 10.0))  # Check at least every 10s

                current_time = time.time()
                if current_time - self.last_auto_flush >= self.flush_interval:
                    logger.debug("Auto-flush triggered")
                    # This would trigger buffer flushes - implementation depends on buffer integration
                    self.last_auto_flush = current_time

            except asyncio.CancelledError:
                logger.info("Auto-flush loop cancelled")
                break
            except Exception as e:
                logger.error(f"Auto-flush loop error: {e}")

        logger.info("Auto-flush loop stopped")
