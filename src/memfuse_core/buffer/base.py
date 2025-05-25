"""Base buffer implementation for the hierarchical memory system."""

from typing import Any, List, Optional, TypeVar, Generic, Callable, Awaitable, Dict
import asyncio
import time
import threading
from collections import deque
from loguru import logger

T = TypeVar('T')  # Item type
R = TypeVar('R')  # Result type


class BufferBase(Generic[T]):
    """Base class for all buffer implementations in the hierarchical memory system.

    Features:
    - Configurable batch size and flush interval
    - Automatic flushing based on size or time
    - Prioritized items
    - Batch optimization based on item types
    - Statistics tracking
    - Thread-safe operations
    """

    def __init__(
        self,
        max_size: int = 10,
        max_age: Optional[float] = None,
        flush_callback: Optional[Callable[[
            List[T]], Awaitable[List[Any]]]] = None,
        auto_flush: bool = True,
        adaptive_batching: bool = False,
        min_batch_size: int = 1,
        thread_safe: bool = True
    ):
        """Initialize the buffer.

        Args:
            max_size: Maximum number of items in the buffer
            max_age: Maximum age of items in seconds before auto-flush
            flush_callback: Async callback function to call when buffer is flushed
            auto_flush: Whether to automatically flush the buffer
            adaptive_batching: Whether to adapt batch size based on performance
            min_batch_size: Minimum batch size for flushing
            thread_safe: Whether to make buffer operations thread-safe
        """
        self.max_size = max_size
        self.max_age = max_age
        self.flush_callback = flush_callback
        self.auto_flush = auto_flush
        self.adaptive_batching = adaptive_batching
        self.min_batch_size = min_batch_size
        self.thread_safe = thread_safe

        # Buffer storage
        self.items: List[T] = []
        self.priority_items: deque[T] = deque()  # High-priority items
        self.last_flush_time = time.time()

        # Lock for thread safety
        self._lock = asyncio.Lock()
        self._thread_lock = threading.RLock() if thread_safe else None

        # Statistics
        self.total_items_added = 0
        self.total_flushes = 0
        self.total_flush_time = 0
        self.last_batch_size = 0
        self.optimal_batch_size = max_size

        # Initialize auto-flush task
        self._auto_flush_task = None
        self._auto_flush_enabled = auto_flush and max_age is not None

    def _acquire_thread_lock(self):
        """Acquire the thread lock if thread safety is enabled."""
        if self._thread_lock:
            self._thread_lock.acquire()

    def _release_thread_lock(self):
        """Release the thread lock if thread safety is enabled."""
        if self._thread_lock:
            self._thread_lock.release()

    async def initialize(self):
        """Initialize the buffer.

        This method should be called after the buffer is created,
        and before it is used. It initializes any async resources
        that need to be created in an event loop.
        """
        # Start auto-flush task if needed
        if self._auto_flush_enabled and self._auto_flush_task is None:
            self._auto_flush_task = asyncio.create_task(self._auto_flush_loop())

    async def _auto_flush_loop(self):
        """Background task that periodically flushes the buffer based on max_age."""
        if self.max_age is None:
            return

        while True:
            try:
                # Check at least every second
                await asyncio.sleep(min(self.max_age, 1.0))

                current_time = time.time()
                if current_time - self.last_flush_time >= self.max_age:
                    if len(self.items) > 0 or len(self.priority_items) > 0:
                        await self.flush()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in auto-flush loop: {e}")

    async def add(self, item: T, priority: bool = False) -> bool:
        """Add an item to the buffer.

        Args:
            item: Item to add
            priority: Whether this is a high-priority item

        Returns:
            True if buffer was flushed, False otherwise
        """
        # Update statistics
        self.total_items_added += 1

        # Add to appropriate buffer
        if priority:
            self.priority_items.append(item)
            # If priority items exceed max_size, remove oldest
            if len(self.priority_items) > self.max_size:
                self.priority_items.popleft()  # Remove oldest item (FIFO)
        else:
            self.items.append(item)
            # If regular items exceed max_size, remove oldest
            if len(self.items) > self.max_size:
                self.items.pop(0)  # Remove oldest item (FIFO)

        # Note: We no longer automatically flush when reaching max_size
        # This allows the buffer to maintain the most recent items

        # Check if we have enough items for a minimum batch and should flush
        # based on adaptive batching
        total_items = len(self.items) + len(self.priority_items)
        should_flush = (self.adaptive_batching and self.auto_flush and
                        total_items >= self.min_batch_size)
        if should_flush:
            if total_items >= self.optimal_batch_size:
                # Only flush if auto_flush is enabled
                await self.flush()
                return True

        return False

    async def add_batch(self, items: List[T], priority: bool = False) -> bool:
        """Add multiple items to the buffer.

        Args:
            items: Items to add
            priority: Whether these are high-priority items

        Returns:
            True if buffer was flushed, False otherwise
        """
        if not items:
            return False

        # Update statistics
        self.total_items_added += len(items)

        # Add to appropriate buffer
        if priority:
            # Add items to priority buffer
            for item in items:
                self.priority_items.append(item)
                # If priority items exceed max_size, remove oldest
                if len(self.priority_items) > self.max_size:
                    self.priority_items.popleft()  # Remove oldest item (FIFO)
        else:
            # Add items to regular buffer
            self.items.extend(items)
            # If regular items exceed max_size, remove oldest items
            while len(self.items) > self.max_size:
                self.items.pop(0)  # Remove oldest item (FIFO)

        # Note: We no longer automatically flush when reaching max_size
        # This allows the buffer to maintain the most recent items

        # Check if we have enough items for a minimum batch and should flush
        # based on adaptive batching
        total_items = len(self.items) + len(self.priority_items)
        should_flush = (self.adaptive_batching and self.auto_flush and
                        total_items >= self.min_batch_size)
        if should_flush:
            if total_items >= self.optimal_batch_size:
                # Only flush if auto_flush is enabled
                await self.flush()
                return True

        return False

    async def flush(self) -> List[Any]:
        """Flush the buffer.

        Returns:
            Results from the flush callback, or empty list if no callback
        """
        logger.debug("Flush method called")

        # Combine priority items and regular items
        all_items = list(self.priority_items) + self.items
        logger.debug(f"Combined {len(all_items)} items")

        # Clear the buffers
        self.priority_items.clear()
        self.items.clear()
        logger.debug("Buffers cleared")

        # Update statistics
        self.total_flushes += 1
        self.last_batch_size = len(all_items)
        self.last_flush_time = time.time()
        logger.debug("Statistics updated")

        # If no items or no callback, return empty list
        if not all_items or not self.flush_callback:
            logger.debug("No items or no callback, returning empty list")
            return []

        logger.debug(f"Calling flush callback with {len(all_items)} items")
        # Call the flush callback
        start_time = time.time()
        try:
            logger.debug("Executing flush callback")
            results = await self.flush_callback(all_items)
            result_count = len(results) if results else 0
            logger.debug(f"Flush callback returned {result_count} results")

            flush_time = time.time() - start_time
            self.total_flush_time += flush_time
            logger.debug(f"Flush took {flush_time:.4f} seconds")

            # Update optimal batch size if adaptive batching is enabled
            if self.adaptive_batching and self.total_flushes > 1:
                # Simple adaptive strategy: adjust based on flush time
                if flush_time < 0.1:  # Very fast flush
                    self.optimal_batch_size = min(
                        self.optimal_batch_size * 1.2,
                        self.max_size
                    )
                elif flush_time > 1.0:  # Slow flush
                    self.optimal_batch_size = max(
                        self.optimal_batch_size * 0.8,
                        self.min_batch_size
                    )
                logger.debug(
                    f"Updated optimal batch size to {self.optimal_batch_size}")

            return results
        except Exception as e:
            logger.error(f"Error in flush callback: {e}", exc_info=True)
            return []

    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics.

        Returns:
            Dictionary of buffer statistics
        """
        avg_flush_time = 0
        if self.total_flushes > 0:
            avg_flush_time = self.total_flush_time / self.total_flushes

        return {
            "total_items_added": self.total_items_added,
            "total_flushes": self.total_flushes,
            "total_flush_time": self.total_flush_time,
            "avg_flush_time": avg_flush_time,
            "last_batch_size": self.last_batch_size,
            "current_size": len(self.items) + len(self.priority_items),
            "optimal_batch_size": self.optimal_batch_size,
        }

    async def close(self):
        """Close the buffer and clean up resources."""
        logger.debug("Closing buffer")

        # Cancel auto-flush task if it exists
        if self._auto_flush_task:
            logger.debug("Cancelling auto-flush task")
            self._auto_flush_task.cancel()
            try:
                await self._auto_flush_task
                logger.debug("Auto-flush task cancelled")
            except asyncio.CancelledError:
                logger.debug("Auto-flush task cancelled with CancelledError")
                pass

        # Flush any remaining items
        remaining_items = len(self.items) + len(self.priority_items)
        if remaining_items > 0:
            logger.debug(f"Flushing {remaining_items} remaining items")
            await self.flush()

        logger.debug("Buffer closed")
