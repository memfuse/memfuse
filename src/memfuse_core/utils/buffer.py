"""Enhanced buffering utilities for MemFuse.

This module provides advanced buffering implementations for batch operations,
with features like automatic flushing, prioritization, and statistics tracking.
"""

from typing import Any, Dict, List, Optional, TypeVar, Generic, Callable, Awaitable
from collections import deque
import time
import asyncio
import threading
from loguru import logger

T = TypeVar('T')  # Item type


class Buffer(Generic[T]):
    """Enhanced buffer implementation for batch operations.

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
        flush_callback: Optional[Callable[[List[T]], Awaitable[bool]]] = None,
        min_batch_size: int = 1,
        thread_safe: bool = True
    ):
        """Initialize the buffer.

        Args:
            max_size: Maximum number of items in the buffer before auto-flush
            max_age: Maximum age of items in seconds before auto-flush
            flush_callback: Async callback function to flush items
            min_batch_size: Minimum batch size for flushing
            thread_safe: Whether to make buffer operations thread-safe
        """
        self.max_size = max_size
        self.max_age = max_age
        self.flush_callback = flush_callback
        self.min_batch_size = min_batch_size
        self.thread_safe = thread_safe

        # Initialize lock if thread-safe
        self._lock = threading.RLock() if thread_safe else None

        # Buffer storage
        self.items = deque()
        self.priority_items = deque()
        self.last_flush_time = time.time()

        # Buffer statistics
        self.total_items_added = 0
        self.total_items_flushed = 0
        self.total_flushes = 0
        self.total_auto_flushes = 0
        self.total_manual_flushes = 0
        self.total_priority_items = 0
        self.total_flush_time = 0

        # Auto-flush timer
        self._flush_timer = None
        if self.max_age is not None:
            self._start_flush_timer()

    def _acquire_lock(self):
        """Acquire the lock if thread-safe is enabled."""
        if self._lock:
            self._lock.acquire()

    def _release_lock(self):
        """Release the lock if thread-safe is enabled."""
        if self._lock:
            self._lock.release()

    def _start_flush_timer(self):
        """Start the auto-flush timer."""
        if self.max_age is not None:
            async def _timer_task():
                while True:
                    await asyncio.sleep(self.max_age)
                    if time.time() - self.last_flush_time >= self.max_age:
                        await self.flush(auto=True)

            self._flush_timer = asyncio.create_task(_timer_task())

    def _cancel_flush_timer(self):
        """Cancel the auto-flush timer."""
        if self._flush_timer is not None:
            self._flush_timer.cancel()
            self._flush_timer = None

    def add(self, item: T, priority: bool = False) -> bool:
        """Add an item to the buffer.

        Args:
            item: Item to add
            priority: Whether the item should be prioritized

        Returns:
            Whether a flush was triggered
        """
        try:
            self._acquire_lock()

            # Add item to buffer
            if priority:
                self.priority_items.append(item)
                self.total_priority_items += 1
            else:
                self.items.append(item)

            self.total_items_added += 1

            # Check if buffer should be flushed
            total_items = len(self.items) + len(self.priority_items)
            if total_items >= self.max_size:
                # Schedule a flush
                asyncio.create_task(self.flush(auto=True))
                return True

            return False
        finally:
            self._release_lock()

    async def add_batch(self, items: List[T], priority: bool = False) -> bool:
        """Add a batch of items to the buffer.

        Args:
            items: Items to add
            priority: Whether the items should be prioritized

        Returns:
            Whether a flush was triggered
        """
        try:
            self._acquire_lock()

            # Add items to buffer
            if priority:
                self.priority_items.extend(items)
                self.total_priority_items += len(items)
            else:
                self.items.extend(items)

            self.total_items_added += len(items)

            # Check if buffer should be flushed
            total_items = len(self.items) + len(self.priority_items)
            if total_items >= self.max_size:
                await self.flush(auto=True)
                return True

            # Check if we have enough items for a minimum batch
            if total_items >= self.min_batch_size:
                await self.flush(auto=True)
                return True

            return False
        finally:
            self._release_lock()

    async def flush(self, auto: bool = False) -> bool:
        """Flush the buffer.

        Args:
            auto: Whether this is an automatic flush

        Returns:
            Whether the flush was successful
        """
        if self.flush_callback is None:
            logger.warning("No flush callback provided, items will be discarded")
            return False

        try:
            self._acquire_lock()

            # Check if there are items to flush
            total_items = len(self.items) + len(self.priority_items)
            if total_items == 0:
                return True

            # Check if we have enough items for a minimum batch
            if total_items < self.min_batch_size and not auto:
                return False

            # Combine items, with priority items first
            combined_items = list(self.priority_items) + list(self.items)

            # Clear buffer
            self.priority_items.clear()
            self.items.clear()

            # Update statistics
            self.total_items_flushed += total_items
            self.total_flushes += 1
            if auto:
                self.total_auto_flushes += 1
            else:
                self.total_manual_flushes += 1

            # Update last flush time
            self.last_flush_time = time.time()
        finally:
            self._release_lock()

        # Call flush callback
        start_time = time.time()
        try:
            result = await self.flush_callback(combined_items)
            self.total_flush_time += time.time() - start_time
            return result
        except Exception as e:
            logger.error(f"Error in flush callback: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics.

        Returns:
            Dictionary with buffer statistics
        """
        try:
            self._acquire_lock()

            stats = {
                "current_size": len(self.items) + len(self.priority_items),
                "current_regular_items": len(self.items),
                "current_priority_items": len(self.priority_items),
                "max_size": self.max_size,
                "max_age": self.max_age,
                "min_batch_size": self.min_batch_size,
                "total_items_added": self.total_items_added,
                "total_items_flushed": self.total_items_flushed,
                "total_flushes": self.total_flushes,
                "total_auto_flushes": self.total_auto_flushes,
                "total_manual_flushes": self.total_manual_flushes,
                "total_priority_items": self.total_priority_items,
                "avg_flush_time": self.total_flush_time / self.total_flushes if self.total_flushes > 0 else 0,
                "time_since_last_flush": time.time() - self.last_flush_time,
            }

            return stats
        finally:
            self._release_lock()

    async def shutdown(self) -> bool:
        """Shut down the buffer.

        Returns:
            Whether the shutdown was successful
        """
        # Cancel auto-flush timer
        self._cancel_flush_timer()

        # Flush any remaining items
        return await self.flush(auto=False)
