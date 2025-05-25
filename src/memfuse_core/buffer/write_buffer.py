"""WriteBuffer implementation for MemFuse.

    The WriteBuffer collects items to be written to storage in batches,
    improving write performance by reducing the number of database operations.

    This is a generic buffer that can handle any type of data items, not just messages.

    CRITICAL FIFO LOGIC - DO NOT CLEAR BUFFER AFTER BATCH WRITE:

    1. Initial Phase: Data accumulates until max_size (e.g., 5 items)
    2. First Batch Write: When reaching max_size, trigger batch write but KEEP all data
    3. FIFO Mode: After first batch write, maintain exactly max_size items:
       - Add 1 new item → Remove 1 oldest item (FIFO)
       - Buffer always contains exactly max_size recent items
    4. Periodic Batch Write: After max_size FIFO operations (full buffer rotation),
       trigger another batch write

    WHY NO CLEARING: SpeculativeBuffer needs constant access to recent items from
    WriteBuffer for context generation. Clearing would break this dependency.

This is a high-performance rewrite of WriteBuffer that maintains all functionality
while eliminating performance bottlenecks:

- No asyncio.create_task() abuse
- No complex background workers
- Simplified FIFO logic
- Direct async operations
- Minimal overhead
"""

import asyncio
from typing import Any, Callable, List, Optional, Set, Dict
from loguru import logger

from ..interfaces import MemoryServiceStorageHandler as StorageHandler


class WriteBuffer:
    """Write buffer with FIFO queue management.

    Maintains all original functionality while optimizing for performance:
    - FIFO logic preserved
    - Update handlers supported
    - Batch writing optimized
    - No background task overhead
    """

    def __init__(
        self,
        max_size: int = 5,
        batch_threshold: int = 5,
        storage_handler: Optional[StorageHandler] = None
    ):
        """Initialize the WriteBuffer.

        Args:
            max_size: Maximum number of items in the buffer
            batch_threshold: Number of items that triggers a batch write
            storage_handler: StorageHandler instance to handle batch writes
        """
        self.max_size = max_size
        self.batch_threshold = batch_threshold
        self.storage_handler = storage_handler
        self.items: List[Any] = []
        self._item_index: Dict[str, Any] = {}
        self.update_handlers: Set[Callable] = set()
        self._lock = asyncio.Lock()
        self._last_batch_results: List[Any] = []

        # FIFO tracking
        self._fifo_operations_count = 0
        self._first_batch_written = False
        self._written_items: Set[str] = set()  # Per-instance tracking

        logger.info(
            f"WriteBuffer: Initialized with max_size={max_size}, "
            f"batch_threshold={batch_threshold}")

    def register_update_handler(self, handler: Callable) -> None:
        """Register a handler to be called when items are added."""
        self.update_handlers.add(handler)

    async def add(self, item: Any) -> bool:
        """Add an item to the buffer with optimized FIFO logic.

        Args:
            item: The item to add

        Returns:
            Whether a batch write was triggered
        """
        async with self._lock:
            # FIFO logic: remove oldest if buffer is full
            if len(self.items) >= self.max_size:
                old_item = self.items.pop(0)
                if isinstance(old_item, dict) and 'id' in old_item:
                    self._item_index.pop(old_item['id'], None)

                if self._first_batch_written:
                    self._fifo_operations_count += 1

            self.items.append(item)

            # Update index
            if isinstance(item, dict) and 'id' in item:
                self._item_index[item['id']] = item

            # Check for batch write trigger
            should_batch_write = (
                (not self._first_batch_written and len(self.items) >= self.batch_threshold) or
                (self._first_batch_written and self._fifo_operations_count >= self.max_size)
            )

            if should_batch_write:
                if self._first_batch_written:
                    self._fifo_operations_count = 0

                # Filter out already written items
                items_to_write = []
                for item in self.items:
                    item_key = self._get_item_key(item)
                    if item_key not in self._written_items:
                        items_to_write.append(item)

                if items_to_write:
                    # Direct async batch write (no task creation)
                    await self._execute_batch_write(items_to_write)
                    self._first_batch_written = True
                    return True

            return False

    def _get_item_key(self, item: Any) -> str:
        """Get unique key for item to prevent duplicates."""
        if isinstance(item, dict):
            return item.get('content', str(item))
        return str(item)

    async def _execute_batch_write(self, items: List[Any]) -> None:
        """Execute batch write directly without background tasks."""
        try:
            if not self.storage_handler:
                logger.error("WriteBuffer: No storage handler available")
                return

            # Direct async call to storage handler
            results = await self.storage_handler.handle_batch(items)

            # Update tracking
            for item in items:
                item_key = self._get_item_key(item)
                self._written_items.add(item_key)

            self._last_batch_results = results or []

            # Notify update handlers directly
            if self.update_handlers:
                await self._notify_handlers(items)

            logger.debug(f"WriteBuffer: Batch write completed, {len(results or [])} results")

        except Exception as e:
            logger.error(f"WriteBuffer: Batch write error: {e}")

    async def _notify_handlers(self, items: List[Any]) -> None:
        """Notify update handlers efficiently."""
        for handler in self.update_handlers:
            try:
                await handler(items)
            except Exception as e:
                logger.error(f"WriteBuffer: Handler error: {e}")

    async def batch_write(self, timeout: float = 30.0) -> List[Any]:
        """Manual batch write of all items in buffer."""
        async with self._lock:
            if not self.items:
                return []

            items_to_process = self.items.copy()

        try:
            await asyncio.wait_for(
                self._execute_batch_write(items_to_process),
                timeout=timeout
            )
            return self._last_batch_results.copy()
        except asyncio.TimeoutError:
            logger.error(f"WriteBuffer: Batch write timeout after {timeout}s")
            return []

    async def find_items_by_ids(self, item_ids: List[str]) -> List[Any]:
        """Find items in buffer by IDs."""
        async with self._lock:
            return [self._item_index[item_id] for item_id in item_ids if item_id in self._item_index]

    async def update_items(self, item_ids: List[str], new_items: List[Any]) -> bool:
        """Update items in buffer."""
        if len(item_ids) != len(new_items):
            return False

        async with self._lock:
            updated = False
            for item_id, new_item in zip(item_ids, new_items):
                if item_id in self._item_index:
                    # Update in list
                    for i, item in enumerate(self.items):
                        if isinstance(item, dict) and item.get('id') == item_id:
                            self.items[i] = new_item
                            break

                    # Update index
                    if isinstance(new_item, dict) and 'id' in new_item:
                        self._item_index[new_item['id']] = new_item
                        if new_item['id'] != item_id:
                            self._item_index.pop(item_id, None)
                    else:
                        self._item_index.pop(item_id, None)

                    updated = True
            return updated

    async def delete_items(self, item_ids: List[str]) -> List[str]:
        """Delete items from buffer."""
        async with self._lock:
            deleted_ids = []
            self.items = [
                item for item in self.items
                if not (isinstance(item, dict) and item.get('id') in item_ids)
            ]

            for item_id in item_ids:
                if item_id in self._item_index:
                    self._item_index.pop(item_id)
                    deleted_ids.append(item_id)

            return deleted_ids

    async def query_items(self, query_text: str, max_results: int = 10) -> List[Any]:
        """Query items in buffer by content."""
        async with self._lock:
            results = []
            query_lower = query_text.lower()

            for item in self.items:
                if len(results) >= max_results:
                    break

                content = ""
                if isinstance(item, dict):
                    content = item.get('content', '')
                else:
                    content = str(item)

                if query_lower in content.lower():
                    results.append(item)

            return results

    async def get_last_batch_results(self) -> List[Any]:
        """Get results from most recent batch operation."""
        async with self._lock:
            return self._last_batch_results.copy()

    def get_last_message_ids(self) -> List[str]:
        """Get message IDs from most recent batch (sync version)."""
        return self._last_batch_results.copy()

    async def clear(self) -> None:
        """Clear all items from buffer."""
        async with self._lock:
            self.items.clear()
            self._item_index.clear()
            self._written_items.clear()
            self._last_batch_results.clear()
            self._fifo_operations_count = 0
            self._first_batch_written = False

    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        return {
            "size": len(self.items),
            "max_size": self.max_size,
            "batch_threshold": self.batch_threshold,
            "fifo_operations": self._fifo_operations_count,
            "first_batch_written": self._first_batch_written,
            "written_items_count": len(self._written_items),
            "update_handlers_count": len(self.update_handlers),
            "has_storage_handler": self.storage_handler is not None
        }
