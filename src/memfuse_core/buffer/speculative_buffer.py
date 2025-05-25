"""SpeculativeBuffer implementation for MemFuse.

    The SpeculativeBuffer predicts which items are likely to be accessed next
    and prefetches them, reducing latency for subsequent accesses.

    It works by taking the most recent items from the WriteBuffer, concatenating
    their content, and using that as a query to retrieve related items from the
    vector store.

This is a high-performance rewrite of SpeculativeBuffer that maintains all functionality
while eliminating performance bottlenecks:

- True async implementation
- Optimized context generation
- Efficient retrieval handling
- Minimal overhead
"""

import asyncio
from typing import Any, Callable, List, Optional, Dict
from loguru import logger

from ..interfaces import BufferComponentInterface


class SpeculativeBuffer(BufferComponentInterface):
    """Speculative buffer for prefetching likely-to-be-accessed items.

    Maintains all original functionality while optimizing for performance:
    - True async operations
    - Efficient context generation
    - Optimized retrieval handling
    - Minimal latency overhead
    """

    def __init__(
        self,
        max_size: int = 10,
        context_window: int = 3,
        retrieval_handler: Optional[Callable] = None
    ):
        """Initialize the SpeculativeBuffer.

        Args:
            max_size: Maximum number of items in the buffer
            context_window: Number of recent items to use for prediction
            retrieval_handler: Async callback function to retrieve items
        """
        self._max_size = max_size
        self.context_window = context_window
        self.retrieval_handler = retrieval_handler
        self._items: List[Any] = []
        self._lock = asyncio.Lock()
        self.total_updates = 0
        self.total_items_processed = 0

        logger.info(
            f"SpeculativeBuffer: Initialized with max_size={max_size}, "
            f"context_window={context_window}")

    async def update(self, recent_items: List[Any]) -> None:
        """Update buffer based on recent items with optimized processing.

        Args:
            recent_items: List of recently accessed items
        """
        if not self.retrieval_handler or not recent_items:
            return

        # Optimized context generation
        context = self._generate_context(recent_items)
        if not context:
            return

        try:
            # Direct async retrieval (no task creation)
            speculative_items = await self.retrieval_handler(context, self._max_size)

            if speculative_items:
                async with self._lock:
                    self._items = speculative_items[:self._max_size]
                    self.total_updates += 1
                    self.total_items_processed += len(speculative_items)

                logger.debug(f"SpeculativeBuffer: Updated with {len(self._items)} items")

        except Exception as e:
            logger.error(f"SpeculativeBuffer: Update error: {e}")

    def _generate_context(self, recent_items: List[Any]) -> str:
        """Generate context string efficiently from recent items."""
        try:
            # Use only the most recent items within context window
            context_items = recent_items[-self.context_window:] if recent_items else []

            if not context_items:
                return ""

            # Optimized content extraction
            contents = []
            for item in context_items:
                content = self._extract_content(item)
                if content:
                    contents.append(content)

            return " ".join(contents) if contents else ""

        except Exception as e:
            logger.error(f"SpeculativeBuffer: Context generation error: {e}")
            return ""

    def _extract_content(self, item: Any) -> str:
        """Extract content from item efficiently."""
        if isinstance(item, dict):
            return item.get('content', '')
        elif hasattr(item, 'content'):
            return getattr(item, 'content', '')
        else:
            return str(item)

    async def update_from_items(self, items: List[Any]) -> None:
        """Update buffer based on a list of items."""
        logger.info(f"SpeculativeBuffer.update_from_items: Called with {len(items) if items else 0} items")
        if items:
            await self.update(items)
        else:
            logger.debug("SpeculativeBuffer.update_from_items: No items provided")

    async def update_from_write_buffer(self, write_buffer) -> None:
        """Update buffer based on items in write buffer."""
        try:
            # Handle different write buffer types efficiently
            if isinstance(write_buffer, list):
                items = write_buffer
            elif hasattr(write_buffer, 'items'):
                items = write_buffer.items
            elif hasattr(write_buffer, 'get_items'):
                items = await write_buffer.get_items()
            else:
                logger.warning("SpeculativeBuffer: Invalid write buffer type")
                return

            await self.update_from_items(items)

        except Exception as e:
            logger.error(f"SpeculativeBuffer: Write buffer update error: {e}")

    @property
    def items(self) -> List[Any]:
        """Get all items in the buffer."""
        return self._items

    @property
    def max_size(self) -> int:
        """Get the maximum size of the buffer."""
        return self._max_size

    async def get_items(self) -> List[Dict[str, Any]]:
        """Get all items in the buffer (async version)."""
        async with self._lock:
            return self._items.copy()

    async def clear(self) -> None:
        """Clear all items from the buffer."""
        async with self._lock:
            self._items.clear()
            logger.debug("SpeculativeBuffer: Buffer cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the buffer."""
        return {
            "size": len(self._items),
            "max_size": self._max_size,
            "context_window": self.context_window,
            "total_updates": self.total_updates,
            "total_items_processed": self.total_items_processed,
            "has_retrieval_handler": self.retrieval_handler is not None,
            "optimization": "true_async_implementation"
        }

    # Additional optimization methods

    async def prefetch_for_query(self, query_text: str) -> List[Any]:
        """Prefetch items that might be relevant to a query.

        This is an optimization that can be called before actual query
        to warm up the speculative buffer.
        """
        if not self.retrieval_handler:
            return []

        try:
            # Use query text directly as context for prefetching
            prefetch_items = await self.retrieval_handler(query_text, self._max_size)

            if prefetch_items:
                async with self._lock:
                    # Merge with existing items, avoiding duplicates
                    existing_ids = set()
                    for item in self._items:
                        if isinstance(item, dict) and 'id' in item:
                            existing_ids.add(item['id'])

                    new_items = []
                    for item in prefetch_items:
                        if isinstance(item, dict) and 'id' in item:
                            if item['id'] not in existing_ids:
                                new_items.append(item)
                        else:
                            new_items.append(item)

                    # Add new items up to max_size
                    available_space = self._max_size - len(self._items)
                    if available_space > 0:
                        self._items.extend(new_items[:available_space])

                logger.debug(f"SpeculativeBuffer: Prefetched {len(new_items)} new items")
                return new_items

            return []

        except Exception as e:
            logger.error(f"SpeculativeBuffer: Prefetch error: {e}")
            return []

    async def get_relevant_items(self, query_text: str, max_results: int = 5) -> List[Any]:
        """Get items from buffer that are relevant to a query.

        This provides a fast lookup in the speculative buffer before
        going to the main storage.
        """
        async with self._lock:
            if not self._items:
                return []

            # Simple relevance scoring based on content similarity
            relevant_items = []
            query_lower = query_text.lower()

            for item in self._items:
                content = self._extract_content(item).lower()
                if query_lower in content or any(word in content for word in query_lower.split()):
                    relevant_items.append(item)

                if len(relevant_items) >= max_results:
                    break

            return relevant_items

    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return len(self._items) == 0

    def get_size(self) -> int:
        """Get current buffer size."""
        return len(self._items)

    async def optimize_for_pattern(self, access_pattern: List[str]) -> None:
        """Optimize buffer based on access patterns.

        This can be used to pre-populate the buffer based on
        observed access patterns.
        """
        if not self.retrieval_handler or not access_pattern:
            return

        try:
            # Create context from access pattern
            pattern_context = " ".join(access_pattern)

            # Retrieve items based on pattern
            pattern_items = await self.retrieval_handler(pattern_context, self._max_size)

            if pattern_items:
                async with self._lock:
                    self._items = pattern_items[:self._max_size]

                logger.debug(f"SpeculativeBuffer: Optimized for pattern with {len(self._items)} items")

        except Exception as e:
            logger.error(f"SpeculativeBuffer: Pattern optimization error: {e}")
