"""QueryBuffer implementation for MemFuse.

    The QueryBuffer performs multi-path retrieval when a query is submitted,
    retrieving items from storage and combining them with items from the
    WriteBuffer and SpeculativeBuffer.

This is a high-performance rewrite of QueryBuffer that maintains all functionality
while eliminating performance bottlenecks:

- Efficient multi-source result combination
- Optimized caching with LRU eviction
- Streamlined metadata handling
- Minimal overhead
"""

import asyncio
from typing import Any, Callable, Dict, List, Optional
from loguru import logger

from ..interfaces import BufferComponentInterface


class QueryBuffer(BufferComponentInterface):
    """Buffer for query processing with multi-path retrieval.

    Maintains all original functionality while optimizing for performance:
    - Efficient multi-source combination
    - Optimized caching strategy
    - Streamlined result processing
    - Minimal latency overhead
    """

    def __init__(
        self,
        retrieval_handler: Optional[Callable] = None,
        max_size: int = 15,
        cache_size: int = 100
    ):
        """Initialize the QueryBuffer.

        Args:
            retrieval_handler: Async callback for multi-path retrieval
            max_size: Maximum number of items to return from a query
            cache_size: Maximum number of queries to cache
        """
        self.retrieval_handler = retrieval_handler
        self._max_size = max_size
        self.cache_size = cache_size
        self._items: List[Any] = []
        self.query_cache: Dict[str, List[Any]] = {}
        self._cache_order: List[str] = []  # LRU tracking
        self._lock = asyncio.Lock()
        self.total_queries = 0
        self.cache_hits = 0
        self.cache_misses = 0

        logger.info(
            f"QueryBuffer: Initialized with max_size={max_size}, "
            f"cache_size={cache_size}")

    async def query(
        self,
        query_text: str,
        write_buffer=None,
        speculative_buffer=None
    ) -> List[Any]:
        """Query buffer and underlying storage with optimized processing.

        Args:
            query_text: Query text
            write_buffer: Optional WriteBuffer to include results from
            speculative_buffer: Optional SpeculativeBuffer to include results from

        Returns:
            List of query results
        """
        query_preview = query_text[:50] + "..." if len(query_text) > 50 else query_text
        logger.info(f"QueryBuffer.query: Called with query: {query_preview}")

        self.total_queries += 1

        # Check cache first with LRU update
        logger.debug(f"QueryBuffer.query: Checking cache for query")
        cached_result = await self._check_cache(query_text)
        if cached_result is not None:
            self.cache_hits += 1
            logger.info(f"QueryBuffer.query: Cache hit! Returning {len(cached_result)} cached results")
            # Don't limit cached results here - let the caller decide
            return cached_result

        self.cache_misses += 1
        logger.info(f"QueryBuffer.query: Cache miss, querying storage")

        if not self.retrieval_handler:
            logger.warning("QueryBuffer: No retrieval handler available")
            return []

        try:
            # Get results from storage
            logger.info(f"QueryBuffer.query: Calling retrieval handler with max_size={self._max_size}")
            storage_results = await self.retrieval_handler(query_text, self._max_size)
            logger.info(f"QueryBuffer.query: Storage returned {len(storage_results) if storage_results else 0} results")

            # Combine results efficiently
            logger.info("QueryBuffer.query: Combining results from all sources")
            all_results = await self._combine_results(
                storage_results or [],
                write_buffer,
                speculative_buffer
            )
            logger.info(f"QueryBuffer.query: Combined results: {len(all_results)} total items")

            # Update cache
            await self._update_cache(query_text, all_results)
            logger.debug(f"QueryBuffer.query: Updated cache with {len(all_results)} results")

            # Update buffer items
            async with self._lock:
                self._items = all_results[:self._max_size]

            logger.info(f"QueryBuffer.query: Returning {len(all_results)} results")
            return all_results

        except Exception as e:
            logger.error(f"QueryBuffer: Query error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    async def _check_cache(self, query_text: str) -> Optional[List[Any]]:
        """Check cache with LRU update."""
        if query_text in self.query_cache:
            # Move to end (most recently used)
            self._cache_order.remove(query_text)
            self._cache_order.append(query_text)
            return self.query_cache[query_text].copy()
        return None

    async def _update_cache(self, query_text: str, results: List[Any]) -> None:
        """Update cache with LRU eviction."""
        # Add/update cache entry
        self.query_cache[query_text] = results.copy()

        if query_text in self._cache_order:
            self._cache_order.remove(query_text)
        self._cache_order.append(query_text)

        # LRU eviction
        while len(self.query_cache) > self.cache_size:
            oldest_key = self._cache_order.pop(0)
            self.query_cache.pop(oldest_key, None)

    async def _combine_results(
        self,
        storage_results: List[Any],
        write_buffer=None,
        speculative_buffer=None
    ) -> List[Any]:
        """Efficiently combine results from multiple sources."""
        logger.debug(f"QueryBuffer._combine_results: Starting with {len(storage_results)} storage results")

        all_results = []
        seen_ids = set()

        # Add storage results with metadata
        storage_added = 0
        for item in storage_results:
            item_id = self._get_item_id(item)
            if item_id not in seen_ids:
                self._add_metadata(item, 'storage', 1.0)
                all_results.append(item)
                seen_ids.add(item_id)
                storage_added += 1

        logger.debug(f"QueryBuffer._combine_results: Added {storage_added} storage items")

        # Add write buffer results
        write_buffer_added = 0
        if write_buffer and hasattr(write_buffer, 'items'):
            logger.debug(f"QueryBuffer._combine_results: WriteBuffer has {len(write_buffer.items)} items")
            for item in write_buffer.items:
                item_id = self._get_item_id(item)
                if item_id not in seen_ids:
                    self._add_metadata(item, 'write_buffer', 1.0)
                    all_results.append(item)
                    seen_ids.add(item_id)
                    write_buffer_added += 1
        else:
            logger.debug("QueryBuffer._combine_results: No WriteBuffer or WriteBuffer has no items")

        logger.debug(f"QueryBuffer._combine_results: Added {write_buffer_added} write buffer items")

        # Add speculative buffer results
        speculative_added = 0
        if speculative_buffer and hasattr(speculative_buffer, 'items'):
            logger.debug(f"QueryBuffer._combine_results: SpeculativeBuffer has {len(speculative_buffer.items)} items")
            for item in speculative_buffer.items:
                item_id = self._get_item_id(item)
                if item_id not in seen_ids:
                    self._add_metadata(item, 'speculative_buffer', 0.8)
                    all_results.append(item)
                    seen_ids.add(item_id)
                    speculative_added += 1
        else:
            logger.debug("QueryBuffer._combine_results: No SpeculativeBuffer or SpeculativeBuffer has no items")

        logger.debug(f"QueryBuffer._combine_results: Added {speculative_added} speculative buffer items")
        logger.info(f"QueryBuffer._combine_results: Total combined: {len(all_results)} items (storage: {storage_added}, write: {write_buffer_added}, speculative: {speculative_added})")

        return all_results

    def _get_item_id(self, item: Any) -> str:
        """Get unique identifier for item."""
        if isinstance(item, dict):
            return item.get('id', str(hash(str(item))))
        return str(hash(str(item)))

    def _add_metadata(self, item: Any, source: str, weight: float) -> None:
        """Add metadata to item efficiently."""
        if isinstance(item, dict):
            if 'metadata' not in item:
                item['metadata'] = {}
            if 'retrieval' not in item['metadata']:
                item['metadata']['retrieval'] = {}
            item['metadata']['retrieval']['source'] = source
            item['weight'] = weight

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
            logger.debug("QueryBuffer: Buffer cleared")

    async def clear_cache(self) -> None:
        """Clear the query cache."""
        self.query_cache.clear()
        self._cache_order.clear()
        logger.debug("QueryBuffer: Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the buffer."""
        cache_hit_rate = (self.cache_hits / self.total_queries * 100) if self.total_queries > 0 else 0

        return {
            "size": len(self._items),
            "max_size": self._max_size,
            "cache_size": self.cache_size,
            "cache_entries": len(self.query_cache),
            "total_queries": self.total_queries,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": f"{cache_hit_rate:.1f}%",
            "has_retrieval_handler": self.retrieval_handler is not None,
            "optimization": "lru_cache_with_efficient_combination"
        }

    # Additional optimization methods

    async def preload_cache(self, common_queries: List[str]) -> None:
        """Preload cache with common queries."""
        if not self.retrieval_handler:
            return

        for query in common_queries:
            if query not in self.query_cache:
                try:
                    results = await self.retrieval_handler(query, self._max_size)
                    await self._update_cache(query, results or [])
                    logger.debug(f"QueryBuffer: Preloaded cache for query: {query[:50]}...")
                except Exception as e:
                    logger.error(f"QueryBuffer: Preload error for query '{query}': {e}")

    async def get_cache_info(self) -> Dict[str, Any]:
        """Get detailed cache information."""
        return {
            "cache_entries": len(self.query_cache),
            "cache_order": self._cache_order.copy(),
            "cache_utilization": f"{len(self.query_cache) / self.cache_size * 100:.1f}%",
            "most_recent_queries": self._cache_order[-5:] if self._cache_order else []
        }

    async def optimize_cache_size(self, target_hit_rate: float = 0.8) -> None:
        """Dynamically optimize cache size based on hit rate."""
        if self.total_queries < 10:  # Need enough data
            return

        current_hit_rate = self.cache_hits / self.total_queries

        if current_hit_rate < target_hit_rate and self.cache_size < 200:
            # Increase cache size
            self.cache_size = min(self.cache_size + 20, 200)
            logger.debug(f"QueryBuffer: Increased cache size to {self.cache_size}")
        elif current_hit_rate > target_hit_rate + 0.1 and self.cache_size > 50:
            # Decrease cache size to save memory
            self.cache_size = max(self.cache_size - 10, 50)
            # Trim cache if needed
            while len(self.query_cache) > self.cache_size:
                oldest_key = self._cache_order.pop(0)
                self.query_cache.pop(oldest_key, None)
            logger.debug(f"QueryBuffer: Decreased cache size to {self.cache_size}")

    async def get_similar_cached_results(self, query_text: str, similarity_threshold: float = 0.7) -> List[Any]:
        """Get cached results for similar queries."""
        # Simple similarity check based on word overlap
        query_words = set(query_text.lower().split())

        for cached_query, cached_results in self.query_cache.items():
            cached_words = set(cached_query.lower().split())

            if not query_words or not cached_words:
                continue

            # Jaccard similarity
            intersection = len(query_words & cached_words)
            union = len(query_words | cached_words)
            similarity = intersection / union if union > 0 else 0

            if similarity >= similarity_threshold:
                logger.debug(f"QueryBuffer: Found similar cached query (similarity: {similarity:.2f})")
                return cached_results.copy()

        return []
