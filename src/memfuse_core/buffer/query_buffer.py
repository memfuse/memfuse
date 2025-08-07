"""Query Buffer implementation for MemFuse.

This buffer provides unified query functionality with sorting support,
combining results from HybridBuffer and persistent storage without reranking.
"""

import asyncio
from typing import Any, Dict, List, Optional, Callable
from loguru import logger

from ..interfaces import BufferComponentInterface
from ..rag.retrieve.buffer import BufferRetrieval


class QueryBuffer(BufferComponentInterface):
    """Unified query buffer with sorting and direct result combination.
    
    This buffer combines results from HybridBuffer and storage with support for:
    - sort_by: 'score' or 'timestamp'
    - order: 'asc' or 'desc'
    - Direct combination without complex reranking
    """
    
    def __init__(
        self,
        retrieval_handler: Optional[Callable] = None,
        rerank_handler: Optional[Callable] = None,
        max_size: int = 15,
        cache_size: int = 100,
        default_sort_by: str = "score",
        default_order: str = "desc"
    ):
        """Initialize the QueryBuffer.

        Args:
            retrieval_handler: Async callback for storage retrieval
            rerank_handler: Async callback for result reranking
            max_size: Maximum number of items to return from a query
            cache_size: Maximum number of queries to cache
            default_sort_by: Default sort field ('score' or 'timestamp')
            default_order: Default sort order ('asc' or 'desc')
        """
        self.retrieval_handler = retrieval_handler
        self.rerank_handler = rerank_handler
        self._max_size = max_size
        self.cache_size = cache_size
        self.default_sort_by = default_sort_by
        self.default_order = default_order

        # Cache for query results
        self.query_cache: Dict[str, List[Any]] = {}
        self._cache_order: List[str] = []  # LRU tracking

        # Rerank cache for performance
        self.rerank_cache: Dict[str, List[Any]] = {}

        # Buffer state
        self._items: List[Any] = []

        # Async lock for thread safety
        self._lock = asyncio.Lock()

        # Statistics
        self.total_queries = 0
        self.total_session_queries = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_hybrid_results = 0
        self.total_storage_results = 0
        self.rerank_operations = 0

        # Buffer references
        self.hybrid_buffer = None
        self.round_buffer = None

        # Initialize modular buffer retrieval system
        self.buffer_retrieval = BufferRetrieval(
            encoder_name="minilm",
            similarity_threshold=0.1
        )

        logger.info(f"QueryBuffer: Initialized with max_size={max_size}, default_sort={default_sort_by}, rerank_enabled={rerank_handler is not None}")

    def set_hybrid_buffer(self, hybrid_buffer):
        """Set the HybridBuffer instance for queries.

        Args:
            hybrid_buffer: HybridBuffer instance to use for queries
        """
        self.hybrid_buffer = hybrid_buffer
        logger.debug("QueryBuffer: HybridBuffer reference set")

    def set_round_buffer(self, round_buffer):
        """Set the RoundBuffer instance for queries.

        Args:
            round_buffer: RoundBuffer instance to use for queries
        """
        self.round_buffer = round_buffer
        logger.debug("QueryBuffer: RoundBuffer reference set")

    async def query(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        sort_by: Optional[str] = None,
        order: Optional[str] = None,
        hybrid_buffer=None,
        use_rerank: bool = True
    ) -> List[Any]:
        """Enhanced query with internal reranking and optimization.

        Args:
            query_text: Query text
            top_k: Maximum number of results (uses max_size if None)
            sort_by: Sort field ('score' or 'timestamp', uses default if None)
            order: Sort order ('asc' or 'desc', uses default if None)
            hybrid_buffer: HybridBuffer instance to include results from
            use_rerank: Whether to apply internal reranking (default: True)

        Returns:
            List of query results sorted and optionally reranked
        """
        # Set defaults
        if top_k is None:
            top_k = self._max_size
        if sort_by is None:
            sort_by = self.default_sort_by
        if order is None:
            order = self.default_order
        
        query_preview = query_text[:50] + "..." if len(query_text) > 50 else query_text
        logger.info(f"QueryBuffer: Query '{query_preview}' with sort_by={sort_by}, order={order}")

        self.total_queries += 1

        # Create cache key including sort parameters
        cache_key = f"{query_text}|{sort_by}|{order}|{top_k}"

        # Check cache first
        cached_result = await self._check_cache(cache_key)
        if cached_result is not None:
            self.cache_hits += 1
            logger.info(f"QueryBuffer: Cache hit! Returning {len(cached_result)} cached results")
            return cached_result[:top_k]

        self.cache_misses += 1
        logger.info("QueryBuffer: Cache miss, using intelligent query routing")

        try:
            # Use intelligent query routing for better performance
            final_results = await self._intelligent_query_routing(
                query_text, top_k, sort_by, order, hybrid_buffer, use_rerank
            )

            # Update cache
            await self._update_cache(cache_key, final_results)

            # Update buffer items
            async with self._lock:
                self._items = final_results.copy()

            logger.info(f"QueryBuffer: Returning {len(final_results)} results")
            return final_results

        except Exception as e:
            logger.error(f"QueryBuffer: Query failed: {e}")
            return []

    async def _intelligent_query_routing(
        self,
        query_text: str,
        top_k: int,
        sort_by: str,
        order: str,
        hybrid_buffer=None,
        use_rerank: bool = True
    ) -> List[Any]:
        """Intelligent query routing with Buffer-first optimization.

        Strategy:
        1. Always check Buffer first (fastest, most recent data)
        2. If Buffer has sufficient results, return early
        3. Otherwise, query storage to supplement
        4. Apply smart result merging
        """
        # Step 1: Query Buffer first (fastest path)
        buffer_results = await self.buffer_retrieval.retrieve(
            query=query_text,
            user_id=None,
            session_id=None,
            top_k=top_k * 2,  # Get more for better selection
            hybrid_buffer=hybrid_buffer or self.hybrid_buffer,
            round_buffer=self.round_buffer
        )

        logger.info(f"QueryBuffer: Got {len(buffer_results)} results from buffers")

        # Step 2: Check if Buffer results are sufficient
        buffer_quality_score = self._assess_buffer_quality(buffer_results, query_text)

        if buffer_quality_score >= 0.7 and len(buffer_results) >= top_k:
            # Buffer has high-quality results, use Buffer-only path
            logger.info("QueryBuffer: Buffer-only path - sufficient high-quality results")
            final_results = await self._process_buffer_only_results(
                buffer_results, top_k, sort_by, order, query_text, use_rerank
            )
        else:
            # Need to supplement with storage results
            logger.info("QueryBuffer: Hybrid path - supplementing with storage")
            final_results = await self._process_hybrid_results(
                buffer_results, query_text, top_k, sort_by, order, use_rerank
            )

        return final_results

    def _assess_buffer_quality(self, buffer_results: List[Any], query_text: str) -> float:
        """Assess the quality of buffer results for the given query.

        Returns:
            Quality score between 0.0 and 1.0
        """
        if not buffer_results:
            return 0.0

        # Simple quality assessment based on:
        # 1. Number of results
        # 2. Average score
        # 3. Recency of results

        total_score = 0.0
        recent_count = 0
        import time
        current_time = time.time()

        for result in buffer_results:
            # Score component
            score = result.get('score', 0.0)
            total_score += score

            # Recency component (results from last 5 minutes get bonus)
            timestamp = result.get('metadata', {}).get('timestamp', 0)
            if current_time - timestamp < 300:  # 5 minutes
                recent_count += 1

        # Calculate quality metrics
        avg_score = total_score / len(buffer_results) if buffer_results else 0.0
        recency_ratio = recent_count / len(buffer_results) if buffer_results else 0.0

        # Combined quality score
        quality_score = (avg_score * 0.6) + (recency_ratio * 0.4)

        logger.debug(f"QueryBuffer: Buffer quality assessment - avg_score={avg_score:.2f}, recency_ratio={recency_ratio:.2f}, quality={quality_score:.2f}")

        return min(quality_score, 1.0)

    async def _process_buffer_only_results(
        self,
        buffer_results: List[Any],
        top_k: int,
        sort_by: str,
        order: str,
        query_text: str,
        use_rerank: bool
    ) -> List[Any]:
        """Process results using Buffer-only path."""
        # Sort buffer results
        sorted_results = self._sort_results(buffer_results, sort_by, order)

        # Apply reranking if enabled
        if use_rerank and self.rerank_handler and sorted_results:
            sorted_results = await self._internal_rerank(sorted_results, query_text)

        # Update statistics
        self.total_hybrid_results += len([r for r in sorted_results if r.get('metadata', {}).get('source', '').startswith('hybrid')])

        return sorted_results[:top_k]

    async def _process_hybrid_results(
        self,
        buffer_results: List[Any],
        query_text: str,
        top_k: int,
        sort_by: str,
        order: str,
        use_rerank: bool
    ) -> List[Any]:
        """Process results using hybrid Buffer + Storage path."""
        # Query storage for additional results
        storage_results = []
        if self.retrieval_handler:
            # Request fewer storage results since we have buffer results
            storage_top_k = max(top_k - len(buffer_results), top_k // 2)
            storage_results = await self.retrieval_handler(query_text, storage_top_k)
            self.total_storage_results += len(storage_results or [])
            logger.info(f"QueryBuffer: Got {len(storage_results or [])} results from storage")

        # Combine and sort results with Buffer priority
        all_results = await self._combine_and_sort_results(
            storage_results or [],
            buffer_results,
            [],  # round_results now included in buffer_results
            sort_by,
            order
        )

        # Apply internal reranking if enabled and handler available
        if use_rerank and self.rerank_handler and all_results:
            all_results = await self._internal_rerank(all_results, query_text)

        # Update statistics
        self.total_hybrid_results += len([r for r in buffer_results if r.get('metadata', {}).get('source', '').startswith('hybrid')])

        return all_results[:top_k]

    def _sort_results(self, results: List[Any], sort_by: str, order: str) -> List[Any]:
        """Sort results by the specified criteria."""
        if sort_by == "score":
            results.sort(
                key=lambda x: x.get("score", 0.0),
                reverse=(order == "desc")
            )
        elif sort_by == "timestamp":
            results.sort(
                key=lambda x: self._normalize_timestamp_for_sorting(x.get("created_at", "")),
                reverse=(order == "desc")
            )

        return results

    async def _combine_and_sort_results(
        self,
        storage_results: List[Any],
        hybrid_results: List[Any],
        round_results: List[Any],
        sort_by: str,
        order: str
    ) -> List[Any]:
        """Combine and sort results from different sources.

        Args:
            storage_results: Results from persistent storage
            hybrid_results: Results from HybridBuffer
            round_results: Results from RoundBuffer
            sort_by: Sort field ('score' or 'timestamp')
            order: Sort order ('asc' or 'desc')

        Returns:
            Combined and sorted results
        """
        # Combine results
        all_results = []
        seen_ids = set()

        # Add storage results
        for result in storage_results:
            result_id = result.get("id", str(hash(str(result))))
            if result_id not in seen_ids:
                all_results.append(result)
                seen_ids.add(result_id)

        # Add hybrid results (avoid duplicates)
        for result in hybrid_results:
            result_id = result.get("id", str(hash(str(result))))
            if result_id not in seen_ids:
                all_results.append(result)
                seen_ids.add(result_id)

        # Add round results (avoid duplicates)
        for result in round_results:
            result_id = result.get("id", str(hash(str(result))))
            if result_id not in seen_ids:
                all_results.append(result)
                seen_ids.add(result_id)
        
        # Sort results
        if sort_by == "score":
            all_results.sort(
                key=lambda x: x.get("score", 0.0),
                reverse=(order == "desc")
            )
        elif sort_by == "timestamp":
            all_results.sort(
                key=lambda x: self._normalize_timestamp_for_sorting(x.get("created_at", "")),
                reverse=(order == "desc")
            )
        
        logger.debug(f"QueryBuffer: Combined {len(all_results)} results, sorted by {sort_by} {order}")
        return all_results
    
    async def _check_cache(self, cache_key: str) -> Optional[List[Any]]:
        """Check cache with LRU update.
        
        Args:
            cache_key: Cache key to check
            
        Returns:
            Cached results if found, None otherwise
        """
        if cache_key in self.query_cache:
            # Move to end (most recently used)
            self._cache_order.remove(cache_key)
            self._cache_order.append(cache_key)
            return self.query_cache[cache_key].copy()
        return None
    
    async def _update_cache(self, cache_key: str, results: List[Any]) -> None:
        """Update cache with LRU eviction.
        
        Args:
            cache_key: Cache key
            results: Results to cache
        """
        # Add/update cache entry
        self.query_cache[cache_key] = results.copy()
        
        if cache_key in self._cache_order:
            self._cache_order.remove(cache_key)
        self._cache_order.append(cache_key)
        
        # LRU eviction
        while len(self.query_cache) > self.cache_size:
            oldest_key = self._cache_order.pop(0)
            self.query_cache.pop(oldest_key, None)
    
    async def get_buffer_metadata(self, hybrid_buffer=None) -> Dict[str, Any]:
        """Get buffer metadata for Query API response.

        Args:
            hybrid_buffer: HybridBuffer instance to check (optional, uses instance variable if None)

        Returns:
            Dictionary with buffer metadata
        """
        metadata = {
            "buffer_messages_available": False,
            "buffer_messages_count": 0,
            "sort_by": self.default_sort_by,
            "order": self.default_order
        }

        buffer_to_use = hybrid_buffer or self.hybrid_buffer
        if buffer_to_use:
            try:
                async with buffer_to_use._data_lock:
                    total_messages = sum(len(round_msgs) for round_msgs in buffer_to_use.original_rounds)
                    metadata.update({
                        "buffer_messages_available": total_messages > 0,
                        "buffer_messages_count": total_messages,
                        "buffer_chunks_count": len(buffer_to_use.chunks)
                    })
            except Exception as e:
                logger.error(f"QueryBuffer: Error getting buffer metadata: {e}")
        
        return metadata
    
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
            "total_session_queries": self.total_session_queries,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": f"{cache_hit_rate:.1f}%",
            "total_hybrid_results": self.total_hybrid_results,
            "total_storage_results": self.total_storage_results,
            "rerank_operations": self.rerank_operations,
            "default_sort_by": self.default_sort_by,
            "default_order": self.default_order,
            "has_retrieval_handler": self.retrieval_handler is not None,
            "has_rerank_handler": self.rerank_handler is not None
        }

    # Enhanced QueryBuffer methods

    async def query_by_session(self, session_id: str, limit: Optional[int] = None,
                              sort_by: str = 'timestamp', order: str = 'desc') -> List[Dict[str, Any]]:
        """Session-specific query with multi-source coordination.

        Args:
            session_id: Session ID to query for
            limit: Maximum number of results (uses max_size if None)
            sort_by: Sort field ('timestamp' or 'id')
            order: Sort order ('asc' or 'desc')

        Returns:
            List of messages for the specified session
        """
        logger.info(f"QueryBuffer: Session query for {session_id}")
        self.total_session_queries += 1

        if limit is None:
            limit = self._max_size

        # 1. Parallel data collection from multiple sources
        hybrid_data, storage_data = await asyncio.gather(
            self._get_session_from_hybrid(session_id, limit, sort_by, order),
            self._get_session_from_storage(session_id, limit, sort_by, order),
            return_exceptions=True
        )

        # Handle exceptions
        if isinstance(hybrid_data, Exception):
            logger.warning(f"QueryBuffer: Hybrid buffer query failed: {hybrid_data}")
            hybrid_data = []
        if isinstance(storage_data, Exception):
            logger.warning(f"QueryBuffer: Storage query failed: {storage_data}")
            storage_data = []

        # 2. Intelligent data merging with deduplication
        merged_data = self._merge_session_data(hybrid_data, storage_data, session_id)

        # 3. Apply session-specific sorting and filtering
        filtered_data = self._apply_session_filters(merged_data, sort_by, order)

        # 4. Apply limit
        if limit and limit > 0:
            filtered_data = filtered_data[:limit]

        logger.info(f"QueryBuffer: Session query returned {len(filtered_data)} messages")
        return filtered_data

    async def _internal_rerank(self, results: List[Any], query_text: str) -> List[Any]:
        """Internal reranking with caching.

        Args:
            results: List of results to rerank
            query_text: Original query text

        Returns:
            Reranked results
        """
        if not self.rerank_handler or not results:
            return results

        # 1. Check rerank cache
        cache_key = self._generate_rerank_cache_key(results, query_text)
        cached_rerank = self._check_rerank_cache(cache_key)

        if cached_rerank:
            logger.debug("QueryBuffer: Rerank cache hit")
            return cached_rerank

        # 2. Execute reranking
        try:
            reranked = await self.rerank_handler(query_text, results)
            self.rerank_operations += 1

            # 3. Cache results
            self._cache_rerank_results(cache_key, reranked)

            logger.debug(f"QueryBuffer: Reranked {len(results)} results")
            return reranked
        except Exception as e:
            logger.error(f"QueryBuffer: Reranking failed: {e}")
            return results

    async def _get_session_from_hybrid(self, session_id: str, limit: Optional[int],
                                      sort_by: str, order: str) -> List[Dict[str, Any]]:
        """Get session data from HybridBuffer.

        Args:
            session_id: Session ID to search for
            limit: Maximum number of results
            sort_by: Sort field
            order: Sort order

        Returns:
            List of messages from HybridBuffer for the session
        """
        if not self.hybrid_buffer or not hasattr(self.hybrid_buffer, 'chunks'):
            return []

        # Search through HybridBuffer chunks for session data
        session_messages = []

        try:
            for chunk in self.hybrid_buffer.chunks:
                if hasattr(chunk, 'metadata') and chunk.metadata.get('session_id') == session_id:
                    # Extract messages from chunk
                    if hasattr(chunk, 'messages'):
                        session_messages.extend(chunk.messages)
                    elif hasattr(chunk, 'content'):
                        # If chunk has content but no messages, create a message-like structure
                        session_messages.append({
                            'id': getattr(chunk, 'id', f'chunk_{id(chunk)}'),
                            'content': chunk.content,
                            'metadata': getattr(chunk, 'metadata', {}),
                            'created_at': getattr(chunk, 'created_at', None)
                        })
        except Exception as e:
            logger.error(f"QueryBuffer: Error querying HybridBuffer for session {session_id}: {e}")

        return self._sort_messages(session_messages, sort_by, order)

    async def _get_session_from_storage(self, session_id: str, limit: Optional[int],
                                       sort_by: str, order: str) -> List[Dict[str, Any]]:
        """Get session data from persistent storage.

        Args:
            session_id: Session ID to search for
            limit: Maximum number of results
            sort_by: Sort field
            order: Sort order

        Returns:
            List of messages from storage for the session
        """
        if not self.retrieval_handler:
            return []

        try:
            # Use session-specific query
            query_text = f"session_id:{session_id}"
            results = await self.retrieval_handler(query_text, limit or 100)

            # Filter and sort results
            session_results = [
                result for result in results
                if isinstance(result, dict) and
                result.get('metadata', {}).get('session_id') == session_id
            ]

            return self._sort_messages(session_results, sort_by, order)
        except Exception as e:
            logger.error(f"QueryBuffer: Storage session query failed: {e}")
            return []

    def _merge_session_data(self, hybrid_data: List, storage_data: List, session_id: str) -> List[Dict[str, Any]]:
        """Merge session data from multiple sources with deduplication.

        Args:
            hybrid_data: Data from HybridBuffer
            storage_data: Data from storage
            session_id: Session ID for context

        Returns:
            Merged and deduplicated data
        """
        # Create lookup for deduplication
        seen_ids = set()
        merged_data = []

        # Process hybrid data first (more recent)
        for message in hybrid_data:
            if isinstance(message, dict):
                msg_id = message.get('id')
                if msg_id and msg_id not in seen_ids:
                    seen_ids.add(msg_id)
                    merged_data.append(message)

        # Process storage data (avoid duplicates)
        for message in storage_data:
            if isinstance(message, dict):
                msg_id = message.get('id')
                if msg_id and msg_id not in seen_ids:
                    seen_ids.add(msg_id)
                    merged_data.append(message)

        return merged_data

    def _apply_session_filters(self, merged_data: List, sort_by: str, order: str) -> List[Dict[str, Any]]:
        """Apply session-specific filtering and sorting.

        Args:
            merged_data: Merged data from multiple sources
            sort_by: Sort field
            order: Sort order

        Returns:
            Filtered and sorted data
        """
        # Additional session-specific filtering can be added here
        filtered_data = [
            msg for msg in merged_data
            if isinstance(msg, dict) and msg.get('content')  # Basic content filter
        ]

        # Apply sorting
        return self._sort_messages(filtered_data, sort_by, order)

    def _sort_messages(self, messages: List[Dict[str, Any]], sort_by: str, order: str) -> List[Dict[str, Any]]:
        """Sort messages by specified criteria.

        Args:
            messages: List of messages to sort
            sort_by: Sort field ('timestamp', 'id', or 'created_at')
            order: Sort order ('asc' or 'desc')

        Returns:
            Sorted messages
        """
        if not messages:
            return messages

        reverse = (order.lower() == 'desc')

        if sort_by == 'timestamp' or sort_by == 'created_at':
            return sorted(
                messages,
                key=lambda x: self._normalize_timestamp_for_sorting(x.get('created_at', '')),
                reverse=reverse
            )
        elif sort_by == 'id':
            return sorted(
                messages,
                key=lambda x: x.get('id', ''),
                reverse=reverse
            )
        else:
            # Default to timestamp
            return sorted(
                messages,
                key=lambda x: self._normalize_timestamp_for_sorting(x.get('created_at', '')),
                reverse=reverse
            )

    def _normalize_timestamp_for_sorting(self, timestamp) -> str:
        """Normalize timestamp to string for consistent sorting.

        Args:
            timestamp: Timestamp (could be datetime, float, string, or None)

        Returns:
            ISO format timestamp string for sorting (empty string if None/invalid)
        """
        if timestamp is None:
            return ''

        try:
            from datetime import datetime

            if isinstance(timestamp, datetime):
                # Convert datetime object to ISO string
                return timestamp.isoformat()
            elif isinstance(timestamp, (int, float)):
                # Convert from Unix timestamp
                return datetime.fromtimestamp(timestamp).isoformat()
            elif isinstance(timestamp, str):
                # Already a string, return as-is
                return timestamp
            else:
                # Convert to string
                return str(timestamp)
        except (ValueError, TypeError, OSError):
            # Return empty string for invalid timestamps (will sort to beginning)
            return ''

    def _generate_rerank_cache_key(self, results: List[Any], query_text: str) -> str:
        """Generate cache key for rerank results.

        Args:
            results: Results to rerank
            query_text: Query text

        Returns:
            Cache key string
        """
        # Create a simple hash based on query and result count
        result_hash = hash(tuple(str(r.get('id', '')) for r in results if isinstance(r, dict)))
        return f"rerank_{hash(query_text)}_{result_hash}_{len(results)}"

    def _check_rerank_cache(self, cache_key: str) -> Optional[List[Any]]:
        """Check rerank cache for cached results.

        Args:
            cache_key: Cache key to check

        Returns:
            Cached results if found, None otherwise
        """
        return self.rerank_cache.get(cache_key)

    def _cache_rerank_results(self, cache_key: str, results: List[Any]) -> None:
        """Cache rerank results.

        Args:
            cache_key: Cache key
            results: Results to cache
        """
        # Simple cache with size limit
        if len(self.rerank_cache) >= 50:  # Limit cache size
            # Remove oldest entry
            oldest_key = next(iter(self.rerank_cache))
            del self.rerank_cache[oldest_key]

        self.rerank_cache[cache_key] = results
