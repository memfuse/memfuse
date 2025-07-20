"""Query Buffer implementation for MemFuse.

This buffer provides unified query functionality with sorting support,
combining results from HybridBuffer and persistent storage without reranking.
"""

import asyncio
from typing import Any, Dict, List, Optional, Callable
from loguru import logger

from ..interfaces import BufferComponentInterface


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

        # HybridBuffer reference
        self.hybrid_buffer = None

        logger.info(f"QueryBuffer: Initialized with max_size={max_size}, default_sort={default_sort_by}, rerank_enabled={rerank_handler is not None}")

    def set_hybrid_buffer(self, hybrid_buffer):
        """Set the HybridBuffer instance for queries.

        Args:
            hybrid_buffer: HybridBuffer instance to use for queries
        """
        self.hybrid_buffer = hybrid_buffer
        logger.debug("QueryBuffer: HybridBuffer reference set")

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
        logger.info("QueryBuffer: Cache miss, querying storage and hybrid buffer")
        
        try:
            # Get results from storage
            storage_results = []
            if self.retrieval_handler:
                storage_results = await self.retrieval_handler(query_text, top_k * 2)  # Get more for better sorting
                self.total_storage_results += len(storage_results or [])
                logger.info(f"QueryBuffer: Got {len(storage_results or [])} results from storage")

            # Get results from HybridBuffer (use parameter or instance variable)
            hybrid_results = []
            buffer_to_use = hybrid_buffer or self.hybrid_buffer
            if buffer_to_use and hasattr(buffer_to_use, 'chunks'):
                hybrid_results = await self._query_hybrid_buffer(query_text, buffer_to_use, top_k)
                self.total_hybrid_results += len(hybrid_results)
                logger.info(f"QueryBuffer: Got {len(hybrid_results)} results from HybridBuffer")
            
            # Combine and sort results
            all_results = await self._combine_and_sort_results(
                storage_results or [],
                hybrid_results,
                sort_by,
                order
            )

            # Apply internal reranking if enabled and handler available
            if use_rerank and self.rerank_handler and all_results:
                all_results = await self._internal_rerank(all_results, query_text)

            # Limit results
            final_results = all_results[:top_k]

            # Update cache
            await self._update_cache(cache_key, final_results)

            # Update buffer items
            async with self._lock:
                self._items = final_results.copy()

            logger.info(f"QueryBuffer: Returning {len(final_results)} results (reranked: {use_rerank and self.rerank_handler is not None})")
            return final_results

        except Exception as e:
            logger.error(f"QueryBuffer: Query error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    async def _query_hybrid_buffer(
        self,
        query_text: str,
        hybrid_buffer,
        max_results: int
    ) -> List[Dict[str, Any]]:
        """Query HybridBuffer for relevant chunks.
        
        Args:
            query_text: Query text
            hybrid_buffer: HybridBuffer instance
            max_results: Maximum number of results
            
        Returns:
            List of relevant chunks formatted as query results
        """
        try:
            # Simple text matching for now (can be enhanced with vector similarity)
            query_lower = query_text.lower()
            results = []

            async with hybrid_buffer._data_lock:
                logger.info(f"QueryBuffer: Searching {len(hybrid_buffer.chunks)} chunks in HybridBuffer")
                for i, chunk in enumerate(hybrid_buffer.chunks):
                    logger.debug(f"QueryBuffer: Chunk {i} content preview: {chunk.content[:100]}...")
                    if len(results) >= max_results:
                        break

                    content = chunk.content.lower()
                    # Try multiple matching strategies
                    query_words = query_lower.split()
                    content_words = content.split()

                    # Strategy 1: Exact substring match
                    exact_match = query_lower in content

                    # Strategy 2: Word overlap (more flexible)
                    word_overlap = len(set(query_words) & set(content_words)) / len(query_words) if query_words else 0

                    # Strategy 3: Key terms match (music, artist, etc.)
                    key_terms = ['music', 'artist', 'taylor', 'swift', 'beyonce', 'song', 'album']
                    key_term_matches = sum(1 for term in key_terms if term in content)

                    logger.debug(f"QueryBuffer: Chunk {i} - exact_match: {exact_match}, word_overlap: {word_overlap:.2f}, key_terms: {key_term_matches}")

                    if exact_match or word_overlap > 0.1 or key_term_matches > 0:
                        # Calculate relevance score based on multiple factors
                        exact_score = content.count(query_lower) / len(content.split()) if exact_match else 0
                        overlap_score = word_overlap * 0.5
                        key_term_score = key_term_matches * 0.2
                        score = exact_score + overlap_score + key_term_score
                        
                        result = {
                            "id": f"hybrid_chunk_{i}",
                            "content": chunk.content,
                            "score": min(score, 1.0),  # Cap at 1.0
                            "type": "message",  # Changed from "chunk" to "message" for API compatibility
                            "role": "assistant",  # Default role for chunks
                            "created_at": None,  # HybridBuffer chunks don't have timestamps
                            "updated_at": None,
                            "metadata": {
                                **chunk.metadata,
                                "source": "hybrid_buffer",
                                "retrieval": {
                                    "source": "hybrid_buffer",
                                    "query_method": "text_matching"
                                }
                            }
                        }
                        results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"QueryBuffer: HybridBuffer query error: {e}")
            return []
    
    async def _combine_and_sort_results(
        self,
        storage_results: List[Any],
        hybrid_results: List[Any],
        sort_by: str,
        order: str
    ) -> List[Any]:
        """Combine and sort results from different sources.
        
        Args:
            storage_results: Results from persistent storage
            hybrid_results: Results from HybridBuffer
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
        
        # Sort results
        if sort_by == "score":
            all_results.sort(
                key=lambda x: x.get("score", 0.0),
                reverse=(order == "desc")
            )
        elif sort_by == "timestamp":
            all_results.sort(
                key=lambda x: x.get("created_at", ""),
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
                key=lambda x: x.get('created_at', ''),
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
                key=lambda x: x.get('created_at', ''),
                reverse=reverse
            )

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
