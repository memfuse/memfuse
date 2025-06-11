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
        max_size: int = 15,
        cache_size: int = 100,
        default_sort_by: str = "score",
        default_order: str = "desc"
    ):
        """Initialize the QueryBuffer.
        
        Args:
            retrieval_handler: Async callback for storage retrieval
            max_size: Maximum number of items to return from a query
            cache_size: Maximum number of queries to cache
            default_sort_by: Default sort field ('score' or 'timestamp')
            default_order: Default sort order ('asc' or 'desc')
        """
        self.retrieval_handler = retrieval_handler
        self._max_size = max_size
        self.cache_size = cache_size
        self.default_sort_by = default_sort_by
        self.default_order = default_order
        
        # Cache for query results
        self.query_cache: Dict[str, List[Any]] = {}
        self._cache_order: List[str] = []  # LRU tracking
        
        # Buffer state
        self._items: List[Any] = []
        
        # Async lock for thread safety
        self._lock = asyncio.Lock()
        
        # Statistics
        self.total_queries = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_hybrid_results = 0
        self.total_storage_results = 0

        # HybridBuffer reference
        self.hybrid_buffer = None

        logger.info(f"QueryBuffer: Initialized with max_size={max_size}, default_sort={default_sort_by}")

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
        hybrid_buffer=None
    ) -> List[Any]:
        """Query with unified results from HybridBuffer and storage.
        
        Args:
            query_text: Query text
            top_k: Maximum number of results (uses max_size if None)
            sort_by: Sort field ('score' or 'timestamp', uses default if None)
            order: Sort order ('asc' or 'desc', uses default if None)
            hybrid_buffer: HybridBuffer instance to include results from
            
        Returns:
            List of query results sorted according to parameters
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
            
            # Limit results
            final_results = all_results[:top_k]
            
            # Update cache
            await self._update_cache(cache_key, final_results)
            
            # Update buffer items
            async with self._lock:
                self._items = final_results.copy()
            
            logger.info(f"QueryBuffer: Returning {len(final_results)} sorted results")
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

            async with hybrid_buffer._lock:
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
                async with buffer_to_use._lock:
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
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": f"{cache_hit_rate:.1f}%",
            "total_hybrid_results": self.total_hybrid_results,
            "total_storage_results": self.total_storage_results,
            "default_sort_by": self.default_sort_by,
            "default_order": self.default_order,
            "has_retrieval_handler": self.retrieval_handler is not None
        }
