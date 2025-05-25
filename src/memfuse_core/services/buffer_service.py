"""Buffer Service implementation for MemFuse.

This service provides a high-performance buffer implementation that preserves
all buffer functionality (WriteBuffer, SpeculativeBuffer, QueryBuffer) while
optimizing for minimal latency overhead.

Key Design Principles:
- Preserve all buffer functionality
- True async implementation without task abuse
- Optimized WriteBuffer, SpeculativeBuffer, QueryBuffer
- User-level singleton pattern
- Minimal performance overhead (<5%)
- FIFO logic maintained correctly
"""

import asyncio
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Callable
from omegaconf import DictConfig
from loguru import logger

from ..interfaces import MemoryInterface, ServiceInterface
from ..buffer.write_buffer import WriteBuffer
from ..buffer.speculative_buffer import SpeculativeBuffer
from ..buffer.query_buffer import QueryBuffer

if TYPE_CHECKING:
    from .memory_service import MemoryService


class BufferService(MemoryInterface, ServiceInterface):
    """Buffer service with full WriteBuffer, SpeculativeBuffer, QueryBuffer functionality.

    This service preserves all buffer functionality while optimizing for performance:
    - WriteBuffer: FIFO queue management with optimized batch writes
    - SpeculativeBuffer: Prefetching with efficient context generation
    - QueryBuffer: Multi-source retrieval with LRU caching
    - True async implementation without task abuse
    - <5% overhead target
    """

    def __init__(
        self,
        memory_service: "MemoryService",
        user: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the Optimized Buffer service.

        Args:
            memory_service: MemoryService instance to delegate operations to
            user: User ID (required for user-level singleton pattern)
            config: Configuration dictionary with settings for the buffer
        """
        self.memory_service = memory_service
        self.user = user
        self.config = config or {}

        # Get the actual user_id (UUID) from memory_service
        self.user_id = getattr(memory_service, '_user_id', user) if memory_service else user

        # Buffer configuration
        buffer_config = self.config.get('buffer', {})
        write_config = buffer_config.get('write', {})
        speculative_config = buffer_config.get('speculative', {})
        query_config = buffer_config.get('query', {})

        # Initialize optimized buffer components
        self.write_buffer = WriteBuffer(
            max_size=write_config.get('max_size', 5),
            batch_threshold=write_config.get('batch_threshold', 5),
            storage_handler=self._create_storage_handler()
        )

        self.speculative_buffer = SpeculativeBuffer(
            max_size=speculative_config.get('max_size', 10),
            context_window=speculative_config.get('context_window', 3),
            retrieval_handler=self._create_retrieval_handler()
        )

        self.query_buffer = QueryBuffer(
            retrieval_handler=self._create_retrieval_handler(),
            max_size=query_config.get('max_size', 15),
            cache_size=query_config.get('cache_size', 100)
        )

        # Register WriteBuffer update handler for SpeculativeBuffer
        self.write_buffer.register_update_handler(self.speculative_buffer.update_from_items)

        # Statistics
        self.total_items_added = 0
        self.total_queries = 0
        self.total_batch_writes = 0

        logger.info(f"BufferService: Initialized for user {user} with full buffer functionality")

    def _create_storage_handler(self):
        """Create storage handler for WriteBuffer."""
        class StorageHandler:
            def __init__(self, memory_service):
                self.memory_service = memory_service

            async def handle_batch(self, items: List[Any]) -> List[str]:
                """Handle batch write to memory service."""
                try:
                    result = await self.memory_service.add_batch(items)
                    if result.get("status") == "success":
                        return result.get("data", {}).get("message_ids", [])
                    return []
                except Exception as e:
                    logger.error(f"StorageHandler: Batch write error: {e}")
                    return []

        return StorageHandler(self.memory_service)

    def _create_retrieval_handler(self):
        """Create retrieval handler for SpeculativeBuffer and QueryBuffer."""
        async def retrieval_handler(query: str, max_results: int) -> List[Any]:
            """Handle retrieval from memory service."""
            query_preview = query[:50] + "..." if len(query) > 50 else query
            logger.info(f"RetrievalHandler: Called with query: {query_preview}, max_results={max_results}")

            try:
                # Direct call to MemoryService to avoid infinite recursion
                # This bypasses the BufferService query method
                logger.debug(f"RetrievalHandler: Calling memory_service.query with top_k={max_results}")
                result = await self.memory_service.query(
                    query=query,
                    top_k=max_results,
                    include_messages=True,
                    include_knowledge=True
                )

                logger.info(f"RetrievalHandler: MemoryService returned status: {result.get('status')}")

                if result.get("status") == "success":
                    data = result.get("data", {})
                    logger.info(f"RetrievalHandler: Data keys: {list(data.keys())}")

                    # MemoryService returns results in data.results, not data.messages/data.knowledge
                    results = data.get("results", [])
                    logger.info(f"RetrievalHandler: Got {len(results)} total results from MemoryService")

                    if results:
                        logger.info(f"RetrievalHandler: First result sample keys: {list(results[0].keys())}")
                        logger.info(f"RetrievalHandler: First result type: {results[0].get('type')}")

                    # Separate messages and knowledge based on type field
                    messages = []
                    knowledge = []

                    for item in results:
                        item_type = item.get("type")
                        if item_type == "message":
                            messages.append(item)
                        elif item_type == "knowledge":
                            knowledge.append(item)
                        else:
                            logger.warning(f"RetrievalHandler: Unknown item type: {item_type}")

                    logger.info(f"RetrievalHandler: Separated into {len(messages)} messages, {len(knowledge)} knowledge items")

                    # Combine all items
                    items = messages + knowledge

                    final_items = items[:max_results]
                    logger.info(f"RetrievalHandler: Returning {len(final_items)} items")
                    return final_items
                else:
                    logger.warning(f"RetrievalHandler: MemoryService query failed: {result.get('message', 'Unknown error')}")
                    return []
            except Exception as e:
                logger.error(f"RetrievalHandler: Query error: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return []

        return retrieval_handler

    async def initialize(self, cfg: Optional[DictConfig] = None) -> bool:
        """Initialize the buffer service.

        Args:
            cfg: Configuration for the service (optional)

        Returns:
            True if initialization was successful, False otherwise
        """
        try:
            if self.memory_service:
                if hasattr(self.memory_service, 'initialize'):
                    if cfg is not None:
                        await self.memory_service.initialize(cfg)
                    else:
                        await self.memory_service.initialize()
            return True
        except Exception as e:
            logger.error(f"Failed to initialize optimized buffer service: {e}")
            return False

    async def shutdown(self) -> bool:
        """Shutdown the buffer service gracefully.

        Returns:
            True if shutdown was successful, False otherwise
        """
        try:
            # Clear buffer
            async with self._buffer_lock:
                self._write_buffer.clear()

            # Shutdown memory service if available
            if self.memory_service and hasattr(self.memory_service, 'shutdown'):
                await self.memory_service.shutdown()

            return True
        except Exception as e:
            logger.error(f"Failed to shutdown optimized buffer service: {e}")
            return False

    def is_initialized(self) -> bool:
        """Check if the buffer service is initialized.

        Returns:
            True if the service is initialized, False otherwise
        """
        return self.memory_service is not None

    async def add(self, messages: List[Dict[str, Any]], session_id: Optional[str] = None) -> Dict[str, Any]:
        """Add messages to memory using optimized WriteBuffer.

        Args:
            messages: List of message dictionaries with role and content
            session_id: Session ID for context (passed as parameter)

        Returns:
            Dictionary with status, code, and message IDs
        """
        if not self.memory_service:
            return {
                "status": "error",
                "code": 500,
                "data": None,
                "message": "No memory service available",
                "errors": [{"field": "general", "message": "No memory service available"}],
            }

        if not messages:
            return {
                "status": "success",
                "code": 200,
                "data": {"message_ids": []},
                "message": "No messages to add",
                "errors": None,
            }

        self.total_items_added += len(messages)

        # Add metadata to messages if needed
        for message in messages:
            if 'metadata' not in message:
                message['metadata'] = {}
            if self.user_id and 'user_id' not in message['metadata']:
                message['metadata']['user_id'] = self.user_id
            if session_id and 'session_id' not in message['metadata']:
                message['metadata']['session_id'] = session_id

        # Use optimized WriteBuffer for batching and FIFO management
        try:
            # Add messages to WriteBuffer one by one (maintains FIFO logic)
            batch_triggered = False
            for message in messages:
                if await self.write_buffer.add(message):
                    batch_triggered = True
                    self.total_batch_writes += 1

            # Get the last batch results for response
            last_results = await self.write_buffer.get_last_batch_results()

            return {
                "status": "success",
                "code": 200,
                "data": {"message_ids": last_results},
                "message": f"Added {len(messages)} messages to buffer",
                "errors": None,
            }

        except Exception as e:
            logger.error(f"BufferService.add: Error adding messages: {e}")
            return {
                "status": "error",
                "code": 500,
                "data": None,
                "message": f"Error adding messages: {str(e)}",
                "errors": [{"field": "general", "message": str(e)}],
            }

    async def add_batch(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Add a batch of messages to memory.

        Args:
            messages: List of message dictionaries with role and content

        Returns:
            Dictionary with status, code, and message IDs
        """
        # Direct delegation to add method (which uses write-through strategy)
        return await self.add(messages)

    async def read(self, item_ids: List[str]) -> Dict[str, Any]:
        """Read items from memory.

        Args:
            item_ids: List of item IDs

        Returns:
            Dictionary with status, code, and items
        """
        if not self.memory_service:
            return {
                "status": "error",
                "code": 500,
                "data": None,
                "message": "No memory service available",
                "errors": [{"field": "general", "message": "No memory service available"}],
            }

        # Direct delegation to memory service (no complex buffer lookup)
        return await self.memory_service.read(item_ids)

    async def update(self, item_ids: List[str], new_items: List[Any]) -> Dict[str, Any]:
        """Update items in memory.

        Args:
            item_ids: List of item IDs
            new_items: List of new items to replace the existing ones

        Returns:
            Dictionary with status, code, and updated item IDs
        """
        if not self.memory_service:
            return {
                "status": "error",
                "code": 500,
                "data": None,
                "message": "No memory service available",
                "errors": [{"field": "general", "message": "No memory service available"}],
            }

        # Add metadata to items if needed
        for item in new_items:
            if isinstance(item, dict) and 'metadata' in item:
                if self.user_id and 'user_id' not in item['metadata']:
                    item['metadata']['user_id'] = self.user_id

        # Direct delegation to memory service
        return await self.memory_service.update(item_ids, new_items)

    async def delete(self, item_ids: List[str]) -> Dict[str, Any]:
        """Delete items from memory.

        Args:
            item_ids: List of item IDs

        Returns:
            Dictionary with status, code, and deleted item IDs
        """
        if not self.memory_service:
            return {
                "status": "error",
                "code": 500,
                "data": None,
                "message": "No memory service available",
                "errors": [{"field": "general", "message": "No memory service available"}],
            }

        # Direct delegation to memory service
        return await self.memory_service.delete(item_ids)

    async def add_knowledge(self, knowledge_items: List[Any]) -> Dict[str, Any]:
        """Add knowledge items to memory.

        Args:
            knowledge_items: List of knowledge items

        Returns:
            Dictionary with status, code, and knowledge IDs
        """
        if not self.memory_service:
            return {
                "status": "error",
                "code": 500,
                "data": None,
                "message": "No memory service available",
                "errors": [{"field": "general", "message": "No memory service available"}],
            }

        # Add metadata to knowledge items if needed
        for item in knowledge_items:
            if isinstance(item, dict) and 'metadata' in item:
                if self.user_id and 'user_id' not in item['metadata']:
                    item['metadata']['user_id'] = self.user_id

        # Direct delegation to memory service
        return await self.memory_service.add_knowledge(knowledge_items)

    async def read_knowledge(self, knowledge_ids: List[str]) -> Dict[str, Any]:
        """Read knowledge items from memory.

        Args:
            knowledge_ids: List of knowledge item IDs

        Returns:
            Dictionary with status, code, and knowledge items
        """
        if not self.memory_service:
            return {
                "status": "error",
                "code": 500,
                "data": None,
                "message": "No memory service available",
                "errors": [{"field": "general", "message": "No memory service available"}],
            }

        # Direct delegation to memory service
        return await self.memory_service.read_knowledge(knowledge_ids)

    async def update_knowledge(self, knowledge_ids: List[str], new_knowledge_items: List[Any]) -> Dict[str, Any]:
        """Update knowledge items in memory.

        Args:
            knowledge_ids: List of knowledge item IDs
            new_knowledge_items: List of new knowledge items

        Returns:
            Dictionary with status, code, and updated knowledge IDs
        """
        if not self.memory_service:
            return {
                "status": "error",
                "code": 500,
                "data": None,
                "message": "No memory service available",
                "errors": [{"field": "general", "message": "No memory service available"}],
            }

        # Add metadata to knowledge items if needed
        for item in new_knowledge_items:
            if isinstance(item, dict) and 'metadata' in item:
                if self.user_id and 'user_id' not in item['metadata']:
                    item['metadata']['user_id'] = self.user_id

        # Direct delegation to memory service
        return await self.memory_service.update_knowledge(knowledge_ids, new_knowledge_items)

    async def delete_knowledge(self, knowledge_ids: List[str]) -> Dict[str, Any]:
        """Delete knowledge items from memory.

        Args:
            knowledge_ids: List of knowledge item IDs

        Returns:
            Dictionary with status, code, and deleted knowledge IDs
        """
        if not self.memory_service:
            return {
                "status": "error",
                "code": 500,
                "data": None,
                "message": "No memory service available",
                "errors": [{"field": "general", "message": "No memory service available"}],
            }

        # Direct delegation to memory service
        return await self.memory_service.delete_knowledge(knowledge_ids)

    async def query(
        self,
        query: str,
        top_k: int = 5,
        store_type: Optional[str] = None,
        session_id: Optional[str] = None,
        scope: str = "all",
        include_messages: bool = True,
        include_knowledge: bool = True,
    ) -> Dict[str, Any]:
        """Query memory for relevant messages.

        Args:
            query: Query string
            top_k: Maximum number of results to return
            store_type: Type of store to query (vector, graph, keyword, or None for all)
            session_id: Session ID to filter results (optional)
            scope: Scope of the query (all, session, or user)
            include_messages: Whether to include messages in results
            include_knowledge: Whether to include knowledge in results

        Returns:
            Dictionary with status, code, and query results
        """
        if not self.memory_service:
            return {
                "status": "error",
                "code": 500,
                "data": None,
                "message": "No memory service available",
                "errors": [{"field": "general", "message": "No memory service available"}],
            }

        self.total_queries += 1

        query_preview = query[:50] + "..." if len(query) > 50 else query
        logger.info(f"BufferService.query: Processing query '{query_preview}' with top_k={top_k}")

        # Use optimized QueryBuffer for multi-source retrieval
        try:
            # Log buffer states before query
            write_buffer_size = len(self.write_buffer.items) if hasattr(self.write_buffer, 'items') else 0
            speculative_buffer_size = len(self.speculative_buffer.items) if hasattr(self.speculative_buffer, 'items') else 0

            logger.info(f"BufferService.query: Buffer states - WriteBuffer: {write_buffer_size} items, SpeculativeBuffer: {speculative_buffer_size} items")

            # QueryBuffer combines results from storage, WriteBuffer, and SpeculativeBuffer
            logger.info("BufferService.query: Calling QueryBuffer.query")
            results = await self.query_buffer.query(
                query_text=query,
                write_buffer=self.write_buffer,
                speculative_buffer=self.speculative_buffer
            )

            logger.info(f"BufferService.query: QueryBuffer returned {len(results) if results else 0} results")
            logger.info(f"BufferService.query: Results type: {type(results)}")
            if results:
                logger.info(f"BufferService.query: First result keys: {list(results[0].keys()) if isinstance(results[0], dict) else 'not dict'}")
            else:
                logger.warning("BufferService.query: QueryBuffer returned empty results!")

            # Apply reranking if we have results (same as original BufferService)
            if results:
                reranked_results = await self._rerank_unified_results(
                    query=query,
                    items=results,
                    top_k=top_k
                )
                logger.info(f"BufferService.query: Reranked to {len(reranked_results)} results")
                limited_results = reranked_results
            else:
                limited_results = []
                logger.info("BufferService.query: No results to rerank")

            # Format response to match MemoryService format exactly
            # MemoryService returns {"data": {"results": [...], "total": ...}}
            response = {
                "status": "success",
                "code": 200,
                "data": {
                    "results": limited_results,  # This is what the test script expects
                    "total": len(limited_results)
                },
                "message": f"Retrieved {len(limited_results)} results",
                "errors": None,
            }

            logger.info(f"BufferService.query: Returning response with {len(limited_results)} results")
            logger.info(f"BufferService.query: Response structure: {{'status': '{response['status']}', 'data': {{'results': {len(response['data']['results'])}, 'total': {response['data']['total']}}}}}")
            if limited_results:
                logger.info(f"BufferService.query: First result sample: {limited_results[0]}")
            return response

        except Exception as e:
            logger.error(f"BufferService.query: Error querying: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "status": "error",
                "code": 500,
                "data": None,
                "message": f"Error querying: {str(e)}",
                "errors": [{"field": "general", "message": str(e)}],
            }

    async def _rerank_unified_results(
        self,
        query: str,
        items: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Unified rerank interface for BufferService.

        This method provides reranking functionality specifically for BufferService,
        identical to the original BufferService implementation.

        Args:
            query: Query string
            items: List of items to rerank (all in Response Schema format)
            top_k: Number of top results to return

        Returns:
            List of reranked items
        """
        if not items:
            return []

        try:
            # Import ServiceFactory to get global reranker instance
            from ..services.service_factory import ServiceFactory

            # Use global pre-loaded reranker instance if available
            reranker = ServiceFactory.get_global_reranker_instance()

            if reranker is None:
                # Fallback: Create new reranker instance only if global one is not available
                from ..rag.rerank import MiniLMReranker
                reranker = MiniLMReranker()
                await reranker.initialize()
                logger.warning("BufferService._rerank_unified_results: Using fallback reranker (global instance not available)")
            else:
                logger.info("BufferService._rerank_unified_results: Using global pre-loaded reranker instance")

            # Rerank all items using the unified interface
            reranked_items = await reranker.rerank(
                query=query,
                items=items,
                top_k=top_k,
                source="buffer_service"
            )

            # Add reranking metadata
            for item in reranked_items:
                if isinstance(item, dict) and "metadata" in item:
                    if "retrieval" not in item["metadata"]:
                        item["metadata"]["retrieval"] = {}
                    item["metadata"]["retrieval"]["reranked"] = True
                    item["metadata"]["retrieval"]["rerank_source"] = "buffer_service"

            return reranked_items

        except Exception as e:
            logger.error(f"BufferService._rerank_unified_results: Error reranking items: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Return original items limited to top_k if reranking fails
            return items[:top_k]

    async def get_buffer_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the optimized buffer system.

        Returns:
            Dictionary with detailed buffer statistics
        """
        write_stats = self.write_buffer.get_stats()
        speculative_stats = self.speculative_buffer.get_stats()
        query_stats = self.query_buffer.get_stats()

        return {
            "total_items_added": self.total_items_added,
            "total_queries": self.total_queries,
            "total_batch_writes": self.total_batch_writes,
            "write_buffer": write_stats,
            "speculative_buffer": speculative_stats,
            "query_buffer": query_stats,
            "performance": {
                "overhead_target": "<5%",
                "strategy": "optimized_three_component_system",
                "complexity": "full_functionality_preserved",
                "optimizations": [
                    "no_asyncio_task_abuse",
                    "true_async_implementation",
                    "efficient_fifo_logic",
                    "lru_cache_with_efficient_combination",
                    "optimized_context_generation"
                ]
            },
            "architecture": {
                "write_buffer": "FIFO queue with optimized batch writes",
                "speculative_buffer": "Prefetching with efficient context generation",
                "query_buffer": "Multi-source retrieval with LRU caching"
            }
        }


async def create_buffer_service_from_config(
    cfg: Any,
    memory_service: "MemoryService",
    user: Optional[str] = None,
) -> BufferService:
    """Create an BufferService from a configuration.

    Args:
        cfg: Configuration dictionary or DictConfig
        memory_service: MemoryService to delegate operations to
        user: User ID (required for user-level singleton pattern)

    Returns:
        BufferService instance
    """
    # Extract buffer configuration
    buffer_config = {}

    # Helper function to safely get attributes with defaults
    def get_attr(obj, attr_path, default_value):
        """Get attribute from nested object path with default value."""
        try:
            attrs = attr_path.split('.')
            current = obj
            for attr in attrs:
                current = getattr(current, attr)
            return current
        except (AttributeError, TypeError):
            return default_value

    # Extract buffer configuration from cfg
    if hasattr(cfg, 'buffer'):
        buffer_config['buffer'] = {
            'size': get_attr(cfg.buffer, 'write.max_size', 10),
        }

    # Create and return the service
    service = BufferService(
        memory_service=memory_service,
        user=user,
        config=buffer_config,
    )

    return service
