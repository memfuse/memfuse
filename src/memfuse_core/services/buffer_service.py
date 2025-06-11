"""Buffer Service implementation for MemFuse.

This service implements the Buffer architecture with RoundBuffer, HybridBuffer,
and QueryBuffer components, providing improved performance and functionality.
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING
from omegaconf import DictConfig
from loguru import logger

from ..interfaces import MemoryInterface, ServiceInterface, MessageInterface, MessageList, MessageBatchList
from ..buffer.round_buffer import RoundBuffer
from ..buffer.hybrid_buffer import HybridBuffer
from ..buffer.query_buffer import QueryBuffer

if TYPE_CHECKING:
    from .memory_service import MemoryService


class BufferService(MemoryInterface, ServiceInterface, MessageInterface):
    """Buffer Service with RoundBuffer, HybridBuffer, and QueryBuffer.

    This service implements the Buffer architecture:
    - RoundBuffer: Token-based short-term storage with automatic transfer
    - HybridBuffer: Dual-format (chunks + rounds) medium-term storage with FIFO
    - QueryBuffer: Unified query with sorting and caching
    """
    
    def __init__(
        self,
        memory_service: "MemoryService",
        user: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the Buffer Service.
        
        Args:
            memory_service: MemoryService instance to delegate operations to
            user: User ID (required for user-level singleton pattern)
            config: Configuration dictionary with Buffer settings
        """
        self.memory_service = memory_service
        self.user = user
        self.config = config or {}
        
        # Get the actual user_id (UUID) from memory_service
        self.user_id = getattr(memory_service, '_user_id', user) if memory_service else user
        
        # Buffer configuration
        buffer_config = self.config.get('buffer', {})
        round_config = buffer_config.get('round_buffer', {})
        hybrid_config = buffer_config.get('hybrid_buffer', {})
        query_config = buffer_config.get('query', {})

        # Rerank configuration
        retrieval_config = self.config.get('retrieval', {})
        self.use_rerank = retrieval_config.get('use_rerank', True)
        
        # Initialize Buffer components
        self.round_buffer = RoundBuffer(
            max_tokens=round_config.get('max_tokens', 800),
            max_size=round_config.get('max_size', 5),
            token_model=round_config.get('token_model', 'gpt-4o-mini')
        )
        
        self.hybrid_buffer = HybridBuffer(
            max_size=hybrid_config.get('max_size', 5),
            chunk_strategy=hybrid_config.get('chunk_strategy', 'message'),
            embedding_model=hybrid_config.get('embedding_model', 'all-MiniLM-L6-v2')
        )
        
        self.query_buffer = QueryBuffer(
            retrieval_handler=self._create_retrieval_handler(),
            max_size=query_config.get('max_size', 15),
            cache_size=query_config.get('cache_size', 100),
            default_sort_by=query_config.get('default_sort_by', 'score'),
            default_order=query_config.get('default_order', 'desc')
        )
        
        # Set up component connections
        self.round_buffer.set_transfer_handler(self.hybrid_buffer.add_from_rounds)
        self.hybrid_buffer.set_storage_handlers(
            self._create_sqlite_handler(),
            self._create_qdrant_handler()
        )
        self.query_buffer.set_hybrid_buffer(self.hybrid_buffer)
        
        # Statistics
        self.total_items_added = 0
        self.total_queries = 0
        self.total_batch_writes = 0
        
        logger.info(f"BufferService: Initialized for user {user} with Buffer architecture")
        logger.info(f"BufferService: Rerank enabled: {self.use_rerank}")
    
    def _create_retrieval_handler(self):
        """Create retrieval handler for QueryBuffer."""
        async def retrieval_handler(query: str, max_results: int) -> List[Any]:
            """Handle retrieval from memory service."""
            try:
                # Direct call to MemoryService for storage retrieval
                result = await self.memory_service.query(
                    query=query,
                    top_k=max_results,
                    include_messages=True,
                    include_knowledge=True,
                    include_chunks=True
                )
                
                if result.get("status") == "success":
                    data = result.get("data", {})
                    results = data.get("results", [])
                    return results
                else:
                    logger.warning(f"BufferService: MemoryService query failed: {result.get('message')}")
                    return []
            except Exception as e:
                logger.error(f"BufferService: Retrieval error: {e}")
                return []
        
        return retrieval_handler
    
    def _create_sqlite_handler(self):
        """Create SQLite handler for HybridBuffer."""
        async def sqlite_handler(rounds: List[MessageList]) -> None:
            """Handle SQLite storage operations."""
            try:
                # Convert rounds to MessageBatchList and store via memory service
                if rounds:
                    result = await self.memory_service.add_batch(rounds)
                    if result.get("status") != "success":
                        logger.error(f"BufferService: SQLite storage failed: {result.get('message')}")
            except Exception as e:
                logger.error(f"BufferService: SQLite handler error: {e}")
        
        return sqlite_handler
    
    def _create_qdrant_handler(self):
        """Create Qdrant handler for HybridBuffer."""
        async def qdrant_handler(points: List[Dict[str, Any]]) -> None:
            """Handle Qdrant storage operations."""
            try:
                # Store chunks via vector store if available
                if hasattr(self.memory_service, 'vector_store') and points:
                    # Convert points to format expected by vector store
                    for point in points:
                        # This would integrate with the actual vector store
                        # For now, we'll log the operation
                        logger.debug(f"BufferService: Would store chunk {point['id']} to Qdrant")
            except Exception as e:
                logger.error(f"BufferService: Qdrant handler error: {e}")
        
        return qdrant_handler
    
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
            logger.error(f"Failed to initialize BufferService: {e}")
            return False
    
    async def shutdown(self) -> bool:
        """Shutdown the buffer service gracefully.
        
        Returns:
            True if shutdown was successful, False otherwise
        """
        try:
            # Flush any remaining data
            await self.hybrid_buffer.flush_to_storage()
            
            # Clear buffers
            await self.round_buffer.clear()
            await self.query_buffer.clear()
            await self.query_buffer.clear_cache()
            
            # Shutdown memory service if available
            if self.memory_service and hasattr(self.memory_service, 'shutdown'):
                await self.memory_service.shutdown()
            
            return True
        except Exception as e:
            logger.error(f"Failed to shutdown BufferService: {e}")
            return False
    
    def is_initialized(self) -> bool:
        """Check if the buffer service is initialized.
        
        Returns:
            True if the service is initialized, False otherwise
        """
        return self.memory_service is not None
    
    async def add(self, messages: MessageList, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Add a single list of messages.
        
        Args:
            messages: List of message dictionaries (MessageList)
            session_id: Session ID for context (passed as parameter)
            
        Returns:
            Dictionary with status, data, and message information
        """
        return await self.add_batch([messages], session_id=session_id)
    
    async def add_batch(self, message_batch_list: MessageBatchList, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Add a batch of message lists.
        
        Args:
            message_batch_list: List of lists of messages (MessageBatchList)
            session_id: Session ID for context (passed as parameter)
            
        Returns:
            Dictionary with status, data, and message information
        """
        if not self.memory_service:
            return self._error_response("No memory service available")
        
        if not message_batch_list:
            return self._success_response([], "No message lists to add")
        
        try:
            transfer_triggered = False
            total_messages = 0

            logger.debug(f"BufferService.add_batch: Processing {len(message_batch_list)} message lists")

            for i, message_list in enumerate(message_batch_list):
                if not message_list:
                    continue

                logger.debug(f"BufferService.add_batch: Processing message_list {i} with {len(message_list)} messages")
                logger.debug(f"BufferService.add_batch: message_list type: {type(message_list)}")

                total_messages += len(message_list)

                # Add metadata to messages if needed
                for j, message in enumerate(message_list):
                    logger.debug(f"BufferService.add_batch: Processing message {j}: type={type(message)}")
                    # Ensure message is a dictionary
                    if isinstance(message, dict):
                        if 'metadata' not in message:
                            message['metadata'] = {}
                        if self.user_id and 'user_id' not in message['metadata']:
                            message['metadata']['user_id'] = self.user_id
                        if session_id and 'session_id' not in message['metadata']:
                            message['metadata']['session_id'] = session_id
                    else:
                        logger.warning(f"BufferService: Skipping non-dict message {j}: {type(message)} - {message}")

                # Add to RoundBuffer (may trigger transfer to HybridBuffer)
                logger.debug(f"BufferService.add_batch: Adding message_list to RoundBuffer")
                if await self.round_buffer.add(message_list, session_id):
                    transfer_triggered = True
                    self.total_transfers += 1

            # CRITICAL: Also store data via MemoryService for persistent storage and contextual chunking
            logger.info(f"BufferService.add_batch: Storing {len(message_batch_list)} message lists via MemoryService")
            memory_result = await self.memory_service.add_batch(message_batch_list, session_id=session_id)

            if memory_result.get("status") == "success":
                logger.info(f"BufferService.add_batch: MemoryService storage successful")
                # Extract message IDs from MemoryService response
                memory_data = memory_result.get("data", {})
                message_ids = memory_data.get("message_ids", []) if isinstance(memory_data, dict) else []
            else:
                logger.error(f"BufferService.add_batch: MemoryService storage failed: {memory_result.get('message')}")
                message_ids = []

            self.total_items_added += total_messages

            return self._success_response(
                {"message_ids": message_ids},  # Return actual message IDs from MemoryService
                f"Added {len(message_batch_list)} message lists to Buffer and MemoryService",
                transfer_triggered=transfer_triggered,
                total_messages=total_messages,
                memory_storage_status=memory_result.get("status", "unknown")
            )
            
        except Exception as e:
            logger.error(f"BufferService.add_batch: Error adding message batch: {e}")
            return self._error_response(f"Error adding message batch: {str(e)}")
    
    async def query(
        self,
        query: str,
        top_k: int = 5,
        store_type: Optional[str] = None,
        session_id: Optional[str] = None,
        scope: str = "all",
        include_messages: bool = True,
        include_knowledge: bool = True,
        include_chunks: bool = True,
        sort_by: Optional[str] = None,
        order: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Query memory for relevant messages with Buffer enhancements.
        
        Args:
            query: Query string
            top_k: Maximum number of results to return
            store_type: Type of store to query (ignored in current implementation)
            session_id: Session ID to filter results (optional)
            scope: Scope of the query (all, session, or user)
            include_messages: Whether to include messages in results
            include_knowledge: Whether to include knowledge in results
            include_chunks: Whether to include chunks in results
            sort_by: Sort field ('score' or 'timestamp')
            order: Sort order ('asc' or 'desc')
            
        Returns:
            Dictionary with status, code, and query results
        """
        if not self.memory_service:
            return self._error_response("No memory service available")
        
        self.total_queries += 1
        
        query_preview = query[:50] + "..." if len(query) > 50 else query
        logger.info(f"BufferService.query: Processing query '{query_preview}' with top_k={top_k}")

        try:
            # Log buffer states before query (similar to V2)
            round_buffer_size = len(self.round_buffer.rounds)
            hybrid_buffer_size = len(self.hybrid_buffer.chunks)

            logger.info(f"BufferService.query: Buffer states - RoundBuffer: {round_buffer_size} rounds, HybridBuffer: {hybrid_buffer_size} chunks")

            # Query using QueryBuffer (similar to V2's QueryBuffer)
            logger.info("BufferService.query: Calling QueryBuffer.query")
            results = await self.query_buffer.query(
                query_text=query,
                top_k=top_k,
                sort_by=sort_by or "score",
                order=order or "desc",
                hybrid_buffer=self.hybrid_buffer
            )

            logger.info(f"BufferService.query: QueryBuffer returned {len(results) if results else 0} results")
            logger.info(f"BufferService.query: Results type: {type(results)}")
            if results:
                logger.info(f"BufferService.query: First result keys: {list(results[0].keys()) if isinstance(results[0], dict) else 'not dict'}")
            else:
                logger.warning("BufferService.query: QueryBuffer returned empty results!")

            # Apply reranking if enabled and we have results
            if results and self.use_rerank:
                reranked_results = await self._rerank_unified_results(
                    query=query,
                    items=results,
                    top_k=top_k
                )
                logger.info(f"BufferService.query: Reranked to {len(reranked_results)} results")
                limited_results = reranked_results
            elif results:
                # No reranking, just limit results
                limited_results = results[:top_k]
                logger.info(f"BufferService.query: No reranking applied, limited to {len(limited_results)} results")
            else:
                limited_results = []
                logger.info("BufferService.query: No results to process")

            # Format response to match MemoryService format exactly (same as V2)
            # MemoryService returns {"data": {"results": [...], "total": ...}}
            response = {
                "status": "success",
                "code": 200,
                "data": {
                    "results": limited_results,  # This is what the test script expects
                    "total": len(limited_results)
                },
                "message": f"Retrieved {len(limited_results)} results using Buffer",
                "errors": None,
            }

            logger.info(f"BufferService.query: Returning response with {len(limited_results)} results")
            logger.info(f"BufferService.query: Response structure: {{'status': '{response['status']}', 'data': {{'results': {len(response['data']['results'])}, 'total': {response['data']['total']}}}}}")
            if limited_results:
                logger.info(f"BufferService.query: First result sample: {limited_results[0]}")
            return response

        except Exception as e:
            logger.error(f"BufferService.query: Error querying: {e}")
            return self._error_response(f"Error querying: {str(e)}")
    
    async def get_messages_by_session(
        self,
        session_id: str,
        limit: Optional[int] = None,
        sort_by: str = 'timestamp',
        order: str = 'desc',
        buffer_only: bool = False
    ) -> List[Dict[str, Any]]:
        """Get messages for a session with Buffer support.
        
        Args:
            session_id: Session ID
            limit: Maximum number of messages to return
            sort_by: Field to sort by ('timestamp' or 'id')
            order: Sort order ('asc' or 'desc')
            buffer_only: If True, only return RoundBuffer data
            
        Returns:
            List of message data
        """
        if not self.memory_service:
            return []
        
        try:
            if buffer_only:
                # Only return RoundBuffer data
                return await self.round_buffer.get_all_messages_for_read_api(
                    limit=limit,
                    sort_by=sort_by,
                    order=order
                )
            else:
                # Return HybridBuffer + SQLite data (excluding RoundBuffer)
                all_messages = []
                
                # Get messages from HybridBuffer
                hybrid_messages = await self.hybrid_buffer.get_all_messages_for_read_api(
                    limit=None,  # Get all first, apply limit later
                    sort_by=sort_by,
                    order=order
                )
                
                # Filter by session_id
                for msg in hybrid_messages:
                    if msg.get('metadata', {}).get('session_id') == session_id:
                        all_messages.append(msg)
                
                # Get messages from storage via memory service
                if hasattr(self.memory_service, 'get_messages_by_session'):
                    stored_messages = await self.memory_service.get_messages_by_session(
                        session_id=session_id,
                        limit=None,
                        sort_by=sort_by,
                        order=order
                    )
                    all_messages.extend(stored_messages)
                
                # Sort combined messages
                if sort_by == 'timestamp':
                    all_messages.sort(
                        key=lambda x: x.get('created_at', ''),
                        reverse=(order == 'desc')
                    )
                elif sort_by == 'id':
                    all_messages.sort(
                        key=lambda x: x.get('id', ''),
                        reverse=(order == 'desc')
                    )
                
                # Apply limit
                if limit is not None and limit > 0:
                    all_messages = all_messages[:limit]
                
                return all_messages
                
        except Exception as e:
            logger.error(f"BufferService.get_messages_by_session: Error: {e}")
            return []
    
    async def get_buffer_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the Buffer system.
        
        Returns:
            Dictionary with detailed buffer statistics
        """
        round_stats = self.round_buffer.get_stats()
        hybrid_stats = self.hybrid_buffer.get_stats()
        query_stats = self.query_buffer.get_stats()
        
        return {
            "version": "0.1.1",
            "total_items_added": self.total_items_added,
            "total_queries": self.total_queries,
            "total_transfers": self.total_transfers,
            "round_buffer": round_stats,
            "hybrid_buffer": hybrid_stats,
            "query_buffer": query_stats,
            "architecture": {
                "round_buffer": "Token-based FIFO with automatic transfer",
                "hybrid_buffer": "Dual-format (chunks + rounds) with FIFO",
                "query_buffer": "Unified query with sorting and caching"
            }
        }
    
    def _success_response(self, data: Any, message: str, **kwargs) -> Dict[str, Any]:
        """Create a success response."""
        return {
            "status": "success",
            "code": 200,
            "data": data,
            "message": message,
            "errors": None,
            **kwargs
        }
    
    def _error_response(self, message: str, code: int = 500) -> Dict[str, Any]:
        """Create an error response."""
        return {
            "status": "error",
            "code": code,
            "data": None,
            "message": message,
            "errors": [{"field": "general", "message": message}]
        }
    
    # Delegate other methods to memory service
    async def read(self, item_ids: List[str]) -> Dict[str, Any]:
        """Read items from memory."""
        if not self.memory_service:
            return self._error_response("No memory service available")
        return await self.memory_service.read(item_ids)
    
    async def update(self, item_ids: List[str], new_items: List[Any]) -> Dict[str, Any]:
        """Update items in memory."""
        if not self.memory_service:
            return self._error_response("No memory service available")
        return await self.memory_service.update(item_ids, new_items)
    
    async def delete(self, item_ids: List[str]) -> Dict[str, Any]:
        """Delete items from memory."""
        if not self.memory_service:
            return self._error_response("No memory service available")
        return await self.memory_service.delete(item_ids)
    
    async def add_knowledge(self, knowledge_items: List[Any]) -> Dict[str, Any]:
        """Add knowledge items to memory."""
        if not self.memory_service:
            return self._error_response("No memory service available")
        return await self.memory_service.add_knowledge(knowledge_items)
    
    async def read_knowledge(self, knowledge_ids: List[str]) -> Dict[str, Any]:
        """Read knowledge items from memory."""
        if not self.memory_service:
            return self._error_response("No memory service available")
        return await self.memory_service.read_knowledge(knowledge_ids)
    
    async def update_knowledge(self, knowledge_ids: List[str], new_knowledge_items: List[Any]) -> Dict[str, Any]:
        """Update knowledge items in memory."""
        if not self.memory_service:
            return self._error_response("No memory service available")
        return await self.memory_service.update_knowledge(knowledge_ids, new_knowledge_items)
    
    async def delete_knowledge(self, knowledge_ids: List[str]) -> Dict[str, Any]:
        """Delete knowledge items from memory."""
        if not self.memory_service:
            return self._error_response("No memory service available")
        return await self.memory_service.delete_knowledge(knowledge_ids)

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
                top_k=top_k
            )

            logger.info(f"BufferService._rerank_unified_results: Reranked {len(items)} items to {len(reranked_items)} results")
            return reranked_items

        except Exception as e:
            logger.error(f"BufferService._rerank_unified_results: Reranking error: {e}")
            # Return original items limited to top_k if reranking fails
            return items[:top_k]
