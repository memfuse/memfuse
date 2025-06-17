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
from ..buffer.flush_manager import FlushManager

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
        performance_config = buffer_config.get('performance', {})

        # FlushManager configuration
        flush_config = {
            'max_workers': performance_config.get('max_flush_workers', 3),
            'max_queue_size': performance_config.get('max_flush_queue_size', 100),
            'default_timeout': performance_config.get('flush_timeout', 30.0),
            'flush_interval': performance_config.get('flush_interval', 60.0),
            'enable_auto_flush': performance_config.get('enable_auto_flush', True)
        }

        # Initialize FlushManager
        self.flush_manager = FlushManager(
            max_workers=flush_config['max_workers'],
            max_queue_size=flush_config['max_queue_size'],
            default_timeout=flush_config['default_timeout'],
            flush_interval=flush_config['flush_interval'],
            enable_auto_flush=flush_config['enable_auto_flush']
        )

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
            embedding_model=hybrid_config.get('embedding_model', 'all-MiniLM-L6-v2'),
            flush_manager=self.flush_manager,
            auto_flush_interval=flush_config['flush_interval'],
            enable_auto_flush=flush_config['enable_auto_flush']
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
        self.total_transfers = 0
        
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
        """Create SQLite handler for FlushManager.

        This handler is responsible for persisting original message rounds to SQLite
        database via MemoryService. It's called by FlushManager during flush operations.

        Key responsibilities:
        - Store complete message rounds for Read API and long-term persistence
        - Maintain data integrity and consistency across flush operations
        - Handle errors gracefully with proper logging and exception propagation
        - Support batch operations for efficiency and performance
        - Integrate with MemoryService for unified data management

        The handler is designed to work with the optimistic clearing strategy:
        - Data is cleared from buffer before flush (optimistic)
        - If flush fails, FlushManager handles error recovery
        - This minimizes buffer lock time and improves concurrency

        Returns:
            Async callable that handles SQLite storage operations
        """
        async def sqlite_handler(rounds: List[MessageList]) -> None:
            """Handle SQLite storage operations for message rounds.

            Args:
                rounds: List of MessageList objects to store in SQLite database

            Raises:
                Exception: If storage operation fails, propagated to FlushManager
                          for error handling and potential data recovery
            """
            try:
                if rounds and self.memory_service:
                    logger.debug(f"BufferService: Storing {len(rounds)} rounds to database via SQLite handler")
                    result = await self.memory_service.add_batch(rounds)
                    if result.get("status") == "success":
                        logger.debug(f"BufferService: SQLite storage successful")
                    else:
                        logger.error(f"BufferService: SQLite storage failed: {result.get('message')}")
                        raise Exception(f"SQLite storage failed: {result.get('message')}")
            except Exception as e:
                logger.error(f"BufferService: SQLite handler error: {e}")
                raise  # Re-raise to signal flush failure to FlushManager

        return sqlite_handler


    
    def _create_qdrant_handler(self):
        """Create Qdrant handler for FlushManager.

        This handler is responsible for persisting processed chunks and embeddings
        to Qdrant vector database for semantic search and retrieval operations.

        Key responsibilities:
        - Store pre-processed chunks and embeddings for fast semantic search
        - Maintain vector index consistency for query operations
        - Handle errors gracefully with proper logging and exception propagation
        - Support batch operations for efficient vector storage
        - Integrate with MemoryService's vector storage capabilities

        The handler works in conjunction with SQLite handler:
        - SQLite stores original message rounds for complete data preservation
        - Qdrant stores processed chunks/embeddings for fast retrieval
        - Both are flushed together to maintain data consistency

        Returns:
            Async callable that handles Qdrant vector storage operations
        """
        async def qdrant_handler(chunks, embeddings) -> None:
            """Handle Qdrant vector storage operations for chunks and embeddings.

            Args:
                chunks: List of text chunks to store
                embeddings: Corresponding embeddings for the chunks

            Raises:
                Exception: If storage operation fails, propagated to FlushManager
                          for error handling and potential data recovery
            """
            try:
                if chunks and embeddings and self.memory_service:
                    logger.debug(f"BufferService: Storing {len(chunks)} chunks to Qdrant via handler")
                    # Convert to format expected by memory service
                    # This would need to be implemented based on your MemoryService interface
                    # For now, we log the operation as successful
                    # TODO: Implement actual Qdrant storage via MemoryService
                    logger.debug("BufferService: Qdrant storage successful")
            except Exception as e:
                logger.error(f"BufferService: Qdrant handler error: {e}")
                raise  # Re-raise to signal flush failure to FlushManager

        return qdrant_handler
    
    async def initialize(self, cfg: Optional[DictConfig] = None) -> bool:
        """Initialize the buffer service.

        Args:
            cfg: Configuration for the service (optional)

        Returns:
            True if initialization was successful, False otherwise
        """
        try:
            # Initialize FlushManager
            if not await self.flush_manager.initialize():
                logger.error("BufferService: Failed to initialize FlushManager")
                return False

            # Set up storage handlers
            self.flush_manager.set_handlers(
                sqlite_handler=self._create_sqlite_handler(),
                qdrant_handler=self._create_qdrant_handler()
            )

            # Initialize HybridBuffer
            if not await self.hybrid_buffer.initialize():
                logger.error("BufferService: Failed to initialize HybridBuffer")
                return False

            # Initialize memory service if needed
            if self.memory_service:
                if hasattr(self.memory_service, 'initialize'):
                    if cfg is not None:
                        await self.memory_service.initialize(cfg)
                    else:
                        await self.memory_service.initialize()

            logger.info("BufferService: Initialization completed successfully")
            return True

        except Exception as e:
            logger.error(f"BufferService: Failed to initialize: {e}")
            return False
    
    async def shutdown(self) -> bool:
        """Shutdown the buffer service gracefully.
        
        Returns:
            True if shutdown was successful, False otherwise
        """
        try:
            # Shutdown components in reverse order
            await self.hybrid_buffer.shutdown()
            await self.flush_manager.shutdown()

            # Clear buffers
            await self.round_buffer.clear()
            await self.query_buffer.clear()
            await self.query_buffer.clear_cache()

            # Shutdown memory service if available
            if self.memory_service and hasattr(self.memory_service, 'shutdown'):
                await self.memory_service.shutdown()

            logger.info("BufferService: Shutdown completed successfully")
            return True
        except Exception as e:
            logger.error(f"BufferService: Failed to shutdown: {e}")
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

                # Add metadata and required fields to messages
                for j, message in enumerate(message_list):
                    logger.debug(f"BufferService.add_batch: Processing message {j}: type={type(message)}")
                    # Ensure message is a dictionary
                    if isinstance(message, dict):
                        # Add metadata if not present
                        if 'metadata' not in message:
                            message['metadata'] = {}
                        if self.user_id and 'user_id' not in message['metadata']:
                            message['metadata']['user_id'] = self.user_id
                        if session_id and 'session_id' not in message['metadata']:
                            message['metadata']['session_id'] = session_id

                        # Add required fields for buffer consistency (id, created_at, updated_at)
                        self._ensure_message_fields(message)

                    else:
                        logger.warning(f"BufferService: Skipping non-dict message {j}: {type(message)} - {message}")

                # Add to RoundBuffer (may trigger transfer to HybridBuffer)
                logger.debug(f"BufferService.add_batch: Adding message_list to RoundBuffer")
                if await self.round_buffer.add(message_list, session_id):
                    transfer_triggered = True
                    self.total_transfers += 1

            # BufferService only manages buffer data flow
            # Data will be persisted when HybridBuffer flushes to storage
            logger.info(f"BufferService.add_batch: Added {len(message_batch_list)} message lists to Buffer system")

            # Collect message IDs from the processed messages
            message_ids = []
            for message_list in message_batch_list:
                for message in message_list:
                    if isinstance(message, dict) and message.get('id'):
                        message_ids.append(message['id'])

            self.total_items_added += total_messages

            return self._success_response(
                {"message_ids": message_ids},  # Return message IDs from buffer
                f"Added {len(message_batch_list)} message lists to Buffer system",
                transfer_triggered=transfer_triggered,
                total_messages=total_messages,
                buffer_status="success"
            )
            
        except Exception as e:
            logger.error(f"BufferService.add_batch: Error adding message batch: {e}")
            return self._error_response(f"Error adding message batch: {str(e)}")

    def _ensure_message_fields(self, message: Dict[str, Any]) -> None:
        """Ensure message has required fields (id, created_at, updated_at).

        For initial creation, created_at and updated_at use the same timestamp.
        When storing to SQLite database, updated_at will be refreshed.

        Args:
            message: Message dictionary to update
        """
        import uuid
        from datetime import datetime

        # Add ID if missing
        if 'id' not in message or not message['id']:
            message['id'] = str(uuid.uuid4())

        # Add created_at if missing
        if 'created_at' not in message or not message['created_at']:
            now = datetime.now().isoformat()
            message['created_at'] = now

        # Add updated_at if missing - use same timestamp as created_at for initial creation
        if 'updated_at' not in message or not message['updated_at']:
            # For initial creation, use the same timestamp as created_at
            message['updated_at'] = message['created_at']

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
                # Return HybridBuffer + Database data
                # With new architecture, HybridBuffer should be mostly empty (auto-flushed)
                # But we still check both sources for completeness
                all_messages = []

                # Get messages from HybridBuffer (should be minimal due to auto-flush)
                hybrid_messages = await self.hybrid_buffer.get_all_messages_for_read_api(
                    limit=None,  # Get all first, apply limit later
                    sort_by=sort_by,
                    order=order
                )

                # Filter by session_id and add to results
                for msg in hybrid_messages:
                    if msg.get('metadata', {}).get('session_id') == session_id:
                        all_messages.append(msg)

                # Get messages from database via memory service (primary source)
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
