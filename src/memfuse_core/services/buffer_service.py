"""Buffer Service implementation for MemFuse.

This service implements the Buffer architecture with RoundBuffer, HybridBuffer,
and QueryBuffer components, providing improved performance and functionality.
"""

import os
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from omegaconf import DictConfig
from loguru import logger

from ..interfaces import MemoryInterface, ServiceInterface, MessageInterface, MessageList, MessageBatchList
from ..buffer.write_buffer import WriteBuffer
from ..buffer.query_buffer import QueryBuffer
from ..buffer.speculative_buffer import SpeculativeBuffer
from ..buffer.config_factory import BufferConfigManager

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

        # Check if buffer is enabled (use global config, not just buffer section)
        # Environment variable can override config file setting
        buffer_config = self.config.get('buffer', {})
        config_enabled = buffer_config.get('enabled', True)
        env_enabled = os.environ.get('MEMFUSE_BUFFER_ENABLED', '').lower()

        if env_enabled in ('true', 'false'):
            self.buffer_enabled = env_enabled == 'true'
            logger.info(f"BufferService: Buffer enabled setting overridden by environment variable: {self.buffer_enabled}")
        else:
            self.buffer_enabled = config_enabled

        # Statistics (always available)
        self.total_items_added = 0
        self.total_queries = 0
        self.total_batch_writes = 0
        self.total_transfers = 0

        if not self.buffer_enabled:
            # Bypass mode: Minimal initialization
            logger.info(f"BufferService: Buffer disabled, operating in bypass mode for user {user}")
            self.write_buffer = None
            self.query_buffer = None
            self.speculative_buffer = None
            self.config_manager = None
            self.use_rerank = False
        else:
            # Normal mode: Full buffer architecture
            logger.info(f"BufferService: Buffer enabled, initializing full architecture for user {user}")

            # Initialize configuration manager for autonomous component configuration
            self.config_manager = BufferConfigManager(self.config)

            # Validate configuration
            if not self.config_manager.validate_configuration():
                logger.warning("BufferService: Configuration validation failed, using defaults")

            # Get component configurations from factory
            buffer_service_config = self.config_manager.get_buffer_service_config()

            # Create MemoryService handler for storage operations
            memory_service_handler = self._create_memory_service_handler()

            # Rerank configuration
            retrieval_config = buffer_service_config.get('retrieval', {})
            self.use_rerank = retrieval_config.get('use_rerank', True)

            # Initialize WriteBuffer with autonomous configuration
            write_buffer_config = buffer_service_config['write_buffer']
            self.write_buffer = WriteBuffer(
                config=write_buffer_config,
                memory_service_handler=memory_service_handler
            )

            # Initialize QueryBuffer with autonomous configuration
            query_config = buffer_service_config['query_buffer']
            self.query_buffer = QueryBuffer(
                retrieval_handler=self._create_retrieval_handler(),
                rerank_handler=self._create_rerank_handler(),
                max_size=query_config.get('max_size', 15),
                cache_size=query_config.get('cache_size', 100),
                default_sort_by=query_config.get('default_sort_by', 'score'),
                default_order=query_config.get('default_order', 'desc')
            )

            # Initialize SpeculativeBuffer with autonomous configuration
            speculative_config = buffer_service_config['speculative_buffer']
            self.speculative_buffer = SpeculativeBuffer(
                max_size=speculative_config.get('max_size', 10),
                context_window=speculative_config.get('context_window', 3),
                retrieval_handler=self._create_retrieval_handler()
            )

            # Set up component connections
            self.query_buffer.set_hybrid_buffer(self.write_buffer.get_hybrid_buffer())
            self.query_buffer.set_round_buffer(self.write_buffer.get_round_buffer())

        logger.info(f"BufferService: Initialized for user {user} with {'bypass' if not self.buffer_enabled else 'buffer'} mode")
        if self.buffer_enabled:
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

    def _create_rerank_handler(self):
        """Create rerank handler for QueryBuffer."""
        if not self.use_rerank:
            return None

        async def rerank_handler(query: str, results: List[Any]) -> List[Any]:
            """Handle reranking using the same logic as BufferService."""
            try:
                return await self._rerank_unified_results(
                    query=query,
                    items=results,
                    top_k=len(results)  # Don't limit here, let QueryBuffer handle limiting
                )
            except Exception as e:
                logger.error(f"BufferService: Rerank handler error: {e}")
                return results

        return rerank_handler

    def _create_memory_service_handler(self):
        """Create unified MemoryService handler for FlushManager.

        This handler delegates all storage operations to MemoryService, which then
        routes data through the MemoryLayer to trigger parallel M0/M1/M2 processing.
        This maintains proper architectural separation and enables the full memory hierarchy.

        Key responsibilities:
        - Route all buffer data through MemoryService.add_batch()
        - Trigger parallel M0/M1/M2 processing via MemoryLayer
        - Maintain architectural separation between buffer and storage layers
        - Handle errors gracefully with proper logging and exception propagation
        - Support batch operations for efficient processing

        Architecture flow:
        BufferService → MemoryService.add_batch() → MemoryLayer → M0/M1/M2 parallel processing

        Returns:
            Async callable that handles unified memory service operations
        """
        async def memory_service_handler(rounds: List[MessageList]) -> None:
            """Handle unified memory service operations for message rounds.

            Args:
                rounds: List of MessageList objects to process through memory hierarchy

            Raises:
                Exception: If storage operation fails, propagated to FlushManager
                          for error handling and potential data recovery
            """
            try:
                if rounds and self.memory_service:
                    logger.info(f"BufferService: Processing {len(rounds)} rounds through MemoryService")
                    result = await self.memory_service.add_batch(rounds)
                    if result.get("status") == "success":
                        logger.info(f"BufferService: MemoryService processing successful - parallel M0/M1/M2 triggered")
                    else:
                        logger.error(f"BufferService: MemoryService processing failed: {result.get('message')}")
                        raise Exception(f"MemoryService processing failed: {result.get('message')}")
                else:
                    logger.warning("BufferService: No rounds or MemoryService not available")
            except Exception as e:
                logger.error(f"BufferService: MemoryService handler error: {e}")
                raise  # Re-raise to signal flush failure to FlushManager

        return memory_service_handler
    
    async def initialize(self, cfg: Optional[DictConfig] = None) -> bool:
        """Initialize the buffer service.

        Args:
            cfg: Configuration for the service (optional)

        Returns:
            True if initialization was successful, False otherwise
        """
        try:
            if not self.buffer_enabled:
                # Bypass mode: Only initialize MemoryService
                logger.info("BufferService: Bypass mode - skipping buffer component initialization")

                if self.memory_service and hasattr(self.memory_service, 'initialize'):
                    if cfg is not None:
                        await self.memory_service.initialize(cfg)
                    else:
                        await self.memory_service.initialize()

                logger.info("BufferService: Bypass mode initialization completed")
                return True
            else:
                # Normal mode: Full initialization
                # Initialize WriteBuffer (manages FlushManager, HybridBuffer, RoundBuffer internally)
                if hasattr(self.write_buffer, 'initialize'):
                    if not await self.write_buffer.initialize():
                        logger.error("BufferService: Failed to initialize WriteBuffer")
                        return False

                # Initialize QueryBuffer
                if hasattr(self.query_buffer, 'initialize'):
                    if not await self.query_buffer.initialize():
                        logger.error("BufferService: Failed to initialize QueryBuffer")
                        return False

                # Initialize SpeculativeBuffer
                if hasattr(self.speculative_buffer, 'initialize'):
                    if not await self.speculative_buffer.initialize():
                        logger.error("BufferService: Failed to initialize SpeculativeBuffer")
                        return False

                # Initialize memory service if needed
                if self.memory_service:
                    if hasattr(self.memory_service, 'initialize'):
                        if cfg is not None:
                            await self.memory_service.initialize(cfg)
                        else:
                            await self.memory_service.initialize()

                logger.info("BufferService: Full initialization completed successfully")
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
            # Shutdown WriteBuffer (manages RoundBuffer, HybridBuffer, FlushManager)
            if hasattr(self.write_buffer, 'shutdown'):
                await self.write_buffer.shutdown()

            # Clear QueryBuffer
            await self.query_buffer.clear()
            await self.query_buffer.clear_cache()

            # Clear SpeculativeBuffer
            if hasattr(self.speculative_buffer, 'clear'):
                await self.speculative_buffer.clear()

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
        """Add a batch of message lists (Service layer: orchestration and response formatting only).

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
            logger.debug(f"BufferService.add_batch: Processing {len(message_batch_list)} message lists")

            if not self.buffer_enabled:
                # Bypass mode: Direct MemoryService integration
                logger.info(f"BufferService: Bypass mode - sending {len(message_batch_list)} message lists directly to MemoryService")

                # Add minimal service-level metadata
                processed_batch = self._add_service_metadata(message_batch_list, session_id)

                # Direct call to MemoryService.add_batch()
                result = await self.memory_service.add_batch(processed_batch)

                # Update statistics
                self.total_batch_writes += 1
                self.total_items_added += len(message_batch_list)

                # Format response for bypass mode
                return self._format_bypass_response(result, len(message_batch_list))
            else:
                # Normal mode: Full buffer processing
                logger.debug(f"BufferService: Buffer mode - processing through WriteBuffer")

                # 1. Pre-processing: Service-level metadata only
                processed_batch = self._add_service_metadata(message_batch_list, session_id)

                # 2. Delegate to WriteBuffer for all concrete processing
                result = await self.write_buffer.add_batch(processed_batch, session_id)

                # 3. Service-level statistics and monitoring
                self._update_service_stats(result)

                # 4. Response formatting
                return self._format_write_response(result, len(message_batch_list))

        except Exception as e:
            logger.error(f"BufferService.add_batch: Error: {e}")
            return self._error_response(f"Batch operation failed: {str(e)}")

    def _add_service_metadata(self, message_batch_list: MessageBatchList, session_id: Optional[str]) -> MessageBatchList:
        """Add only service-level metadata (minimal processing).

        Args:
            message_batch_list: List of lists of messages
            session_id: Session ID for context

        Returns:
            Message batch list with service-level metadata added
        """
        if not self.user_id and not session_id:
            return message_batch_list

        # Only add user_id and session_id at service level
        for message_list in message_batch_list:
            for message in message_list:
                if isinstance(message, dict):
                    if 'metadata' not in message:
                        message['metadata'] = {}

                    # Add user_id if available and not present
                    if self.user_id and 'user_id' not in message['metadata']:
                        message['metadata']['user_id'] = self.user_id

                    # Add session_id if provided and not present
                    if session_id and 'session_id' not in message['metadata']:
                        message['metadata']['session_id'] = session_id

        return message_batch_list

    def _update_service_stats(self, result: Dict[str, Any]) -> None:
        """Update service-level statistics.

        Args:
            result: Result from WriteBuffer.add_batch()
        """
        # Update service-level statistics
        self.total_batch_writes += 1
        self.total_items_added += result.get('total_messages', 0)
        self.total_transfers += result.get('transfers_triggered', 0)

        logger.info(f"BufferService: Batch completed - messages: {result.get('total_messages', 0)}, "
                   f"transfers: {result.get('transfers_triggered', 0)}, "
                   f"strategy: {result.get('strategy_used', 'unknown')}")

    def _format_bypass_response(self, result: Dict[str, Any], batch_size: int) -> Dict[str, Any]:
        """Format response for bypass mode add_batch.

        Args:
            result: MemoryService result dictionary
            batch_size: Number of message lists in the batch

        Returns:
            Formatted service response
        """
        if result.get("status") == "success":
            # Extract message_ids from MemoryService result for API compatibility
            memory_data = result.get("data", {})
            message_ids = memory_data.get("message_ids", [])

            return self._success_response(
                data={
                    "mode": "bypass",
                    "batch_size": batch_size,
                    "message_ids": message_ids,  # Include message_ids at top level for API compatibility
                    "memory_service_result": memory_data,
                    "message": f"Processed {batch_size} message lists via MemoryService"
                },
                message=f"Successfully processed {batch_size} message lists in bypass mode"
            )
        else:
            return self._error_response(f"MemoryService processing failed: {result.get('message', 'Unknown error')}")

    def _format_bypass_query_response(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Format response for bypass mode query.

        Args:
            result: MemoryService query result dictionary

        Returns:
            Formatted service response
        """
        if result.get("status") == "success":
            return {
                "status": "success",
                "code": 200,
                "data": {
                    "mode": "bypass",
                    "results": result.get("data", {}).get("results", []),
                    "total": result.get("data", {}).get("total", 0),
                    "memory_service_result": result.get("data", {})
                },
                "message": "Query processed via MemoryService in bypass mode",
                "errors": None
            }
        else:
            return self._error_response(f"MemoryService query failed: {result.get('message', 'Unknown error')}")

    def _format_write_response(self, result: Dict[str, Any], batch_size: int) -> Dict[str, Any]:
        """Format WriteBuffer result into service response.

        Args:
            result: Result from WriteBuffer.add_batch()
            batch_size: Original batch size

        Returns:
            Formatted service response
        """
        # Extract message IDs if available (would need to be added to WriteBuffer result)
        message_ids = result.get('message_ids', [])

        return self._success_response(
            data={
                "message_ids": message_ids,
                "batch_size": batch_size,
                "total_messages": result.get('total_messages', 0),
                "transfers_triggered": result.get('transfers_triggered', 0),
                "strategy_used": result.get('strategy_used', 'unknown'),
                "processing_time": result.get('processing_time', 0)
            },
            message=f"Added {batch_size} message lists to Buffer system",
            transfer_triggered=result.get('transfers_triggered', 0) > 0,
            total_messages=result.get('total_messages', 0),
            buffer_status="success"
        )

    def _ensure_message_fields(self, message: Dict[str, Any]) -> None:
        """Ensure message has required fields (id, created_at, updated_at).

        For initial creation, created_at and updated_at use the same timestamp.
        When storing to database, updated_at will be refreshed.

        Args:
            message: Message dictionary to update
        """
        import uuid
        from datetime import datetime

        # Add ID if missing
        if 'id' not in message or not message['id']:
            message['id'] = str(uuid.uuid4())

        # Generate timestamp once for efficiency
        needs_timestamp = ('created_at' not in message or not message['created_at'] or
                          'updated_at' not in message or not message['updated_at'])

        if needs_timestamp:
            now = datetime.now().isoformat()

            # Add created_at if missing
            if 'created_at' not in message or not message['created_at']:
                message['created_at'] = now

            # Add updated_at if missing - use same timestamp as created_at for initial creation
            if 'updated_at' not in message or not message['updated_at']:
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
        """Query memory for relevant messages (Service layer: orchestration and response formatting only).

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
            if not self.buffer_enabled:
                # Bypass mode: Direct MemoryService query
                logger.info(f"BufferService: Bypass mode - querying MemoryService directly")

                result = await self.memory_service.query(
                    query=query,
                    top_k=top_k,
                    store_type=store_type,
                    session_id=session_id,
                    include_messages=include_messages,
                    include_knowledge=include_knowledge,
                    include_chunks=include_chunks
                )

                return self._format_bypass_query_response(result)
            else:
                # Normal mode: QueryBuffer processing
                logger.debug(f"BufferService: Buffer mode - processing through QueryBuffer")

                # Delegate all query logic to QueryBuffer (with internal reranking)
                results = await self.query_buffer.query(
                    query_text=query,
                    top_k=top_k,
                    sort_by=sort_by or "score",
                    order=order or "desc",
                    use_rerank=self.use_rerank
                )

                logger.info(f"BufferService.query: QueryBuffer returned {len(results) if results else 0} results")

                # Format response to match MemoryService format
                response = {
                    "status": "success",
                    "code": 200,
                    "data": {
                        "results": results,
                        "total": len(results)
                    },
                    "message": f"Retrieved {len(results)} results using Buffer",
                    "errors": None,
                }

                logger.info(f"BufferService.query: Returning response with {len(results)} results")
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
            if not self.buffer_enabled:
                # Bypass mode: Delegate to MemoryService
                return await self.memory_service.get_messages_by_session(
                    session_id=session_id,
                    limit=limit,
                    sort_by=sort_by,
                    order=order
                )
            else:
                # Buffer enabled mode
                if buffer_only:
                    # Only return RoundBuffer data
                    round_buffer = self.get_write_buffer().get_round_buffer()
                    return await round_buffer.get_messages_by_session(
                        session_id=session_id,
                        limit=limit,
                        sort_by=sort_by,
                        order=order
                    )
                else:
                    # Query all data sources: RoundBuffer + HybridBuffer + Database
                    return await self._get_from_all_sources(
                        session_id=session_id,
                        limit=limit,
                        sort_by=sort_by,
                        order=order
                    )

        except Exception as e:
            logger.error(f"BufferService.get_messages_by_session: Error: {e}")
            return []

    async def _get_from_all_sources(
        self,
        session_id: str,
        limit: int = 20,
        sort_by: str = "created_at",
        order: str = "desc"
    ) -> List[Dict[str, Any]]:
        """Get messages from all data sources: RoundBuffer + HybridBuffer + Database.

        This method implements the correct logic for buffer_only=false:
        1. Get messages from RoundBuffer (latest, in-memory data)
        2. Get messages from HybridBuffer (intermediate cached data)
        3. Get messages from Database (persisted data)
        4. Merge and deduplicate by message ID (priority: RoundBuffer > HybridBuffer > Database)
        5. Sort by specified criteria and apply limit

        Args:
            session_id: Session ID to query
            limit: Maximum number of messages to return
            sort_by: Field to sort by (created_at, updated_at, etc.)
            order: Sort order (asc or desc)

        Returns:
            List of message dictionaries, merged and sorted from all sources
        """
        try:
            logger.info(f"BufferService._get_from_all_sources: Querying all sources for session {session_id}")

            # Step 1: Get messages from RoundBuffer (highest priority)
            round_buffer_messages = []
            if self.write_buffer:
                round_buffer = self.write_buffer.get_round_buffer()
                if round_buffer:
                    round_buffer_messages = await round_buffer.get_messages_by_session(
                        session_id=session_id,
                        limit=limit * 3,  # Get more to account for deduplication
                        sort_by=sort_by,
                        order=order
                    )
                    logger.info(f"BufferService._get_from_all_sources: Found {len(round_buffer_messages)} messages in RoundBuffer")

            # Step 2: Get messages from HybridBuffer (medium priority)
            hybrid_buffer_messages = []
            if self.write_buffer:
                hybrid_buffer = self.write_buffer.get_hybrid_buffer()
                if hybrid_buffer:
                    # HybridBuffer doesn't have session-specific query, get all and filter
                    all_hybrid_messages = await hybrid_buffer.get_all_messages_for_read_api(
                        limit=limit * 3,
                        sort_by=sort_by,
                        order=order
                    )
                    # Filter by session_id
                    hybrid_buffer_messages = [
                        msg for msg in all_hybrid_messages
                        if self._message_belongs_to_session(msg, session_id)
                    ]
                    logger.info(f"BufferService._get_from_all_sources: Found {len(hybrid_buffer_messages)} messages in HybridBuffer")

            # Step 3: Get messages from Database (lowest priority)
            database_messages = []
            if self.memory_service:
                database_messages = await self.memory_service.get_messages_by_session(
                    session_id=session_id,
                    limit=limit * 3,  # Get more to account for deduplication
                    sort_by=sort_by,
                    order=order
                )
                logger.info(f"BufferService._get_from_all_sources: Found {len(database_messages)} messages in Database")

            # Step 4: Merge and deduplicate messages (priority order: RoundBuffer > HybridBuffer > Database)
            merged_messages = self._merge_messages_from_all_sources(
                round_buffer_messages, hybrid_buffer_messages, database_messages, sort_by, order
            )

            # Step 5: Apply final limit
            if limit > 0:
                merged_messages = merged_messages[:limit]

            logger.info(f"BufferService._get_from_all_sources: Merged {len(round_buffer_messages)}+{len(hybrid_buffer_messages)}+{len(database_messages)} = {len(merged_messages)} messages")
            return merged_messages

        except Exception as e:
            logger.error(f"BufferService._get_from_all_sources: Error: {e}")
            # Fallback to database-only query
            if self.memory_service:
                return await self.memory_service.get_messages_by_session(
                    session_id=session_id,
                    limit=limit,
                    sort_by=sort_by,
                    order=order
                )
            return []

    def _message_belongs_to_session(self, message: Dict[str, Any], session_id: str) -> bool:
        """Check if a message belongs to the specified session.

        Args:
            message: Message dictionary
            session_id: Target session ID

        Returns:
            True if message belongs to session
        """
        # Check session_id in metadata first
        metadata = message.get("metadata", {})
        message_session_id = metadata.get("session_id")

        # If not in metadata, check message directly
        if not message_session_id:
            message_session_id = message.get("session_id")

        return message_session_id == session_id

    def _merge_messages_from_all_sources(
        self,
        round_buffer_messages: List[Dict[str, Any]],
        hybrid_buffer_messages: List[Dict[str, Any]],
        database_messages: List[Dict[str, Any]],
        sort_by: str = "created_at",
        order: str = "desc"
    ) -> List[Dict[str, Any]]:
        """Merge and deduplicate messages from all three sources.

        Priority order: RoundBuffer > HybridBuffer > Database
        Messages with the same ID from higher priority sources override lower priority ones.

        Args:
            round_buffer_messages: Messages from RoundBuffer (highest priority)
            hybrid_buffer_messages: Messages from HybridBuffer (medium priority)
            database_messages: Messages from Database (lowest priority)
            sort_by: Field to sort by
            order: Sort order (asc or desc)

        Returns:
            Merged and deduplicated list of messages
        """
        # Create a dictionary to track messages by ID
        message_dict = {}

        # Add messages in priority order (lowest to highest priority)
        # Database messages (lowest priority)
        for msg in database_messages:
            msg_id = msg.get('id')
            if msg_id:
                message_dict[msg_id] = msg

        # HybridBuffer messages (medium priority) - overwrites database if same ID
        for msg in hybrid_buffer_messages:
            msg_id = msg.get('id')
            if msg_id:
                message_dict[msg_id] = msg

        # RoundBuffer messages (highest priority) - overwrites others if same ID
        for msg in round_buffer_messages:
            msg_id = msg.get('id')
            if msg_id:
                message_dict[msg_id] = msg

        # Convert back to list
        merged_messages = list(message_dict.values())

        # Sort the merged messages
        reverse_order = (order.lower() == "desc")

        try:
            if sort_by == "created_at":
                merged_messages.sort(
                    key=lambda x: x.get('created_at', ''),
                    reverse=reverse_order
                )
            elif sort_by == "updated_at":
                merged_messages.sort(
                    key=lambda x: x.get('updated_at', ''),
                    reverse=reverse_order
                )
            elif sort_by == "timestamp":  # Backward compatibility
                merged_messages.sort(
                    key=lambda x: x.get('created_at', ''),
                    reverse=reverse_order
                )
            else:
                # Default sorting by created_at if sort_by field is not recognized
                merged_messages.sort(
                    key=lambda x: x.get('created_at', ''),
                    reverse=reverse_order
                )
        except Exception as e:
            logger.warning(f"Error sorting messages by {sort_by}: {e}")
            # Return unsorted if sorting fails

        return merged_messages

    async def get_buffer_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the Buffer system.
        
        Returns:
            Dictionary with detailed buffer statistics
        """
        # Get stats through proper abstraction layers
        write_buffer_stats = self.write_buffer.get_stats()
        query_stats = self.query_buffer.get_stats()
        speculative_stats = self.speculative_buffer.get_stats()
        
        return {
            "version": "0.3.0",
            "architecture": "Refactored with proper abstraction layers",
            "total_items_added": self.total_items_added,
            "total_queries": self.total_queries,
            "total_transfers": self.total_transfers,
            "write_buffer": write_buffer_stats,
            "query_buffer": query_stats,
            "speculative_buffer": speculative_stats,
            "abstraction_layers": {
                "write_buffer": "Manages RoundBuffer + HybridBuffer + FlushManager",
                "query_buffer": "Unified query with sorting and caching",
                "speculative_buffer": "Predictive prefetching (placeholder)"
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
    
    # Component access methods for controlled operations
    def get_write_buffer(self) -> WriteBuffer:
        """Get WriteBuffer instance for write operations.

        Returns:
            WriteBuffer instance
        """
        return self.write_buffer

    def get_query_buffer(self) -> QueryBuffer:
        """Get QueryBuffer instance for query operations.

        Returns:
            QueryBuffer instance
        """
        return self.query_buffer

    def get_speculative_buffer(self) -> SpeculativeBuffer:
        """Get SpeculativeBuffer instance for predictive operations.

        Returns:
            SpeculativeBuffer instance
        """
        return self.speculative_buffer

    # Legacy component access (for backward compatibility)
    def get_round_buffer(self):
        """Get RoundBuffer instance (via WriteBuffer)."""
        return self.write_buffer.get_round_buffer()

    def get_hybrid_buffer(self):
        """Get HybridBuffer instance (via WriteBuffer)."""
        return self.write_buffer.get_hybrid_buffer()

    def get_flush_manager(self):
        """Get FlushManager instance (via WriteBuffer)."""
        return self.write_buffer.get_flush_manager()

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
