"""Memory service for MemFuse server."""

import asyncio
from loguru import logger
from typing import Dict, List, Any, Optional, Union

from ..models import Item, Query, Node, QueryResult, StoreType
from ..store.factory import StoreFactory
from ..utils.config import config_manager
from ..utils.path_manager import PathManager
from ..rag.rerank import MiniLMReranker
from ..interfaces import MessageInterface, MessageBatchList, MemoryLayer
from ..rag.chunk import ChunkStrategy, MessageChunkStrategy
from ..rag.chunk.base import ChunkData
from ..hierarchy.memory_layer_impl import MemoryLayerImpl
from ..utils.config import ConfigManager



class MemoryService(MessageInterface):
    """Memory service for managing user-agent interactions."""

    def __init__(
        self,
        cfg=None,
        user: str = "user_default",
        agent: Optional[str] = None,
        session: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        """Initialize the Memory service.

        Args:
            cfg: Configuration object (optional)
            user: User name (default: "user_default")
            agent: Agent name (optional)
            session: Session name (optional)
            session_id: Session ID (optional, takes precedence if provided)
        """
        # Store parameters for async initialization
        self.user = user
        self.agent = agent or "agent_default"
        self.session = session
        self.session_id = session_id

        # Get configuration
        if cfg is not None:
            # If cfg is provided, use it
            if hasattr(cfg, 'to_container'):
                # If it's a DictConfig, convert it to a dict
                self.config = cfg.to_container()
            else:
                # Otherwise, use it as is
                self.config = cfg
        else:
            # Otherwise, use the default configuration
            self.config = config_manager.get_config()

        # These will be set during async initialization
        self.db = None
        self.user_id = None
        self.agent_id = None
        self._async_initialized = False

        # Session handling will be done in async initialization
        self._session_id = None

        # User directory will be set during async initialization
        self.user_dir = None

        # Initialize store and retrieval (will be set in initialize method)
        self.vector_store = None
        self.graph_store = None
        self.keyword_store = None
        self.multi_path_retrieval = None
        self.reranker = None

        # Initialize chunk strategy (will be configured in initialize method)
        self.chunk_strategy = MessageChunkStrategy()

        # Initialize Memory Layer for M0/M1/M2 parallel processing
        self.memory_layer: Optional[MemoryLayer] = None
        self.use_parallel_layers = self._should_use_parallel_layers()

        # Log initialization
        logger.info(f"MemoryService: Initialized for user: {user}")
        logger.info(f"MemoryService: Parallel layers enabled: {self.use_parallel_layers}")

    def _should_use_parallel_layers(self) -> bool:
        """Determine if parallel memory layers should be used based on configuration."""
        try:
            # Check for memory_service config at top level first (new structure)
            memory_service_config = self.config.get("memory_service", {})
            if memory_service_config:
                return memory_service_config.get("parallel_enabled", False)

            # Fallback to old structure for backward compatibility
            memory_config = self.config.get("memory", {})
            memory_service_config = memory_config.get("memory_service", {})
            return memory_service_config.get("parallel_enabled", False)
        except Exception as e:
            logger.warning(f"MemoryService: Error checking parallel layers config: {e}")
            return False

    async def _async_init(self):
        """Perform async initialization of database-related components."""
        if self._async_initialized:
            return

        logger.info("MemoryService._async_init: Starting async initialization")
        logger.info(f"MemoryService._async_init: config type={type(self.config)}, config={self.config}")

        # Get database instance
        from .database_service import DatabaseService
        logger.info("MemoryService._async_init: Getting database instance")
        self.db = await DatabaseService.get_instance()
        logger.info("MemoryService._async_init: Database instance obtained")

        # Ensure user exists and get user_id
        self.user_id = await self.db.get_or_create_user_by_name(self.user)

        # Ensure agent exists and get agent_id
        self.agent_id = await self.db.get_or_create_agent_by_name(self.agent)

        # Handle session creation/retrieval
        if self.session_id is not None:
            # If session_id is provided, use it directly
            session_data = await self.db.get_session(self.session_id)
            if not session_data:
                # Create a new session with the provided ID
                session_id = await self.db.create_session(
                    self.user_id, self.agent_id, self.session_id)
                self._session_id = session_id
                self.session = session_id  # Use session_id as name
            else:
                self._session_id = self.session_id
                self.session = session_data["name"]
        elif self.session is not None:
            # If session name is provided, check if it already exists for this user
            session_data = await self.db.get_session_by_name(self.session, user_id=self.user_id)
            if session_data is not None:
                # Session with this name already exists for this user - raise error
                raise ValueError(
                    f"Session with name '{self.session}' already exists for user '{self.user}'. "
                    f"Session names must be unique within each user's scope."
                )
            else:
                # Session not found, create a new one
                session_id = await self.db.create_session_with_name(
                    self.user_id, self.agent_id, self.session)
                self._session_id = session_id
        else:
            # For cross-session queries, we don't need a specific session
            self._session_id = None

        # Set up user directory path now that we have user_id
        data_dir = self.config.get("data_dir", "data")
        self.user_dir = str(PathManager.get_user_dir(data_dir, self.user_id))

        self._async_initialized = True

    async def initialize(self):
        """Initialize the store and retrieval components asynchronously."""
        # First, perform async database initialization
        if not self._async_initialized:
            await self._async_init()

        # Make sure user directory exists
        PathManager.ensure_directory(self.user_dir)

        # Try to get the pre-loaded model from the server
        existing_model = None
        try:
            # Import the server module directly
            from memfuse_core.server import _model_manager
            if _model_manager is not None:
                existing_model = _model_manager.get_embedding_model()
                if existing_model is not None:
                    logger.info("Using pre-loaded embedding model from server")
        except (ImportError, AttributeError) as e:
            logger.debug(f"Could not get pre-loaded model: {e}")
            pass

        # Initialize store components with the pre-loaded model
        try:
            logger.info(f"Creating vector store with data_dir={self.user_dir}, existing_model={existing_model is not None}")
            self.vector_store = await StoreFactory.create_vector_store(
                data_dir=self.user_dir,
                existing_model=existing_model
            )
            logger.info("Vector store created successfully")
        except Exception as e:
            logger.error(f"Failed to create vector store: {e}")
            import traceback
            logger.error(f"Vector store creation traceback: {traceback.format_exc()}")
            self.vector_store = None

        # Only create graph store if enabled in configuration
        if self.config.get("store", {}).get("multi_path", {}).get("use_graph", False):
            try:
                self.graph_store = await StoreFactory.create_graph_store(
                    data_dir=self.user_dir,
                    existing_model=existing_model
                )
                logger.info("Graph store created successfully")
            except Exception as e:
                logger.error(f"Failed to create graph store: {e}")
                self.graph_store = None
        else:
            logger.info("Graph store disabled in configuration")
            self.graph_store = None

        try:
            self.keyword_store = await StoreFactory.create_keyword_store(data_dir=self.user_dir)
        except Exception as e:
            logger.error(f"Failed to create keyword store: {e}")
            self.keyword_store = None

        # Initialize multi-path retrieval
        cache_size = self.config.get("store", {}).get("cache_size", 100)
        try:
            self.multi_path_retrieval = await StoreFactory.create_multi_path_retrieval(
                data_dir=self.user_dir,
                vector_store=self.vector_store,
                graph_store=self.graph_store,
                keyword_store=self.keyword_store,
                cache_size=cache_size
            )
        except Exception as e:
            logger.error(f"Failed to create multi-path retrieval: {e}")
            self.multi_path_retrieval = None

        # Initialize rerank manager
        rerank_strategy = self.config.get(
            "retrieval", {}).get("rerank_strategy", "rrf")

        # Get rerank settings from config
        use_rerank = self.config.get("retrieval", {}).get("use_rerank", False)
        cross_encoder_model = self.config.get("retrieval", {}).get(
            "rerank_model", "cross-encoder/ms-marco-MiniLM-L6-v2")
        cross_encoder_batch_size = self.config.get(
            "retrieval", {}).get("cross_encoder_batch_size", 16)
        cross_encoder_max_length = self.config.get(
            "retrieval", {}).get("cross_encoder_max_length", 256)
        normalize_scores = self.config.get(
            "retrieval", {}).get("normalize_scores", True)
        rrf_k = self.config.get("retrieval", {}).get("rrf_k", 60)

        # Get fusion weights from config
        fusion_weights = self.config.get(
            "retrieval", {}).get("fusion_weights", None)

        logger.info(
            f"Initializing MiniLMReranker with strategy: {rerank_strategy}")
        # Always log rerank status, whether enabled or disabled
        logger.info(f"Reranking is {'enabled' if use_rerank else 'disabled'}")
        if use_rerank:
            logger.info(f"Using rerank model: {cross_encoder_model}")
        if normalize_scores:
            logger.info("Score normalization enabled for reranking")

        # Only initialize reranker if reranking is enabled
        if use_rerank:
            # Check for global reranker instance first
            from .service_factory import ServiceFactory
            global_reranker_instance = ServiceFactory.get_global_reranker_instance()

            if global_reranker_instance is not None:
                logger.info("Using global pre-loaded reranker instance")
                self.reranker = global_reranker_instance
                # Update configuration if needed
                if hasattr(self.reranker, 'rerank_strategy'):
                    self.reranker.rerank_strategy = rerank_strategy
                if hasattr(self.reranker, 'use_rerank'):
                    self.reranker.use_rerank = use_rerank
                if hasattr(self.reranker, 'normalize_scores'):
                    self.reranker.normalize_scores = normalize_scores
                if hasattr(self.reranker, 'rrf_k'):
                    self.reranker.rrf_k = rrf_k
                if hasattr(self.reranker, 'fusion_weights') and fusion_weights:
                    self.reranker.fusion_weights = fusion_weights
            elif hasattr(self, 'reranker') and self.reranker is not None:
                logger.info("Using existing reranker instance")
                # Update configuration if needed
                if hasattr(self.reranker, 'rerank_strategy'):
                    self.reranker.rerank_strategy = rerank_strategy
                if hasattr(self.reranker, 'use_rerank'):
                    self.reranker.use_rerank = use_rerank
                if hasattr(self.reranker, 'normalize_scores'):
                    self.reranker.normalize_scores = normalize_scores
                if hasattr(self.reranker, 'rrf_k'):
                    self.reranker.rrf_k = rrf_k
                if hasattr(self.reranker, 'fusion_weights') and fusion_weights:
                    self.reranker.fusion_weights = fusion_weights
            else:
                # Create new reranker only if we don't have one and reranking is enabled
                logger.info("Creating new MiniLMReranker instance")
                self.reranker = MiniLMReranker(
                    rerank_strategy=rerank_strategy,
                    thread_safe=True,
                    use_rerank=use_rerank,
                    cross_encoder_model=cross_encoder_model,
                    cross_encoder_batch_size=cross_encoder_batch_size,
                    cross_encoder_max_length=cross_encoder_max_length,
                    normalize_scores=normalize_scores,
                    rrf_k=rrf_k,
                    fusion_weights=fusion_weights
                )
                await self.reranker.initialize()

                # Store the newly created reranker as global instance for future use
                ServiceFactory.set_global_models(reranker_instance=self.reranker)
        else:
            # Reranking is disabled, don't create reranker instance
            logger.info("Reranking disabled, skipping reranker initialization")
            self.reranker = None

        # Configure chunk strategy based on configuration
        await self._configure_chunk_strategy()

        # Initialize Memory Layer if enabled
        if self.use_parallel_layers:
            await self._initialize_memory_layer()

        return self

    async def _initialize_memory_layer(self):
        """Initialize the memory layer for M0/M1/M2 parallel processing."""
        try:
            logger.info("MemoryService: Initializing Memory Layer...")

            # Create config manager for hierarchy components
            hierarchy_config_manager = ConfigManager()

            # Create memory layer implementation
            self.memory_layer = MemoryLayerImpl(
                user_id=str(self.user_id),
                config_manager=hierarchy_config_manager
            )

            # Initialize the memory layer
            memory_config = self.config.get("memory", {})
            if await self.memory_layer.initialize(memory_config):
                logger.info("MemoryService: Memory Layer initialized successfully")
            else:
                logger.error("MemoryService: Failed to initialize Memory Layer")
                self.memory_layer = None
                self.use_parallel_layers = False

        except Exception as e:
            logger.error(f"MemoryService: Error initializing Memory Layer: {e}")
            self.memory_layer = None
            self.use_parallel_layers = False

    async def _configure_chunk_strategy(self):
        """Configure chunk strategy based on configuration and inject dependencies."""
        try:
            # Get buffer configuration to determine chunk strategy
            buffer_config = self.config.get("buffer", {})
            hybrid_config = buffer_config.get("hybrid_buffer", {})
            chunk_strategy_name = hybrid_config.get("chunk_strategy", "message")

            logger.info(f"Configuring chunk strategy: {chunk_strategy_name}")

            if chunk_strategy_name == "contextual":
                # Use advanced ContextualChunkStrategy
                from ..rag.chunk.contextual import ContextualChunkStrategy

                # Get chunking configuration
                chunking_config = buffer_config.get("chunking", {})
                strategy_config = chunking_config.get(chunk_strategy_name, {})

                # Create strategy with configuration and inject dependencies
                self.chunk_strategy = ContextualChunkStrategy(
                    max_words_per_group=strategy_config.get("max_words_per_group", 800),
                    max_words_per_chunk=strategy_config.get("max_words_per_chunk", 800),
                    role_format=strategy_config.get("role_format", "[{role}]"),
                    chunk_separator=strategy_config.get("chunk_separator", "\n\n"),
                    enable_contextual=strategy_config.get("enable_contextual", True),
                    context_window_size=strategy_config.get("context_window_size", 2),
                    gpt_model=strategy_config.get("gpt_model", "gpt-4o-mini"),
                    vector_store=self.vector_store,  # Inject vector store for context retrieval
                    llm_provider=None  # TODO: Inject LLM provider when available
                )

                logger.info(f"Configured {chunk_strategy_name} strategy with contextual enhancement: "
                           f"enable_contextual={strategy_config.get('enable_contextual', True)}")
            else:
                # Keep existing MessageChunkStrategy for "message" strategy
                logger.info("Using basic MessageChunkStrategy")

        except Exception as e:
            logger.error(f"Failed to configure chunk strategy: {e}")
            # Keep the default strategy on error





    async def _process_with_parallel_layers(self, message_batch_list: MessageBatchList, **kwargs) -> Dict[str, Any]:
        """Process message batch using Memory Layer (M0/M1/M2 parallel processing)."""
        try:
            if not self.memory_layer:
                logger.error("MemoryService: Memory Layer not initialized")
                return self._error_response("Memory Layer not available")

            # Prepare session and round information
            session_id, round_id = await self._prepare_session_and_round(message_batch_list)
            logger.info(f"MemoryService._process_with_parallel_layers: session_id={session_id}, round_id={round_id}")

            # Store original messages to database first
            message_ids = await self._store_original_messages_with_round(
                message_batch_list, session_id, round_id
            )
            logger.info(f"MemoryService._process_with_parallel_layers: Stored {len(message_ids)} messages")

            # Process through Memory Layer (M0/M1/M2 parallel processing)
            metadata = {
                "session_id": session_id,
                "round_id": round_id,
                "user_id": self.user_id,
                "agent_id": self.agent_id,
                "message_ids": message_ids,
                **kwargs
            }

            write_result = await self.memory_layer.write_parallel(
                message_batch_list=message_batch_list,
                session_id=session_id,
                metadata=metadata
            )

            if write_result.success:
                logger.info("MemoryService._process_with_parallel_layers: Successfully processed through memory layer")

                # Extract processing statistics
                layer_results = write_result.layer_results
                total_processed = sum(
                    result.get("processed_count", 0)
                    for result in layer_results.values()
                    if isinstance(result, dict)
                )

                return self._success_response(
                    message_ids,
                    f"Processed {len(message_batch_list)} message lists through M0/M1/M2 parallel processing",
                    chunk_count=total_processed,
                    layer_results=layer_results,
                    processing_method="parallel_layers"
                )
            else:
                logger.error(f"MemoryService._process_with_parallel_layers: Processing failed: {write_result.message}")
                return self._error_response(f"Memory layer processing failed: {write_result.message}")

        except Exception as e:
            logger.error(f"MemoryService._process_with_parallel_layers: Error: {e}")
            return self._error_response(f"Memory layer processing error: {str(e)}")

    async def _process_with_traditional_method(self, message_batch_list: MessageBatchList, **kwargs) -> Dict[str, Any]:
        """Process message batch using traditional sequential method."""
        # This contains the original add_batch logic
        session_id, round_id = await self._prepare_session_and_round(message_batch_list)
        logger.info(f"MemoryService.add_batch: Prepared session_id={session_id}, round_id={round_id}")

        # Sequential processing to reduce connection pool pressure during flush operations
        # Store original messages to database first
        message_ids = await self._store_original_messages_with_round(message_batch_list, session_id, round_id)

        # Then create and store chunks
        chunks = await self.chunk_strategy.create_chunks(message_batch_list)
        logger.info(f"MemoryService.add_batch: Created {len(chunks)} chunks")
        await self._store_chunks_enhanced(chunks, session_id, round_id)

        logger.info(f"MemoryService.add_batch: Successfully processed {len(chunks)} chunks and {len(message_ids)} messages")



        return self._success_response(
            message_ids,
            f"Processed {len(message_batch_list)} message lists into {len(chunks)} chunks",
            chunk_count=len(chunks)
        )



    def _get_retrieval_method(
        self,
        store_type: Optional[StoreType],
        explicit_store_type: Optional[StoreType] = None,
        original_retrieval_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Get the retrieval method information based on the store type.

        Args:
            store_type: The store type from the result (vector, graph, keyword, or None)
            explicit_store_type: The store type explicitly requested by the user (if any)
            original_retrieval_info: Original retrieval info from the result metadata

        Returns:
            Dictionary with retrieval method information
        """
        # Get configuration
        cfg = config_manager.get_config()

        # Default retrieval info
        retrieval_info = {
            "method": "unknown",
            "fusion_strategy": None
        }

        # If we have original retrieval info, use it as a base
        if original_retrieval_info:
            # Copy original retrieval info
            for key, value in original_retrieval_info.items():
                retrieval_info[key] = value

        # If the user explicitly requested a specific store type, use that
        if explicit_store_type is not None:
            if explicit_store_type == StoreType.VECTOR:
                retrieval_info["method"] = "vector"
            elif explicit_store_type == StoreType.GRAPH:
                retrieval_info["method"] = "graph"
            elif explicit_store_type == StoreType.KEYWORD:
                retrieval_info["method"] = "keyword"
        # Otherwise, use the store type from the result
        elif store_type == StoreType.VECTOR:
            retrieval_info["method"] = "vector"
        elif store_type == StoreType.GRAPH:
            retrieval_info["method"] = "graph"
        elif store_type == StoreType.KEYWORD:
            retrieval_info["method"] = "keyword"
        else:
            # When store_type is None, it means multi-path retrieval was used
            retrieval_info["method"] = "multi_path"
            # Add fusion strategy from config if not already present
            if "fusion_strategy" not in retrieval_info:
                if "store" in cfg and "multi_path" in cfg["store"] and "fusion_strategy" in cfg["store"]["multi_path"]:
                    retrieval_info["fusion_strategy"] = cfg["store"]["multi_path"]["fusion_strategy"]

        return retrieval_info

    async def add_batch(self, message_batch_list: MessageBatchList, **kwargs) -> Dict[str, Any]:
        """Add a batch of message lists.

        This is the core processing method that handles MessageBatchList.
        Applies chunking strategy and stores chunks to various stores.
        Uses async parallel processing to reduce latency.

        Args:
            message_batch_list: List of lists of messages (MessageBatchList)
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary with status, data, and message information
        """
        try:
            if not message_batch_list:
                return self._success_response([], "No message lists to process")

            logger.info(f"MemoryService.add_batch: Processing {len(message_batch_list)} message lists")

            # Choose processing method based on configuration
            if self.use_parallel_layers and self.memory_layer:
                logger.info("MemoryService.add_batch: Using Memory Layer (M0/M1/M2 parallel processing)")
                return await self._process_with_parallel_layers(message_batch_list, **kwargs)
            else:
                logger.info("MemoryService.add_batch: Using traditional method (M0-only processing)")
                return await self._process_with_traditional_method(message_batch_list, **kwargs)

        except Exception as e:
            logger.error(f"MemoryService.add_batch: Error processing message batch: {e}")
            return self._error_response(f"Error processing message batch: {str(e)}")

    # Chunks Query Methods
    async def get_chunks_by_session(self, session_id: str, store_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all chunks for a specific session from all or specific stores.

        Args:
            session_id: Session ID to filter chunks
            store_type: Optional store type filter ('vector', 'keyword', 'graph', 'hybrid')

        Returns:
            List of chunk dictionaries with store_type metadata
        """
        all_chunks = []

        try:
            # Determine which stores to query based on store_type
            if store_type == "hybrid":
                # Hybrid: prioritize vector > keyword > graph
                stores_to_query = [
                    ("vector", self.vector_store),
                    ("keyword", self.keyword_store),
                    ("graph", self.graph_store)
                ]
            elif store_type is None:
                # All stores
                stores_to_query = [
                    ("vector", self.vector_store),
                    ("keyword", self.keyword_store),
                    ("graph", self.graph_store)
                ]
            else:
                # Specific store
                store_map = {
                    "vector": self.vector_store,
                    "keyword": self.keyword_store,
                    "graph": self.graph_store
                }
                stores_to_query = [(store_type, store_map.get(store_type))]

            # Query stores in order (important for hybrid)
            for store_name, store in stores_to_query:
                if store and hasattr(store, 'get_chunks_by_session'):
                    store_chunks = await store.get_chunks_by_session(session_id)
                    for chunk in store_chunks:
                        chunk.metadata["store_type"] = store_name
                        all_chunks.append({
                            "chunk_id": chunk.chunk_id,
                            "content": chunk.content,
                            "metadata": chunk.metadata
                        })

            logger.info(f"Retrieved {len(all_chunks)} chunks for session {session_id} (store_type: {store_type or 'all'})")
            return all_chunks

        except Exception as e:
            logger.error(f"Error getting chunks by session {session_id}: {e}")
            return []

    async def get_chunks_by_round(self, round_id: str, store_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all chunks for a specific round from all or specific stores.

        Args:
            round_id: Round ID to filter chunks
            store_type: Optional store type filter ('vector', 'keyword', 'graph')

        Returns:
            List of chunk dictionaries with store_type metadata
        """
        all_chunks = []

        try:
            if store_type is None or store_type == "vector":
                if self.vector_store and hasattr(self.vector_store, 'get_chunks_by_round'):
                    vector_chunks = await self.vector_store.get_chunks_by_round(round_id)
                    for chunk in vector_chunks:
                        chunk.metadata["store_type"] = "vector"
                        all_chunks.append({
                            "chunk_id": chunk.chunk_id,
                            "content": chunk.content,
                            "metadata": chunk.metadata
                        })

            if store_type is None or store_type == "keyword":
                if self.keyword_store and hasattr(self.keyword_store, 'get_chunks_by_round'):
                    keyword_chunks = await self.keyword_store.get_chunks_by_round(round_id)
                    for chunk in keyword_chunks:
                        chunk.metadata["store_type"] = "keyword"
                        all_chunks.append({
                            "chunk_id": chunk.chunk_id,
                            "content": chunk.content,
                            "metadata": chunk.metadata
                        })

            if store_type is None or store_type == "graph":
                if self.graph_store and hasattr(self.graph_store, 'get_chunks_by_round'):
                    graph_chunks = await self.graph_store.get_chunks_by_round(round_id)
                    for chunk in graph_chunks:
                        chunk.metadata["store_type"] = "graph"
                        all_chunks.append({
                            "chunk_id": chunk.chunk_id,
                            "content": chunk.content,
                            "metadata": chunk.metadata
                        })

            logger.info(f"Retrieved {len(all_chunks)} chunks for round {round_id}")
            return all_chunks

        except Exception as e:
            logger.error(f"Error getting chunks by round {round_id}: {e}")
            return []

    async def get_chunks_stats(self, store_type: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics about chunks from all or specific stores.

        Args:
            store_type: Optional store type filter ('vector', 'keyword', 'graph')

        Returns:
            Dictionary containing aggregated statistics
        """
        all_stats = {}

        try:
            if store_type is None or store_type == "vector":
                if self.vector_store and hasattr(self.vector_store, 'get_chunks_stats'):
                    vector_stats = await self.vector_store.get_chunks_stats()
                    all_stats["vector"] = vector_stats

            if store_type is None or store_type == "keyword":
                if self.keyword_store and hasattr(self.keyword_store, 'get_chunks_stats'):
                    keyword_stats = await self.keyword_store.get_chunks_stats()
                    all_stats["keyword"] = keyword_stats

            if store_type is None or store_type == "graph":
                if self.graph_store and hasattr(self.graph_store, 'get_chunks_stats'):
                    graph_stats = await self.graph_store.get_chunks_stats()
                    all_stats["graph"] = graph_stats

            # Aggregate total chunks
            total_chunks = sum(stats.get("total_chunks", 0) for stats in all_stats.values())

            return {
                "total_chunks": total_chunks,
                "by_store": all_stats,
                "store_type_filter": store_type or "all"
            }

        except Exception as e:
            logger.error(f"Error getting chunks stats: {e}")
            return {
                "total_chunks": 0,
                "by_store": {},
                "store_type_filter": store_type or "all",
                "error": str(e)
            }

    async def _prepare_session_and_round(self, message_batch_list: MessageBatchList) -> tuple[str, str]:
        """Fast preparation: extract session_id and create round_id without DB writes.

        Args:
            message_batch_list: List of lists of messages

        Returns:
            Tuple of (session_id, round_id)

        Raises:
            ValueError: If no session_id found in messages
        """
        # Flatten message_batch_list to get all messages
        all_messages = []
        for message_list in message_batch_list:
            all_messages.extend(message_list)

        if not all_messages:
            raise ValueError("No messages found in message_batch_list")

        # Get session_id from first message metadata
        session_id = None
        first_message = all_messages[0]
        if isinstance(first_message, dict) and 'metadata' in first_message:
            metadata = first_message['metadata']
            if isinstance(metadata, dict) and 'session_id' in metadata:
                session_id = metadata['session_id']



        if session_id is None:
            raise ValueError("No session_id found in messages")

        # Create round_id immediately (just a UUID, no DB operation)
        import uuid
        round_id = str(uuid.uuid4())

        return session_id, round_id

    async def _store_original_messages_with_round(self, message_batch_list: MessageBatchList,
                                                 session_id: str, round_id: str) -> List[str]:
        """Store original messages to database with pre-created round_id.

        Args:
            message_batch_list: List of lists of messages
            session_id: Pre-extracted session ID
            round_id: Pre-created round ID

        Returns:
            List of message IDs
        """
        message_ids = []

        # Flatten message_batch_list to get all messages
        all_messages = []
        for message_list in message_batch_list:
            all_messages.extend(message_list)

        if not all_messages:
            return message_ids

        # Add metadata to all messages
        for message in all_messages:
            if 'metadata' not in message:
                message['metadata'] = {}
            if 'session_id' not in message['metadata']:
                message['metadata']['session_id'] = session_id
            if 'user_id' not in message['metadata']:
                message['metadata']['user_id'] = self.user_id
            if 'agent_id' not in message['metadata']:
                message['metadata']['agent_id'] = self.agent_id

        # Create the round in database with pre-created round_id
        await self.db.create_round(session_id, round_id)

        # Store messages to database using helper method
        message_ids.extend(await self._store_messages_to_db(all_messages, round_id))

        return message_ids

    async def _store_messages_to_db(self, messages: List[Dict], round_id: str) -> List[str]:
        """Helper method to store messages to database with proper timestamp handling.

        Args:
            messages: List of message dictionaries
            round_id: Round ID for the messages

        Returns:
            List of message IDs
        """
        message_ids = []

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            # Extract existing id and created_at from message if available
            existing_id = message.get("id")
            existing_created_at = message.get("created_at")

            # Check if message already exists to avoid UNIQUE constraint failures
            if existing_id:
                existing_message = await self.db.get_message(existing_id)
                if existing_message:
                    logger.debug(f"Message {existing_id} already exists in database, skipping insert")
                    message_ids.append(existing_id)
                    continue

            # Always update the updated_at timestamp when storing to database
            from datetime import datetime
            updated_at = datetime.now().isoformat()

            # Convert created_at to proper format if it's a numeric timestamp
            if existing_created_at is not None:
                if isinstance(existing_created_at, (int, float)):
                    # Convert numeric timestamp to ISO format
                    existing_created_at = datetime.fromtimestamp(existing_created_at).isoformat()
                elif isinstance(existing_created_at, str):
                    # Try to parse as numeric timestamp if it's a string of digits
                    try:
                        timestamp = float(existing_created_at)
                        existing_created_at = datetime.fromtimestamp(timestamp).isoformat()
                    except (ValueError, TypeError):
                        # If it's not a numeric string, assume it's already in correct format
                        pass

            try:
                message_id = await self.db.add_message(
                    round_id=round_id,
                    role=role,
                    content=content,
                    message_id=existing_id,
                    created_at=existing_created_at,
                    updated_at=updated_at
                )
                message_ids.append(message_id)
            except Exception as e:
                # Handle UNIQUE constraint failures gracefully
                error_msg = str(e).lower()
                if 'unique' in error_msg and existing_id:
                    logger.warning(f"Message {existing_id} already exists (race condition), using existing ID")
                    message_ids.append(existing_id)
                else:
                    logger.error(f"Failed to store message: {e}")
                    raise

        return message_ids

    async def _store_original_messages(self, message_batch_list: MessageBatchList) -> List[str]:
        """Store original messages to database and return message IDs.

        Args:
            message_batch_list: List of lists of messages

        Returns:
            List of message IDs
        """
        message_ids = []

        # Flatten message_batch_list to get all messages
        all_messages = []
        for message_list in message_batch_list:
            all_messages.extend(message_list)

        if not all_messages:
            return message_ids

        # Get session_id from first message metadata
        session_id = None
        first_message = all_messages[0]
        if isinstance(first_message, dict) and 'metadata' in first_message:
            metadata = first_message['metadata']
            if isinstance(metadata, dict) and 'session_id' in metadata:
                session_id = metadata['session_id']



        if session_id is None:
            logger.error("MemoryService._store_original_messages: No session_id found")
            return message_ids

        # Add metadata to all messages
        for message in all_messages:
            if 'metadata' not in message:
                message['metadata'] = {}
            if 'session_id' not in message['metadata']:
                message['metadata']['session_id'] = session_id
            if 'user_id' not in message['metadata']:
                message['metadata']['user_id'] = self.user_id
            if 'agent_id' not in message['metadata']:
                message['metadata']['agent_id'] = self.agent_id

        # Create a new round for these messages
        round_id = await self.db.create_round(session_id)

        # Store messages to database using helper method
        message_ids.extend(await self._store_messages_to_db(all_messages, round_id))

        return message_ids

    async def _store_chunks_enhanced(self, chunks, session_id: str, round_id: str) -> None:
        """Store chunks to various stores with enhanced metadata including session_id and round_id.

        Args:
            chunks: List of ChunkData objects
            session_id: Session ID for the chunks
            round_id: Round ID for the chunks
        """
        if not chunks:
            return

        # Add comprehensive metadata to chunk metadata for all stores
        from datetime import datetime
        user_chunks = []
        for chunk in chunks:
            user_chunk = ChunkData(
                content=chunk.content,
                chunk_id=chunk.chunk_id,
                metadata={
                    **chunk.metadata,  # Preserve original strategy-specific metadata
                    "type": "chunk",
                    "user_id": self.user_id,
                    "session_id": session_id,      # New: session association
                    "round_id": round_id,          # New: round association
                    "agent_id": self.agent_id,    # New: agent association
                    "created_at": datetime.now().isoformat(),  # New: timestamp
                }
            )
            user_chunks.append(user_chunk)

        # Store chunks to vector store
        if self.vector_store:
            try:
                await asyncio.wait_for(self.vector_store.add(user_chunks), timeout=30.0)
                logger.debug(f"Successfully stored {len(user_chunks)} chunks to vector store")
            except Exception as e:
                logger.error(f"Error adding chunks to vector store: {e}")

        # Store chunks to keyword store
        if self.keyword_store:
            try:
                await asyncio.wait_for(self.keyword_store.add(user_chunks), timeout=30.0)
                logger.debug(f"Successfully stored {len(user_chunks)} chunks to keyword store")
            except Exception as e:
                logger.error(f"Error adding chunks to keyword store: {e}")

        # Store chunks to graph store
        if self.graph_store:
            try:
                await asyncio.wait_for(self.graph_store.add(user_chunks), timeout=30.0)
                logger.debug(f"Successfully stored {len(user_chunks)} chunks to graph store")
            except Exception as e:
                logger.error(f"Error adding chunks to graph store: {e}")



    async def read(self, message_ids: List[str]) -> Dict[str, Any]:
        """Read messages from memory.

        Args:
            message_ids: List of message IDs

        Returns:
            Dictionary with status, code, and messages
        """
        messages = []
        not_found_ids = []

        for message_id in message_ids:
            message = await self.db.get_message(message_id)
            if message:
                messages.append({
                    "id": message["id"],
                    "role": message["role"],
                    "content": message["content"],
                    "created_at": message["created_at"],
                    "updated_at": message["updated_at"],
                })
            else:
                not_found_ids.append(message_id)

        # If some messages were not found, return error
        if not_found_ids:
            return {
                "status": "error",
                "code": 404,
                "data": {"not_found_ids": not_found_ids},
                "message": f"Some message IDs were not found",
                "errors": [
                    {"field": "message_ids",
                     "message": f"Message IDs not found: {', '.join(not_found_ids)}"}
                ],
            }

        return {
            "status": "success",
            "code": 200,
            "data": {"messages": messages},
            "message": f"Read {len(messages)} messages",
            "errors": None,
        }

    async def get_messages_by_session(
        self,
        session_id: str,
        limit: Optional[int] = None,
        sort_by: str = 'timestamp',
        order: str = 'desc',
        buffer_only: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """Get messages for a session with optional limit and sorting.

        Args:
            session_id: Session ID
            limit: Maximum number of messages to return (optional)
            sort_by: Field to sort by, either 'timestamp' or 'id' (default: 'timestamp')
            order: Sort order, either 'asc' or 'desc' (default: 'desc')
            buffer_only: Ignored in MemoryService (for compatibility with BufferService)

        Returns:
            List of message data
        """
        try:
            logger.debug(f"MemoryService.get_messages_by_session: session_id={session_id}, limit={limit}, sort_by={sort_by}, order={order}")
            messages = await self.db.get_messages_by_session(
                session_id=session_id,
                limit=limit,
                sort_by=sort_by,
                order=order
            )
            logger.debug(f"MemoryService.get_messages_by_session: Retrieved {len(messages)} messages")
            return messages
        except Exception as e:
            logger.error(f"MemoryService.get_messages_by_session: Error: {e}")
            import traceback
            logger.error(f"MemoryService.get_messages_by_session: Traceback: {traceback.format_exc()}")
            return []

    async def update(self, message_ids: List[str], new_messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Update messages in memory.

        Args:
            message_ids: List of message IDs
            new_messages: List of new message dictionaries

        Returns:
            Dictionary with status, code, and updated message IDs
        """
        if len(message_ids) != len(new_messages):
            return {
                "status": "error",
                "code": 400,
                "data": None,
                "message": "Number of message IDs and new messages must match",
                "errors": [{"field": "general", "message": "Number of message IDs and new messages must match"}],
            }

        updated_ids = []
        not_found_ids = []

        for i, message_id in enumerate(message_ids):
            message = await self.db.get_message(message_id)
            if not message:
                not_found_ids.append(message_id)
                continue

            new_message = new_messages[i]
            content = new_message.get("content", "")

            # Update message in database
            if await self.db.update_message(message_id, content):
                updated_ids.append(message_id)

                # Update message in vector store
                if self.vector_store:
                    await self.vector_store.update(message_id, Item(
                        id=message_id,
                        content=content,
                        metadata={
                            "type": "message",
                            "role": message["role"],
                            "user_id": self.user_id,
                            "agent_id": self.agent_id,
                            "session_id": self._session_id,
                            "round_id": message["round_id"],
                        }
                    ))

                # Update message in graph store
                if self.graph_store:
                    await self.graph_store.update_node(message_id, Node(
                        id=message_id,
                        content=content,
                        metadata={
                            "type": "message",
                            "role": message["role"],
                            "user_id": self.user_id,
                            "agent_id": self.agent_id,
                            "session_id": self._session_id,
                            "round_id": message["round_id"],
                        }
                    ))

                # Update message in keyword store
                if self.keyword_store:
                    await self.keyword_store.update(message_id, Item(
                        id=message_id,
                        content=content,
                        metadata={
                            "type": "message",
                            "role": message["role"],
                            "user_id": self.user_id,
                            "agent_id": self.agent_id,
                            "session_id": self._session_id,
                            "round_id": message["round_id"],
                        }
                    ))

        # Check if any messages were not found
        if not_found_ids:
            return {
                "status": "error",
                "code": 404,
                "data": {"not_found_ids": not_found_ids},
                "message": f"Some message IDs were not found",
                "errors": [
                    {"field": "message_ids",
                     "message": f"Message IDs not found: {', '.join(not_found_ids)}"}
                ],
            }

        return {
            "status": "success",
            "code": 200,
            "data": {"message_ids": updated_ids},
            "message": f"Updated {len(updated_ids)} messages",
            "errors": None,
        }

    async def delete(self, message_ids: Union[str, List[str]]) -> Union[bool, Dict[str, Any]]:
        """Delete message(s) from memory.

        This method can handle both single message deletion and batch deletion.
        For single message: returns bool
        For multiple messages: returns Dict with status information

        Args:
            message_ids: Message ID(s) to delete - can be str or List[str]

        Returns:
            For single message: True if the message was deleted, False otherwise
            For multiple messages: Dictionary with status, code, and deletion results
        """
        # Handle single message deletion (backward compatibility)
        if isinstance(message_ids, str):
            message_id = message_ids
            # Delete message from database
            if not await self.db.delete_message(message_id):
                return False

            # Delete message from vector store
            if self.vector_store:
                await self.vector_store.delete(message_id)

            # Delete message from graph store
            if self.graph_store:
                await self.graph_store.delete_node(message_id)

            # Delete message from keyword store
            if self.keyword_store:
                await self.keyword_store.delete(message_id)

            return True

        # Handle multiple message deletion
        elif isinstance(message_ids, list):
            deleted_ids = []
            not_found_ids = []
            failed_ids = []

            for message_id in message_ids:
                try:
                    # Check if message exists first
                    message = await self.db.get_message(message_id)
                    if not message:
                        not_found_ids.append(message_id)
                        continue

                    # Delete message from database
                    if await self.db.delete_message(message_id):
                        # Delete message from vector store
                        if self.vector_store:
                            await self.vector_store.delete(message_id)

                        # Delete message from graph store
                        if self.graph_store:
                            await self.graph_store.delete_node(message_id)

                        # Delete message from keyword store
                        if self.keyword_store:
                            await self.keyword_store.delete(message_id)

                        deleted_ids.append(message_id)
                    else:
                        # Message exists but deletion failed
                        failed_ids.append(message_id)
                except Exception as e:
                    logger.error(f"Failed to delete message {message_id}: {e}")
                    failed_ids.append(message_id)

            # If any messages were not found, return an error
            if not_found_ids:
                return {
                    "status": "error",
                    "code": 404,
                    "data": {
                        "deleted_ids": deleted_ids,
                        "not_found_ids": not_found_ids,
                        "failed_ids": failed_ids
                    },
                    "message": f"Some message IDs were not found: {', '.join(not_found_ids)}",
                    "errors": [{"field": "message_ids", "message": f"Message IDs not found: {', '.join(not_found_ids)}"}],
                }

            # If there were failures but no not_found messages, return error with different code
            if failed_ids:
                return {
                    "status": "error",
                    "code": 500,
                    "data": {
                        "deleted_ids": deleted_ids,
                        "failed_ids": failed_ids
                    },
                    "message": f"Failed to delete {len(failed_ids)} messages",
                    "errors": [{"field": "message_ids", "message": f"Failed to delete message IDs: {', '.join(failed_ids)}"}],
                }

            # Return success response
            return {
                "status": "success",
                "code": 200,
                "data": {
                    "deleted_ids": deleted_ids,
                    "not_found_ids": [],
                    "failed_ids": []
                },
                "message": f"Deleted {len(deleted_ids)} messages",
                "errors": None,
            }
        else:
            raise ValueError(f"message_ids must be str or List[str], got {type(message_ids)}")

    async def delete_messages_by_session(self, session_id: str) -> Dict[str, Any]:
        """Delete all messages for a specific session.

        Args:
            session_id: Session ID

        Returns:
            Dictionary with status, code, and deletion results
        """
        try:
            # Get all messages for this session
            messages = await self.get_messages_by_session(session_id)
            if not messages:
                return {
                    "status": "success",
                    "code": 200,
                    "data": {"deleted_ids": [], "message": "No messages found for session"},
                    "message": "No messages to delete",
                    "errors": None,
                }

            # Extract message IDs
            message_ids = [msg['id'] for msg in messages]
            logger.info(f"MemoryService.delete_messages_by_session: Deleting {len(message_ids)} messages for session {session_id}")

            # Use existing delete method for batch deletion
            result = await self.delete(message_ids)
            logger.info(f"MemoryService.delete_messages_by_session: Deletion result: {result}")
            return result

        except Exception as e:
            logger.error(f"MemoryService.delete_messages_by_session: Error deleting messages for session {session_id}: {e}")
            return {
                "status": "error",
                "code": 500,
                "data": {"error": str(e)},
                "message": f"Failed to delete messages for session {session_id}",
                "errors": [{"field": "session_id", "message": str(e)}],
            }

    async def add_knowledge(self, knowledge: List[str]) -> Dict[str, Any]:
        """Add knowledge to memory.

        Args:
            knowledge: List of knowledge strings

        Returns:
            Dictionary with status, code, and knowledge IDs
        """
        knowledge_ids = []
        for item in knowledge:
            # Add knowledge to database
            knowledge_id = await self.db.add_knowledge(self.user_id, item)
            knowledge_ids.append(knowledge_id)

            # Add knowledge to vector store
            await self.vector_store.add(Item(
                id=knowledge_id,
                content=item,
                metadata={
                    "type": "knowledge",
                    "user_id": self.user_id,
                }
            ))

            # Add knowledge to graph store
            await self.graph_store.add_node(Node(
                id=knowledge_id,
                content=item,
                metadata={
                    "type": "knowledge",
                    "user_id": self.user_id,
                }
            ))

            # Add knowledge to keyword store
            await self.keyword_store.add(Item(
                id=knowledge_id,
                content=item,
                metadata={
                    "type": "knowledge",
                    "user_id": self.user_id,
                }
            ))

        return {
            "status": "success",
            "code": 200,
            "data": {"knowledge_ids": knowledge_ids},
            "message": f"Added {len(knowledge_ids)} knowledge items",
            "errors": None,
        }

    async def read_knowledge(self, knowledge_ids: List[str]) -> Dict[str, Any]:
        """Read knowledge from memory.

        Args:
            knowledge_ids: List of knowledge IDs

        Returns:
            Dictionary with status, code, and knowledge items
        """
        knowledge_items = []
        for knowledge_id in knowledge_ids:
            knowledge = await self.db.get_knowledge(knowledge_id)
            if knowledge:
                knowledge_items.append({
                    "id": knowledge["id"],
                    "content": knowledge["content"],
                    "created_at": knowledge["created_at"],
                    "updated_at": knowledge["updated_at"],
                })

        return {
            "status": "success",
            "code": 200,
            "data": {"knowledge_items": knowledge_items},
            "message": f"Read {len(knowledge_items)} knowledge items",
            "errors": None,
        }

    async def update_knowledge(self, knowledge_ids: List[str], new_knowledge: List[str]) -> Dict[str, Any]:
        """Update knowledge in memory.

        Args:
            knowledge_ids: List of knowledge IDs
            new_knowledge: List of new knowledge strings

        Returns:
            Dictionary with status, code, and updated knowledge IDs
        """
        if len(knowledge_ids) != len(new_knowledge):
            return {
                "status": "error",
                "code": 400,
                "data": None,
                "message": "Number of knowledge IDs and new knowledge items must match",
                "errors": [{"field": "general", "message": "Number of knowledge IDs and new knowledge items must match"}],
            }

        updated_ids = []
        for i, knowledge_id in enumerate(knowledge_ids):
            knowledge = await self.db.get_knowledge(knowledge_id)
            if not knowledge:
                continue

            content = new_knowledge[i]

            # Update knowledge in database
            if await self.db.update_knowledge(knowledge_id, content):
                updated_ids.append(knowledge_id)

                # Update knowledge in vector store
                await self.vector_store.update(knowledge_id, Item(
                    id=knowledge_id,
                    content=content,
                    metadata={
                        "type": "knowledge",
                        "user_id": self.user_id,
                    }
                ))

                # Update knowledge in graph store
                await self.graph_store.update_node(knowledge_id, Node(
                    id=knowledge_id,
                    content=content,
                    metadata={
                        "type": "knowledge",
                        "user_id": self.user_id,
                    }
                ))

                # Update knowledge in keyword store
                await self.keyword_store.update(knowledge_id, Item(
                    id=knowledge_id,
                    content=content,
                    metadata={
                        "type": "knowledge",
                        "user_id": self.user_id,
                    }
                ))

        return {
            "status": "success",
            "code": 200,
            "data": {"knowledge_ids": updated_ids},
            "message": f"Updated {len(updated_ids)} knowledge items",
            "errors": None,
        }

    async def delete_knowledge(self, knowledge_id: str) -> bool:
        """Delete a knowledge item from memory.

        This is a core method that deletes a knowledge item from all store components.
        It does not include error handling or validation, which should be done
        by the caller.

        Args:
            knowledge_id: Knowledge ID to delete

        Returns:
            True if the knowledge item was deleted, False otherwise
        """
        # Delete knowledge from database
        if not await self.db.delete_knowledge(knowledge_id):
            return False

        # Delete knowledge from vector store
        if self.vector_store:
            await self.vector_store.delete(knowledge_id)

        # Delete knowledge from graph store
        if self.graph_store:
            await self.graph_store.delete_node(knowledge_id)

        # Delete knowledge from keyword store
        if self.keyword_store:
            await self.keyword_store.delete(knowledge_id)

        return True

    async def query(
        self,
        query: str,
        top_k: int = 15,  # Default return 15 results (rerank stage)
        first_stage_top_k: Optional[int] = None,  # If None, use top_k * 2
        store_type: Optional[str] = None,
        include_messages: bool = True,
        include_knowledge: bool = True,
        include_chunks: bool = True,  # Include chunks by default
        use_rerank: bool = True,
        session_id: Optional[str] = None,  # P1 OPTIMIZATION: Allow session_id override
    ) -> Dict[str, Any]:
        """Query memory for relevant information.

        Args:
            query: Query string
            top_k: Number of final results to return after reranking (default: 15)
            first_stage_top_k: Number of results to retrieve in the first stage.
                               If None, defaults to top_k * 2 (default: None)
            store_type: Type of store to query (vector, graph, keyword, or None for all)
            include_messages: Whether to include messages in the results
            include_knowledge: Whether to include knowledge in the results
            use_rerank: Whether to use reranking on the results
            session_id: Session ID to filter results (optional, overrides instance session_id)

        Returns:
            Dictionary with status, code, and query results
        """
        # Convert store_type to enum if provided
        store_type_enum = None
        if store_type:
            try:
                store_type_enum = StoreType(store_type)
            except ValueError:
                return {
                    "status": "error",
                    "code": 400,
                    "data": None,
                    "message": f"Invalid store type: {store_type}",
                    "errors": [{"field": "store_type", "message": f"Invalid store type: {store_type}"}],
                }

        # P1 OPTIMIZATION: Use provided session_id or fall back to instance session_id
        effective_session_id = session_id if session_id is not None else self._session_id

        # Calculate first_stage_top_k if not provided
        if first_stage_top_k is None:
            first_stage_top_k = top_k * 2

        # Log the top_k values
        logger.info(
            f"MemoryService.query: Using first_stage_top_k={first_stage_top_k}, final_top_k={top_k}")

        # Create query object
        query_obj = Query(
            text=query,
            metadata={
                "include_messages": include_messages,
                "include_knowledge": include_knowledge,
                "include_chunks": include_chunks,
                "user_id": self.user_id,
            }
        )

        # Get configuration
        cfg = config_manager.get_config()

        # Query using multi-path retrieval
        if store_type_enum == StoreType.VECTOR:
            # Query only vector store
            all_results = await self.multi_path_retrieval.retrieve(
                query=query_obj.text,
                user_id=self.user_id,
                session_id=effective_session_id,
                top_k=first_stage_top_k,  # Use first stage top_k value
                use_vector=True,
                use_graph=False,
                use_keyword=False
            )
            # Convert to QueryResult objects
            all_results = [
                QueryResult(
                    id=result["id"],
                    content=result["content"],
                    metadata=result["metadata"],
                    score=result["score"],
                    store_type=StoreType.VECTOR
                )
                for result in all_results
            ]
        elif store_type_enum == StoreType.GRAPH:
            # Query only graph store
            all_results = await self.multi_path_retrieval.retrieve(
                query=query_obj.text,
                user_id=self.user_id,
                session_id=effective_session_id,
                top_k=first_stage_top_k,  # Use first stage top_k value
                use_vector=False,
                use_graph=True,
                use_keyword=False
            )
            # Convert to QueryResult objects
            all_results = [
                QueryResult(
                    id=result["id"],
                    content=result["content"],
                    metadata=result["metadata"],
                    score=result["score"],
                    store_type=StoreType.GRAPH
                )
                for result in all_results
            ]
        elif store_type_enum == StoreType.KEYWORD:
            # Query only keyword store
            all_results = await self.multi_path_retrieval.retrieve(
                query=query_obj.text,
                user_id=self.user_id,
                session_id=effective_session_id,
                top_k=first_stage_top_k,  # Use first stage top_k value
                use_vector=False,
                use_graph=False,
                use_keyword=True
            )
            # Convert to QueryResult objects
            all_results = [
                QueryResult(
                    id=result["id"],
                    content=result["content"],
                    metadata=result["metadata"],
                    score=result["score"],
                    store_type=StoreType.KEYWORD
                )
                for result in all_results
            ]
        else:
            # Query based on server configuration
            use_vector = cfg["store"]["multi_path"]["use_vector"]
            use_graph = cfg["store"]["multi_path"]["use_graph"]
            use_keyword = cfg["store"]["multi_path"]["use_keyword"]

            # logger.info(
            #     f"MemoryService.query: Using server configuration for multi-path retrieval")
            # logger.info(
            #     f"MemoryService.query: Config settings: use_vector={use_vector}, use_graph={use_graph}, use_keyword={use_keyword}")
            # logger.info(
            #     f"MemoryService.query: Available stores: vector_store={self.vector_store is not None}, graph_store={self.graph_store is not None}, keyword_store={self.keyword_store is not None}")

            # Double check that at least one store is enabled in the configuration
            if not (use_vector or use_graph or use_keyword):
                logger.warning(
                    "MemoryService.query: No stores enabled in server configuration, enabling vector store by default")
                use_vector = True

            # Double check that the enabled stores exist
            if use_vector and self.vector_store is None:
                logger.warning(
                    "MemoryService.query: Vector store enabled in config but not available, disabling")
                use_vector = False
            if use_graph and self.graph_store is None:
                logger.warning(
                    "MemoryService.query: Graph store enabled in config but not available, disabling")
                use_graph = False
            if use_keyword and self.keyword_store is None:
                logger.warning(
                    "MemoryService.query: Keyword store enabled in config but not available, disabling")
                use_keyword = False

            # Final check to ensure at least one store is enabled and available
            if not (use_vector or use_graph or use_keyword):
                logger.error(
                    "MemoryService.query: No stores enabled and available in server config, falling back to vector store")
                use_vector = True  # Force vector store as fallback

            logger.info(
                f"MemoryService.query: Final server configuration: use_vector={use_vector}, use_graph={use_graph}, use_keyword={use_keyword}")

            # Use the server configuration for multi-path retrieval
            all_results = await self.multi_path_retrieval.retrieve(
                query=query_obj.text,
                user_id=self.user_id,
                session_id=effective_session_id,
                top_k=first_stage_top_k,  # Use first stage top_k value
                use_vector=use_vector,
                use_graph=use_graph,
                use_keyword=use_keyword
            )

            logger.info(
                f"MemoryService.query: multi_path_retrieval returned {len(all_results)} results")
            # Convert to QueryResult objects
            all_results = [
                QueryResult(
                    id=result["id"],
                    content=result["content"],
                    metadata=result["metadata"],
                    score=result["score"],
                    store_type=StoreType(result["store_type"]) if result.get(
                        "store_type") else None
                )
                for result in all_results
            ]



        # Convert to dictionaries
        result_dicts = []
        for result in all_results:
            # Get full item from database
            if result.metadata.get("type") == "message":
                item = await self.db.get_message(result.id)
                if item:
                    # Get round and session information
                    round_data = await self.db.get_round(
                        item["round_id"]) if item.get("round_id") else None
                    session_id = round_data.get(
                        "session_id") if round_data else None

                    # Get the actual session data to ensure correct metadata
                    actual_session = await self.db.get_session(
                        session_id) if session_id else None

                    # Create result dictionary with metadata
                    result_dict = {
                        "id": result.id,
                        "content": result.content,
                        "score": result.score,
                        "type": "message",
                        "role": item["role"],
                        "created_at": item["created_at"],
                        "updated_at": item["updated_at"],
                        "metadata": {
                            "user_id": self.user_id,
                            "agent_id": actual_session["agent_id"] if actual_session else None,
                            "session_id": session_id,
                            "session_name": actual_session["name"] if actual_session else None,
                            "level": 0,  # Default level is 0
                            "retrieval": self._get_retrieval_method(
                                result.store_type,
                                store_type_enum,
                                result.metadata.get("retrieval", {})
                            )
                        }
                    }
                    result_dicts.append(result_dict)
            elif result.metadata.get("type") == "knowledge":
                item = await self.db.get_knowledge(result.id)
                if item:
                    # Create result dictionary with metadata
                    result_dict = {
                        "id": result.id,
                        "content": result.content,
                        "score": result.score,
                        "type": "knowledge",
                        "role": None,  # Knowledge items don't have roles
                        "created_at": item["created_at"],
                        "updated_at": item["updated_at"],
                        "metadata": {
                            "user_id": self.user_id,
                            "agent_id": None,  # Knowledge is not associated with agents
                            "session_id": None,  # Knowledge is not associated with sessions
                            "level": 0,  # Default level is 0
                            "retrieval": self._get_retrieval_method(
                                result.store_type,
                                store_type_enum,
                                result.metadata.get("retrieval", {})
                            )
                        }
                    }
                    result_dicts.append(result_dict)

        # Apply reranking if requested
        if use_rerank and self.reranker and result_dicts:
            logger.info(
                f"Applying reranking to {len(result_dicts)} results with top_k={top_k}")

            # Apply reranking directly with the new simplified interface
            reranked_results = await self.reranker.rerank(
                query=query,
                items=result_dicts,
                top_k=top_k  # Use final top_k value (default 15)
            )

            # Replace result_dicts with reranked results
            if reranked_results:
                logger.info(
                    f"Reranking returned {len(reranked_results)} results")

                # Update result_dicts with reranked results
                result_dicts = reranked_results

                # Add reranking info to metadata
                for result in result_dicts:
                    if isinstance(result, dict) and "metadata" in result:
                        # Mark as reranked
                        result["metadata"]["retrieval"]["reranked"] = True

                        # Copy any reranking scores to the retrieval metadata
                        if "rrf_score" in result.get("metadata", {}):
                            result["metadata"]["retrieval"]["rrf_score"] = result["metadata"]["rrf_score"]
            else:
                logger.warning(
                    "Reranking returned no results, using original results")

        return {
            "status": "success",
            "code": 200,
            "data": {
                "results": result_dicts,
                "total": len(result_dicts),
            },
            "message": f"Found {len(result_dicts)} results",
            "errors": None,
        }

    async def shutdown(self):
        """Shutdown the memory service and cleanup resources."""
        try:
            logger.info("Shutting down MemoryService...")

            # Shutdown memory layer if it exists
            if hasattr(self, 'memory_layer') and self.memory_layer:
                if hasattr(self.memory_layer, 'cleanup'):
                    await self.memory_layer.cleanup()
                elif hasattr(self.memory_layer, 'shutdown'):
                    await self.memory_layer.shutdown()
                elif hasattr(self.memory_layer, 'close'):
                    await self.memory_layer.close()

            # Shutdown storage backends
            if hasattr(self, 'storage_manager') and self.storage_manager:
                if hasattr(self.storage_manager, 'shutdown'):
                    await self.storage_manager.shutdown()
                elif hasattr(self.storage_manager, 'close'):
                    await self.storage_manager.close()

            logger.info("MemoryService shutdown completed")

        except Exception as e:
            logger.error(f"Error during MemoryService shutdown: {e}")
