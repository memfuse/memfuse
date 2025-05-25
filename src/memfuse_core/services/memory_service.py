"""Memory service for MemFuse server."""

import asyncio
from loguru import logger
from typing import Dict, List, Any, Optional

from ..models import Item, Query, Node, QueryResult, StoreType
from ..store.factory import StoreFactory
from ..utils.config import config_manager
from ..utils.path_manager import PathManager
from ..rag.rerank import MiniLMReranker


class MemoryService:
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
        # Use the global database instance from DatabaseService
        from .database_service import DatabaseService
        self.db = DatabaseService.get_instance()

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

        # Ensure user exists and get user_id
        user_id = self.db.get_or_create_user_by_name(user)

        # Ensure agent exists and get agent_id if provided
        if agent is None:
            # Use a default agent name for all users
            agent = "agent_default"
        agent_id = self.db.get_or_create_agent_by_name(agent)

        # Get session_id - no longer creating sessions directly
        if session_id is not None:
            # If session_id is provided, use it directly
            session_data = self.db.get_session(session_id)
            if not session_data:
                # Create a new session with the provided ID
                session_id = self.db.create_session(
                    user_id, agent_id, session_id)
                self._session_id = session_id
                session = session_id  # Use session_id as name
            else:
                self._session_id = session_id
                session = session_data["name"]
        elif session is not None:
            # If session name is provided, check if it already exists for this user
            session_data = self.db.get_session_by_name(session, user_id=user_id)
            if session_data is not None:
                # Session with this name already exists for this user - raise error
                raise ValueError(
                    f"Session with name '{session}' already exists for user '{user}'. "
                    f"Session names must be unique within each user's scope."
                )
            else:
                # Session not found, create a new one
                session_id = self.db.create_session_with_name(
                    user_id, agent_id, session)
                self._session_id = session_id
        else:
            # For cross-session queries, we don't need a specific session
            self._session_id = None

        # Store both the names and IDs for internal use
        self.user = user
        self.agent = agent
        self.session = session
        self._user_id = user_id
        self._agent_id = agent_id

        # Store the user directory path
        data_dir = self.config.get("data_dir", "data")
        self.user_dir = str(PathManager.get_user_dir(data_dir, self._user_id))

        # Initialize store and retrieval (will be set in initialize method)
        self.vector_store = None
        self.graph_store = None
        self.keyword_store = None
        self.multi_path_retrieval = None
        self.reranker = None

        # Log initialization
        logger.info(f"MemoryService: Initialized for user: {user}")

    async def initialize(self):
        """Initialize the store and retrieval components asynchronously."""
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
            self.vector_store = await StoreFactory.create_vector_store(
                data_dir=self.user_dir,
                existing_model=existing_model
            )
        except Exception as e:
            logger.error(f"Failed to create vector store: {e}")
            self.vector_store = None

        try:
            self.graph_store = await StoreFactory.create_graph_store(
                data_dir=self.user_dir,
                existing_model=existing_model
            )
        except Exception as e:
            logger.error(f"Failed to create graph store: {e}")
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
            # Don't return early - continue with normal initialization
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
            # Create new reranker only if we don't have one
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

        return self

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

    async def add_batch(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Add a batch of messages to memory in a single operation.

        This method implements true batch processing, writing all messages in a single
        database transaction where possible, significantly improving performance.

        Args:
            messages: List of message dictionaries with role and content

        Returns:
            Dictionary with status, code, and message IDs
        """
        # P1 OPTIMIZATION: Get session_id from message metadata (not from self._session_id)
        session_id = None

        # Try to get session_id from message metadata
        if messages and len(messages) > 0:
            # Check first message metadata for session_id
            first_message = messages[0]
            logger.debug(f"MemoryService.add_batch: First message: {first_message}")
            if isinstance(first_message, dict) and 'metadata' in first_message:
                metadata = first_message['metadata']
                if isinstance(metadata, dict) and 'session_id' in metadata:
                    session_id = metadata['session_id']
                    logger.debug(f"MemoryService.add_batch: Found session_id in metadata: {session_id}")

            # Also check the message itself for session_id (for backward compatibility)
            if session_id is None and isinstance(first_message, dict) and 'session_id' in first_message:
                session_id = first_message['session_id']
                logger.debug(f"MemoryService.add_batch: Found session_id in message: {session_id}")

        # Still no session_id, return error
        if session_id is None:
            logger.error("MemoryService.add_batch: No session_id found, returning error")
            return {
                "status": "error",
                "code": 400,
                "data": None,
                "message": "Cannot add messages without a session",
                "errors": [{"field": "general", "message": "Cannot add messages without a session"}],
            }

        # Return early if no messages
        if not messages:
            return {
                "status": "success",
                "code": 200,
                "data": {"message_ids": []},
                "message": "No messages to add",
                "errors": None,
            }

        # Add session_id to message metadata if not already present
        for message in messages:
            if 'metadata' not in message:
                message['metadata'] = {}

            # Add session_id to metadata (use resolved session_id, not self._session_id)
            if 'session_id' not in message['metadata']:
                message['metadata']['session_id'] = session_id

            # Add user_id and agent_id to metadata
            if 'user_id' not in message['metadata']:
                message['metadata']['user_id'] = self._user_id

            if 'agent_id' not in message['metadata']:
                message['metadata']['agent_id'] = self._agent_id

        # Create a new round for these messages
        logger.debug(f"MemoryService.add_batch: Session ID: {session_id}")
        if not session_id:
            logger.error("MemoryService.add_batch: Session ID is None or empty")
            return {
                "status": "error",
                "code": 400,
                "data": None,
                "message": "Cannot add messages without a session",
                "errors": [{"field": "general", "message": "Cannot add messages without a session"}],
            }

        round_id = self.db.create_round(session_id)

        # Prepare data for batch operations
        message_ids = []
        vector_items = []
        graph_nodes = []
        keyword_items = []

        # Process all messages and prepare data structures for batch operations
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            # Add message to database (this is still individual since SQLite doesn't have true batch insert)
            message_id = self.db.add_message(round_id, role, content)
            message_ids.append(message_id)

            # Get user_id and agent_id from message metadata if available
            user_id = message.get('metadata', {}).get('user_id', self._user_id)
            agent_id = message.get('metadata', {}).get(
                'agent_id', self._agent_id)

            # Prepare item for vector store
            vector_items.append(Item(
                id=message_id,
                content=content,
                metadata={
                    "type": "message",
                    "role": role,
                    "user_id": user_id,
                    "agent_id": agent_id,
                    "session_id": session_id,
                    "round_id": round_id,
                }
            ))

            # Prepare node for graph store
            graph_nodes.append(Node(
                id=message_id,
                content=content,
                metadata={
                    "type": "message",
                    "role": role,
                    "user_id": user_id,
                    "agent_id": agent_id,
                    "session_id": session_id,
                    "round_id": round_id,
                }
            ))

            # Prepare item for keyword store
            keyword_items.append(Item(
                id=message_id,
                content=content,
                metadata={
                    "type": "message",
                    "role": role,
                    "user_id": user_id,
                    "agent_id": agent_id,
                    "session_id": session_id,
                    "round_id": round_id,
                }
            ))

        # Perform batch operations on each store
        try:
            # Batch add to vector store with timeout
            if self.vector_store and hasattr(self.vector_store, 'add_batch'):
                logger.info(
                    f"MemoryService.add_batch: Adding {len(vector_items)} items to vector store in batch")
                try:
                    await asyncio.wait_for(self.vector_store.add_batch(vector_items), timeout=30.0)
                except asyncio.TimeoutError:
                    logger.error(
                        "MemoryService.add_batch: Vector store batch operation timed out after 30.0 seconds")
                except Exception as e:
                    logger.error(
                        f"MemoryService.add_batch: Error adding items to vector store: {e}")
            elif self.vector_store:

                for item in vector_items:
                    try:
                        await asyncio.wait_for(self.vector_store.add(item), timeout=5.0)
                    except asyncio.TimeoutError:
                        logger.error(
                            f"MemoryService.add_batch: Vector store add operation timed out after 5.0 seconds for item {item.id}")
                    except Exception as e:
                        logger.error(
                            f"MemoryService.add_batch: Error adding item to vector store: {e}")

            # Batch add to graph store with timeout
            if self.graph_store and hasattr(self.graph_store, 'add_nodes'):
                logger.info(
                    f"MemoryService.add_batch: Adding {len(graph_nodes)} nodes to graph store in batch")
                try:
                    await asyncio.wait_for(self.graph_store.add_nodes(graph_nodes), timeout=30.0)
                except asyncio.TimeoutError:
                    logger.error(
                        "MemoryService.add_batch: Graph store batch operation timed out after 30.0 seconds")
                except Exception as e:
                    logger.error(
                        f"MemoryService.add_batch: Error adding nodes to graph store: {e}")
            elif self.graph_store:

                for node in graph_nodes:
                    try:
                        await asyncio.wait_for(self.graph_store.add_node(node), timeout=5.0)
                    except asyncio.TimeoutError:
                        logger.error(
                            f"MemoryService.add_batch: Graph store add operation timed out after 5.0 seconds for node {node.id}")
                    except Exception as e:
                        logger.error(
                            f"MemoryService.add_batch: Error adding node to graph store: {e}")

            # Batch add to keyword store with timeout
            if self.keyword_store and hasattr(self.keyword_store, 'add_batch'):
                logger.info(
                    f"MemoryService.add_batch: Adding {len(keyword_items)} items to keyword store in batch")
                try:
                    await asyncio.wait_for(self.keyword_store.add_batch(keyword_items), timeout=30.0)
                except asyncio.TimeoutError:
                    logger.error(
                        "MemoryService.add_batch: Keyword store batch operation timed out after 30.0 seconds")
                except Exception as e:
                    logger.error(
                        f"MemoryService.add_batch: Error adding items to keyword store: {e}")
            elif self.keyword_store:

                for item in keyword_items:
                    try:
                        await asyncio.wait_for(self.keyword_store.add(item), timeout=5.0)
                    except asyncio.TimeoutError:
                        logger.error(
                            f"MemoryService.add_batch: Keyword store add operation timed out after 5.0 seconds for item {item.id}")
                    except Exception as e:
                        logger.error(
                            f"MemoryService.add_batch: Error adding item to keyword store: {e}")

            logger.info(
                f"MemoryService.add_batch: Successfully added {len(messages)} messages in batch")
            return {
                "status": "success",
                "code": 200,
                "data": {"message_ids": message_ids},
                "message": f"Added {len(messages)} messages in batch",
                "errors": None,
            }
        except Exception as e:
            logger.error(
                f"MemoryService.add_batch: Error adding messages in batch: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "status": "error",
                "code": 500,
                "data": None,
                "message": f"Error adding messages in batch: {str(e)}",
                "errors": [{"field": "general", "message": f"Error adding messages in batch: {str(e)}"}],
            }

    async def add(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Add messages to memory.

        This method delegates to add_batch for better performance.

        Args:
            messages: List of message dictionaries with role and content

        Returns:
            Dictionary with status, code, and message IDs
        """
        # Ensure messages contain necessary metadata
        for message in messages:
            if 'metadata' not in message:
                message['metadata'] = {}

            # Ensure metadata contains session ID
            if 'session_id' not in message['metadata'] and self._session_id:
                message['metadata']['session_id'] = self._session_id

            # Ensure metadata contains user ID
            if 'user_id' not in message['metadata'] and self._user_id:
                message['metadata']['user_id'] = self._user_id

            # Ensure metadata contains agent ID
            if 'agent_id' not in message['metadata'] and self._agent_id:
                message['metadata']['agent_id'] = self._agent_id

        # Use direct batch store
        logger.info(
            f"MemoryService.add: Using batch store for {len(messages)} messages")
        return await self.add_batch(messages)

    async def read(self, message_ids: List[str]) -> Dict[str, Any]:
        """Read messages from memory.

        Args:
            message_ids: List of message IDs

        Returns:
            Dictionary with status, code, and messages
        """
        messages = []
        for message_id in message_ids:
            message = self.db.get_message(message_id)
            if message:
                messages.append({
                    "id": message["id"],
                    "role": message["role"],
                    "content": message["content"],
                    "created_at": message["created_at"],
                    "updated_at": message["updated_at"],
                })

        return {
            "status": "success",
            "code": 200,
            "data": {"messages": messages},
            "message": f"Read {len(messages)} messages",
            "errors": None,
        }

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
        for i, message_id in enumerate(message_ids):
            message = self.db.get_message(message_id)
            if not message:
                continue

            new_message = new_messages[i]
            content = new_message.get("content", "")

            # Update message in database
            if self.db.update_message(message_id, content):
                updated_ids.append(message_id)

                # Update message in vector store
                await self.vector_store.update(message_id, Item(
                    id=message_id,
                    content=content,
                    metadata={
                        "type": "message",
                        "role": message["role"],
                        "user_id": self._user_id,
                        "agent_id": self._agent_id,
                        "session_id": self._session_id,
                        "round_id": message["round_id"],
                    }
                ))

                # Update message in graph store
                await self.graph_store.update_node(message_id, Node(
                    id=message_id,
                    content=content,
                    metadata={
                        "type": "message",
                        "role": message["role"],
                        "user_id": self._user_id,
                        "agent_id": self._agent_id,
                        "session_id": self._session_id,
                        "round_id": message["round_id"],
                    }
                ))

                # Update message in keyword store
                await self.keyword_store.update(message_id, Item(
                    id=message_id,
                    content=content,
                    metadata={
                        "type": "message",
                        "role": message["role"],
                        "user_id": self._user_id,
                        "agent_id": self._agent_id,
                        "session_id": self._session_id,
                        "round_id": message["round_id"],
                    }
                ))

        return {
            "status": "success",
            "code": 200,
            "data": {"message_ids": updated_ids},
            "message": f"Updated {len(updated_ids)} messages",
            "errors": None,
        }

    async def delete(self, message_id: str) -> bool:
        """Delete a message from memory.

        This is a core method that deletes a message from all store components.
        It does not include error handling or validation, which should be done
        by the caller.

        Args:
            message_id: Message ID to delete

        Returns:
            True if the message was deleted, False otherwise
        """
        # Delete message from database
        if not self.db.delete_message(message_id):
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
            knowledge_id = self.db.add_knowledge(self._user_id, item)
            knowledge_ids.append(knowledge_id)

            # Add knowledge to vector store
            await self.vector_store.add(Item(
                id=knowledge_id,
                content=item,
                metadata={
                    "type": "knowledge",
                    "user_id": self._user_id,
                }
            ))

            # Add knowledge to graph store
            await self.graph_store.add_node(Node(
                id=knowledge_id,
                content=item,
                metadata={
                    "type": "knowledge",
                    "user_id": self._user_id,
                }
            ))

            # Add knowledge to keyword store
            await self.keyword_store.add(Item(
                id=knowledge_id,
                content=item,
                metadata={
                    "type": "knowledge",
                    "user_id": self._user_id,
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
            knowledge = self.db.get_knowledge(knowledge_id)
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
            knowledge = self.db.get_knowledge(knowledge_id)
            if not knowledge:
                continue

            content = new_knowledge[i]

            # Update knowledge in database
            if self.db.update_knowledge(knowledge_id, content):
                updated_ids.append(knowledge_id)

                # Update knowledge in vector store
                await self.vector_store.update(knowledge_id, Item(
                    id=knowledge_id,
                    content=content,
                    metadata={
                        "type": "knowledge",
                        "user_id": self._user_id,
                    }
                ))

                # Update knowledge in graph store
                await self.graph_store.update_node(knowledge_id, Node(
                    id=knowledge_id,
                    content=content,
                    metadata={
                        "type": "knowledge",
                        "user_id": self._user_id,
                    }
                ))

                # Update knowledge in keyword store
                await self.keyword_store.update(knowledge_id, Item(
                    id=knowledge_id,
                    content=content,
                    metadata={
                        "type": "knowledge",
                        "user_id": self._user_id,
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
        if not self.db.delete_knowledge(knowledge_id):
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
                "user_id": self._user_id,
            }
        )

        # Get configuration
        cfg = config_manager.get_config()

        # Query using multi-path retrieval
        if store_type_enum == StoreType.VECTOR:
            # Query only vector store
            all_results = await self.multi_path_retrieval.retrieve(
                query=query_obj.text,
                user_id=self._user_id,
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
                user_id=self._user_id,
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
                user_id=self._user_id,
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
                user_id=self._user_id,
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
                item = self.db.get_message(result.id)
                if item:
                    # Get round and session information
                    round_data = self.db.get_round(
                        item["round_id"]) if item.get("round_id") else None
                    session_id = round_data.get(
                        "session_id") if round_data else None

                    # Get the actual session data to ensure correct metadata
                    actual_session = self.db.get_session(
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
                            "user_id": self._user_id,
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
                item = self.db.get_knowledge(result.id)
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
                            "user_id": self._user_id,
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
