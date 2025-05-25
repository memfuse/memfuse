"""Service factory for MemFuse server."""

from typing import Optional, Dict, Any
from loguru import logger

from .memory_service import MemoryService
from .memory_service_proxy import MemoryServiceProxy


class ServiceFactory:
    """Factory class for creating service instances.

    This class provides methods for creating and accessing service instances
    such as MemoryService and BufferService.

    P1 OPTIMIZED Design Philosophy:
    - Global singletons: Models, Database, Reranker instances - ONE instance globally
    - User-specific cached: MemoryService, MemoryServiceProxy, BufferService (per user)
    - Session context: Passed as parameters, not stored in instances

    P1 Optimized Architecture:
    ┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐
    │ Global Singleton│    │ User-Specific    │    │ Session Context  │
    │                 │    │ Cached Services  │    │ (Parameters)     │
    │ RerankModel     │    │ MemoryService    │    │ session_id       │
    │ EmbeddingModel  │    │ MemoryServiceProxy│    │ agent_id         │
    │ Database        │    │ BufferService    │    │ context_params   │
    └─────────────────┘    └──────────────────┘    └──────────────────┘
    """

    # Global singletons (truly global, shared across all users)
    _global_rerank_model: Optional[Any] = None
    _global_embedding_model: Optional[Any] = None
    _global_reranker_instance: Optional[Any] = None

    # User-specific cached instances (one per user due to user-specific dirs)
    _memory_service_instances: Dict[str, MemoryService] = {}

    # P1 OPTIMIZATION: User-level cached instances (no longer session-specific)
    _memory_service_proxy_instances: Dict[str, MemoryServiceProxy] = {}
    _buffer_service_instances: Dict[str, Any] = {}

    @classmethod
    def set_global_models(
        cls,
        rerank_model: Optional[Any] = None,
        embedding_model: Optional[Any] = None,
        reranker_instance: Optional[Any] = None
    ) -> None:
        """Set global model instances that should be shared across all services.

        Args:
            rerank_model: Pre-loaded rerank model instance
            embedding_model: Pre-loaded embedding model instance
            reranker_instance: Pre-loaded reranker instance (MiniLMReranker)
        """
        if rerank_model is not None:
            cls._global_rerank_model = rerank_model
            logger.info("Set global rerank model instance")

        if embedding_model is not None:
            cls._global_embedding_model = embedding_model
            logger.info("Set global embedding model instance")

        if reranker_instance is not None:
            cls._global_reranker_instance = reranker_instance
            logger.info("Set global reranker instance")

    @classmethod
    def get_global_rerank_model(cls) -> Optional[Any]:
        """Get the global rerank model instance.

        Returns:
            Global rerank model instance or None
        """
        return cls._global_rerank_model

    @classmethod
    def get_global_embedding_model(cls) -> Optional[Any]:
        """Get the global embedding model instance.

        Returns:
            Global embedding model instance or None
        """
        return cls._global_embedding_model

    @classmethod
    def get_global_reranker_instance(cls) -> Optional[Any]:
        """Get the global reranker instance.

        Returns:
            Global reranker instance or None
        """
        return cls._global_reranker_instance

    @classmethod
    def set_global_memory_service(cls, memory_service: MemoryService) -> None:
        """Set the global memory service instance.

        Args:
            memory_service: MemoryService instance to use globally
        """
        cls._global_memory_service = memory_service
        logger.info("Set global memory service instance")

    @classmethod
    def get_memory_service_for_user(
        cls,
        user: str = "user_default",
        cfg: Optional[Any] = None
    ) -> Optional[MemoryService]:
        """Get a MemoryService instance for the specified user.

        This method returns a user-specific MemoryService instance that is cached
        for performance. Each user gets their own MemoryService due to user-specific
        data directories and storage components.

        Args:
            user: User name (default: "user_default")
            cfg: Configuration object (optional)

        Returns:
            MemoryService instance for the user
        """
        # Check if we already have a MemoryService instance for this user
        if user in cls._memory_service_instances:
            logger.debug(f"Using existing MemoryService instance for user {user}")
            return cls._memory_service_instances[user]

        # Create a new MemoryService instance for this user
        logger.info(f"Creating new MemoryService instance for user {user}")
        memory_service = MemoryService(cfg=cfg, user=user)

        # Store the instance for future use
        cls._memory_service_instances[user] = memory_service
        logger.debug(f"Cached MemoryService instance for user {user}")

        return memory_service

    @classmethod
    async def get_memory_service_proxy_for_user(
        cls,
        user: str = "user_default",
    ) -> Optional[MemoryServiceProxy]:
        """Get a user-level MemoryServiceProxy instance.

        P1 OPTIMIZATION: Returns a user-level proxy that can handle multiple sessions
        via parameter passing instead of creating session-specific instances.

        Args:
            user: User name (default: "user_default")

        Returns:
            MemoryServiceProxy instance for the user or None if memory service is not available
        """
        # Check if we already have a MemoryServiceProxy instance for this user
        if user in cls._memory_service_proxy_instances:
            logger.debug(f"Using existing user-level MemoryServiceProxy for user {user}")
            return cls._memory_service_proxy_instances[user]

        # Get the user-specific MemoryService instance
        memory_service = cls.get_memory_service_for_user(user)
        if memory_service is None:
            return None

        # Create a new user-level proxy for the user-specific MemoryService instance
        proxy = MemoryServiceProxy(
            memory_service=memory_service,
            user=user,
            agent=None,  # P1: Agent passed as parameter to methods
            session=None,  # P1: Session passed as parameter to methods
            session_id=None,  # P1: Session ID passed as parameter to methods
        )

        # Initialize the proxy (ensures underlying MemoryService is initialized)
        await proxy.initialize()
        logger.debug(f"User-level MemoryServiceProxy initialized for user {user}")

        # Store the instance for future use
        cls._memory_service_proxy_instances[user] = proxy
        logger.debug(f"Created new user-level MemoryServiceProxy for user {user}")

        return proxy

    @classmethod
    async def get_memory_service(
        cls,
        user: str = "user_default",
        agent: Optional[str] = None,
        session: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Optional[MemoryServiceProxy]:
        """Get a MemoryServiceProxy instance for the specified user, agent, and session.

        P1 OPTIMIZATION: This method now returns a user-level proxy and session context
        should be passed to individual method calls rather than stored in the proxy.

        Args:
            user: User name (default: "user_default")
            agent: Agent name (optional, for backward compatibility)
            session: Session name (optional, for backward compatibility)
            session_id: Session ID (optional, for backward compatibility)

        Returns:
            MemoryServiceProxy instance or None if memory service is not available
        """
        # P1 OPTIMIZATION: Return user-level proxy instead of session-specific
        return await cls.get_memory_service_proxy_for_user(user)

    @classmethod
    async def get_buffer_service_for_user(
        cls,
        user: str = "user_default",
    ) -> Optional[Any]:
        """Get a user-level BufferService instance.

        OPTIMIZED REFACTOR: Returns a user-level optimized buffer service that provides
        minimal overhead (<5%) while maintaining full MemoryInterface compatibility.

        Args:
            user: User name (default: "user_default")

        Returns:
            BufferService instance for the user or None if memory service is not available
        """
        # Import here to avoid circular imports
        from .buffer_service import BufferService
        from ..utils.config import config_manager
        from omegaconf import OmegaConf

        # Check if we already have an BufferService instance for this user
        if user in cls._buffer_service_instances:
            logger.info(f"✅ Using existing user-level BufferService for user {user}")
            return cls._buffer_service_instances[user]

        logger.info(f"🔄 Creating new user-level BufferService for user {user}")
        logger.info(f"🔍 Current cached users: {list(cls._buffer_service_instances.keys())}")

        # Get configuration
        config_dict = config_manager.get_config()
        cfg = OmegaConf.create(config_dict)
        buffer_config = cfg.get("buffer", {})

        # Get the user-specific MemoryService instance
        memory_service = cls.get_memory_service_for_user(user)
        if memory_service is None:
            logger.error(f"Cannot create BufferService for user {user}: MemoryService not available")
            return None

        # Ensure MemoryService is properly initialized
        try:
            # Check if multi_path_retrieval is None to determine if initialization is needed
            if memory_service.multi_path_retrieval is None:
                await memory_service.initialize()
                logger.debug(f"MemoryService initialized for BufferService user {user}")
            else:
                logger.debug(f"MemoryService already initialized for BufferService user {user}")
        except Exception as e:
            logger.error(f"Failed to initialize MemoryService for BufferService user {user}: {e}")
            return None

        # Create a new user-level BufferService instance
        buffer_service = BufferService(
            memory_service=memory_service,
            user=user,
            config=buffer_config,
        )

        # Store the instance for future use
        cls._buffer_service_instances[user] = buffer_service
        logger.debug(f"Created new user-level BufferService for user {user}")

        return buffer_service

    @classmethod
    async def get_buffer_service(
        cls,
        user: str = "user_default",
        agent: Optional[str] = None,
        session: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Optional[Any]:
        """Get an BufferService instance for the specified user, agent, and session.

        OPTIMIZED REFACTOR: This method now returns a user-level optimized buffer service
        with minimal overhead (<5%) and session context should be passed to individual
        method calls rather than stored in the service.

        Args:
            user: User name (default: "user_default")
            agent: Agent name (optional, for backward compatibility)
            session: Session name (optional, for backward compatibility)
            session_id: Session ID (optional, for backward compatibility)

        Returns:
            BufferService instance or None if memory service is not available
        """
        # OPTIMIZED REFACTOR: Return user-level optimized buffer service instead of session-specific
        return await cls.get_buffer_service_for_user(user)

    @classmethod
    def set_memory_service(cls, memory_service: MemoryService) -> None:
        """Set the global memory service instance.

        Args:
            memory_service: MemoryService instance
        """
        cls._global_memory_service = memory_service
        logger.debug("Global memory service instance set")

    @classmethod
    def reset(cls) -> None:
        """Reset all service instances.

        This method is primarily used for testing.
        """
        cls._global_rerank_model = None
        cls._global_embedding_model = None
        cls._global_reranker_instance = None
        cls._memory_service_instances = {}
        cls._memory_service_proxy_instances = {}
        cls._buffer_service_instances = {}
        logger.debug("Service factory reset")
