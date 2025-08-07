"""Service factory for MemFuse server."""

from typing import Optional, Dict, Any, List
from loguru import logger
import asyncio

from .memory_service import MemoryService
from .memory_service_proxy import MemoryServiceProxy
from .global_model_manager import get_global_model_manager


class ServiceFactory:
    """Factory class for creating service instances.

    This class provides methods for creating and accessing service instances
    such as MemoryService and BufferService.

    P1 OPTIMIZED Design Philosophy:
    - Global singletons: Models, Database, Reranker instances - ONE instance globally
    - User-specific cached: MemoryService, MemoryServiceProxy, BufferService (per user)
    - Session context: Passed as parameters, not stored in instances
    - Service pre-caching: Common services pre-created during startup for faster access

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

    # P1 OPTIMIZATION: Service pre-caching configuration
    _pre_cache_enabled: bool = True
    _common_users: List[str] = ["user_default", "alice", "bob", "demo_user"]
    _warmup_completed: bool = False

    @classmethod
    def set_global_models(
        cls,
        rerank_model: Optional[Any] = None,
        embedding_model: Optional[Any] = None,
        reranker_instance: Optional[Any] = None
    ) -> None:
        """Set global model instances that should be shared across all services.
        
        DEPRECATED: Use GlobalModelManager instead.

        Args:
            rerank_model: Pre-loaded rerank model instance
            embedding_model: Pre-loaded embedding model instance
            reranker_instance: Pre-loaded reranker instance (MiniLMReranker)
        """
        logger.warning("ServiceFactory.set_global_models is deprecated. Use GlobalModelManager instead.")
        
        if rerank_model is not None:
            cls._global_rerank_model = rerank_model
            logger.info("Set global rerank model instance (legacy)")

        if embedding_model is not None:
            cls._global_embedding_model = embedding_model
            logger.info("Set global embedding model instance (legacy)")

        if reranker_instance is not None:
            cls._global_reranker_instance = reranker_instance
            logger.info("Set global reranker instance (legacy)")

    @classmethod
    def get_global_rerank_model(cls) -> Optional[Any]:
        """Get the global rerank model instance.

        Returns:
            Global rerank model instance or None
        """
        # Try GlobalModelManager first
        global_model_manager = get_global_model_manager()
        model = global_model_manager.get_reranking_model()
        if model is not None:
            return model
        
        # Fallback to legacy storage
        return cls._global_rerank_model

    @classmethod
    def get_global_embedding_model(cls) -> Optional[Any]:
        """Get the global embedding model instance.

        Returns:
            Global embedding model instance or None
        """
        # Try GlobalModelManager first
        global_model_manager = get_global_model_manager()
        model = global_model_manager.get_embedding_model()
        if model is not None:
            return model
        
        # Fallback to legacy storage
        return cls._global_embedding_model

    @classmethod
    def get_global_reranker_instance(cls) -> Optional[Any]:
        """Get the global reranker instance.

        Returns:
            Global reranker instance or None
        """
        # Try GlobalModelManager first
        global_model_manager = get_global_model_manager()
        model = global_model_manager.get_reranking_model()
        if model is not None:
            return model
        
        # Fallback to legacy storage
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

        BUFFER: Returns BufferService with RoundBuffer, HybridBuffer, and QueryBuffer
        for improved performance and functionality.

        Args:
            user: User name (default: "user_default")

        Returns:
            BufferService instance for the user or None if memory service is not available
        """
        # Import here to avoid circular imports
        from .buffer_service import BufferService
        from ..utils.config import config_manager
        from omegaconf import OmegaConf

        # Check if we already have a BufferService instance for this user
        if user in cls._buffer_service_instances:
            logger.info(f"✅ Using existing user-level BufferService for user {user}")
            return cls._buffer_service_instances[user]

        logger.info(f"🔄 Creating new user-level BufferService for user {user}")
        logger.info(f"🔍 Current cached users: {list(cls._buffer_service_instances.keys())}")

        # Get configuration
        config_dict = config_manager.get_config()
        cfg = OmegaConf.create(config_dict)

        # Get the user-specific MemoryService instance first
        memory_service = cls.get_memory_service_for_user(user)
        if memory_service is None:
            logger.error(f"Cannot create BufferService for user {user}: MemoryService not available")
            return None

        # Ensure MemoryService is properly initialized
        try:
            if memory_service.multi_path_retrieval is None:
                await memory_service.initialize()
                logger.debug(f"MemoryService initialized for BufferService user {user}")
        except Exception as e:
            logger.error(f"Failed to initialize MemoryService for BufferService user {user}: {e}")
            return None

        # Create BufferService with bypass support (enabled/disabled handled in BufferService.__init__)
        # Convert OmegaConf to dict to ensure proper config access
        config_dict = OmegaConf.to_container(cfg, resolve=True) if hasattr(cfg, 'to_container') else cfg
        buffer_service = BufferService(
            memory_service=memory_service,
            user=user,
            config=config_dict,  # Pass full config as dict including buffer.enabled setting
        )

        # Initialize the BufferService
        try:
            if not await buffer_service.initialize():
                logger.error(f"Failed to initialize BufferService for user {user}")
                return None
            logger.debug(f"BufferService initialized successfully for user {user}")
        except Exception as e:
            logger.error(f"BufferService initialization error for user {user}: {e}")
            return None

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
        """Get a BufferService instance for the specified user, agent, and session.

        BUFFER: This method now returns BufferService with RoundBuffer, HybridBuffer,
        and QueryBuffer for improved performance and functionality.

        Args:
            user: User name (default: "user_default")
            agent: Agent name (optional, for backward compatibility)
            session: Session name (optional, for backward compatibility)
            session_id: Session ID (optional, for backward compatibility)

        Returns:
            BufferService instance or None if memory service is not available
        """
        # BUFFER: Return user-level BufferService instead of session-specific
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
    async def cleanup_all_services(cls) -> None:
        """Cleanup all service instances and shared resources.

        Optimized cleanup with concurrent operations where safe.
        
        This method should be called during application shutdown to ensure
        proper cleanup of connection pools and other shared resources.
        """
        logger.info("Cleaning up all service instances...")

        # Step 1: Shutdown memory services (can be done concurrently)
        memory_shutdown_tasks = []
        for user, memory_service in cls._memory_service_instances.items():
            async def shutdown_memory_service(service_user, service):
                try:
                    if hasattr(service, 'shutdown'):
                        await service.shutdown()
                        logger.debug(f"Shutdown memory service for user {service_user}")
                    elif hasattr(service, 'close'):
                        await service.close()
                        logger.debug(f"Closed memory service for user {service_user}")
                except Exception as e:
                    logger.error(f"Error shutting down memory service for user {service_user}: {e}")
            
            memory_shutdown_tasks.append(shutdown_memory_service(user, memory_service))

        if memory_shutdown_tasks:
            logger.info(f"ServiceFactory: Shutting down {len(memory_shutdown_tasks)} memory services concurrently")
            await asyncio.gather(*memory_shutdown_tasks, return_exceptions=True)

        # Step 2: Shutdown buffer services (critical - contains FlushManager)
        buffer_shutdown_tasks = []
        for user, buffer_service in cls._buffer_service_instances.items():
            async def shutdown_buffer_service(service_user, service):
                try:
                    if hasattr(service, 'shutdown'):
                        await service.shutdown()
                        logger.debug(f"Shutdown buffer service for user {service_user}")
                    elif hasattr(service, 'close'):
                        await service.close()
                        logger.debug(f"Closed buffer service for user {service_user}")
                except Exception as e:
                    logger.error(f"Error shutting down buffer service for user {service_user}: {e}")
            
            buffer_shutdown_tasks.append(shutdown_buffer_service(user, buffer_service))

        if buffer_shutdown_tasks:
            logger.info(f"ServiceFactory: Shutting down {len(buffer_shutdown_tasks)} buffer services concurrently")
            await asyncio.gather(*buffer_shutdown_tasks, return_exceptions=True)

        # Step 3: Shutdown memory service proxies (can be done concurrently)
        proxy_shutdown_tasks = []
        for user, proxy in cls._memory_service_proxy_instances.items():
            async def shutdown_proxy(service_user, service_proxy):
                try:
                    if hasattr(service_proxy, 'close'):
                        await service_proxy.close()
                        logger.debug(f"Closed memory service proxy for user {service_user}")
                except Exception as e:
                    logger.error(f"Error closing memory service proxy for user {service_user}: {e}")
            
            proxy_shutdown_tasks.append(shutdown_proxy(user, proxy))

        if proxy_shutdown_tasks:
            logger.info(f"ServiceFactory: Closing {len(proxy_shutdown_tasks)} service proxies concurrently")
            await asyncio.gather(*proxy_shutdown_tasks, return_exceptions=True)

        # Step 4: Close global connection pools (final step)
        try:
            from .global_connection_manager import get_global_connection_manager
            connection_manager = get_global_connection_manager()
            await connection_manager.close_all_pools(force=True)
            logger.info("Closed all global connection pools")
        except Exception as e:
            logger.error(f"Error closing global connection pools: {e}")

        # Step 5: Clear all cached instances
        cls._memory_service_instances.clear()
        cls._buffer_service_instances.clear()
        cls._memory_service_proxy_instances.clear()
        cls._global_memory_service = None

        logger.info("Service cleanup completed")

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
        cls._warmup_completed = False
        logger.debug("Service factory reset")

    @classmethod
    async def warmup_common_services(cls, cfg: Optional[Any] = None) -> bool:
        """Pre-cache service instances for common users to improve first-access performance.

        This method should be called during application startup after global models
        are initialized to pre-create service instances for common users.

        Args:
            cfg: Configuration object (optional)

        Returns:
            True if warmup completed successfully, False otherwise
        """
        if not cls._pre_cache_enabled:
            logger.info("ServiceFactory: Service pre-caching disabled")
            return True

        if cls._warmup_completed:
            logger.debug("ServiceFactory: Service warmup already completed")
            return True

        try:
            logger.info(f"ServiceFactory: Starting service warmup for {len(cls._common_users)} common users...")
            warmup_start_time = asyncio.get_event_loop().time()

            # Pre-create services for common users in parallel
            warmup_tasks = []
            for user in cls._common_users:
                warmup_tasks.append(cls._warmup_user_services(user, cfg))

            # Execute warmup tasks concurrently
            results = await asyncio.gather(*warmup_tasks, return_exceptions=True)

            # Check results
            successful_warmups = 0
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.warning(f"ServiceFactory: Failed to warmup services for user {cls._common_users[i]}: {result}")
                elif result:
                    successful_warmups += 1

            warmup_end_time = asyncio.get_event_loop().time()
            warmup_duration = warmup_end_time - warmup_start_time

            cls._warmup_completed = True
            logger.info(f"ServiceFactory: Service warmup completed - {successful_warmups}/{len(cls._common_users)} users warmed up in {warmup_duration:.3f}s")

            # Log cache statistics
            cls._log_cache_statistics()

            return successful_warmups > 0

        except Exception as e:
            logger.error(f"ServiceFactory: Service warmup failed: {e}")
            return False

    @classmethod
    async def _warmup_user_services(cls, user: str, cfg: Optional[Any] = None) -> bool:
        """Warmup services for a specific user.

        Args:
            user: User name to warmup services for
            cfg: Configuration object (optional)

        Returns:
            True if warmup successful, False otherwise
        """
        try:
            logger.debug(f"ServiceFactory: Warming up services for user {user}")

            # Pre-create MemoryService (this will also cache it)
            memory_service = cls.get_memory_service_for_user(user, cfg)
            if memory_service:
                await memory_service.initialize()
                logger.debug(f"ServiceFactory: MemoryService warmed up for user {user}")

            # Pre-create MemoryServiceProxy (this will also cache it)
            memory_proxy = await cls.get_memory_service_proxy_for_user(user)
            if memory_proxy:
                logger.debug(f"ServiceFactory: MemoryServiceProxy warmed up for user {user}")

            # Pre-create BufferService (this will also cache it)
            buffer_service = await cls.get_buffer_service_for_user(user)
            if buffer_service:
                logger.debug(f"ServiceFactory: BufferService warmed up for user {user}")

            logger.debug(f"ServiceFactory: Successfully warmed up all services for user {user}")
            return True

        except Exception as e:
            logger.warning(f"ServiceFactory: Failed to warmup services for user {user}: {e}")
            return False

    @classmethod
    def _log_cache_statistics(cls) -> None:
        """Log current cache statistics."""
        memory_services = len(cls._memory_service_instances)
        memory_proxies = len(cls._memory_service_proxy_instances)
        buffer_services = len(cls._buffer_service_instances)

        logger.info(f"ServiceFactory: Cache statistics - MemoryServices: {memory_services}, "
                    f"MemoryProxies: {memory_proxies}, BufferServices: {buffer_services}")

    @classmethod
    def configure_warmup(cls, enabled: bool = True, common_users: Optional[List[str]] = None) -> None:
        """Configure service warmup settings.

        Args:
            enabled: Whether to enable service pre-caching
            common_users: List of common users to pre-cache services for
        """
        cls._pre_cache_enabled = enabled
        if common_users is not None:
            cls._common_users = common_users

        logger.info(f"ServiceFactory: Warmup configured - enabled: {enabled}, users: {cls._common_users}")

    @classmethod
    def is_warmup_completed(cls) -> bool:
        """Check if service warmup has been completed.

        Returns:
            True if warmup completed, False otherwise
        """
        return cls._warmup_completed
