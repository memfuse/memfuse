"""Service initializer for MemFuse server.

This module provides centralized service initialization and management.
"""

from typing import List, Optional, Type
from omegaconf import DictConfig
from loguru import logger

from .base_service import BaseService, ServiceRegistry
from .app_service import get_app_service
from .logging_service import get_logging_service
from .model_service import get_model_service
from .memory_service import MemoryService
from .service_factory import ServiceFactory


class ServiceInitializer:
    """Service for initializing and managing all application services."""

    def __init__(self):
        """Initialize the service initializer."""
        self._services: List[BaseService] = []
        self._initialized = False

    async def initialize_all_services(self, cfg: DictConfig) -> bool:
        """Initialize all services in the correct order.

        Args:
            cfg: Configuration from Hydra

        Returns:
            True if all services were initialized successfully, False otherwise
        """
        try:
            logger.info("Starting service initialization")

            # Initialize services in dependency order
            success = await self._initialize_core_services(cfg)
            if not success:
                logger.error("Failed to initialize core services")
                return False

            success = await self._initialize_business_services(cfg)
            if not success:
                logger.error("Failed to initialize business services")
                return False

            success = await self._initialize_app_services(cfg)
            if not success:
                logger.error("Failed to initialize app services")
                return False

            self._initialized = True
            logger.info("All services initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Error during service initialization: {e}")
            await self.shutdown_all_services()
            return False

    async def _initialize_core_services(self, cfg: DictConfig) -> bool:
        """Initialize core services (logging, config, etc.).

        Args:
            cfg: Configuration from Hydra

        Returns:
            True if initialization was successful, False otherwise
        """
        # 1. Initialize logging service
        logging_service = get_logging_service()
        if not await logging_service.initialize(cfg):
            return False
        self._services.append(logging_service)
        ServiceRegistry.register(logging_service)

        logger.info("Core services initialized")
        return True

    async def _initialize_business_services(self, cfg: DictConfig) -> bool:
        """Initialize business services (models, memory, etc.).

        Args:
            cfg: Configuration from Hydra

        Returns:
            True if initialization was successful, False otherwise
        """
        # 1. Initialize model service and preload models
        model_service = get_model_service()
        if not await model_service.initialize(cfg):
            return False
        ServiceRegistry.register(model_service)

        # 2. Set global models in ServiceFactory BEFORE creating MemoryService
        await self._set_global_models_in_factory(model_service)

        # 3. Initialize memory service (will now use global models)
        memory_service = await self._initialize_memory_service(cfg)
        if memory_service is None:
            return False

        logger.info("Business services initialized")
        return True

    async def _initialize_app_services(self, cfg: DictConfig) -> bool:
        """Initialize application services (FastAPI, etc.).

        Args:
            cfg: Configuration from Hydra

        Returns:
            True if initialization was successful, False otherwise
        """
        # 1. Initialize app service
        app_service = get_app_service()
        if not await app_service.initialize(cfg):
            return False
        self._services.append(app_service)
        ServiceRegistry.register(app_service)

        logger.info("Application services initialized")
        return True

    async def _set_global_models_in_factory(self, model_service) -> None:
        """Set global models in ServiceFactory before creating services.

        Args:
            model_service: Initialized model service with preloaded models
        """
        try:
            logger.info("Setting global models in ServiceFactory")

            # Get pre-loaded models from model service
            embedding_model = model_service.get_embedding_model()
            rerank_model = model_service.get_rerank_model()

            # Set global models in ServiceFactory for reuse
            from .service_factory import ServiceFactory
            ServiceFactory.set_global_models(
                rerank_model=rerank_model,
                embedding_model=embedding_model
            )

            # If we have a rerank model, create a global reranker instance
            if rerank_model is not None:
                logger.info("Creating global reranker instance with pre-loaded model")
                from ..rag.rerank import MiniLMReranker
                global_reranker = MiniLMReranker(
                    existing_model=rerank_model,  # Use the pre-loaded model
                    model_name="cross-encoder/ms-marco-MiniLM-L6-v2"
                )

                # Initialize the reranker (should be quick since model is already loaded)
                await global_reranker.initialize()

                # Store as global instance
                ServiceFactory.set_global_models(reranker_instance=global_reranker)
                logger.info("Global reranker instance created and cached")

            logger.info("Global models set in ServiceFactory successfully")

        except Exception as e:
            logger.warning(f"Failed to set global models in ServiceFactory: {e}")

    async def _initialize_memory_service(self, cfg: DictConfig) -> Optional[MemoryService]:
        """Initialize the memory service with model integration.

        Args:
            cfg: Configuration from Hydra

        Returns:
            Initialized memory service or None if failed
        """
        try:
            logger.info("Initializing memory service")

            # Create a default memory service for user_default
            # This will be used as a template and cached in ServiceFactory
            # Global models are already set, so MemoryService will use them
            memory_service = ServiceFactory.get_memory_service_for_user(
                user="user_default",
                cfg=cfg
            )

            # Initialize memory service (will use global models)
            if memory_service is not None:
                await memory_service.initialize()

            logger.info("Memory service initialized successfully")
            return memory_service

        except Exception as e:
            logger.error(f"Failed to initialize memory service: {e}")
            return None

    async def shutdown_all_services(self) -> bool:
        """Shutdown all services gracefully.

        Returns:
            True if all services were shutdown successfully, False otherwise
        """
        try:
            logger.info("Shutting down all services")

            # Shutdown services in reverse order
            for service in reversed(self._services):
                try:
                    await service.shutdown()
                except Exception as e:
                    logger.error(f"Error shutting down service {service.name}: {e}")

            # Clear registry
            ServiceRegistry.clear()
            self._services.clear()
            self._initialized = False

            logger.info("All services shutdown successfully")
            return True

        except Exception as e:
            logger.error(f"Error during service shutdown: {e}")
            return False

    def is_initialized(self) -> bool:
        """Check if all services are initialized.

        Returns:
            True if all services are initialized, False otherwise
        """
        return self._initialized


# Global service initializer instance
_service_initializer: Optional[ServiceInitializer] = None


def get_service_initializer() -> ServiceInitializer:
    """Get the global service initializer instance.

    Returns:
        ServiceInitializer instance
    """
    global _service_initializer
    if _service_initializer is None:
        _service_initializer = ServiceInitializer()
    return _service_initializer
