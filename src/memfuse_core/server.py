"""MemFuse server implementation.

This module provides the main server entry point and orchestrates
service initialization and server startup.
"""

import asyncio
import threading
from typing import Optional, Any
from fastapi import FastAPI
from loguru import logger
from omegaconf import DictConfig
import uvicorn

from .utils.path_manager import PathManager
from .utils.config import config_manager

# Import services
from .services import (
    get_app_service,
    get_service_initializer,
    ServiceFactory
)


# ============================================================================
# Service Access Functions
# ============================================================================

def get_memory_service(
    user: str = "user_default",
    agent: Optional[str] = None,
    session: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Optional[Any]:
    """Get a memory service instance for the specified user, agent, and session.

    This function returns a lightweight proxy around the global memory service
    that is configured for the specified user, agent, and session.

    Args:
        user: User name (default: "user_default")
        agent: Agent name (optional)
        session: Session name (optional)
        session_id: Session ID (optional, takes precedence if provided)

    Returns:
        Memory service instance or None if memory service is not initialized
    """
    return ServiceFactory.get_memory_service(
        user=user,
        agent=agent,
        session=session,
        session_id=session_id,
    )


async def get_buffer_service(
    user: str = "user_default",
    agent: Optional[str] = None,
    session: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Optional[Any]:
    """Get a BufferService instance for the specified user, agent, and session.

    Args:
        user: User name (default: "user_default")
        agent: Agent name (optional)
        session: Session name (optional)
        session_id: Session ID (optional, takes precedence if provided)

    Returns:
        BufferService instance or None if buffer manager is not initialized
    """
    return await ServiceFactory.get_buffer_service(
        user=user,
        agent=agent,
        session=session,
        session_id=session_id,
    )


# ============================================================================
# Server Management
# ============================================================================

def run_server(cfg: Optional[DictConfig] = None):
    """Run the MemFuse server with the given configuration.

    Args:
        cfg: Configuration from Hydra (DictConfig)
    """
    # If no configuration provided, use __main__.main to run server
    if cfg is None:
        logger.info("No configuration provided, using __main__.main to run server")
        try:
            # Import main function
            from . import __main__
            # Call main function (this will trigger Hydra decorator)
            __main__.main()
            return
        except Exception as e:
            logger.error(f"Error running server via __main__.main: {e}")
            raise ValueError(f"Failed to run server: {e}") from e

    # Use provided configuration to run server
    logger.info("Using provided configuration to run server")

    # 1. Set configuration
    config_manager.set_config(cfg)
    logger.info("Configuration set successfully in ConfigManager")

    # 2. Create necessary directories
    PathManager.get_logs_dir()
    data_dir = cfg.get("data_dir", "data")
    PathManager.get_data_dir(data_dir)

    # 3. Log server configuration
    server_config = cfg.get("server", {})
    host = server_config.get("host", "localhost")
    port = server_config.get("port", 8000)
    logger.info(f"Starting MemFuse server on {host}:{port}")
    logger.info(f"Using data directory: {data_dir}")

    # 4. Initialize all services
    service_initializer = get_service_initializer()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Initialize services
    success = loop.run_until_complete(service_initializer.initialize_all_services(cfg))
    if not success:
        logger.error("Failed to initialize services, shutting down")
        return

    # Keep the event loop running in a separate thread
    def run_event_loop():
        loop.run_forever()

    event_loop_thread = threading.Thread(target=run_event_loop)
    event_loop_thread.daemon = True
    event_loop_thread.start()

    # 5. Start the server
    reload = server_config.get("reload", False)
    uvicorn.run(
        "memfuse_core.server:create_app",
        host=host,
        port=port,
        reload=reload,
        factory=True,
    )


# ============================================================================
# Factory Functions
# ============================================================================

def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    This function is used by uvicorn as a factory function.

    Returns:
        Configured FastAPI application
    """
    app_service = get_app_service()
    app = app_service.get_app()

    if app is None:
        logger.warning("App service not initialized, creating app directly")
        # Fallback: create app directly if service not initialized
        from .services.app_service import AppService
        temp_service = AppService()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(temp_service.initialize())
        app = temp_service.get_app()

        if app is None:
            raise RuntimeError("Failed to create FastAPI application")

    return app


# ============================================================================
# Entry Point
# ============================================================================

def main():
    """Entry point for the memfuse-core command.

    This function is called when running:
    - `poetry run memfuse-core`
    - `python -m memfuse_core` (via __main__.py)
    """
    run_server()
