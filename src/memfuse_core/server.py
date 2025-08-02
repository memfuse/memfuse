"""MemFuse server implementation.

This module provides the main server entry point and orchestrates
service initialization and server startup.
"""

import asyncio
import os
import signal
import threading
from typing import Optional, Any
from fastapi import FastAPI
from loguru import logger
from omegaconf import DictConfig
import uvicorn
import hydra

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load .env file from project root
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
        logger.info(f"Loaded environment variables from {env_path}")
    else:
        logger.info("No .env file found, using system environment variables")
except ImportError:
    logger.warning("python-dotenv not installed, using system environment variables only")

from .utils.path_manager import PathManager
from .utils.config import config_manager
from .utils.global_config_manager import get_global_config_manager
from .services.global_model_manager import get_global_model_manager

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

async def initialize_global_singletons(cfg: DictConfig) -> None:
    """Initialize global singletons for optimal performance.
    
    Args:
        cfg: Configuration from Hydra (DictConfig)
    """
    # Initialize global configuration manager
    global_config = get_global_config_manager()
    await global_config.initialize(cfg)
    
    # Also set in legacy config manager for backward compatibility
    config_manager.set_config(cfg)
    
    # Log configuration performance statistics
    config_stats = global_config.get_performance_stats()
    logger.info(f"Global configuration initialized: {config_stats['cached_keys']} keys cached in {config_stats['load_time_seconds']:.3f}s")
    
    # Initialize global model manager
    global_model_manager = get_global_model_manager()
    config_dict = global_config.get_raw_config()
    await global_model_manager.initialize_models(config_dict)
    
    # Log model performance statistics
    model_stats = global_model_manager.get_performance_stats()
    logger.info(f"Global models initialized: {model_stats['total_models']} models loaded in {model_stats['initialization_time_seconds']:.3f}s")


def run_server(cfg: DictConfig):
    """Run the MemFuse server with the given configuration.

    Args:
        cfg: Configuration from Hydra (DictConfig)
    """
    if cfg is None:
        raise ValueError("Configuration is required to run the server")

    # Use provided configuration to run server
    logger.info("Using provided configuration to run server")

    # 1. Initialize global singletons for optimal performance
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Initialize global singletons first
    loop.run_until_complete(initialize_global_singletons(cfg))
    logger.info("Global singletons initialized successfully")

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

    # Initialize services
    success = loop.run_until_complete(service_initializer.initialize_all_services(cfg))
    if not success:
        logger.error("Failed to initialize services, shutting down")
        return

    # Keep the event loop running in a separate thread
    def run_event_loop():
        try:
            loop.run_forever()
        except Exception as e:
            logger.error(f"Event loop error: {e}")
        finally:
            # Clean up pending tasks
            try:
                if loop and not loop.is_closed():
                    pending = asyncio.all_tasks(loop)
                    if pending:
                        logger.info(f"Cleaning up {len(pending)} pending tasks...")
                        for task in pending:
                            task.cancel()

                        # Only try to run cleanup if loop is still running
                        if loop.is_running():
                            try:
                                loop.run_until_complete(asyncio.wait_for(
                                    asyncio.gather(*pending, return_exceptions=True),
                                    timeout=3.0
                                ))
                            except (asyncio.TimeoutError, RuntimeError) as e:
                                logger.warning(f"Task cleanup timed out or failed: {e}")
                        else:
                            logger.info("Event loop not running, skipping task cleanup")
            except Exception as e:
                logger.warning(f"Error during task cleanup: {e}")
            finally:
                try:
                    if loop and not loop.is_closed():
                        loop.close()
                        logger.debug("Event loop closed successfully")
                except Exception as e:
                    logger.warning(f"Error closing event loop: {e}")

    event_loop_thread = threading.Thread(target=run_event_loop)
    event_loop_thread.daemon = True
    event_loop_thread.start()

    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down gracefully...")

        # Schedule service shutdown in the event loop
        if loop and not loop.is_closed() and loop.is_running():
            try:
                async def cleanup_services():
                    try:
                        from .services.service_initializer import ServiceInitializer
                        initializer = ServiceInitializer()
                        await initializer.shutdown_all_services()
                        logger.info("Service cleanup completed")
                    except Exception as e:
                        logger.error(f"Error during service cleanup: {e}")
                    finally:
                        loop.stop()

                # Schedule the cleanup task
                try:
                    asyncio.run_coroutine_threadsafe(cleanup_services(), loop)
                    logger.info("Cleanup task scheduled")
                except RuntimeError as e:
                    logger.warning(f"Could not schedule cleanup task: {e}")
                    loop.call_soon_threadsafe(loop.stop)
            except Exception as e:
                logger.error(f"Error scheduling cleanup: {e}")
                if loop and not loop.is_closed():
                    loop.call_soon_threadsafe(loop.stop)
        else:
            logger.info("Event loop not available for cleanup, stopping immediately")

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # 5. Start the server
        reload = server_config.get("reload", False)
        uvicorn.run(
            "memfuse_core.server:create_app",
            host=host,
            port=port,
            reload=reload,
            factory=True,
        )
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    finally:
        # Cleanup services
        try:
            from .services.service_initializer import ServiceInitializer
            initializer = ServiceInitializer()

            # Try to cleanup services
            if loop and not loop.is_closed() and not loop.is_running():
                # Loop exists but not running, try to run cleanup in new loop
                try:
                    logger.info("Running service cleanup in new event loop")
                    asyncio.run(initializer.shutdown_all_services())
                except Exception as e:
                    logger.error(f"Error in new loop cleanup: {e}")
            elif loop and not loop.is_closed() and loop.is_running():
                # Loop is running, schedule cleanup
                async def final_cleanup():
                    try:
                        await initializer.shutdown_all_services()
                        logger.info("Final service cleanup completed")
                    except Exception as e:
                        logger.error(f"Error in final cleanup: {e}")

                try:
                    future = asyncio.run_coroutine_threadsafe(final_cleanup(), loop)
                    future.result(timeout=5)  # Wait up to 5 seconds for cleanup
                except (asyncio.TimeoutError, RuntimeError) as e:
                    logger.warning(f"Final cleanup timed out or failed: {e}")
            else:
                # Loop is closed or unavailable, try direct cleanup
                try:
                    logger.info("Running direct service cleanup")
                    asyncio.run(initializer.shutdown_all_services())
                except Exception as e:
                    logger.error(f"Error in direct cleanup: {e}")
        except Exception as e:
            logger.error(f"Error during final cleanup: {e}")

        # Ensure event loop cleanup
        if loop and not loop.is_closed():
            loop.call_soon_threadsafe(loop.stop)
        if event_loop_thread.is_alive():
            event_loop_thread.join(timeout=5)


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

        # Check if we're already in an event loop
        try:
            loop = asyncio.get_running_loop()
            # If we're in a running loop, we can't use run_until_complete
            raise RuntimeError(
                "Cannot initialize app service from within a running event loop. "
                "Use create_app_async() instead."
            )
        except RuntimeError as e:
            if "no running event loop" in str(e):
                # No running loop, safe to create new one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(temp_service.initialize())
                app = temp_service.get_app()
            else:
                # Re-raise the original error
                raise

        if app is None:
            raise RuntimeError("Failed to create FastAPI application")

    return app


async def create_app_async() -> FastAPI:
    """Create and configure the FastAPI application asynchronously.

    This function can be used when already within an async context.

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
        await temp_service.initialize()
        app = temp_service.get_app()

        if app is None:
            raise RuntimeError("Failed to create FastAPI application")

    return app


# ============================================================================
# Entry Point
# ============================================================================

@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Entry point for the memfuse-core command.

    This function is called when running:
    - `poetry run memfuse-core`
    - `python -m memfuse_core` (via __main__.py)

    Args:
        cfg: Configuration from Hydra
    """
    run_server(cfg)
