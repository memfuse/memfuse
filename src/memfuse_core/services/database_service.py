"""Database service for MemFuse server."""

from typing import Optional
import asyncio
from loguru import logger

import os
from ..utils.config import config_manager
from ..database import Database, PostgresDB


class DatabaseService:
    """Database service for MemFuse server.

    This class provides a singleton instance of the Database class
    to avoid creating multiple database connections.
    """

    _instance: Optional[Database] = None
    _init_lock: Optional[asyncio.Lock] = None

    @classmethod
    async def get_instance(cls) -> Database:
        """Get the singleton Database instance.

        Returns:
            Database instance
        """
        if cls._instance is None:
            # Lazily create the init lock bound to current event loop
            if cls._init_lock is None:
                cls._init_lock = asyncio.Lock()

            async with cls._init_lock:
                # Double-checked locking: another waiter may have created it
                if cls._instance is not None:
                    return cls._instance

                logger.debug("Creating new Database instance")

                # Get database configuration from global config manager
                from ..utils.global_config_manager import get_global_config_manager
                config_manager_instance = get_global_config_manager()
                config_dict = config_manager_instance.get_config()
                db_config = config_dict.get("database", {})

                # PostgreSQL backend (pgai enhanced) - check environment variables first
                postgres_config = db_config.get("postgres", {})
                host = os.getenv("POSTGRES_HOST", postgres_config.get("host", "localhost"))
                port = int(os.getenv("POSTGRES_PORT", postgres_config.get("port", 5432)))
                database = os.getenv("POSTGRES_DB", postgres_config.get("database", "memfuse"))
                user = os.getenv("POSTGRES_USER", postgres_config.get("user", "postgres"))
                password = os.getenv("POSTGRES_PASSWORD", postgres_config.get("password", ""))

                try:
                    backend = PostgresDB(host, port, database, user, password)
                    logger.info(f"Using PostgreSQL backend (pgai enhanced) at {host}:{port}/{database}")
                except ImportError as e:
                    logger.error(f"PostgreSQL backend not available: {e}")
                    raise RuntimeError(f"PostgreSQL backend required but not available: {e}. Please install psycopg.")
                except Exception as e:
                    logger.error(f"Failed to connect to PostgreSQL: {e}")
                    raise RuntimeError(f"Failed to connect to PostgreSQL at {host}:{port}/{database}: {e}")

                # Create base database instance with simplified connection management
                base_database = Database(backend)
                # Initialize database tables asynchronously
                await base_database.initialize()
                
                # Use database directly without queue wrapper for streaming optimization
                cls._instance = base_database
                
                logger.info("Database instance created with simplified connection management")
        return cls._instance

    @classmethod
    async def reset_instance(cls) -> None:
        """Reset the singleton Database instance.

        This method is primarily used for testing.
        """
        if cls._instance is not None:
            await cls._instance.close()
            cls._instance = None
            logger.debug("Database instance reset")

    @classmethod
    def reset_instance_sync(cls) -> None:
        """Reset the singleton Database instance synchronously.

        This method is for testing environments where async context is not available.
        WARNING: This does not properly close connections - use reset_instance() when possible.
        """
        if cls._instance is not None:
            # Force reset without proper cleanup - only for testing
            cls._instance = None
            logger.warning("Database instance reset synchronously without proper cleanup")
