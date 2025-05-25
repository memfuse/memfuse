"""Database service for MemFuse server."""

from typing import Optional, Dict, Any
from loguru import logger

import os
from ..utils.config import config_manager
from ..utils.path_manager import PathManager
from ..database import Database, SQLiteDB, PostgresDB


class DatabaseService:
    """Database service for MemFuse server.
    
    This class provides a singleton instance of the Database class
    to avoid creating multiple database connections.
    """
    
    _instance: Optional[Database] = None
    
    @classmethod
    def get_instance(cls) -> Database:
        """Get the singleton Database instance.
        
        Returns:
            Database instance
        """
        if cls._instance is None:
            logger.debug("Creating new Database instance")
            
            # Get database configuration from config
            config_dict = config_manager.get_config()
            db_config = config_dict.get("database", {})
            db_type = db_config.get("type", "sqlite")
            
            # Create appropriate backend based on configuration
            if db_type == "postgres":
                # PostgreSQL backend
                host = db_config.get("host", "localhost")
                port = db_config.get("port", 5432)
                database = db_config.get("database", "memfuse")
                user = db_config.get("user", "postgres")
                password = db_config.get("password", "")
                
                try:
                    backend = PostgresDB(host, port, database, user, password)
                    logger.info(f"Using PostgreSQL backend at {host}:{port}/{database}")
                except ImportError:
                    logger.warning("PostgreSQL backend not available, falling back to SQLite")
                    db_path = cls._get_sqlite_path(config_dict)
                    backend = SQLiteDB(db_path)
                    logger.info(f"Using SQLite backend at {db_path}")
            else:
                # SQLite backend (default)
                db_path = cls._get_sqlite_path(config_dict)
                backend = SQLiteDB(db_path)
                logger.info(f"Using SQLite backend at {db_path}")
            
            cls._instance = Database(backend)
        return cls._instance
        
    @classmethod
    def _get_sqlite_path(cls, config_dict):
        """Get the SQLite database path from configuration.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            SQLite database path
        """
        data_dir = config_dict.get("data_dir", "data")
        db_path = os.path.join(data_dir, "metadata.db")
        # Create directory if it doesn't exist
        PathManager.ensure_directory(os.path.dirname(db_path))
        return db_path
    
    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton Database instance.
        
        This method is primarily used for testing.
        """
        if cls._instance is not None:
            cls._instance.close()
            cls._instance = None
            logger.debug("Database instance reset")
