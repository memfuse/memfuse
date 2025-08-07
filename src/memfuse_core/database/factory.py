"""
LEGACY - Database Factory with Queue Management.

This module contains legacy database factory functions that create database instances
with queue management. These are no longer used in the current implementation.

Current approach uses direct database creation through DatabaseService.
See docs/connection_pool_optimization.md for details.
"""

from typing import Optional, Dict, Any
from loguru import logger

from .base import Database
from .postgres import PostgresDB
from .sqlite import SQLiteDB
from .queued_database import QueuedDatabase


def create_database(
    backend_type: str = "postgres",
    enable_queue: bool = True,
    **kwargs
) -> Database:
    """
    Create a database instance with optional queue management.
    
    Args:
        backend_type: Type of database backend ("postgres" or "sqlite")
        enable_queue: Whether to enable queue management for flow control
        **kwargs: Additional arguments for database initialization
    
    Returns:
        Database instance (optionally wrapped with queue management)
    """
    # Create the base database instance
    if backend_type.lower() == "postgres":
        base_db = PostgresDB(**kwargs)
    elif backend_type.lower() == "sqlite":
        base_db = SQLiteDB(**kwargs)
    else:
        raise ValueError(f"Unsupported database backend: {backend_type}")
    
    # Wrap with queue management if enabled
    if enable_queue:
        logger.info(f"Database factory: Creating {backend_type} database with queue management")
        return QueuedDatabase(base_db)
    else:
        logger.info(f"Database factory: Creating {backend_type} database without queue management")
        return base_db


def create_queued_database(
    backend_type: str = "postgres",
    max_concurrent_operations: int = 15,
    rate_limit_delay: float = 0.01,
    **kwargs
) -> QueuedDatabase:
    """
    Create a database instance with customized queue management.
    
    Args:
        backend_type: Type of database backend ("postgres" or "sqlite")
        max_concurrent_operations: Maximum concurrent database operations
        rate_limit_delay: Delay between operations in seconds
        **kwargs: Additional arguments for database initialization
    
    Returns:
        QueuedDatabase instance with customized settings
    """
    # Create the base database instance
    base_db = create_database(backend_type, enable_queue=False, **kwargs)
    
    # Create queue manager with custom settings
    # from ..services.database_queue_manager import DatabaseQueueManager
    # NOTE: DatabaseQueueManager was removed - this code is no longer functional
    
    logger.warning("create_queued_database is legacy code and no longer functional. "
                  "Use DatabaseService.get_instance() for current implementation.")
    return None  # This function is no longer operational
    
    # Create queued database with custom queue manager
    queued_db = QueuedDatabase(base_db)
    queued_db.queue_manager = queue_manager
    
    logger.info(f"Database factory: Created {backend_type} database with custom queue management "
               f"(max_concurrent={max_concurrent_operations}, rate_limit={rate_limit_delay}s)")
    
    return queued_db


# Convenience functions for common configurations
def create_high_throughput_database(backend_type: str = "postgres", **kwargs) -> QueuedDatabase:
    """Create a database optimized for high throughput operations."""
    return create_queued_database(
        backend_type=backend_type,
        max_concurrent_operations=25,
        rate_limit_delay=0.005,  # 5ms delay
        **kwargs
    )


def create_conservative_database(backend_type: str = "postgres", **kwargs) -> QueuedDatabase:
    """Create a database with conservative settings to prevent overload."""
    return create_queued_database(
        backend_type=backend_type,
        max_concurrent_operations=10,
        rate_limit_delay=0.02,  # 20ms delay
        **kwargs
    )


def create_batch_optimized_database(backend_type: str = "postgres", **kwargs) -> QueuedDatabase:
    """Create a database optimized for batch operations."""
    return create_queued_database(
        backend_type=backend_type,
        max_concurrent_operations=5,  # Fewer concurrent operations
        rate_limit_delay=0.05,  # 50ms delay for batch processing
        **kwargs
    )