"""
Global Connection Manager for MemFuse

This module implements a singleton-based global connection manager that follows
the MemFuse singleton optimization strategy. It provides centralized management
of PostgreSQL connection pools across all services and users.

Design Principles:
- Tier 1 Global Singleton: One connection pool per database URL
- Configuration-driven: Reads from config hierarchy
- Resource efficiency: Shared pools across all users
- Proper lifecycle management: Cleanup and monitoring
"""

import asyncio
import weakref
from typing import Dict, Optional, Any, Set
from loguru import logger
from dataclasses import dataclass
from threading import Lock

try:
    from psycopg_pool import AsyncConnectionPool
    from pgvector.psycopg import register_vector_async
    PSYCOPG_AVAILABLE = True
except ImportError:
    PSYCOPG_AVAILABLE = False
    AsyncConnectionPool = None
    logger.warning("psycopg dependencies not available for global connection manager")


@dataclass
class ConnectionPoolConfig:
    """Configuration for PostgreSQL connection pools."""
    min_size: int = 1  # Minimal connections for stability
    max_size: int = 2  # Very conservative max to avoid pool exhaustion
    timeout: float = 5.0  # Short timeout to fail fast
    recycle: int = 1800  # Shorter recycle time
    connection_timeout: float = 10.0  # Short connection timeout
    keepalives_idle: int = 600  # Longer idle time for stability
    keepalives_interval: int = 60  # Longer interval
    keepalives_count: int = 3
    
    @classmethod
    def from_memfuse_config(cls, config: Dict[str, Any]) -> 'ConnectionPoolConfig':
        """Create configuration from MemFuse config hierarchy.
        
        Configuration priority (highest to lowest):
        1. store.database.postgres.*
        2. database.postgres.*
        3. postgres.*
        4. Default values
        """
        postgres_config = {}
        
        # Layer 1: Base postgres config
        if "postgres" in config:
            postgres_config.update(config["postgres"])
        
        # Layer 2: Database postgres config (higher priority)
        if "database" in config and "postgres" in config["database"]:
            postgres_config.update(config["database"]["postgres"])
        
        # Layer 3: Store database postgres config (highest priority)
        if ("store" in config and 
            "database" in config["store"] and 
            "postgres" in config["store"]["database"]):
            postgres_config.update(config["store"]["database"]["postgres"])
        
        # Extract values with defaults
        pool_size = postgres_config.get("pool_size", 10)  # Match test expectations
        max_overflow = postgres_config.get("max_overflow", 40)  # 50 - 10 = 40
        max_size = pool_size + max_overflow
        
        logger.debug(f"ConnectionPoolConfig: min_size={pool_size}, max_size={max_size}, "
                    f"timeout={postgres_config.get('pool_timeout', 30.0)}, "
                    f"recycle={postgres_config.get('pool_recycle', 3600)}")
        
        return cls(
            min_size=pool_size,
            max_size=max_size,
            timeout=postgres_config.get("pool_timeout", 60.0),
            recycle=postgres_config.get("pool_recycle", 7200),
            connection_timeout=postgres_config.get("connection_timeout", 30.0),
            keepalives_idle=postgres_config.get("keepalives_idle", 600),
            keepalives_interval=postgres_config.get("keepalives_interval", 30),
            keepalives_count=postgres_config.get("keepalives_count", 3)
        )


class GlobalConnectionManager:
    """
    Tier 1 Global Singleton: PostgreSQL Connection Pool Manager
    
    This class implements the singleton pattern for managing PostgreSQL connection
    pools across the entire MemFuse application. It ensures that all services,
    users, and stores share connection pools based on database URL.
    
    Features:
    - Single connection pool per database URL
    - Configuration-driven pool sizing
    - Automatic cleanup and lifecycle management
    - Connection monitoring and statistics
    - Thread-safe operations
    """
    
    _instance: Optional['GlobalConnectionManager'] = None
    _lock = Lock()
    
    def __init__(self):
        """Initialize the global connection manager."""
        if not PSYCOPG_AVAILABLE:
            raise ImportError("psycopg dependencies required for global connection manager")

        self._pools: Dict[str, AsyncConnectionPool] = {}
        self._pool_configs: Dict[str, ConnectionPoolConfig] = {}
        self._store_references: Dict[str, Set[weakref.ref]] = {}
        self._async_lock = None  # Will be created when needed
        self._initialized = False

        logger.info("GlobalConnectionManager: Singleton instance created")
    
    def __new__(cls) -> 'GlobalConnectionManager':
        """Ensure singleton pattern with thread safety."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def get_instance(cls) -> 'GlobalConnectionManager':
        """Get the singleton instance."""
        return cls()
    
    async def get_connection_pool(
        self,
        db_url: str,
        config: Optional[Dict[str, Any]] = None,
        store_ref: Optional[Any] = None
    ) -> AsyncConnectionPool:
        """
        Get or create a shared connection pool for the database URL.

        This method implements the core singleton pattern for connection pools:
        - One pool per unique database URL
        - Configuration-driven pool settings
        - Automatic reference tracking for cleanup

        Args:
            db_url: Database connection URL
            config: MemFuse configuration dictionary
            store_ref: Reference to the store using this pool

        Returns:
            Shared AsyncConnectionPool instance
        """
        # Create async lock if not already created
        if self._async_lock is None:
            self._async_lock = asyncio.Lock()

        async with self._async_lock:
            # Check if pool already exists
            if db_url in self._pools:
                pool = self._pools[db_url]
                
                # Add store reference for tracking
                if store_ref is not None:
                    self._add_store_reference(db_url, store_ref)
                
                logger.debug(f"GlobalConnectionManager: Reusing pool for {self._mask_url(db_url)}")
                return pool
            
            # Create new pool
            logger.info(f"GlobalConnectionManager: Creating new pool for {self._mask_url(db_url)}")
            
            # Get pool configuration from MemFuse config
            if config is None:
                # Try to get config from global config manager
                try:
                    from ..utils.config import config_manager
                    config = config_manager.get_config() or {}
                except Exception as e:
                    logger.warning(f"Could not get global config: {e}")
                    config = {}
            
            pool_config = ConnectionPoolConfig.from_memfuse_config(config)
            self._pool_configs[db_url] = pool_config
            
            # Create the pool with enhanced connection parameters
            # Add keepalive parameters to the connection URL
            enhanced_db_url = self._enhance_db_url_with_keepalives(db_url, pool_config)

            pool = AsyncConnectionPool(
                enhanced_db_url,
                min_size=pool_config.min_size,
                max_size=pool_config.max_size,
                open=False,
                configure=self._configure_connection,
                timeout=pool_config.connection_timeout
            )
            
            # Open the pool
            try:
                await asyncio.wait_for(pool.open(), timeout=pool_config.timeout)
                logger.info(f"GlobalConnectionManager: Pool opened successfully "
                           f"(min={pool_config.min_size}, max={pool_config.max_size})")
            except asyncio.TimeoutError:
                logger.error(f"GlobalConnectionManager: Pool opening timed out for {self._mask_url(db_url)}")
                raise
            except Exception as e:
                logger.error(f"GlobalConnectionManager: Failed to open pool: {e}")
                raise
            
            # Store the pool
            self._pools[db_url] = pool
            self._store_references[db_url] = set()
            
            # Add store reference for tracking
            if store_ref is not None:
                self._add_store_reference(db_url, store_ref)
            
            return pool
    
    def _enhance_db_url_with_keepalives(self, db_url: str, config: ConnectionPoolConfig) -> str:
        """Add keepalive parameters to database URL for better connection stability."""
        try:
            from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

            parsed = urlparse(db_url)
            query_params = parse_qs(parsed.query)

            # Add keepalive parameters
            query_params['keepalives_idle'] = [str(config.keepalives_idle)]
            query_params['keepalives_interval'] = [str(config.keepalives_interval)]
            query_params['keepalives_count'] = [str(config.keepalives_count)]
            query_params['connect_timeout'] = [str(int(config.connection_timeout))]

            # Rebuild URL
            new_query = urlencode(query_params, doseq=True)
            enhanced_url = urlunparse((
                parsed.scheme, parsed.netloc, parsed.path,
                parsed.params, new_query, parsed.fragment
            ))

            logger.debug(f"Enhanced DB URL with keepalive parameters")
            return enhanced_url

        except Exception as e:
            logger.warning(f"Failed to enhance DB URL with keepalives: {e}")
            return db_url

    async def _configure_connection(self, conn):
        """Configure a new connection with pgvector support."""
        try:
            await register_vector_async(conn)
            logger.debug("GlobalConnectionManager: pgvector registered on connection")
        except Exception as e:
            logger.warning(f"GlobalConnectionManager: Failed to register pgvector: {e}")
    
    def _add_store_reference(self, db_url: str, store_ref: Any):
        """Add a weak reference to a store using this pool."""
        if db_url not in self._store_references:
            self._store_references[db_url] = set()
        
        # Create weak reference with cleanup callback
        def cleanup_ref(ref):
            if db_url in self._store_references:
                self._store_references[db_url].discard(ref)
                logger.debug(f"GlobalConnectionManager: Cleaned up store reference for {self._mask_url(db_url)}")
        
        weak_ref = weakref.ref(store_ref, cleanup_ref)
        self._store_references[db_url].add(weak_ref)
        
        logger.debug(f"GlobalConnectionManager: Added store reference for {self._mask_url(db_url)} "
                    f"(total refs: {len(self._store_references[db_url])})")
    
    async def close_pool(self, db_url: str, force: bool = False):
        """Close a specific connection pool."""
        async with self._async_lock:
            if db_url not in self._pools:
                logger.debug(f"GlobalConnectionManager: Pool for {self._mask_url(db_url)} not found")
                return
            
            # Check if there are active references
            active_refs = self._get_active_references(db_url)
            if active_refs > 0 and not force:
                logger.info(f"GlobalConnectionManager: Not closing pool for {self._mask_url(db_url)} - "
                           f"{active_refs} active references")
                return
            
            logger.info(f"GlobalConnectionManager: Closing pool for {self._mask_url(db_url)}")
            
            pool = self._pools[db_url]
            try:
                await pool.close()
                logger.debug(f"GlobalConnectionManager: Pool closed for {self._mask_url(db_url)}")
            except Exception as e:
                logger.error(f"GlobalConnectionManager: Error closing pool: {e}")
            
            # Clean up tracking data
            del self._pools[db_url]
            if db_url in self._pool_configs:
                del self._pool_configs[db_url]
            if db_url in self._store_references:
                del self._store_references[db_url]
    
    async def close_all_pools(self, force: bool = False):
        """Close all connection pools."""
        logger.info("GlobalConnectionManager: Closing all connection pools...")
        
        # Get list of URLs to avoid modifying dict during iteration
        db_urls = list(self._pools.keys())
        
        for db_url in db_urls:
            await self.close_pool(db_url, force=force)
        
        logger.info("GlobalConnectionManager: All connection pools closed")
    
    def _get_active_references(self, db_url: str) -> int:
        """Get count of active store references for a pool."""
        if db_url not in self._store_references:
            return 0
        
        # Clean up dead references and count active ones
        active_refs = set()
        for ref in self._store_references[db_url]:
            if ref() is not None:  # Reference is still alive
                active_refs.add(ref)
        
        # Update the set with only active references
        self._store_references[db_url] = active_refs
        return len(active_refs)
    
    def get_pool_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get comprehensive statistics for all connection pools."""
        stats = {}
        
        for db_url, pool in self._pools.items():
            config = self._pool_configs.get(db_url, ConnectionPoolConfig())
            active_refs = self._get_active_references(db_url)
            
            stats[self._mask_url(db_url)] = {
                "min_size": config.min_size,
                "max_size": config.max_size,
                "timeout": config.timeout,
                "recycle": config.recycle,
                "active_references": active_refs,
                "pool_closed": getattr(pool, '_closed', False)
            }
        
        return stats
    
    def _mask_url(self, db_url: str) -> str:
        """Mask sensitive information in database URL for logging."""
        try:
            # Simple masking - replace password with ***
            if "://" in db_url and "@" in db_url:
                parts = db_url.split("://")
                if len(parts) == 2:
                    scheme = parts[0]
                    rest = parts[1]
                    if "@" in rest:
                        auth_part, host_part = rest.split("@", 1)
                        if ":" in auth_part:
                            user, _ = auth_part.split(":", 1)
                            return f"{scheme}://{user}:***@{host_part}"
            return db_url
        except Exception:
            return "***masked***"


# Global singleton instance
_global_connection_manager: Optional[GlobalConnectionManager] = None


def get_global_connection_manager() -> GlobalConnectionManager:
    """Get the global connection manager singleton instance."""
    global _global_connection_manager
    if _global_connection_manager is None:
        _global_connection_manager = GlobalConnectionManager.get_instance()
    return _global_connection_manager
