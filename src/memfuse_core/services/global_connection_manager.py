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
- Lock-free optimization: Read-write locks for high concurrency
- Connection warmup: Pre-created pools for common databases
"""

import asyncio
import weakref
import time
from typing import Dict, Optional, Any, Set, List
from loguru import logger
from dataclasses import dataclass
from threading import Lock

from ..monitoring.performance_monitor import get_performance_monitor

try:
    from psycopg_pool import AsyncConnectionPool
    from pgvector.psycopg import register_vector_async
    from urllib.parse import urlparse, urlunparse, urlencode, parse_qs
    PSYCOPG_AVAILABLE = True
except ImportError:
    PSYCOPG_AVAILABLE = False
    AsyncConnectionPool = None
    logger.warning("psycopg dependencies not available for global connection manager")


class AsyncRWLock:
    """Async read-write lock for high-concurrency scenarios."""

    def __init__(self):
        self._readers = 0
        self._writers = 0
        self._read_ready = asyncio.Condition()
        self._write_ready = asyncio.Condition()

    async def acquire_read(self):
        """Acquire read lock."""
        async with self._read_ready:
            while self._writers > 0:
                await self._read_ready.wait()
            self._readers += 1

    async def release_read(self):
        """Release read lock."""
        async with self._read_ready:
            self._readers -= 1
            if self._readers == 0:
                self._read_ready.notify_all()
                async with self._write_ready:
                    self._write_ready.notify_all()

    async def acquire_write(self):
        """Acquire write lock."""
        async with self._write_ready:
            while self._writers > 0 or self._readers > 0:
                await self._write_ready.wait()
            self._writers += 1

    async def release_write(self):
        """Release write lock."""
        async with self._write_ready:
            self._writers -= 1
            self._write_ready.notify_all()
            async with self._read_ready:
                self._read_ready.notify_all()

    def reader(self):
        """Context manager for read lock."""
        return _ReadLockContext(self)

    def writer(self):
        """Context manager for write lock."""
        return _WriteLockContext(self)


class _ReadLockContext:
    def __init__(self, lock: AsyncRWLock):
        self._lock = lock

    async def __aenter__(self):
        await self._lock.acquire_read()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._lock.release_read()


class _WriteLockContext:
    def __init__(self, lock: AsyncRWLock):
        self._lock = lock

    async def __aenter__(self):
        await self._lock.acquire_write()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._lock.release_write()


@dataclass
class ConnectionPoolConfig:
    """Configuration for PostgreSQL connection pools."""
    min_size: int = 2    # Reduced minimum pool size to prevent connection hoarding
    max_size: int = 8    # Reduced maximum pool size for better connection management
    timeout: float = 15.0  # Reduced timeout for faster failure detection
    recycle: int = 1800  # 30 minutes recycle time for more aggressive connection recycling
    connection_timeout: float = 10.0  # Faster connection timeout
    keepalives_idle: int = 300  # Shorter idle time for faster dead connection detection
    keepalives_interval: int = 15  # More frequent keepalive checks
    keepalives_count: int = 3

    @classmethod
    def from_memfuse_config(cls, config: Dict[str, Any]) -> 'ConnectionPoolConfig':
        """Create configuration from MemFuse config hierarchy.

        Configuration priority (highest to lowest):
        1. Environment variables (for test configuration)
        2. store.database.postgres.*
        3. database.postgres.*
        4. postgres.*
        5. Default values
        """
        import os
        
        logger.debug(f"ConnectionPoolConfig.from_memfuse_config: config type={type(config)}")
        postgres_config = {}

        # Layer 1: Base postgres config
        if "postgres" in config and config["postgres"] is not None:
            logger.debug(f"ConnectionPoolConfig: Found base postgres config")
            postgres_config.update(config["postgres"])

        # Layer 2: Database postgres config (higher priority)
        if ("database" in config and
            "postgres" in config["database"] and
            config["database"]["postgres"] is not None):
            logger.debug(f"ConnectionPoolConfig: Found database.postgres config")
            postgres_config.update(config["database"]["postgres"])

        # Layer 3: Store database postgres config (highest priority)
        if ("store" in config and
            "database" in config["store"] and
            "postgres" in config["store"]["database"] and
            config["store"]["database"]["postgres"] is not None):
            logger.debug(f"ConnectionPoolConfig: Found store.database.postgres config")
            postgres_config.update(config["store"]["database"]["postgres"])
        
        # Highest priority: Environment variables (for test configuration)
        # These override all config file settings
        env_pool_size = os.environ.get("POSTGRES_POOL_SIZE")
        env_max_overflow = os.environ.get("POSTGRES_MAX_OVERFLOW")
        env_pool_timeout = os.environ.get("POSTGRES_POOL_TIMEOUT")
        env_connection_timeout = os.environ.get("POSTGRES_CONNECTION_TIMEOUT")
        
        # Extract values with environment variable support
        pool_size = int(env_pool_size) if env_pool_size else postgres_config.get("pool_size", 5)
        max_overflow = int(env_max_overflow) if env_max_overflow else postgres_config.get("max_overflow", 15)
        max_size = pool_size + max_overflow
        
        # Use environment timeout if available, otherwise use config
        pool_timeout = float(env_pool_timeout) if env_pool_timeout else postgres_config.get("pool_timeout", 300.0)
        connection_timeout = float(env_connection_timeout) if env_connection_timeout else postgres_config.get("connection_timeout", 60.0)
        
        # Debug logging
        logger.debug(f"ConnectionPoolConfig: pool_size={pool_size}, max_overflow={max_overflow}, max_size={max_size}")
        logger.debug(f"ConnectionPoolConfig: pool_timeout={pool_timeout}, connection_timeout={connection_timeout}")
        logger.debug(f"ConnectionPoolConfig: env_pool_timeout={env_pool_timeout}, env_connection_timeout={env_connection_timeout}")
        logger.debug(f"ConnectionPoolConfig: postgres_config pool_timeout={postgres_config.get('pool_timeout')}, connection_timeout={postgres_config.get('connection_timeout')}")
        
        logger.debug(f"ConnectionPoolConfig: min_size={pool_size}, max_size={max_size}, "
                    f"timeout={pool_timeout}, "
                    f"recycle={postgres_config.get('pool_recycle', 3600)}")
        
        return cls(
            min_size=pool_size,
            max_size=max_size,
            timeout=pool_timeout,
            recycle=postgres_config.get("pool_recycle", 7200),
            connection_timeout=connection_timeout,
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
    - Lock-free high-concurrency operations with read-write locks
    - Connection pool warmup for common databases
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
        self._warm_pools: Dict[str, AsyncConnectionPool] = {}  # Pre-warmed pools
        self._rw_lock = AsyncRWLock()  # Replace single async lock with read-write lock
        self._initialized = False
        self._warmup_completed = False

        # P4 OPTIMIZATION: Health check optimization
        self._health_check_cache: Dict[str, float] = {}  # db_url -> last_check_time
        self._health_check_interval = 300.0  # 5 minutes between health checks
        self._performance_monitor = get_performance_monitor()
        
        # Monitoring disabled for simplicity

        logger.info("GlobalConnectionManager: Optimized singleton instance created")
    
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

        This method implements optimized connection pool access:
        - Lock-free fast path for existing pools (read lock only)
        - Write lock only when creating new pools
        - Connection pool warmup for common databases
        - Automatic reference tracking for cleanup
        - P4 OPTIMIZATION: Performance monitoring and health check optimization

        Args:
            db_url: Database connection URL
            config: MemFuse configuration dictionary
            store_ref: Reference to the store using this pool

        Returns:
            Shared AsyncConnectionPool instance
        """
        # P4 OPTIMIZATION: Track connection pool access performance
        async with self._performance_monitor.track_operation("connection_pool_access", {"db_url": self._mask_url(db_url)}):
            # Fast path: Check existing pools with read lock only
            async with self._rw_lock.reader():
                if db_url in self._pools:
                    pool = self._pools[db_url]
                    try:
                        if not pool.closed:
                            # Perform sophisticated health check
                            if await self.is_pool_healthy(pool, db_url):
                                # Add store reference for tracking (this is safe under read lock)
                                if store_ref is not None:
                                    self._add_store_reference_safe(db_url, store_ref)

                                logger.debug(f"GlobalConnectionManager: Fast path - reusing healthy pool for {self._mask_url(db_url)}")
                                return pool
                            else:
                                # Pool is unhealthy, fall through to slow path for recreation
                                logger.debug(f"GlobalConnectionManager: Pool unhealthy, will recreate for {self._mask_url(db_url)}")
                        else:
                            # Pool is closed, fall through to slow path
                            logger.debug(f"GlobalConnectionManager: Pool closed, will recreate for {self._mask_url(db_url)}")
                    except Exception as e:
                        logger.debug(f"GlobalConnectionManager: Pool health check failed, will recreate: {e}")
                        # Fall through to slow path

            # Slow path: Create or recreate pool with write lock
            async with self._rw_lock.writer():
                # Double-check under write lock (pool might have been created by another thread)
                if db_url in self._pools:
                    pool = self._pools[db_url]
                    try:
                        if not pool.closed:
                            # Perform sophisticated health check
                            if await self.is_pool_healthy(pool, db_url):
                                # Add store reference for tracking
                                if store_ref is not None:
                                    self._add_store_reference(db_url, store_ref)

                                logger.debug(f"GlobalConnectionManager: Double-check - reusing healthy pool for {self._mask_url(db_url)}")
                                return pool
                            else:
                                # Pool is unhealthy, remove it
                                logger.warning(f"GlobalConnectionManager: Removing unhealthy pool for {self._mask_url(db_url)}")
                                del self._pools[db_url]
                                if db_url in self._store_references:
                                    del self._store_references[db_url]
                        else:
                            # Pool is closed, remove it
                            logger.warning(f"GlobalConnectionManager: Removing closed pool for {self._mask_url(db_url)}")
                            del self._pools[db_url]
                            if db_url in self._store_references:
                                del self._store_references[db_url]
                    except Exception as e:
                        logger.warning(f"GlobalConnectionManager: Error checking pool health: {e}")
                        # Don't immediately remove pool on health check errors
                        # Only remove if pool is structurally broken (closed)
                        if pool.closed:
                            logger.warning(f"GlobalConnectionManager: Pool is closed, removing for {self._mask_url(db_url)}")
                            try:
                                await pool.close()
                            except Exception:
                                pass
                            if db_url in self._pools:
                                del self._pools[db_url]
                            if db_url in self._store_references:
                                del self._store_references[db_url]
                        else:
                            logger.info(f"GlobalConnectionManager: Pool structure OK, keeping pool despite health check error for {self._mask_url(db_url)}")
                            # Add store reference even if health check failed - pool structure is OK
                            if store_ref is not None:
                                self._add_store_reference(db_url, store_ref)
                            return pool

                # Create new pool
                logger.info(f"GlobalConnectionManager: Creating new pool for {self._mask_url(db_url)}")

                # Get pool configuration from MemFuse config
                if config is None or config == {}:
                    # Try to get config from global config manager
                    try:
                        from ..utils.config import config_manager
                        global_config = config_manager.get_config()
                        if global_config and global_config != {}:
                            config = global_config
                        else:
                            # Try to load config directly with Hydra
                            try:
                                import hydra
                                from hydra import compose, initialize
                                import os
                                # Use absolute path to config directory
                                config_path = os.path.join(os.getcwd(), 'config')
                                if os.path.exists(config_path):
                                    initialize(version_base=None, config_path='config')
                                    cfg = compose(config_name='config')
                                    config = cfg
                                    logger.info("Loaded config directly with Hydra")
                                else:
                                    logger.warning(f"Config directory not found: {config_path}")
                                    config = {}
                            except Exception as e2:
                                logger.warning(f"Could not load config with Hydra: {e2}")
                                config = {}
                    except Exception as e:
                        logger.warning(f"Could not get global config: {e}")
                        config = {}

                logger.info(f"GlobalConnectionManager: Creating pool config from config type={type(config)}")
                logger.info(f"GlobalConnectionManager: Raw config keys: {list(config.keys()) if isinstance(config, dict) else 'Not a dict'}")
                pool_config = ConnectionPoolConfig.from_memfuse_config(config)
                logger.info(f"GlobalConnectionManager: Pool config created: min_size={pool_config.min_size}, max_size={pool_config.max_size}, timeout={pool_config.timeout}, connection_timeout={pool_config.connection_timeout}")
                self._pool_configs[db_url] = pool_config

                # Create the pool with enhanced connection parameters
                # Add keepalive parameters to the connection URL
                enhanced_db_url = self._enhance_db_url_with_keepalives(db_url, pool_config)
                
                # For debugging, use the raw URL to check if URL enhancement is causing issues
                logger.info(f"GlobalConnectionManager: Creating pool with URL: {self._mask_url(db_url)}")
                logger.info(f"GlobalConnectionManager: Enhanced URL: {self._mask_url(enhanced_db_url)}")

                pool = AsyncConnectionPool(
                    db_url,  # Use raw URL for debugging
                    min_size=pool_config.min_size,
                    max_size=pool_config.max_size,
                    open=False,
                    configure=self._configure_connection,
                    timeout=min(pool_config.timeout, 30.0)  # Cap timeout to 30 seconds for faster failure
                )

                # Open the pool with reduced timeout for faster failure
                try:
                    open_timeout = min(pool_config.timeout, 30.0)  # Cap to 30 seconds
                    logger.info(f"GlobalConnectionManager: Opening pool with timeout={open_timeout} seconds")
                    await asyncio.wait_for(pool.open(), timeout=open_timeout)
                    logger.info(f"GlobalConnectionManager: Pool opened successfully "
                               f"(min={pool_config.min_size}, max={pool_config.max_size})")
                except asyncio.TimeoutError:
                    logger.error(f"GlobalConnectionManager: Pool opening timed out for {self._mask_url(db_url)} after {open_timeout} seconds")
                    # Try to close the pool to clean up resources
                    try:
                        await pool.close()
                    except Exception:
                        pass
                    raise
                except Exception as e:
                    logger.error(f"GlobalConnectionManager: Failed to open pool: {e}")
                    # Try to close the pool to clean up resources
                    try:
                        await pool.close()
                    except Exception:
                        pass
                    raise

                # Store the pool
                self._pools[db_url] = pool
                self._store_references[db_url] = set()

                # Add store reference for tracking
                if store_ref is not None:
                    self._add_store_reference(db_url, store_ref)

                # Smart monitoring disabled

                return pool

    async def warmup_common_pools(self, common_db_urls: Optional[List[str]] = None):
        """Pre-warm connection pools for common database URLs."""
        if self._warmup_completed:
            logger.debug("GlobalConnectionManager: Pool warmup already completed")
            return

        if common_db_urls is None:
            # Try to get common URLs from config
            try:
                from ..utils.config import config_manager
                config = config_manager.get_config() or {}
                database_config = config.get("database", {})

                # Build common URLs from config
                common_db_urls = []
                if "url" in database_config:
                    common_db_urls.append(database_config["url"])

                # Add any additional common patterns
                postgres_config = database_config.get("postgres", {})
                if postgres_config:
                    host = postgres_config.get("host", "localhost")
                    port = postgres_config.get("port", 5432)
                    database = postgres_config.get("database", "memfuse")
                    user = postgres_config.get("user", "postgres")
                    password = postgres_config.get("password", "")

                    if password:
                        url = f"postgresql://{user}:{password}@{host}:{port}/{database}"
                    else:
                        url = f"postgresql://{user}@{host}:{port}/{database}"

                    if url not in common_db_urls:
                        common_db_urls.append(url)

            except Exception as e:
                logger.warning(f"GlobalConnectionManager: Could not determine common URLs from config: {e}")
                common_db_urls = []

        if not common_db_urls:
            logger.info("GlobalConnectionManager: No common database URLs to warm up")
            self._warmup_completed = True
            return

        logger.info(f"GlobalConnectionManager: Starting warmup for {len(common_db_urls)} database URLs")

        warmup_tasks = []
        for db_url in common_db_urls:
            task = asyncio.create_task(self._warmup_single_pool(db_url))
            warmup_tasks.append(task)

        # Wait for all warmup tasks to complete
        results = await asyncio.gather(*warmup_tasks, return_exceptions=True)

        successful = sum(1 for r in results if not isinstance(r, Exception))
        failed = len(results) - successful

        logger.info(f"GlobalConnectionManager: Pool warmup completed - {successful} successful, {failed} failed")
        self._warmup_completed = True

    async def _warmup_single_pool(self, db_url: str):
        """Warm up a single connection pool."""
        try:
            logger.debug(f"GlobalConnectionManager: Warming up pool for {self._mask_url(db_url)}")

            # Create the pool without store reference
            pool = await self.get_connection_pool(db_url)

            # Test the pool with a simple query
            conn = await pool.getconn()
            try:
                await conn.execute("SELECT 1")
                logger.debug(f"GlobalConnectionManager: Pool warmup successful for {self._mask_url(db_url)}")
            finally:
                await pool.putconn(conn)

        except Exception as e:
            logger.warning(f"GlobalConnectionManager: Pool warmup failed for {self._mask_url(db_url)}: {e}")
            raise

    def _enhance_db_url_with_keepalives(self, db_url: str, config: ConnectionPoolConfig) -> str:
        """Add keepalive parameters to database URL for better connection stability."""
        try:
            from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

            parsed = urlparse(db_url)
            query_params = parse_qs(parsed.query or "")

            # Add keepalive parameters (only valid ones)
            query_params['keepalives_idle'] = [str(config.keepalives_idle)]
            query_params['keepalives_interval'] = [str(config.keepalives_interval)]
            query_params['keepalives_count'] = [str(config.keepalives_count)]
            
            # Note: connect_timeout is not a valid URL parameter for psycopg
            # It should be set in the connection pool configuration instead

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
        """
        Configure a new connection with pgvector support.

        P4 OPTIMIZATION: Optimized health check with caching to reduce overhead.
        """
        # P4 OPTIMIZATION: Check if we need to perform health check
        conn_info = str(conn.info.dsn) if hasattr(conn, 'info') and hasattr(conn.info, 'dsn') else "unknown"
        current_time = time.time()

        # Check if we've recently performed health check for this connection type
        last_check = self._health_check_cache.get(conn_info, 0)
        if current_time - last_check < self._health_check_interval:
            logger.debug(f"GlobalConnectionManager: Skipping health check for {conn_info} (cached)")
            return

        try:
            # Perform pgvector registration with performance tracking
            async with self._performance_monitor.track_operation("pgvector_registration", {"conn_info": conn_info}):
                await register_vector_async(conn)

            # Update health check cache
            self._health_check_cache[conn_info] = current_time
            logger.debug(f"GlobalConnectionManager: pgvector registered on connection {conn_info}")
        except Exception as e:
            logger.warning(f"GlobalConnectionManager: Failed to register pgvector on {conn_info}: {e}")
    
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

    def _add_store_reference_safe(self, db_url: str, store_ref: Any):
        """Thread-safe version of _add_store_reference for use under read locks."""
        try:
            # Initialize the set if it doesn't exist (this is thread-safe for dict access)
            if db_url not in self._store_references:
                # Use a lock-free approach: create the set if missing
                # This is safe because dict access is atomic in Python
                if db_url not in self._store_references:
                    self._store_references[db_url] = set()

            # Create weak reference with cleanup callback
            def cleanup_ref(ref):
                # This cleanup will be called later and doesn't need immediate consistency
                if db_url in self._store_references:
                    self._store_references[db_url].discard(ref)
                    logger.debug(f"GlobalConnectionManager: Cleaned up store reference for {self._mask_url(db_url)}")

            weak_ref = weakref.ref(store_ref, cleanup_ref)

            # Add the reference to the set (set.add is thread-safe)
            self._store_references[db_url].add(weak_ref)

            logger.debug(f"GlobalConnectionManager: Added store reference for {self._mask_url(db_url)} "
                        f"(total refs: {len(self._store_references[db_url])})")

        except Exception as e:
            logger.debug(f"GlobalConnectionManager: Could not add store reference safely: {e}")
            # This is non-critical, so we continue
    
    async def close_pool(self, db_url: str, force: bool = False):
        """Close a specific connection pool."""
        async with self._rw_lock.writer():
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
        async with self._rw_lock.reader():
            db_urls = list(self._pools.keys())

        for db_url in db_urls:
            await self.close_pool(db_url, force=force)

        logger.info("GlobalConnectionManager: All connection pools closed")
    
    async def reset_all_pools(self):
        """Force reset all connection pools and clear caches.
        
        This method is useful for forcing recreation of connection pools
        when configuration changes have been made.
        """
        logger.info("GlobalConnectionManager: Resetting all connection pools...")
        
        # Close all pools first
        await self.close_all_pools(force=True)
        
        # Clear all caches
        self._health_check_cache.clear()
        self._pool_configs.clear()
        self._store_references.clear()
        self._warm_pools.clear()
        
        # Reset initialization flags
        self._warmup_completed = False
        
        logger.info("GlobalConnectionManager: All connection pools reset successfully")
    
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
    
    async def is_pool_healthy(self, pool: AsyncConnectionPool, db_url: str) -> bool:
        """Perform a sophisticated health check on a connection pool.
        
        This method implements a more sophisticated health check that:
        1. Uses cached results to avoid excessive checking
        2. Tests actual connectivity with a simple query
        3. Handles temporary failures gracefully
        4. Provides detailed logging for diagnostics
        
        Args:
            pool: The connection pool to check
            db_url: The database URL for logging and caching
            
        Returns:
            True if the pool is healthy, False otherwise
        """
        current_time = time.time()
        
        # Check if we have a recent cached health check result
        if db_url in self._health_check_cache:
            last_check = self._health_check_cache[db_url]
            if current_time - last_check < self._health_check_interval:
                # Use cached result - pool was recently healthy
                logger.debug(f"GlobalConnectionManager: Using cached health check for {self._mask_url(db_url)}")
                return True
        
        try:
            # Perform actual connectivity test
            async with pool.connection() as conn:
                await conn.execute("SELECT 1")
                
            # Update health check cache
            self._health_check_cache[db_url] = current_time
            logger.debug(f"GlobalConnectionManager: Health check passed for {self._mask_url(db_url)}")
            return True
            
        except Exception as e:
            logger.warning(f"GlobalConnectionManager: Health check failed for {self._mask_url(db_url)}: {e}")
            
            # Don't immediately fail - check if this is a temporary issue
            if "connection" in str(e).lower() or "timeout" in str(e).lower():
                # For connection issues, give it a chance to recover
                logger.info(f"GlobalConnectionManager: Connection issue detected, will retry on next access for {self._mask_url(db_url)}")
                # Don't update cache - will retry on next access
                return False
            else:
                # For other errors, clear cache and mark as unhealthy
                if db_url in self._health_check_cache:
                    del self._health_check_cache[db_url]
                return False
    
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


async def warmup_global_connection_pools(common_db_urls: Optional[List[str]] = None):
    """Warm up global connection pools for better performance.

    This function should be called during application startup to pre-create
    connection pools for common database URLs, reducing latency for the first
    database operations.

    Args:
        common_db_urls: List of database URLs to warm up. If None, will try
                       to determine from configuration.
    """
    manager = get_global_connection_manager()
    await manager.warmup_common_pools(common_db_urls)


async def reset_global_connection_pools():
    """Reset all global connection pools and clear caches.
    
    This function is useful for forcing recreation of connection pools
    when configuration changes have been made. It will close all existing
    pools and clear all caches, ensuring that new pools will be created
    with the latest configuration.
    """
    manager = get_global_connection_manager()
    await manager.reset_all_pools()
