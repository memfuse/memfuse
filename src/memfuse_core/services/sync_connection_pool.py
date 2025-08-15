"""
Synchronous Database Connection Pool for MemFuse

This module provides a thread-safe connection pool to handle high-concurrency database operations
while avoiding connection exhaustion issues.
"""

import threading
import time
from typing import Dict, Any, Optional
from contextlib import contextmanager
from loguru import logger

import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from psycopg2.extras import RealDictCursor


class SyncConnectionPool:
    """Thread-safe synchronous database connection pool."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        
        self.pool = None
        self.db_config = None
        self._initialized = False
        self._stats = {
            'connections_created': 0,
            'connections_used': 0,
            'errors': 0,
            'lock_timeouts': 0
        }
    
    def initialize(self, db_config: Dict[str, Any]) -> None:
        """Initialize the connection pool."""
        if self._initialized:
            return
            
        with SyncConnectionPool._lock:
            if self._initialized:
                return
                
            self.db_config = db_config.copy()
            
            try:
                # Create connection pool with optimized settings
                self.pool = ThreadedConnectionPool(
                    minconn=5,  # Minimum connections
                    maxconn=50,  # Maximum connections
                    host=db_config.get('host', 'localhost'),
                    port=db_config.get('port', 5432),
                    database=db_config.get('database', 'memfuse'),
                    user=db_config.get('user', 'postgres'),
                    password=db_config.get('password', 'postgres'),
                    # Connection-level optimizations
                    options='-c lock_timeout=30s -c statement_timeout=60s -c idle_in_transaction_session_timeout=300s'
                )
                
                self._initialized = True
                logger.info(f"✅ SyncConnectionPool initialized: 5-50 connections")
                
                # Test the pool
                self._test_pool()
                
            except Exception as e:
                logger.error(f"❌ Failed to initialize SyncConnectionPool: {e}")
                raise
    
    def _test_pool(self) -> None:
        """Test the connection pool."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1 as test_value")
                    result = cur.fetchone()
                    # Simple tuple access
                    if result and result[0] == 1:
                        logger.info("✅ SyncConnectionPool test successful")
                    else:
                        raise Exception(f"Pool test failed: got {result}")
        except Exception as e:
            logger.error(f"❌ SyncConnectionPool test failed: {e}")
            raise
    
    @contextmanager
    def get_connection(self):
        """Get a connection from the pool with proper error handling."""
        if not self._initialized:
            raise RuntimeError("SyncConnectionPool not initialized")
        
        conn = None
        start_time = time.time()
        
        try:
            # Get connection from pool with timeout
            conn = self.pool.getconn()
            if conn is None:
                raise Exception("Failed to get connection from pool")
            
            # Set connection-specific parameters for better concurrency
            with conn.cursor() as cur:
                cur.execute("SET lock_timeout = '30s'")
                cur.execute("SET statement_timeout = '60s'")
                cur.execute("SET idle_in_transaction_session_timeout = '300s'")
            
            self._stats['connections_used'] += 1
            yield conn
            
        except Exception as e:
            self._stats['errors'] += 1
            if 'lock timeout' in str(e).lower():
                self._stats['lock_timeouts'] += 1
                logger.warning(f"🔒 Lock timeout detected: {e}")
            else:
                logger.error(f"❌ Database connection error: {e}")
            
            # Rollback on error
            if conn:
                try:
                    conn.rollback()
                except:
                    pass
            raise
        finally:
            if conn:
                try:
                    # Return connection to pool
                    self.pool.putconn(conn)
                    elapsed = time.time() - start_time
                    if elapsed > 5.0:  # Log slow operations
                        logger.warning(f"⚠️ Slow database operation: {elapsed:.2f}s")
                except Exception as e:
                    logger.error(f"❌ Error returning connection to pool: {e}")
    
    def execute_batch(self, query: str, args_list: list, max_retries: int = 3) -> None:
        """Execute multiple queries in batches with retry logic."""
        retry_delay = 0.1
        
        for attempt in range(max_retries):
            try:
                with self.get_connection() as conn:
                    with conn.cursor() as cur:
                        # Process in smaller batches to reduce lock time
                        batch_size = 5
                        for i in range(0, len(args_list), batch_size):
                            batch = args_list[i:i + batch_size]
                            for args in batch:
                                cur.execute(query, args)
                            # Commit each small batch
                            conn.commit()
                    return
                    
            except Exception as e:
                if 'lock timeout' in str(e).lower() and attempt < max_retries - 1:
                    logger.warning(f"🔄 Retrying batch operation after lock timeout (attempt {attempt + 1})")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                else:
                    raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        pool_stats = {}
        if self.pool:
            # Note: ThreadedConnectionPool doesn't expose detailed stats
            # We track basic usage statistics
            pool_stats = {
                'min_connections': 5,
                'max_connections': 50,
                'pool_type': 'ThreadedConnectionPool'
            }
        
        return {
            **self._stats,
            **pool_stats,
            'initialized': self._initialized
        }
    
    def close(self) -> None:
        """Close the connection pool."""
        if self.pool:
            try:
                self.pool.closeall()
                logger.info("✅ SyncConnectionPool closed")
            except Exception as e:
                logger.warning(f"⚠️ Error closing connection pool: {e}")
            finally:
                self.pool = None
                self._initialized = False


# Global instance
sync_connection_pool = SyncConnectionPool()
