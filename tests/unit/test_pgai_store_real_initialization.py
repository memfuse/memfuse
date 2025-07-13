#!/usr/bin/env python3
"""
Real PgaiStore initialization tests that don't mock the connection pool.

These tests are designed to catch the actual connection pool hanging issue
that was missed by existing mocked tests.
"""

import pytest
import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import patch, AsyncMock
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from memfuse_core.store.pgai_store.pgai_store import PgaiStore

class TestPgaiStoreRealInitialization:
    """Test PgaiStore initialization without mocking connection pool."""
    
    @pytest.fixture
    def real_db_config(self):
        """Real database configuration for testing."""
        return {
            "postgres": {
                "host": "localhost",
                "port": 5432,
                "database": "memfuse",
                "user": "postgres",
                "password": "postgres",
                "pool_size": 2,
                "max_overflow": 1
            },
            "pgai": {
                "immediate_trigger": True,
                "auto_embedding": True
            }
        }
    
    @pytest.fixture
    def mock_db_config(self):
        """Mock database configuration that should fail."""
        return {
            "postgres": {
                "host": "nonexistent-host",
                "port": 9999,
                "database": "nonexistent",
                "user": "fake",
                "password": "fake"
            }
        }
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not os.getenv('POSTGRES_AVAILABLE'), reason="PostgreSQL not available")
    async def test_real_initialization_success(self, real_db_config):
        """Test that PgaiStore can initialize with real database."""
        store = None
        try:
            # Create store with real config
            store = PgaiStore(table_name="test_real_init", config=real_db_config)
            
            # Test initialization with timeout
            start_time = time.time()
            success = await asyncio.wait_for(store.initialize(), timeout=60.0)
            end_time = time.time()
            
            # Verify results
            assert success, "Store initialization should succeed"
            assert store.initialized, "Store should be marked as initialized"
            assert store.pool is not None, "Connection pool should be created"
            
            # Verify initialization time is reasonable (not hanging)
            init_time = end_time - start_time
            assert init_time < 45.0, f"Initialization took too long: {init_time}s"
            
            print(f"✅ Real initialization succeeded in {init_time:.2f}s")
            
        except asyncio.TimeoutError:
            pytest.fail("Store initialization timed out - likely hanging on connection pool")
        except Exception as e:
            pytest.fail(f"Store initialization failed with error: {e}")
        finally:
            if store and store.pool:
                try:
                    await store.pool.close()
                except:
                    pass
    
    @pytest.mark.asyncio
    async def test_initialization_with_invalid_config(self, mock_db_config):
        """Test that initialization fails gracefully with invalid config."""
        store = None
        try:
            store = PgaiStore(table_name="test_invalid_config", config=mock_db_config)
            
            # This should fail quickly, not hang
            start_time = time.time()
            success = await asyncio.wait_for(store.initialize(), timeout=30.0)
            end_time = time.time()
            
            # Should fail but not hang
            assert not success, "Initialization should fail with invalid config"
            assert not store.initialized, "Store should not be marked as initialized"
            
            # Should fail quickly
            fail_time = end_time - start_time
            assert fail_time < 25.0, f"Failure took too long: {fail_time}s"
            
            print(f"✅ Invalid config failed gracefully in {fail_time:.2f}s")
            
        except asyncio.TimeoutError:
            pytest.fail("Store initialization timed out even with invalid config")
        finally:
            if store and store.pool:
                try:
                    await store.pool.close()
                except:
                    pass
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not os.getenv('POSTGRES_AVAILABLE'), reason="PostgreSQL not available")
    async def test_concurrent_initialization(self, real_db_config):
        """Test that multiple stores can initialize concurrently without deadlock."""
        stores = []
        try:
            # Create multiple stores
            store_configs = [
                (f"test_concurrent_{i}", real_db_config) 
                for i in range(3)
            ]
            
            # Initialize all stores concurrently
            start_time = time.time()
            
            async def init_store(table_name, config):
                store = PgaiStore(table_name=table_name, config=config)
                success = await store.initialize()
                return store, success
            
            # Run concurrent initialization
            results = await asyncio.gather(
                *[init_store(name, config) for name, config in store_configs],
                return_exceptions=True
            )
            
            end_time = time.time()
            init_time = end_time - start_time
            
            # Verify all succeeded
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    pytest.fail(f"Store {i} failed with exception: {result}")
                
                store, success = result
                stores.append(store)
                assert success, f"Store {i} initialization should succeed"
                assert store.initialized, f"Store {i} should be marked as initialized"
            
            # Should complete in reasonable time
            assert init_time < 90.0, f"Concurrent initialization took too long: {init_time}s"
            
            print(f"✅ Concurrent initialization of {len(stores)} stores succeeded in {init_time:.2f}s")
            
        except asyncio.TimeoutError:
            pytest.fail("Concurrent initialization timed out - likely deadlock")
        finally:
            # Cleanup all stores
            for store in stores:
                if store and store.pool:
                    try:
                        await store.pool.close()
                    except:
                        pass
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not os.getenv('POSTGRES_AVAILABLE'), reason="PostgreSQL not available")
    async def test_connection_pool_configuration(self, real_db_config):
        """Test that connection pool is configured correctly."""
        store = None
        try:
            # Test with specific pool configuration
            config = real_db_config.copy()
            config["postgres"]["pool_size"] = 1
            config["postgres"]["max_overflow"] = 0
            
            store = PgaiStore(table_name="test_pool_config", config=config)
            success = await asyncio.wait_for(store.initialize(), timeout=45.0)
            
            assert success, "Store initialization should succeed"
            assert store.pool is not None, "Connection pool should be created"
            
            # Test basic pool operations
            async with store.pool.connection() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute("SELECT 1")
                    result = await cursor.fetchone()
                    assert result == (1,), "Basic query should work"
            
            print("✅ Connection pool configuration test passed")
            
        except asyncio.TimeoutError:
            pytest.fail("Connection pool test timed out")
        finally:
            if store and store.pool:
                try:
                    await store.pool.close()
                except:
                    pass
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not os.getenv('POSTGRES_AVAILABLE'), reason="PostgreSQL not available")
    async def test_pgvector_registration(self, real_db_config):
        """Test that pgvector registration works correctly."""
        store = None
        try:
            store = PgaiStore(table_name="test_pgvector", config=real_db_config)
            success = await asyncio.wait_for(store.initialize(), timeout=45.0)
            
            assert success, "Store initialization should succeed"
            
            # Test that vector operations work
            async with store.pool.connection() as conn:
                async with conn.cursor() as cursor:
                    # Test vector type is available
                    await cursor.execute("SELECT '[1,2,3]'::vector")
                    result = await cursor.fetchone()
                    assert result is not None, "Vector type should be available"
            
            print("✅ pgvector registration test passed")
            
        except asyncio.TimeoutError:
            pytest.fail("pgvector registration test timed out")
        finally:
            if store and store.pool:
                try:
                    await store.pool.close()
                except:
                    pass


if __name__ == "__main__":
    # Set environment variable for testing
    os.environ['POSTGRES_AVAILABLE'] = '1'
    
    # Run tests
    pytest.main([__file__, "-v", "-s"])
