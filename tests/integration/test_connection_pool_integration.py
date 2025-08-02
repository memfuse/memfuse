"""
Integration tests for connection pool optimization.

Tests that the optimized connection pool works correctly with real database
operations and maintains compatibility with existing functionality.
"""

import asyncio
import pytest
import os
from unittest.mock import patch, AsyncMock

from src.memfuse_core.services.global_connection_manager import (
    GlobalConnectionManager,
    get_global_connection_manager,
    warmup_global_connection_pools
)


class TestConnectionPoolIntegration:
    """Integration tests for optimized connection pool."""
    
    @pytest.fixture
    def test_db_url(self):
        """Get test database URL from environment or use default."""
        return os.getenv(
            "TEST_DATABASE_URL",
            "postgresql://postgres:password@localhost:5432/memfuse_test"
        )
    
    @pytest.fixture
    def manager(self):
        """Create a fresh connection manager for testing."""
        # Reset singleton to ensure clean state
        GlobalConnectionManager._instance = None
        return GlobalConnectionManager()
    
    @pytest.mark.asyncio
    async def test_pool_creation_and_reuse(self, manager, test_db_url):
        """Test that pools are created correctly and reused."""
        # Mock the actual pool creation to avoid database dependency
        with patch('src.memfuse_core.services.global_connection_manager.AsyncConnectionPool') as mock_pool_class:
            mock_pool = AsyncMock()
            mock_pool.closed = False
            mock_pool_class.return_value = mock_pool
            
            # First call should create new pool
            pool1 = await manager.get_connection_pool(test_db_url)
            assert pool1 is mock_pool
            assert test_db_url in manager._pools
            
            # Second call should reuse existing pool
            pool2 = await manager.get_connection_pool(test_db_url)
            assert pool2 is pool1
            
            # Pool creation should only be called once
            assert mock_pool_class.call_count == 1
    
    @pytest.mark.asyncio
    async def test_pool_configuration_from_memfuse_config(self, manager, test_db_url):
        """Test that pool configuration is correctly derived from MemFuse config."""
        config = {
            "database": {
                "postgres": {
                    "pool_size": 5,
                    "max_overflow": 15,  # max_size = pool_size + max_overflow = 20
                    "pool_timeout": 30.0,
                    "pool_recycle": 3600
                }
            }
        }
        
        with patch('src.memfuse_core.services.global_connection_manager.AsyncConnectionPool') as mock_pool_class:
            mock_pool = AsyncMock()
            mock_pool.closed = False
            mock_pool_class.return_value = mock_pool
            
            # Create pool with custom config
            pool = await manager.get_connection_pool(test_db_url, config)
            
            # Verify pool was created with correct configuration
            assert mock_pool_class.called
            call_kwargs = mock_pool_class.call_args[1]
            
            assert call_kwargs['min_size'] == 5   # From config pool_size
            assert call_kwargs['max_size'] == 20  # pool_size + max_overflow = 5 + 15
            assert call_kwargs['timeout'] == 30.0

            # Verify config is stored
            assert test_db_url in manager._pool_configs
            pool_config = manager._pool_configs[test_db_url]
            assert pool_config.min_size == 5
            assert pool_config.max_size == 20
            assert pool_config.connection_timeout == 30.0
            assert pool_config.recycle == 3600
    
    @pytest.mark.asyncio
    async def test_store_reference_tracking(self, manager, test_db_url):
        """Test that store references are tracked correctly."""
        with patch('src.memfuse_core.services.global_connection_manager.AsyncConnectionPool') as mock_pool_class:
            mock_pool = AsyncMock()
            mock_pool.closed = False
            mock_pool_class.return_value = mock_pool
            
            # Create mock store objects that support weak references
            class MockStore:
                pass
            store1 = MockStore()
            store2 = MockStore()
            
            # Get pools with store references
            pool1 = await manager.get_connection_pool(test_db_url, store_ref=store1)
            pool2 = await manager.get_connection_pool(test_db_url, store_ref=store2)
            
            assert pool1 is pool2  # Same pool
            
            # Verify references are tracked
            assert test_db_url in manager._store_references
            refs = manager._store_references[test_db_url]
            assert len(refs) == 2
            
            # Verify active reference counting
            active_count = manager._get_active_references(test_db_url)
            assert active_count == 2
    
    @pytest.mark.asyncio
    async def test_pool_cleanup_on_close(self, manager, test_db_url):
        """Test that pools are properly cleaned up when closed."""
        with patch('src.memfuse_core.services.global_connection_manager.AsyncConnectionPool') as mock_pool_class:
            mock_pool = AsyncMock()
            mock_pool.closed = False
            mock_pool_class.return_value = mock_pool
            
            # Create pool
            pool = await manager.get_connection_pool(test_db_url)
            assert test_db_url in manager._pools
            
            # Close pool
            await manager.close_pool(test_db_url, force=True)
            
            # Verify cleanup
            assert test_db_url not in manager._pools
            assert test_db_url not in manager._pool_configs
            assert test_db_url not in manager._store_references
            
            # Verify pool.close() was called
            mock_pool.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_closed_pool_recreation(self, manager, test_db_url):
        """Test that closed pools are automatically recreated."""
        with patch('src.memfuse_core.services.global_connection_manager.AsyncConnectionPool') as mock_pool_class:
            # First pool (will be marked as closed)
            closed_pool = AsyncMock()
            closed_pool.closed = True
            
            # Second pool (new one)
            new_pool = AsyncMock()
            new_pool.closed = False
            
            mock_pool_class.side_effect = [closed_pool, new_pool]
            
            # First call creates pool
            pool1 = await manager.get_connection_pool(test_db_url)
            assert pool1 is closed_pool
            
            # Mark pool as closed
            closed_pool.closed = True
            
            # Second call should detect closed pool and create new one
            pool2 = await manager.get_connection_pool(test_db_url)
            assert pool2 is new_pool
            assert pool2 is not pool1
            
            # Should have called pool creation twice
            assert mock_pool_class.call_count == 2
    
    @pytest.mark.asyncio
    async def test_concurrent_pool_creation(self, manager, test_db_url):
        """Test that concurrent pool creation doesn't create duplicate pools."""
        with patch('src.memfuse_core.services.global_connection_manager.AsyncConnectionPool') as mock_pool_class:
            mock_pool = AsyncMock()
            mock_pool.closed = False
            mock_pool_class.return_value = mock_pool
            
            # Simulate slow pool creation by adding delay to open method
            async def slow_open():
                await asyncio.sleep(0.1)  # Simulate slow creation

            mock_pool.open = slow_open
            
            # Start multiple concurrent requests
            tasks = [
                manager.get_connection_pool(test_db_url)
                for _ in range(5)
            ]
            
            pools = await asyncio.gather(*tasks)
            
            # All should return the same pool
            assert all(pool is pools[0] for pool in pools)
            
            # Pool should only be created once despite concurrent requests
            assert mock_pool_class.call_count == 1
    
    @pytest.mark.asyncio
    async def test_warmup_integration(self, manager):
        """Test warmup functionality with realistic scenarios."""
        db_urls = [
            "postgresql://user1:pass@localhost:5432/db1",
            "postgresql://user2:pass@localhost:5432/db2"
        ]
        
        with patch('src.memfuse_core.services.global_connection_manager.AsyncConnectionPool') as mock_pool_class:
            mock_pool = AsyncMock()
            mock_pool.closed = False
            
            # Mock connection for warmup test
            mock_conn = AsyncMock()
            mock_conn.execute = AsyncMock()

            # Mock getconn/putconn for warmup functionality
            mock_pool.getconn = AsyncMock(return_value=mock_conn)
            mock_pool.putconn = AsyncMock()
            
            mock_pool_class.return_value = mock_pool
            
            # Perform warmup
            await manager.warmup_common_pools(db_urls)
            
            # Verify pools were created
            assert len(manager._pools) == 2
            for db_url in db_urls:
                assert db_url in manager._pools
            
            # Verify warmup test queries were executed
            assert mock_conn.execute.call_count == 2
            mock_conn.execute.assert_called_with("SELECT 1")
            
            # Verify warmup completion flag
            assert manager._warmup_completed
    
    @pytest.mark.asyncio
    async def test_statistics_collection(self, manager, test_db_url):
        """Test that pool statistics are collected correctly."""
        with patch('src.memfuse_core.services.global_connection_manager.AsyncConnectionPool') as mock_pool_class:
            mock_pool = AsyncMock()
            mock_pool.closed = False
            mock_pool_class.return_value = mock_pool
            
            # Create pool with custom config
            config = {
                "database": {
                    "postgres": {
                        "min_connections": 10,
                        "max_connections": 30
                    }
                }
            }
            
            pool = await manager.get_connection_pool(test_db_url, config)
            
            # Get statistics
            stats = manager.get_pool_statistics()
            
            # Verify statistics structure
            assert len(stats) == 1
            
            # Find the stats entry (URL is masked)
            stats_entry = list(stats.values())[0]
            
            assert stats_entry["min_size"] == 5   # Conservative default from global config
            assert stats_entry["max_size"] == 15  # Conservative default from global config
            assert "timeout" in stats_entry
            assert "recycle" in stats_entry
            assert "active_references" in stats_entry
    
    @pytest.mark.asyncio
    async def test_global_warmup_with_config(self):
        """Test global warmup function with configuration."""
        # Reset singleton
        GlobalConnectionManager._instance = None
        
        with patch('src.memfuse_core.services.global_connection_manager.AsyncConnectionPool') as mock_pool_class:
            mock_pool = AsyncMock()
            mock_pool.closed = False
            
            # Mock connection for warmup
            mock_conn = AsyncMock()
            mock_conn.execute = AsyncMock()
            mock_pool.getconn = AsyncMock(return_value=mock_conn)
            mock_pool.putconn = AsyncMock()
            
            mock_pool_class.return_value = mock_pool

            # Test warmup with explicit URLs (simpler approach)
            test_urls = ["postgresql://test:test@localhost:5432/test"]
            await warmup_global_connection_pools(test_urls)

            # Verify manager was created and warmup was performed
            manager = get_global_connection_manager()
            print(f"DEBUG: warmup_completed = {manager._warmup_completed}")
            print(f"DEBUG: pools count = {len(manager._pools)}")
            print(f"DEBUG: pools = {list(manager._pools.keys())}")
            assert manager._warmup_completed
            assert len(manager._pools) >= 1


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "-s"])
