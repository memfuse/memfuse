"""
Test Global Connection Manager Singleton

This test verifies that the GlobalConnectionManager properly implements
the singleton pattern and manages connection pools correctly.
"""

import pytest
import asyncio
import sys
import os
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from memfuse_core.services.global_connection_manager import (
    GlobalConnectionManager, 
    get_global_connection_manager,
    ConnectionPoolConfig
)


class TestGlobalConnectionManager:
    """Test suite for GlobalConnectionManager singleton."""
    
    def test_singleton_pattern(self):
        """Test that GlobalConnectionManager follows singleton pattern."""
        # Get multiple instances
        manager1 = GlobalConnectionManager()
        manager2 = GlobalConnectionManager.get_instance()
        manager3 = get_global_connection_manager()
        
        # All should be the same instance
        assert manager1 is manager2
        assert manager2 is manager3
        assert manager1 is manager3
        
        print("✅ Singleton pattern verified")
    
    def test_connection_pool_config_hierarchy(self):
        """Test configuration hierarchy for connection pools."""
        # Test configuration priority
        config = {
            "postgres": {
                "pool_size": 3,
                "max_overflow": 5,
                "pool_timeout": 20.0
            },
            "database": {
                "postgres": {
                    "pool_size": 7,  # Should override base postgres
                    "pool_timeout": 25.0
                }
            },
            "store": {
                "database": {
                    "postgres": {
                        "pool_size": 10,  # Should have highest priority
                        "pool_recycle": 7200
                    }
                }
            }
        }
        
        pool_config = ConnectionPoolConfig.from_memfuse_config(config)
        
        # Verify highest priority values are used
        assert pool_config.min_size == 10  # From store.database.postgres
        assert pool_config.max_size == 15  # 10 + 5 (max_overflow from database.postgres)
        assert pool_config.timeout == 25.0  # From database.postgres
        assert pool_config.recycle == 7200  # From store.database.postgres
        
        print("✅ Configuration hierarchy working correctly")
    
    @pytest.mark.asyncio
    async def test_connection_pool_sharing(self):
        """Test that multiple stores share the same connection pool."""
        manager = get_global_connection_manager()
        
        # Use existing memfuse database instead of non-existent test database
        db_url = "postgresql://postgres:postgres@localhost:5432/memfuse"
        
        # Mock configuration
        config = {
            "database": {
                "postgres": {
                    "pool_size": 2,
                    "max_overflow": 3,
                    "pool_timeout": 10.0
                }
            }
        }
        
        # Create mock store references
        class MockStore:
            def __init__(self, name):
                self.name = name
        
        store1 = MockStore("store1")
        store2 = MockStore("store2")
        
        try:
            # Get pools for both stores
            pool1 = await manager.get_connection_pool(db_url, config, store1)
            pool2 = await manager.get_connection_pool(db_url, config, store2)
            
            # Should be the same pool instance
            assert pool1 is pool2
            
            # Check statistics
            stats = manager.get_pool_statistics()
            assert len(stats) == 1  # Only one pool
            
            pool_stat = list(stats.values())[0]
            assert pool_stat["active_references"] == 2  # Two store references
            assert pool_stat["min_size"] == 2
            assert pool_stat["max_size"] == 5  # 2 + 3
            
            print("✅ Connection pool sharing verified")
            
        finally:
            # Cleanup
            await manager.close_all_pools(force=True)
    
    @pytest.mark.asyncio
    async def test_connection_pool_cleanup(self):
        """Test that connection pools are properly cleaned up."""
        manager = get_global_connection_manager()
        
        # Use existing memfuse database instead of non-existent test database
        db_url = "postgresql://postgres:postgres@localhost:5432/memfuse"
        
        # Mock configuration
        config = {
            "database": {
                "postgres": {
                    "pool_size": 1,
                    "max_overflow": 1,
                    "pool_timeout": 5.0
                }
            }
        }
        
        # Create mock store reference
        class MockStore:
            def __init__(self, name):
                self.name = name
        
        store = MockStore("test_store")
        
        try:
            # Get pool
            pool = await manager.get_connection_pool(db_url, config, store)
            
            # Verify pool exists
            stats = manager.get_pool_statistics()
            assert len(stats) == 1
            
            # Close all pools
            await manager.close_all_pools(force=True)
            
            # Verify pools are cleaned up
            stats = manager.get_pool_statistics()
            assert len(stats) == 0
            
            print("✅ Connection pool cleanup verified")
            
        except Exception as e:
            # Cleanup on error
            await manager.close_all_pools(force=True)
            raise
    
    def test_url_masking(self):
        """Test that database URLs are properly masked in logs."""
        manager = get_global_connection_manager()
        
        # Test URL with password
        url_with_password = "postgresql://user:secret123@localhost:5432/database"
        masked = manager._mask_url(url_with_password)
        
        assert "secret123" not in masked
        assert "user:***@localhost:5432/database" in masked
        
        # Test URL without password
        url_without_password = "postgresql://user@localhost:5432/database"
        masked = manager._mask_url(url_without_password)
        
        assert masked == url_without_password
        
        print("✅ URL masking working correctly")


if __name__ == "__main__":
    # Run tests directly
    test_instance = TestGlobalConnectionManager()
    
    print("Running GlobalConnectionManager tests...")
    
    test_instance.test_singleton_pattern()
    test_instance.test_connection_pool_config_hierarchy()
    test_instance.test_url_masking()
    
    # Run async tests
    async def run_async_tests():
        await test_instance.test_connection_pool_sharing()
        await test_instance.test_connection_pool_cleanup()
    
    asyncio.run(run_async_tests())
    
    print("✅ All GlobalConnectionManager tests passed!")
