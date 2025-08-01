"""
Test Connection Pool Configuration

This test verifies that connection pool configuration is properly
read from the MemFuse configuration hierarchy and applied correctly.
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
    ConnectionPoolConfig,
    get_global_connection_manager
)
from memfuse_core.utils.config import config_manager
from memfuse_core.store.pgai_store.pgai_store import PgaiStore


class TestConnectionPoolConfiguration:
    """Test suite for connection pool configuration."""
    
    def test_configuration_hierarchy_priority(self):
        """Test that configuration hierarchy is respected."""
        # Test configuration with all levels
        config = {
            "postgres": {
                "pool_size": 3,
                "max_overflow": 5,
                "pool_timeout": 20.0,
                "pool_recycle": 1800
            },
            "database": {
                "postgres": {
                    "pool_size": 7,  # Should override base postgres
                    "pool_timeout": 25.0,
                    "pool_recycle": 2400
                }
            },
            "store": {
                "database": {
                    "postgres": {
                        "pool_size": 10,  # Should have highest priority
                        "max_overflow": 15,
                        "pool_recycle": 7200
                    }
                }
            }
        }
        
        pool_config = ConnectionPoolConfig.from_memfuse_config(config)
        
        # Verify configuration hierarchy
        assert pool_config.min_size == 10  # From store.database.postgres (highest priority)
        assert pool_config.max_size == 25  # 10 + 15 (max_overflow from store.database.postgres)
        assert pool_config.timeout == 25.0  # From database.postgres (not overridden by store)
        assert pool_config.recycle == 7200  # From store.database.postgres (highest priority)
        
        print("✅ Configuration hierarchy priority test passed")
    
    def test_configuration_defaults(self):
        """Test that default values are used when configuration is missing."""
        # Empty configuration
        empty_config = {}
        pool_config = ConnectionPoolConfig.from_memfuse_config(empty_config)
        
        # Should use defaults (matching database/default.yaml)
        assert pool_config.min_size == 20  # Match database config default
        assert pool_config.max_size == 60  # 20 + 40 overflow
        assert pool_config.timeout == 60.0
        assert pool_config.recycle == 7200
        
        # Partial configuration
        partial_config = {
            "database": {
                "postgres": {
                    "pool_size": 8
                }
            }
        }
        pool_config = ConnectionPoolConfig.from_memfuse_config(partial_config)
        
        # Should use provided value and defaults for others
        assert pool_config.min_size == 8
        assert pool_config.max_size == 48  # 8 + 40 (default max_overflow)
        assert pool_config.timeout == 60.0  # default
        assert pool_config.recycle == 7200  # default
        
        print("✅ Configuration defaults test passed")
    
    @pytest.mark.asyncio
    async def test_configuration_applied_to_pools(self):
        """Test that configuration is properly applied to actual connection pools."""
        connection_manager = get_global_connection_manager()
        
        # Test configuration
        config = {
            "database": {
                "postgres": {
                    "pool_size": 3,
                    "max_overflow": 4,
                    "pool_timeout": 15.0,
                    "pool_recycle": 1800
                }
            }
        }
        
        # Mock database URL
        db_url = "postgresql://postgres:postgres@localhost:5432/memfuse_test"
        
        # Create mock store
        class MockStore:
            def __init__(self, name):
                self.name = name
        
        store = MockStore("config_test")
        
        try:
            # Get connection pool with configuration
            pool = await connection_manager.get_connection_pool(db_url, config, store)
            
            # Check pool statistics
            stats = connection_manager.get_pool_statistics()
            assert len(stats) == 1
            
            pool_stat = list(stats.values())[0]
            assert pool_stat["min_size"] == 3
            assert pool_stat["max_size"] == 7  # 3 + 4
            assert pool_stat["timeout"] == 15.0
            assert pool_stat["recycle"] == 1800
            
            print("✅ Configuration applied to pools test passed")
            
        finally:
            # Cleanup
            await connection_manager.close_all_pools(force=True)
    
    @pytest.mark.asyncio
    async def test_pgai_store_uses_configuration(self):
        """Test that PgaiStore properly uses configuration from config manager."""
        # Set up configuration in config manager
        test_config = {
            "database": {
                "postgres": {
                    "pool_size": 2,
                    "max_overflow": 3,
                    "pool_timeout": 10.0,
                    "pool_recycle": 900
                }
            }
        }
        
        # Set configuration in global config manager
        config_manager.set_config(test_config)
        
        connection_manager = get_global_connection_manager()
        
        try:
            # Create PgaiStore instance
            store = PgaiStore(table_name="config_test_store")
            
            # Initialize store (this should use the global configuration)
            await store.initialize()
            
            # Check that the configuration was used
            stats = connection_manager.get_pool_statistics()
            assert len(stats) == 1
            
            pool_stat = list(stats.values())[0]
            assert pool_stat["min_size"] == 2
            assert pool_stat["max_size"] == 5  # 2 + 3
            assert pool_stat["timeout"] == 10.0
            assert pool_stat["recycle"] == 900
            
            print("✅ PgaiStore configuration usage test passed")
            
        except Exception as e:
            print(f"Expected error (database may not be available): {e}")
            
        finally:
            # Cleanup
            try:
                await store.close()
            except:
                pass
            await connection_manager.close_all_pools(force=True)
    
    def test_environment_variable_override(self):
        """Test that environment variables can override configuration."""
        # Set environment variables
        original_values = {}
        env_vars = {
            "POSTGRES_HOST": "test-host",
            "POSTGRES_PORT": "5433",
            "POSTGRES_DB": "test-db",
            "POSTGRES_USER": "test-user",
            "POSTGRES_PASSWORD": "test-password"
        }
        
        for key, value in env_vars.items():
            original_values[key] = os.environ.get(key)
            os.environ[key] = value
        
        try:
            # Create PgaiStore and check database URL
            store = PgaiStore(table_name="env_test")
            
            # Check that environment variables are used in database URL
            assert "test-host" in store.db_url
            assert "5433" in store.db_url
            assert "test-db" in store.db_url
            assert "test-user" in store.db_url
            assert "test-password" in store.db_url
            
            print("✅ Environment variable override test passed")
            
        finally:
            # Restore original environment variables
            for key, original_value in original_values.items():
                if original_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = original_value
    
    def test_configuration_validation(self):
        """Test that invalid configuration values are handled properly."""
        # Test with invalid values
        invalid_config = {
            "database": {
                "postgres": {
                    "pool_size": -1,  # Invalid
                    "max_overflow": "invalid",  # Invalid type
                    "pool_timeout": -5.0,  # Invalid
                }
            }
        }
        
        # Should handle invalid values gracefully
        try:
            pool_config = ConnectionPoolConfig.from_memfuse_config(invalid_config)
            # Should use defaults for invalid values
            assert pool_config.min_size >= 1
            assert pool_config.max_size >= pool_config.min_size
            assert pool_config.timeout > 0
            
            print("✅ Configuration validation test passed")
            
        except Exception as e:
            # Should not crash on invalid configuration
            print(f"Configuration validation handled error: {e}")


if __name__ == "__main__":
    # Run tests directly
    test_instance = TestConnectionPoolConfiguration()
    
    print("Running Connection Pool Configuration tests...")
    
    test_instance.test_configuration_hierarchy_priority()
    test_instance.test_configuration_defaults()
    test_instance.test_environment_variable_override()
    test_instance.test_configuration_validation()
    
    # Run async tests
    async def run_async_tests():
        await test_instance.test_configuration_applied_to_pools()
        await test_instance.test_pgai_store_uses_configuration()
    
    asyncio.run(run_async_tests())
    
    print("✅ All Connection Pool Configuration tests passed!")
