"""Performance tests for GlobalConfigManager optimization.

This module tests the performance improvements achieved by the GlobalConfigManager
compared to the legacy configuration system.
"""

import asyncio
import time
import pytest
from typing import Dict, Any
from omegaconf import DictConfig, OmegaConf

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from memfuse_core.utils.global_config_manager import GlobalConfigManager, get_global_config_manager
from memfuse_core.utils.config import ConfigManager


class TestGlobalConfigPerformance:
    """Test suite for GlobalConfigManager performance optimization."""

    @pytest.fixture
    def sample_config(self) -> Dict[str, Any]:
        """Create a sample configuration for testing."""
        return {
            "server": {
                "host": "localhost",
                "port": 8000,
                "reload": False
            },
            "database": {
                "type": "postgres",
                "postgres": {
                    "host": "localhost",
                    "port": 5432,
                    "database": "memfuse",
                    "user": "postgres",
                    "password": "postgres",
                    "pool_size": 20,
                    "max_overflow": 40,
                    "pool_timeout": 60.0,
                    "pool_recycle": 7200
                }
            },
            "buffer": {
                "enabled": True,
                "round_buffer": {
                    "max_tokens": 800,
                    "max_size": 5
                },
                "hybrid_buffer": {
                    "max_size": 5
                },
                "query": {
                    "max_size": 15,
                    "cache_size": 100
                }
            },
            "memory": {
                "memory_service": {
                    "parallel_enabled": True
                },
                "layers": {
                    "m0": {"enabled": True},
                    "m1": {"enabled": False},
                    "m2": {"enabled": False}
                }
            },
            "store": {
                "backend": "qdrant",
                "top_k": 5,
                "similarity_threshold": 0.3
            },
            "embedding": {
                "dimension": 384,
                "model": "all-MiniLM-L6-v2"
            },
            "data_dir": "data"
        }

    @pytest.fixture
    async def initialized_global_config(self, sample_config) -> GlobalConfigManager:
        """Create and initialize a GlobalConfigManager instance."""
        # Reset singleton for testing
        GlobalConfigManager._instance = None
        GlobalConfigManager._initialized = False
        
        global_config = GlobalConfigManager()
        await global_config.initialize(sample_config)
        return global_config

    @pytest.fixture
    def legacy_config(self, sample_config) -> ConfigManager:
        """Create a legacy ConfigManager instance."""
        config_manager = ConfigManager()
        config_manager.set_config(sample_config)
        return config_manager

    async def test_initialization_performance(self, sample_config):
        """Test that GlobalConfigManager initializes faster than legacy system."""
        # Reset singleton for testing
        GlobalConfigManager._instance = None
        GlobalConfigManager._initialized = False
        
        # Test GlobalConfigManager initialization
        start_time = time.time()
        global_config = GlobalConfigManager()
        await global_config.initialize(sample_config)
        global_init_time = time.time() - start_time
        
        # Test legacy ConfigManager initialization
        start_time = time.time()
        legacy_config = ConfigManager()
        legacy_config.set_config(sample_config)
        legacy_init_time = time.time() - start_time
        
        # GlobalConfigManager should initialize quickly (under 100ms)
        assert global_init_time < 0.1, f"Global config init took {global_init_time:.3f}s, expected < 0.1s"
        
        # Log performance comparison
        print(f"Global config init: {global_init_time:.3f}s")
        print(f"Legacy config init: {legacy_init_time:.3f}s")
        
        # Get performance stats
        stats = global_config.get_performance_stats()
        assert stats["initialized"] is True
        assert stats["cached_keys"] > 0
        assert stats["load_time_seconds"] == global_init_time

    async def test_access_performance_comparison(self, initialized_global_config, legacy_config):
        """Test that GlobalConfigManager provides faster configuration access."""
        global_config = initialized_global_config
        
        # Common configuration keys to test
        test_keys = [
            ("server.host", ["server", "host"]),
            ("database.postgres.pool_size", ["database", "postgres", "pool_size"]),
            ("buffer.enabled", ["buffer", "enabled"]),
            ("memory.memory_service.parallel_enabled", ["memory", "memory_service", "parallel_enabled"]),
            ("store.top_k", ["store", "top_k"]),
            ("embedding.dimension", ["embedding", "dimension"])
        ]
        
        # Test GlobalConfigManager access performance
        start_time = time.time()
        for _ in range(1000):  # 1000 accesses
            for key, _ in test_keys:
                global_config.get(key)
        global_access_time = time.time() - start_time
        
        # Test legacy ConfigManager access performance
        start_time = time.time()
        for _ in range(1000):  # 1000 accesses
            for _, path in test_keys:
                config = legacy_config.get_config()
                current = config
                for key in path:
                    if isinstance(current, dict) and key in current:
                        current = current[key]
                    else:
                        break
        legacy_access_time = time.time() - start_time
        
        # GlobalConfigManager should be significantly faster
        improvement_ratio = legacy_access_time / global_access_time
        assert improvement_ratio > 2.0, f"Expected >2x improvement, got {improvement_ratio:.2f}x"
        
        print(f"Global config access (1000x6): {global_access_time:.3f}s")
        print(f"Legacy config access (1000x6): {legacy_access_time:.3f}s")
        print(f"Performance improvement: {improvement_ratio:.2f}x")
        
        # Check cache performance
        stats = global_config.get_performance_stats()
        assert stats["cache_hit_rate"] > 0.8, f"Cache hit rate too low: {stats['cache_hit_rate']:.2f}"

    async def test_cache_effectiveness(self, initialized_global_config):
        """Test that the configuration cache is effective."""
        global_config = initialized_global_config
        
        # Access the same keys multiple times
        test_keys = [
            "server.host",
            "database.postgres.pool_size",
            "buffer.enabled",
            "store.top_k"
        ]
        
        # First access (should miss cache)
        for key in test_keys:
            global_config.get(key)
        
        # Get initial stats
        initial_stats = global_config.get_performance_stats()
        
        # Second access (should hit cache)
        for key in test_keys:
            global_config.get(key)
        
        # Get final stats
        final_stats = global_config.get_performance_stats()
        
        # Cache hits should have increased
        cache_hits_increase = final_stats["cache_hits"] - initial_stats["cache_hits"]
        assert cache_hits_increase >= len(test_keys), "Cache hits did not increase as expected"
        
        # Cache hit rate should be high
        assert final_stats["cache_hit_rate"] > 0.5, f"Cache hit rate too low: {final_stats['cache_hit_rate']:.2f}"

    async def test_memory_usage_efficiency(self, sample_config):
        """Test that GlobalConfigManager uses memory efficiently."""
        # Reset singleton for testing
        GlobalConfigManager._instance = None
        GlobalConfigManager._initialized = False
        
        global_config = GlobalConfigManager()
        await global_config.initialize(sample_config)
        
        # Check that cache is populated but not excessive
        stats = global_config.get_performance_stats()
        
        # Should have cached common keys
        assert stats["cached_keys"] > 10, "Too few keys cached"
        assert stats["cached_keys"] < 100, "Too many keys cached (potential memory waste)"
        
        # Should be initialized quickly
        assert stats["load_time_seconds"] < 0.1, "Initialization took too long"

    async def test_concurrent_access_performance(self, initialized_global_config):
        """Test performance under concurrent access."""
        global_config = initialized_global_config
        
        async def access_config():
            """Simulate concurrent configuration access."""
            for _ in range(100):
                global_config.get("server.host")
                global_config.get("database.postgres.pool_size")
                global_config.get("buffer.enabled")
                await asyncio.sleep(0.001)  # Small delay to simulate real usage
        
        # Run multiple concurrent tasks
        start_time = time.time()
        tasks = [access_config() for _ in range(10)]
        await asyncio.gather(*tasks)
        concurrent_time = time.time() - start_time
        
        # Should complete quickly even under concurrent load
        assert concurrent_time < 5.0, f"Concurrent access took too long: {concurrent_time:.3f}s"
        
        # Check that cache hit rate is high
        stats = global_config.get_performance_stats()
        assert stats["cache_hit_rate"] > 0.9, f"Cache hit rate under load: {stats['cache_hit_rate']:.2f}"

    async def test_hot_reload_functionality(self, initialized_global_config):
        """Test hot-reload functionality."""
        global_config = initialized_global_config
        
        # Get initial value
        initial_value = global_config.get("server.port", 8000)
        assert initial_value == 8000
        
        # Create updated configuration
        updated_config = {
            "server": {
                "host": "localhost",
                "port": 9000,  # Changed value
                "reload": False
            },
            "database": {
                "type": "postgres",
                "postgres": {
                    "pool_size": 25  # Changed value
                }
            }
        }
        
        # Hot reload
        await global_config.hot_reload(updated_config)
        
        # Check that values are updated
        new_port = global_config.get("server.port", 8000)
        assert new_port == 9000, "Hot reload did not update server.port"
        
        new_pool_size = global_config.get("database.postgres.pool_size", 20)
        assert new_pool_size == 25, "Hot reload did not update pool_size"
        
        # Check that cache was rebuilt
        stats = global_config.get_performance_stats()
        assert stats["cache_hits"] == 0, "Cache should be reset after hot reload"


async def run_performance_tests():
    """Run all performance tests."""
    test_instance = TestGlobalConfigPerformance()
    
    # Create sample config
    sample_config = {
        "server": {"host": "localhost", "port": 8000},
        "database": {"postgres": {"pool_size": 20}},
        "buffer": {"enabled": True},
        "store": {"top_k": 5}
    }
    
    print("Running GlobalConfigManager performance tests...")
    
    # Test initialization performance
    await test_instance.test_initialization_performance(sample_config)
    print("✓ Initialization performance test passed")
    
    # Test access performance
    GlobalConfigManager._instance = None
    GlobalConfigManager._initialized = False
    global_config = GlobalConfigManager()
    await global_config.initialize(sample_config)
    
    legacy_config = ConfigManager()
    legacy_config.set_config(sample_config)
    
    await test_instance.test_access_performance_comparison(global_config, legacy_config)
    print("✓ Access performance test passed")
    
    print("All performance tests completed successfully!")


if __name__ == "__main__":
    asyncio.run(run_performance_tests())