#!/usr/bin/env python3
"""Simple test script for GlobalConfigManager performance validation."""

import asyncio
import time
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from memfuse_core.utils.global_config_manager import GlobalConfigManager
from memfuse_core.utils.config import ConfigManager


async def test_basic_functionality():
    """Test basic GlobalConfigManager functionality."""
    print("Testing GlobalConfigManager basic functionality...")
    
    # Sample configuration
    sample_config = {
        "server": {
            "host": "localhost",
            "port": 8000,
            "reload": False
        },
        "database": {
            "postgres": {
                "pool_size": 20,
                "max_overflow": 40
            }
        },
        "buffer": {
            "enabled": True
        }
    }
    
    # Reset singleton for testing
    GlobalConfigManager._instance = None
    GlobalConfigManager._initialized = False
    
    # Test initialization
    start_time = time.time()
    global_config = GlobalConfigManager()
    await global_config.initialize(sample_config)
    init_time = time.time() - start_time
    
    print(f"✓ Initialization completed in {init_time:.3f}s")
    
    # Test basic access
    host = global_config.get("server.host")
    port = global_config.get("server.port")
    pool_size = global_config.get("database.postgres.pool_size")
    
    assert host == "localhost", f"Expected 'localhost', got '{host}'"
    assert port == 8000, f"Expected 8000, got {port}"
    assert pool_size == 20, f"Expected 20, got {pool_size}"
    
    print("✓ Basic configuration access working")
    
    # Test performance stats
    stats = global_config.get_performance_stats()
    print(f"✓ Performance stats: {stats['cached_keys']} keys cached, {stats['cache_hit_rate']:.2f} hit rate")
    
    return True


async def test_performance_comparison():
    """Test performance comparison between global and legacy config."""
    print("\nTesting performance comparison...")
    
    sample_config = {
        "server": {"host": "localhost", "port": 8000},
        "database": {"postgres": {"pool_size": 20}},
        "buffer": {"enabled": True},
        "store": {"top_k": 5}
    }
    
    # Reset and initialize global config
    GlobalConfigManager._instance = None
    GlobalConfigManager._initialized = False
    global_config = GlobalConfigManager()
    await global_config.initialize(sample_config)
    
    # Initialize legacy config
    legacy_config = ConfigManager()
    legacy_config.set_config(sample_config)
    
    # Test access performance
    test_iterations = 1000
    
    # Global config performance
    start_time = time.time()
    for _ in range(test_iterations):
        global_config.get("server.host")
        global_config.get("database.postgres.pool_size")
        global_config.get("buffer.enabled")
    global_time = time.time() - start_time
    
    # Legacy config performance
    start_time = time.time()
    for _ in range(test_iterations):
        config = legacy_config.get_config()
        config.get("server", {}).get("host")
        config.get("database", {}).get("postgres", {}).get("pool_size")
        config.get("buffer", {}).get("enabled")
    legacy_time = time.time() - start_time
    
    improvement = legacy_time / global_time if global_time > 0 else 0
    
    print(f"✓ Global config: {global_time:.3f}s for {test_iterations * 3} accesses")
    print(f"✓ Legacy config: {legacy_time:.3f}s for {test_iterations * 3} accesses")
    print(f"✓ Performance improvement: {improvement:.2f}x")
    
    # Get final stats
    stats = global_config.get_performance_stats()
    print(f"✓ Cache hit rate: {stats['cache_hit_rate']:.2f}")
    
    return improvement > 1.0


async def main():
    """Run all tests."""
    print("GlobalConfigManager Performance Test")
    print("=" * 40)
    
    try:
        # Test basic functionality
        success1 = await test_basic_functionality()
        
        # Test performance comparison
        success2 = await test_performance_comparison()
        
        if success1 and success2:
            print("\n✅ All tests passed!")
            print("GlobalConfigManager is working correctly and provides performance improvements.")
            return True
        else:
            print("\n❌ Some tests failed!")
            return False
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)