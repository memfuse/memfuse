#!/usr/bin/env python3
"""Simple performance test for GlobalConfigManager."""

import asyncio
import sys
import os
import time

# Add src to path
sys.path.insert(0, 'src')

from memfuse_core.utils.global_config_manager import GlobalConfigManager
from memfuse_core.utils.config import ConfigManager


async def test_performance():
    """Test GlobalConfigManager performance."""
    # Sample config
    config = {
        'server': {'host': 'localhost', 'port': 8000},
        'database': {'postgres': {'pool_size': 20, 'max_overflow': 40}},
        'buffer': {'enabled': True, 'round_buffer': {'max_tokens': 800}},
        'store': {'top_k': 5, 'similarity_threshold': 0.3}
    }
    
    print("Testing GlobalConfigManager performance...")
    
    # Test GlobalConfigManager
    GlobalConfigManager._instance = None
    GlobalConfigManager._initialized = False
    
    start = time.time()
    global_config = GlobalConfigManager()
    await global_config.initialize(config)
    global_init_time = time.time() - start
    
    # Test access performance
    start = time.time()
    for _ in range(1000):
        global_config.get('server.host')
        global_config.get('database.postgres.pool_size')
        global_config.get('buffer.enabled')
        global_config.get('store.top_k')
    global_access_time = time.time() - start
    
    # Test legacy ConfigManager
    start = time.time()
    legacy_config = ConfigManager()
    legacy_config.set_config(config)
    legacy_init_time = time.time() - start
    
    start = time.time()
    for _ in range(1000):
        cfg = legacy_config.get_config()
        cfg.get('server', {}).get('host')
        cfg.get('database', {}).get('postgres', {}).get('pool_size')
        cfg.get('buffer', {}).get('enabled')
        cfg.get('store', {}).get('top_k')
    legacy_access_time = time.time() - start
    
    print(f'Global config init: {global_init_time:.4f}s')
    print(f'Legacy config init: {legacy_init_time:.4f}s')
    print(f'Global config access (4000 ops): {global_access_time:.4f}s')
    print(f'Legacy config access (4000 ops): {legacy_access_time:.4f}s')
    
    if global_access_time > 0:
        improvement = legacy_access_time / global_access_time
        print(f'Access performance improvement: {improvement:.2f}x')
    
    stats = global_config.get_performance_stats()
    print(f'Cache hit rate: {stats["cache_hit_rate"]:.2f}')
    print(f'Cached keys: {stats["cached_keys"]}')
    print(f'Total accesses: {stats["total_accesses"]}')
    
    print("✓ GlobalConfigManager performance test completed!")


if __name__ == "__main__":
    asyncio.run(test_performance())