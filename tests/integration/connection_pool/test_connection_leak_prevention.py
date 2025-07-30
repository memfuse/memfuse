"""
Test Connection Leak Prevention

This test verifies that the connection pool optimization prevents
PostgreSQL connection leaks under various usage scenarios.
"""

import pytest
import asyncio
import aiohttp
import psycopg
import time
import sys
import os
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from memfuse_core.services.global_connection_manager import get_global_connection_manager
from memfuse_core.store.pgai_store.pgai_store import PgaiStore

# Test configuration
API_BASE_URL = "http://localhost:8000"
API_KEY = os.getenv("MEMFUSE_API_KEY", "test-api-key")
POSTGRES_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/memfuse")


class ConnectionMonitor:
    """Monitor PostgreSQL connection count."""
    
    def __init__(self, postgres_url: str):
        self.postgres_url = postgres_url
        
    async def get_connection_count(self) -> int:
        """Get current PostgreSQL connection count."""
        try:
            async with await psycopg.AsyncConnection.connect(self.postgres_url) as conn:
                async with conn.cursor() as cur:
                    await cur.execute("""
                        SELECT count(*) 
                        FROM pg_stat_activity 
                        WHERE state IN ('active', 'idle', 'idle in transaction')
                          AND datname = current_database()
                    """)
                    result = await cur.fetchone()
                    return result[0] if result else 0
        except Exception as e:
            print(f"Failed to get connection count: {e}")
            return -1


class TestConnectionLeakPrevention:
    """Test suite for connection leak prevention."""
    
    @pytest.mark.asyncio
    async def test_store_instance_creation_no_leak(self):
        """Test that creating multiple store instances doesn't leak connections."""
        monitor = ConnectionMonitor(POSTGRES_URL)
        connection_manager = get_global_connection_manager()
        
        # Get initial connection count
        initial_count = await monitor.get_connection_count()
        if initial_count == -1:
            pytest.skip("Cannot connect to PostgreSQL")
        
        print(f"Initial connections: {initial_count}")
        
        # Create multiple store instances
        stores = []
        for i in range(5):
            config = {
                "database": {
                    "postgres": {
                        "pool_size": 2,
                        "max_overflow": 3,
                        "pool_timeout": 10.0
                    }
                }
            }
            store = PgaiStore(config=config, table_name=f"test_table_{i}")
            stores.append(store)
        
        try:
            # Initialize all stores
            for i, store in enumerate(stores):
                await store.initialize()
                print(f"Store {i} initialized")
            
            # Check connection count after initialization
            after_init_count = await monitor.get_connection_count()
            print(f"Connections after init: {after_init_count}")
            
            # Verify connection pool sharing
            stats = connection_manager.get_pool_statistics()
            print(f"Pool statistics: {stats}")
            
            # Should have only one pool with multiple references
            assert len(stats) == 1, f"Expected 1 pool, got {len(stats)}"
            
            pool_stat = list(stats.values())[0]
            assert pool_stat["active_references"] == 5, f"Expected 5 references, got {pool_stat['active_references']}"
            
            # Connection increase should be minimal (just the pool size)
            connection_increase = after_init_count - initial_count
            assert connection_increase <= 10, f"Too many connections created: {connection_increase}"
            
            print("✅ Store instance creation test passed")
            
        finally:
            # Cleanup stores
            for store in stores:
                await store.close()
            
            # Force cleanup of pools
            await connection_manager.close_all_pools(force=True)
            
            # Check final connection count
            final_count = await monitor.get_connection_count()
            print(f"Final connections: {final_count}")
    
    @pytest.mark.asyncio
    async def test_rapid_store_creation_destruction(self):
        """Test rapid creation and destruction of stores doesn't leak connections."""
        monitor = ConnectionMonitor(POSTGRES_URL)
        connection_manager = get_global_connection_manager()
        
        # Get initial connection count
        initial_count = await monitor.get_connection_count()
        if initial_count == -1:
            pytest.skip("Cannot connect to PostgreSQL")
        
        print(f"Initial connections: {initial_count}")
        
        # Rapid creation and destruction cycles
        for cycle in range(3):
            print(f"Cycle {cycle + 1}/3")
            
            stores = []
            for i in range(3):
                config = {
                    "database": {
                        "postgres": {
                            "pool_size": 1,
                            "max_overflow": 2,
                            "pool_timeout": 5.0
                        }
                    }
                }
                store = PgaiStore(config=config, table_name=f"rapid_test_{cycle}_{i}")
                stores.append(store)
            
            # Initialize stores
            for store in stores:
                try:
                    await store.initialize()
                except Exception as e:
                    print(f"Store initialization error (expected): {e}")
            
            # Check connection count
            cycle_count = await monitor.get_connection_count()
            print(f"  Connections in cycle: {cycle_count}")
            
            # Close stores
            for store in stores:
                await store.close()
            
            # Small delay for cleanup
            await asyncio.sleep(0.1)
        
        # Force cleanup
        await connection_manager.close_all_pools(force=True)
        await asyncio.sleep(0.5)
        
        # Check final connection count
        final_count = await monitor.get_connection_count()
        print(f"Final connections: {final_count}")
        
        # Should not have significant increase
        connection_increase = final_count - initial_count
        assert connection_increase <= 5, f"Connection leak detected: {connection_increase} extra connections"
        
        print("✅ Rapid creation/destruction test passed")
    
    @pytest.mark.asyncio
    async def test_api_requests_no_leak(self):
        """Test that multiple API requests don't leak connections."""
        monitor = ConnectionMonitor(POSTGRES_URL)
        
        # Get initial connection count
        initial_count = await monitor.get_connection_count()
        if initial_count == -1:
            pytest.skip("Cannot connect to PostgreSQL")
        
        print(f"Initial connections: {initial_count}")
        
        # Make multiple API requests
        async with aiohttp.ClientSession(
            headers={"X-API-Key": API_KEY},
            timeout=aiohttp.ClientTimeout(total=30)
        ) as session:
            
            # Create a test user
            try:
                async with session.post(
                    f"{API_BASE_URL}/api/v1/users",
                    json={"name": f"leak_test_user_{int(time.time())}"}
                ) as response:
                    if response.status == 201:
                        user_data = await response.json()
                        user_id = user_data["data"]["user"]["id"]
                        print(f"Created test user: {user_id}")
                    else:
                        pytest.skip(f"Cannot create test user: {response.status}")
            except Exception as e:
                pytest.skip(f"Cannot connect to API: {e}")
            
            # Make multiple query requests
            for i in range(10):
                try:
                    async with session.post(
                        f"{API_BASE_URL}/api/v1/users/{user_id}/query",
                        json={
                            "query": f"test query {i}",
                            "top_k": 5,
                            "include_messages": True,
                            "include_knowledge": True
                        }
                    ) as response:
                        if response.status == 200:
                            print(f"Query {i+1}/10 successful")
                        else:
                            print(f"Query {i+1}/10 failed: {response.status}")
                except Exception as e:
                    print(f"Query {i+1}/10 error: {e}")
                
                # Small delay between requests
                await asyncio.sleep(0.1)
        
        # Check final connection count
        final_count = await monitor.get_connection_count()
        print(f"Final connections: {final_count}")
        
        # Should not have significant increase
        connection_increase = final_count - initial_count
        assert connection_increase <= 10, f"API request connection leak: {connection_increase} extra connections"
        
        print("✅ API requests test passed")


if __name__ == "__main__":
    # Run tests directly
    test_instance = TestConnectionLeakPrevention()
    
    print("Running Connection Leak Prevention tests...")
    
    async def run_tests():
        await test_instance.test_store_instance_creation_no_leak()
        await test_instance.test_rapid_store_creation_destruction()
        await test_instance.test_api_requests_no_leak()
    
    asyncio.run(run_tests())
    
    print("✅ All Connection Leak Prevention tests passed!")
