#!/usr/bin/env python3
"""
Direct Connection Pool Stress Test

This test directly tests the connection pool optimization without requiring
the API server, focusing on the core connection management functionality.
"""

import asyncio
import psycopg
import time
import sys
import os
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from memfuse_core.services.global_connection_manager import get_global_connection_manager
from memfuse_core.store.pgai_store.pgai_store import PgaiStore

# Test configuration
POSTGRES_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/memfuse")
STRESS_DURATION = 60  # 1 minute for direct test
CONCURRENT_STORES = 20
OPERATIONS_PER_STORE = 10


class ConnectionMonitor:
    """Monitor PostgreSQL connections during stress test."""
    
    def __init__(self, postgres_url: str):
        self.postgres_url = postgres_url
        self.connection_history = []
        
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
    
    async def get_detailed_info(self) -> Dict[str, Any]:
        """Get detailed connection information."""
        try:
            async with await psycopg.AsyncConnection.connect(self.postgres_url) as conn:
                async with conn.cursor() as cur:
                    # Total connections
                    await cur.execute("""
                        SELECT count(*) 
                        FROM pg_stat_activity 
                        WHERE datname = current_database()
                    """)
                    total = (await cur.fetchone())[0]
                    
                    # Max connections
                    await cur.execute("SHOW max_connections")
                    max_conn = int((await cur.fetchone())[0])
                    
                    # State breakdown
                    await cur.execute("""
                        SELECT state, count(*) 
                        FROM pg_stat_activity 
                        WHERE datname = current_database()
                        GROUP BY state
                    """)
                    states = {row[0]: row[1] for row in await cur.fetchall()}
                    
                    return {
                        "total_connections": total,
                        "max_connections": max_conn,
                        "usage_percentage": (total / max_conn) * 100,
                        "state_breakdown": states,
                        "timestamp": time.time()
                    }
        except Exception as e:
            return {"error": str(e), "timestamp": time.time()}


async def stress_test_store_operations():
    """Stress test store operations directly."""
    print("🔥 Starting Direct Connection Pool Stress Test")
    print(f"Concurrent stores: {CONCURRENT_STORES}, Operations per store: {OPERATIONS_PER_STORE}")
    
    monitor = ConnectionMonitor(POSTGRES_URL)
    connection_manager = get_global_connection_manager()
    
    # Get initial state
    initial_count = await monitor.get_connection_count()
    initial_info = await monitor.get_detailed_info()
    
    print(f"Initial connections: {initial_count}")
    print(f"Max connections: {initial_info.get('max_connections', 'unknown')}")
    print(f"Initial usage: {initial_info.get('usage_percentage', 0):.1f}%")
    
    # Test configuration
    config = {
        "database": {
            "postgres": {
                "pool_size": 5,
                "max_overflow": 10,
                "pool_timeout": 30.0,
                "pool_recycle": 3600
            }
        }
    }
    
    async def stress_test_single_store(store_id: int) -> Dict[str, Any]:
        """Stress test a single store."""
        store = PgaiStore(config=config, table_name=f"stress_test_{store_id}")
        operations_completed = 0
        errors = 0
        
        try:
            # Initialize store
            await store.initialize()
            operations_completed += 1
            
            # Perform operations
            for i in range(OPERATIONS_PER_STORE):
                try:
                    # Test basic operations
                    count = await store.count()
                    operations_completed += 1
                    
                    # Small delay
                    await asyncio.sleep(0.01)
                    
                except Exception as e:
                    errors += 1
                    if "too many clients already" in str(e):
                        print(f"🚨 Store {store_id}: CONNECTION POOL EXHAUSTION!")
                        break
            
            return {
                "store_id": store_id,
                "operations_completed": operations_completed,
                "errors": errors,
                "success": errors == 0
            }
            
        except Exception as e:
            errors += 1
            if "too many clients already" in str(e):
                print(f"🚨 Store {store_id}: CONNECTION POOL EXHAUSTION during init!")
            return {
                "store_id": store_id,
                "operations_completed": operations_completed,
                "errors": errors,
                "success": False,
                "error": str(e)
            }
        finally:
            try:
                await store.close()
            except Exception as e:
                print(f"Error closing store {store_id}: {e}")
    
    # Run stress test
    print(f"\n🚀 Starting stress test with {CONCURRENT_STORES} concurrent stores...")
    
    start_time = time.time()
    
    # Execute all stores concurrently
    tasks = [stress_test_single_store(i) for i in range(CONCURRENT_STORES)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    end_time = time.time()
    
    # Get final state
    final_count = await monitor.get_connection_count()
    final_info = await monitor.get_detailed_info()
    
    # Analyze results
    print(f"\n📊 Stress Test Results (Duration: {end_time - start_time:.1f}s)")
    
    successful_stores = 0
    total_operations = 0
    total_errors = 0
    connection_exhaustion_errors = 0
    
    for result in results:
        if isinstance(result, Exception):
            total_errors += 1
            if "too many clients already" in str(result):
                connection_exhaustion_errors += 1
        elif isinstance(result, dict):
            if result.get("success", False):
                successful_stores += 1
            total_operations += result.get("operations_completed", 0)
            total_errors += result.get("errors", 0)
            if "too many clients already" in result.get("error", ""):
                connection_exhaustion_errors += 1
    
    print(f"Successful stores: {successful_stores}/{CONCURRENT_STORES}")
    print(f"Total operations: {total_operations}")
    print(f"Total errors: {total_errors}")
    print(f"Connection exhaustion errors: {connection_exhaustion_errors}")
    
    # Connection analysis
    print(f"\n🔍 Connection Analysis:")
    print(f"Initial connections: {initial_count}")
    print(f"Final connections: {final_count}")
    print(f"Connection increase: {final_count - initial_count}")
    print(f"Final usage: {final_info.get('usage_percentage', 0):.1f}%")
    
    # Pool statistics
    pool_stats = connection_manager.get_pool_statistics()
    print(f"\n🏊 Connection Pool Statistics:")
    for url, stats in pool_stats.items():
        print(f"  {url}:")
        print(f"    Pool size: {stats['min_size']}-{stats['max_size']}")
        print(f"    Active references: {stats['active_references']}")
        print(f"    Pool closed: {stats['pool_closed']}")
    
    # Cleanup
    await connection_manager.close_all_pools(force=True)
    
    # Final check after cleanup
    await asyncio.sleep(1)
    cleanup_count = await monitor.get_connection_count()
    print(f"Connections after cleanup: {cleanup_count}")
    
    # Determine if test passed
    connection_increase = final_count - initial_count
    final_usage = final_info.get('usage_percentage', 0)
    
    success_criteria = [
        connection_exhaustion_errors == 0,
        connection_increase < 30,  # Reasonable increase
        final_usage < 70,  # Under 70% usage
        successful_stores >= CONCURRENT_STORES * 0.9  # 90% success rate
    ]
    
    if all(success_criteria):
        print(f"\n✅ DIRECT STRESS TEST PASSED!")
        print(f"   - No connection pool exhaustion errors")
        print(f"   - Connection increase: {connection_increase} (acceptable)")
        print(f"   - Final usage: {final_usage:.1f}% (under 70%)")
        print(f"   - Success rate: {successful_stores}/{CONCURRENT_STORES}")
        return True
    else:
        print(f"\n❌ DIRECT STRESS TEST FAILED!")
        if connection_exhaustion_errors > 0:
            print(f"   - {connection_exhaustion_errors} connection exhaustion errors")
        if connection_increase >= 30:
            print(f"   - Excessive connection increase: {connection_increase}")
        if final_usage >= 70:
            print(f"   - High connection usage: {final_usage:.1f}%")
        if successful_stores < CONCURRENT_STORES * 0.9:
            print(f"   - Low success rate: {successful_stores}/{CONCURRENT_STORES}")
        return False


if __name__ == "__main__":
    print("Direct Connection Pool Stress Test")
    print("=" * 50)
    
    # Run stress test
    success = asyncio.run(stress_test_store_operations())
    exit(0 if success else 1)
