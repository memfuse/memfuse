#!/usr/bin/env python3
"""
Connection Pool Stress Test

This test reproduces the original connection pool exhaustion issue
and verifies that it has been completely resolved.

Original Issue:
- Multiple API calls resulted in "too many clients already" error
- Connections accumulated until PostgreSQL limit was reached
- Required service restart to recover

Test Scenarios:
1. High-frequency API requests
2. Multiple concurrent users
3. Long-running stress test
4. Connection monitoring throughout
"""

import asyncio
import aiohttp
import psycopg
import time
import sys
import os
from pathlib import Path
from typing import List, Dict, Any
import json

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from memfuse_core.services.global_connection_manager import get_global_connection_manager

# Test configuration
API_BASE_URL = "http://localhost:8000"
API_KEY = os.getenv("MEMFUSE_API_KEY", "test-api-key")
POSTGRES_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/memfuse")

# Stress test parameters
STRESS_DURATION = 300  # 5 minutes
CONCURRENT_USERS = 10
REQUESTS_PER_USER = 50
REQUEST_INTERVAL = 0.1  # 100ms between requests


class ConnectionMonitor:
    """Enhanced connection monitor for stress testing."""
    
    def __init__(self, postgres_url: str):
        self.postgres_url = postgres_url
        self.connection_history = []
        
    async def get_detailed_connection_info(self) -> Dict[str, Any]:
        """Get detailed PostgreSQL connection information."""
        try:
            async with await psycopg.AsyncConnection.connect(self.postgres_url) as conn:
                async with conn.cursor() as cur:
                    # Get connection count by state
                    await cur.execute("""
                        SELECT 
                            state,
                            count(*) as count
                        FROM pg_stat_activity 
                        WHERE datname = current_database()
                        GROUP BY state
                        ORDER BY count DESC
                    """)
                    state_counts = {row[0]: row[1] for row in await cur.fetchall()}
                    
                    # Get total connections
                    await cur.execute("""
                        SELECT count(*) 
                        FROM pg_stat_activity 
                        WHERE datname = current_database()
                    """)
                    total_connections = (await cur.fetchone())[0]
                    
                    # Get max connections setting
                    await cur.execute("SHOW max_connections")
                    max_connections = int((await cur.fetchone())[0])
                    
                    # Get long-running connections
                    await cur.execute("""
                        SELECT count(*)
                        FROM pg_stat_activity 
                        WHERE datname = current_database()
                          AND state = 'idle'
                          AND now() - state_change > interval '5 minutes'
                    """)
                    long_idle = (await cur.fetchone())[0]
                    
                    return {
                        "total_connections": total_connections,
                        "max_connections": max_connections,
                        "usage_percentage": (total_connections / max_connections) * 100,
                        "state_counts": state_counts,
                        "long_idle_connections": long_idle,
                        "timestamp": time.time()
                    }
        except Exception as e:
            print(f"Failed to get connection info: {e}")
            return {"error": str(e), "timestamp": time.time()}
    
    def record_connection_info(self, info: Dict[str, Any]):
        """Record connection information for analysis."""
        self.connection_history.append(info)
    
    def analyze_connection_trends(self) -> Dict[str, Any]:
        """Analyze connection usage trends."""
        if not self.connection_history:
            return {"error": "No connection history available"}
        
        valid_records = [r for r in self.connection_history if "total_connections" in r]
        if not valid_records:
            return {"error": "No valid connection records"}
        
        connections = [r["total_connections"] for r in valid_records]
        usage_percentages = [r["usage_percentage"] for r in valid_records]
        
        return {
            "initial_connections": connections[0],
            "final_connections": connections[-1],
            "peak_connections": max(connections),
            "min_connections": min(connections),
            "connection_increase": connections[-1] - connections[0],
            "peak_usage_percentage": max(usage_percentages),
            "average_connections": sum(connections) / len(connections),
            "total_samples": len(valid_records),
            "duration_minutes": (valid_records[-1]["timestamp"] - valid_records[0]["timestamp"]) / 60
        }


class StressTestClient:
    """Client for stress testing API endpoints."""
    
    def __init__(self, base_url: str, api_key: str, user_id: int):
        self.base_url = base_url
        self.api_key = api_key
        self.user_id = user_id
        self.session = None
        self.memfuse_user_id = None
        self.request_count = 0
        self.error_count = 0
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            headers={"X-API-Key": self.api_key},
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def setup_user(self) -> bool:
        """Create a test user for this client."""
        try:
            async with self.session.post(
                f"{self.base_url}/api/v1/users",
                json={"name": f"stress_test_user_{self.user_id}_{int(time.time())}"}
            ) as response:
                if response.status == 201:
                    data = await response.json()
                    self.memfuse_user_id = data["data"]["user"]["id"]
                    print(f"Client {self.user_id}: Created user {self.memfuse_user_id}")
                    return True
                else:
                    print(f"Client {self.user_id}: Failed to create user: {response.status}")
                    return False
        except Exception as e:
            print(f"Client {self.user_id}: Error creating user: {e}")
            return False
    
    async def make_query_request(self, query_id: int) -> bool:
        """Make a single query request."""
        try:
            async with self.session.post(
                f"{self.base_url}/api/v1/users/{self.memfuse_user_id}/query",
                json={
                    "query": f"stress test query {query_id} from user {self.user_id}",
                    "top_k": 5,
                    "include_messages": True,
                    "include_knowledge": True
                }
            ) as response:
                self.request_count += 1
                if response.status == 200:
                    return True
                else:
                    self.error_count += 1
                    if response.status == 500:
                        text = await response.text()
                        if "too many clients already" in text:
                            print(f"🚨 Client {self.user_id}: CONNECTION POOL EXHAUSTION DETECTED!")
                            return False
                    print(f"Client {self.user_id}: Query failed: {response.status}")
                    return False
        except Exception as e:
            self.error_count += 1
            if "too many clients already" in str(e):
                print(f"🚨 Client {self.user_id}: CONNECTION POOL EXHAUSTION DETECTED!")
            print(f"Client {self.user_id}: Query error: {e}")
            return False
    
    async def run_stress_test(self, num_requests: int, interval: float) -> Dict[str, Any]:
        """Run stress test for this client."""
        if not await self.setup_user():
            return {"error": "Failed to setup user"}
        
        start_time = time.time()
        successful_requests = 0
        
        for i in range(num_requests):
            success = await self.make_query_request(i)
            if success:
                successful_requests += 1
            
            # Small delay between requests
            await asyncio.sleep(interval)
        
        end_time = time.time()
        
        return {
            "user_id": self.user_id,
            "total_requests": self.request_count,
            "successful_requests": successful_requests,
            "error_count": self.error_count,
            "success_rate": (successful_requests / num_requests) * 100 if num_requests > 0 else 0,
            "duration_seconds": end_time - start_time,
            "requests_per_second": self.request_count / (end_time - start_time) if end_time > start_time else 0
        }


async def run_connection_pool_stress_test():
    """Run comprehensive connection pool stress test."""
    print("🔥 Starting Connection Pool Stress Test")
    print(f"Duration: {STRESS_DURATION}s, Users: {CONCURRENT_USERS}, Requests per user: {REQUESTS_PER_USER}")
    
    monitor = ConnectionMonitor(POSTGRES_URL)
    connection_manager = get_global_connection_manager()
    
    # Get initial state
    initial_info = await monitor.get_detailed_connection_info()
    monitor.record_connection_info(initial_info)
    
    print(f"Initial connections: {initial_info.get('total_connections', 'unknown')}")
    print(f"Max connections: {initial_info.get('max_connections', 'unknown')}")
    print(f"Initial usage: {initial_info.get('usage_percentage', 0):.1f}%")
    
    # Start monitoring task
    monitoring_task = asyncio.create_task(monitor_connections_continuously(monitor))
    
    try:
        # Create stress test clients
        clients = []
        for i in range(CONCURRENT_USERS):
            client = StressTestClient(API_BASE_URL, API_KEY, i)
            clients.append(client)
        
        # Run stress test
        print(f"\n🚀 Starting stress test with {CONCURRENT_USERS} concurrent users...")
        
        async def run_client_stress_test(client):
            async with client:
                return await client.run_stress_test(REQUESTS_PER_USER, REQUEST_INTERVAL)
        
        # Execute all clients concurrently
        start_time = time.time()
        results = await asyncio.gather(*[run_client_stress_test(client) for client in clients])
        end_time = time.time()
        
        # Stop monitoring
        monitoring_task.cancel()
        try:
            await monitoring_task
        except asyncio.CancelledError:
            pass
        
        # Get final state
        final_info = await monitor.get_detailed_connection_info()
        monitor.record_connection_info(final_info)
        
        # Analyze results
        print(f"\n📊 Stress Test Results (Duration: {end_time - start_time:.1f}s)")
        
        total_requests = sum(r.get("total_requests", 0) for r in results)
        total_successful = sum(r.get("successful_requests", 0) for r in results)
        total_errors = sum(r.get("error_count", 0) for r in results)
        
        print(f"Total requests: {total_requests}")
        print(f"Successful requests: {total_successful}")
        print(f"Failed requests: {total_errors}")
        print(f"Overall success rate: {(total_successful / total_requests * 100) if total_requests > 0 else 0:.1f}%")
        
        # Connection analysis
        connection_analysis = monitor.analyze_connection_trends()
        print(f"\n🔍 Connection Analysis:")
        print(f"Initial connections: {connection_analysis.get('initial_connections', 'unknown')}")
        print(f"Final connections: {connection_analysis.get('final_connections', 'unknown')}")
        print(f"Peak connections: {connection_analysis.get('peak_connections', 'unknown')}")
        print(f"Connection increase: {connection_analysis.get('connection_increase', 'unknown')}")
        print(f"Peak usage: {connection_analysis.get('peak_usage_percentage', 0):.1f}%")
        
        # Pool statistics
        pool_stats = connection_manager.get_pool_statistics()
        print(f"\n🏊 Connection Pool Statistics:")
        for url, stats in pool_stats.items():
            print(f"  {url}:")
            print(f"    Pool size: {stats['min_size']}-{stats['max_size']}")
            print(f"    Active references: {stats['active_references']}")
            print(f"    Pool closed: {stats['pool_closed']}")
        
        # Determine if test passed
        connection_increase = connection_analysis.get('connection_increase', 0)
        peak_usage = connection_analysis.get('peak_usage_percentage', 0)
        
        if total_errors == 0 and connection_increase < 50 and peak_usage < 80:
            print(f"\n✅ STRESS TEST PASSED!")
            print(f"   - No connection pool exhaustion errors")
            print(f"   - Connection increase: {connection_increase} (acceptable)")
            print(f"   - Peak usage: {peak_usage:.1f}% (under 80%)")
            return True
        else:
            print(f"\n❌ STRESS TEST FAILED!")
            if total_errors > 0:
                print(f"   - {total_errors} errors occurred")
            if connection_increase >= 50:
                print(f"   - Excessive connection increase: {connection_increase}")
            if peak_usage >= 80:
                print(f"   - High connection usage: {peak_usage:.1f}%")
            return False
            
    except Exception as e:
        print(f"❌ Stress test failed with exception: {e}")
        monitoring_task.cancel()
        return False
    
    finally:
        # Cleanup
        await connection_manager.close_all_pools(force=True)


async def monitor_connections_continuously(monitor: ConnectionMonitor):
    """Continuously monitor connections during stress test."""
    try:
        while True:
            await asyncio.sleep(10)  # Monitor every 10 seconds
            info = await monitor.get_detailed_connection_info()
            monitor.record_connection_info(info)
            
            if "total_connections" in info:
                print(f"⏱️  Connections: {info['total_connections']} ({info['usage_percentage']:.1f}% usage)")
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    print("Connection Pool Stress Test")
    print("=" * 50)
    
    # Check if API server is running
    import requests
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print("❌ MemFuse API server is not running!")
            print("Please start the server with: poetry run memfuse-core")
            exit(1)
    except requests.exceptions.RequestException:
        print("❌ Cannot connect to MemFuse API server!")
        print("Please start the server with: poetry run memfuse-core")
        exit(1)
    
    # Run stress test
    success = asyncio.run(run_connection_pool_stress_test())
    exit(0 if success else 1)
