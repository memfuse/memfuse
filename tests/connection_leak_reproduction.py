#!/usr/bin/env python3
"""
Connection Leak Reproduction Script

This script reproduces the PostgreSQL connection leak issue by making
multiple API calls to the /users/{user_id}/query endpoint and monitoring
the database connection count.

Usage:
    poetry run python tests/connection_leak_reproduction.py

Requirements:
    - MemFuse server running on localhost:8000
    - PostgreSQL database accessible
    - API key configured
"""

import asyncio
import aiohttp
import psycopg
import time
import os
from typing import List, Dict, Any
from loguru import logger

# Configuration
API_BASE_URL = "http://localhost:8000"
API_KEY = os.getenv("MEMFUSE_API_KEY", "test-api-key")
POSTGRES_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/memfuse")

# Test parameters
NUM_USERS = 5
REQUESTS_PER_USER = 10
CONCURRENT_REQUESTS = 3


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
                        WHERE state = 'active' OR state = 'idle'
                    """)
                    result = await cur.fetchone()
                    return result[0] if result else 0
        except Exception as e:
            logger.error(f"Failed to get connection count: {e}")
            return -1
    
    async def get_connection_details(self) -> List[Dict[str, Any]]:
        """Get detailed connection information."""
        try:
            async with await psycopg.AsyncConnection.connect(self.postgres_url) as conn:
                async with conn.cursor() as cur:
                    await cur.execute("""
                        SELECT 
                            pid,
                            usename,
                            application_name,
                            client_addr,
                            state,
                            query_start,
                            state_change
                        FROM pg_stat_activity 
                        WHERE state IN ('active', 'idle', 'idle in transaction')
                        ORDER BY query_start DESC
                    """)
                    rows = await cur.fetchall()
                    
                    columns = ['pid', 'usename', 'application_name', 'client_addr', 
                              'state', 'query_start', 'state_change']
                    return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            logger.error(f"Failed to get connection details: {e}")
            return []


class APITester:
    """Test API endpoints and monitor connection usage."""
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            headers={"X-API-Key": self.api_key},
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def create_user(self, user_name: str) -> str:
        """Create a user and return user_id."""
        try:
            async with self.session.post(
                f"{self.base_url}/api/v1/users",
                json={"name": user_name}
            ) as response:
                if response.status == 201:
                    data = await response.json()
                    return data["data"]["user"]["id"]
                else:
                    logger.error(f"Failed to create user {user_name}: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error creating user {user_name}: {e}")
            return None
    
    async def query_memory(self, user_id: str, query: str) -> bool:
        """Query user memory and return success status."""
        try:
            async with self.session.post(
                f"{self.base_url}/users/{user_id}/query",
                json={
                    "query": query,
                    "top_k": 5,
                    "include_messages": True,
                    "include_knowledge": True
                }
            ) as response:
                success = response.status == 200
                if not success:
                    text = await response.text()
                    logger.error(f"Query failed for user {user_id}: {response.status} - {text}")
                return success
        except Exception as e:
            logger.error(f"Error querying memory for user {user_id}: {e}")
            return False


async def test_connection_leak():
    """Main test function to reproduce connection leak."""
    logger.info("Starting connection leak reproduction test")
    
    monitor = ConnectionMonitor(POSTGRES_URL)
    
    # Get initial connection count
    initial_connections = await monitor.get_connection_count()
    logger.info(f"Initial PostgreSQL connections: {initial_connections}")
    
    if initial_connections == -1:
        logger.error("Cannot connect to PostgreSQL. Ensure database is running.")
        return
    
    async with APITester(API_BASE_URL, API_KEY) as api:
        # Create test users
        logger.info(f"Creating {NUM_USERS} test users...")
        user_ids = []
        
        for i in range(NUM_USERS):
            user_name = f"test_user_{i}_{int(time.time())}"
            user_id = await api.create_user(user_name)
            if user_id:
                user_ids.append(user_id)
                logger.info(f"Created user {user_name} with ID {user_id}")
            else:
                logger.error(f"Failed to create user {user_name}")
        
        if not user_ids:
            logger.error("No users created successfully. Cannot proceed with test.")
            return
        
        # Monitor connections during API calls
        connection_counts = []
        
        for round_num in range(REQUESTS_PER_USER):
            logger.info(f"Round {round_num + 1}/{REQUESTS_PER_USER}")
            
            # Make concurrent requests for all users
            tasks = []
            for user_id in user_ids:
                query = f"test query round {round_num} for user {user_id}"
                task = api.query_memory(user_id, query)
                tasks.append(task)
            
            # Execute requests in batches to control concurrency
            for i in range(0, len(tasks), CONCURRENT_REQUESTS):
                batch = tasks[i:i + CONCURRENT_REQUESTS]
                results = await asyncio.gather(*batch, return_exceptions=True)
                
                # Log any failures
                for j, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(f"Request failed: {result}")
                    elif not result:
                        logger.warning(f"Request {i + j} returned False")
            
            # Check connection count after each round
            current_connections = await monitor.get_connection_count()
            connection_counts.append(current_connections)
            logger.info(f"Connections after round {round_num + 1}: {current_connections}")
            
            # Small delay between rounds
            await asyncio.sleep(1)
        
        # Final connection count
        final_connections = await monitor.get_connection_count()
        logger.info(f"Final PostgreSQL connections: {final_connections}")
        
        # Analyze results
        connection_increase = final_connections - initial_connections
        logger.info(f"Connection increase: {connection_increase}")
        
        if connection_increase > NUM_USERS * 2:  # Expected: ~2 connections per user max
            logger.error(f"🚨 CONNECTION LEAK DETECTED! Increased by {connection_increase} connections")
            
            # Get detailed connection info
            details = await monitor.get_connection_details()
            logger.info("Current connection details:")
            for detail in details:
                logger.info(f"  PID: {detail['pid']}, User: {detail['usename']}, "
                           f"State: {detail['state']}, App: {detail['application_name']}")
        else:
            logger.info("✅ No significant connection leak detected")
        
        # Connection count progression
        logger.info("Connection count progression:")
        for i, count in enumerate(connection_counts):
            logger.info(f"  Round {i + 1}: {count} connections")


if __name__ == "__main__":
    asyncio.run(test_connection_leak())
