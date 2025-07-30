#!/usr/bin/env python3
"""
PostgreSQL Connection Monitor

Simple script to monitor PostgreSQL connections and identify potential leaks.

Usage:
    poetry run python tests/connection_monitor.py
"""

import asyncio
import psycopg
import os
import time
from loguru import logger

POSTGRES_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/memfuse")


async def monitor_connections():
    """Monitor PostgreSQL connections."""
    try:
        async with await psycopg.AsyncConnection.connect(POSTGRES_URL) as conn:
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
                
                logger.info("Connection count by state:")
                total_connections = 0
                async for row in cur:
                    state, count = row
                    total_connections += count
                    logger.info(f"  {state}: {count}")
                
                logger.info(f"Total connections to current database: {total_connections}")
                
                # Get detailed connection info
                await cur.execute("""
                    SELECT 
                        pid,
                        usename,
                        application_name,
                        client_addr,
                        state,
                        query_start,
                        state_change,
                        query
                    FROM pg_stat_activity 
                    WHERE datname = current_database()
                      AND state IN ('active', 'idle', 'idle in transaction')
                    ORDER BY query_start DESC
                    LIMIT 20
                """)
                
                logger.info("\nRecent connections:")
                async for row in cur:
                    pid, usename, app_name, client_addr, state, query_start, state_change, query = row
                    query_preview = (query[:50] + "...") if query and len(query) > 50 else query
                    logger.info(f"  PID {pid}: {usename}@{client_addr} [{state}] - {app_name}")
                    if query_preview:
                        logger.info(f"    Query: {query_preview}")
                
                # Check for long-running idle connections
                await cur.execute("""
                    SELECT 
                        pid,
                        usename,
                        application_name,
                        state,
                        now() - state_change as idle_duration
                    FROM pg_stat_activity 
                    WHERE datname = current_database()
                      AND state = 'idle'
                      AND now() - state_change > interval '5 minutes'
                    ORDER BY idle_duration DESC
                """)
                
                idle_connections = await cur.fetchall()
                if idle_connections:
                    logger.warning(f"\nFound {len(idle_connections)} long-running idle connections:")
                    for row in idle_connections:
                        pid, usename, app_name, state, duration = row
                        logger.warning(f"  PID {pid}: {usename} [{app_name}] idle for {duration}")
                else:
                    logger.info("\n✅ No long-running idle connections found")
                
                return total_connections
                
    except Exception as e:
        logger.error(f"Failed to monitor connections: {e}")
        return -1


async def continuous_monitor(interval: int = 10, duration: int = 60):
    """Continuously monitor connections for a specified duration."""
    logger.info(f"Starting continuous monitoring for {duration} seconds (interval: {interval}s)")
    
    start_time = time.time()
    connection_history = []
    
    while time.time() - start_time < duration:
        logger.info(f"\n--- Monitor check at {time.strftime('%H:%M:%S')} ---")
        count = await monitor_connections()
        
        if count > 0:
            connection_history.append((time.time(), count))
        
        await asyncio.sleep(interval)
    
    # Summary
    if connection_history:
        logger.info("\n=== MONITORING SUMMARY ===")
        initial_count = connection_history[0][1]
        final_count = connection_history[-1][1]
        max_count = max(count for _, count in connection_history)
        min_count = min(count for _, count in connection_history)
        
        logger.info(f"Initial connections: {initial_count}")
        logger.info(f"Final connections: {final_count}")
        logger.info(f"Peak connections: {max_count}")
        logger.info(f"Minimum connections: {min_count}")
        logger.info(f"Net change: {final_count - initial_count}")
        
        if final_count > initial_count + 5:
            logger.warning("🚨 Potential connection leak detected!")
        else:
            logger.info("✅ Connection count appears stable")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "continuous":
        # Continuous monitoring mode
        interval = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        duration = int(sys.argv[3]) if len(sys.argv) > 3 else 60
        asyncio.run(continuous_monitor(interval, duration))
    else:
        # Single check mode
        asyncio.run(monitor_connections())
