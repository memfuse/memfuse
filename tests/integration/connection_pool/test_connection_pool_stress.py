#!/usr/bin/env python3
"""
Connection Pool Stress Test

This test analyzes connection pool performance under various stress scenarios
to identify bottlenecks and connection leaks.
"""

import asyncio
import time
import sys
import os
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass
from loguru import logger

# Add src path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from memfuse_core.services.global_connection_manager import get_global_connection_manager
from memfuse_core.services.database_service import DatabaseService
from memfuse_core.services.memory_service import MemoryService
from memfuse_core.services.buffer_service import BufferService
from memfuse_core.store.pgai_store import PgaiStore
from memfuse_core.utils.config import config_manager


@dataclass
class StressTestResult:
    """Results from a stress test scenario."""
    scenario_name: str
    duration: float
    operations_completed: int
    operations_per_second: float
    peak_connections: int
    final_connections: int
    connection_leaks: int
    errors: List[str]
    pool_stats: Dict[str, Any]


class ConnectionPoolStressTester:
    """Stress tester for connection pool analysis."""
    
    def __init__(self):
        self.connection_manager = get_global_connection_manager()
        self.initial_connections = 0
        self.peak_connections = 0
        
    async def setup(self):
        """Setup test environment."""
        logger.info("Setting up stress test environment...")
        
        # Reset any existing instances
        DatabaseService.reset_instance_sync()
        
        # Get initial connection count
        self.initial_connections = await self._get_connection_count()
        logger.info(f"Initial connections: {self.initial_connections}")
        
    async def cleanup(self):
        """Cleanup test environment."""
        logger.info("Cleaning up stress test environment...")
        
        # Close all pools
        await self.connection_manager.close_all_pools(force=True)
        
        # Reset instances
        DatabaseService.reset_instance_sync()
        
        # Wait for cleanup
        await asyncio.sleep(2)
        
    async def _get_connection_count(self) -> int:
        """Get current PostgreSQL connection count."""
        try:
            db = await DatabaseService.get_instance()
            result = await db.execute_query(
                "SELECT count(*) FROM pg_stat_activity WHERE datname = current_database()"
            )
            return result[0][0] if result else 0
        except Exception as e:
            logger.warning(f"Could not get connection count: {e}")
            return 0
            
    async def _monitor_connections(self, duration: float) -> int:
        """Monitor connection count during test and return peak."""
        peak = self.initial_connections
        start_time = time.time()
        
        while time.time() - start_time < duration:
            current = await self._get_connection_count()
            peak = max(peak, current)
            await asyncio.sleep(0.5)
            
        return peak
        
    async def test_concurrent_database_service_creation(self, concurrent_count: int = 20) -> StressTestResult:
        """Test concurrent DatabaseService instance creation."""
        logger.info(f"Testing concurrent DatabaseService creation ({concurrent_count} instances)")
        
        errors = []
        start_time = time.time()
        
        # Monitor connections in background
        monitor_task = asyncio.create_task(self._monitor_connections(30))
        
        async def create_db_service():
            try:
                # Reset to force new instance creation
                DatabaseService.reset_instance_sync()
                db = await DatabaseService.get_instance()
                # Perform a simple query
                await db.execute_query("SELECT 1")
                return True
            except Exception as e:
                errors.append(str(e))
                return False
                
        # Create concurrent tasks
        tasks = [create_db_service() for _ in range(concurrent_count)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Wait for monitoring to complete
        peak_connections = await monitor_task
        
        duration = time.time() - start_time
        successful = sum(1 for r in results if r is True)
        
        final_connections = await self._get_connection_count()
        connection_leaks = final_connections - self.initial_connections
        
        return StressTestResult(
            scenario_name="Concurrent DatabaseService Creation",
            duration=duration,
            operations_completed=successful,
            operations_per_second=successful / duration,
            peak_connections=peak_connections,
            final_connections=final_connections,
            connection_leaks=connection_leaks,
            errors=errors,
            pool_stats=self.connection_manager.get_pool_statistics()
        )
        
    async def test_concurrent_pgai_store_operations(self, concurrent_count: int = 10) -> StressTestResult:
        """Test concurrent PgaiStore operations."""
        logger.info(f"Testing concurrent PgaiStore operations ({concurrent_count} stores)")
        
        errors = []
        start_time = time.time()
        
        # Monitor connections in background
        monitor_task = asyncio.create_task(self._monitor_connections(60))
        
        async def create_and_use_store(store_id: int):
            try:
                config = config_manager.get_config()
                store = PgaiStore(
                    table_name=f"test_store_{store_id}",
                    config=config
                )
                
                # Initialize store
                await store.initialize()
                
                # Perform operations
                test_data = {
                    "content": f"Test content for store {store_id}",
                    "metadata": {"store_id": store_id, "test": True}
                }
                
                # Add data
                await store.add(test_data)
                
                # Query data
                results = await store.query("test content", limit=5)
                
                # Cleanup
                await store.close()
                
                return True
            except Exception as e:
                errors.append(f"Store {store_id}: {str(e)}")
                return False
                
        # Create concurrent tasks
        tasks = [create_and_use_store(i) for i in range(concurrent_count)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Wait for monitoring to complete
        peak_connections = await monitor_task
        
        duration = time.time() - start_time
        successful = sum(1 for r in results if r is True)
        
        final_connections = await self._get_connection_count()
        connection_leaks = final_connections - self.initial_connections
        
        return StressTestResult(
            scenario_name="Concurrent PgaiStore Operations",
            duration=duration,
            operations_completed=successful,
            operations_per_second=successful / duration,
            peak_connections=peak_connections,
            final_connections=final_connections,
            connection_leaks=connection_leaks,
            errors=errors,
            pool_stats=self.connection_manager.get_pool_statistics()
        )
        
    async def test_memory_service_parallel_processing(self, batch_count: int = 5) -> StressTestResult:
        """Test MemoryService parallel M0/M1/M2 processing."""
        logger.info(f"Testing MemoryService parallel processing ({batch_count} batches)")

        errors = []
        start_time = time.time()

        # Monitor connections in background
        monitor_task = asyncio.create_task(self._monitor_connections(120))

        try:
            # Create MemoryService with parallel processing enabled
            config = config_manager.get_config()
            # Force enable parallel processing for this test
            if "memory" not in config:
                config["memory"] = {}
            if "memory_service" not in config["memory"]:
                config["memory"]["memory_service"] = {}
            config["memory"]["memory_service"]["parallel_enabled"] = True

            memory_service = MemoryService(
                user="stress_test_user",
                agent="stress_test_agent",
                cfg=config
            )

            await memory_service.initialize()

            async def process_batch(batch_id: int):
                try:
                    # Create test message batch
                    messages = [
                        [
                            {
                                "role": "user",
                                "content": f"Test message {i} in batch {batch_id}",
                                "metadata": {"batch_id": batch_id, "message_id": i}
                            }
                        ]
                        for i in range(10)  # 10 message lists per batch
                    ]

                    # Process through MemoryService (triggers M0/M1/M2 parallel processing)
                    result = await memory_service.add_batch(messages)

                    return result.get("status") == "success"
                except Exception as e:
                    errors.append(f"Batch {batch_id}: {str(e)}")
                    return False

            # Create concurrent batch processing tasks
            tasks = [process_batch(i) for i in range(batch_count)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Cleanup
            await memory_service.close()

        except Exception as e:
            errors.append(f"MemoryService setup: {str(e)}")
            results = []

        # Wait for monitoring to complete
        peak_connections = await monitor_task

        duration = time.time() - start_time
        successful = sum(1 for r in results if r is True)

        final_connections = await self._get_connection_count()
        connection_leaks = final_connections - self.initial_connections

        return StressTestResult(
            scenario_name="MemoryService Parallel Processing",
            duration=duration,
            operations_completed=successful,
            operations_per_second=successful / duration,
            peak_connections=peak_connections,
            final_connections=final_connections,
            connection_leaks=connection_leaks,
            errors=errors,
            pool_stats=self.connection_manager.get_pool_statistics()
        )

    async def test_parallel_vs_sequential_comparison(self) -> Dict[str, StressTestResult]:
        """Compare parallel vs sequential processing performance and connection usage."""
        logger.info("Testing parallel vs sequential processing comparison")

        results = {}

        # Test 1: Sequential processing (parallel_enabled=False)
        await self.setup()
        config = config_manager.get_config()
        if "memory" not in config:
            config["memory"] = {}
        if "memory_service" not in config["memory"]:
            config["memory"]["memory_service"] = {}
        config["memory"]["memory_service"]["parallel_enabled"] = False

        sequential_result = await self._test_processing_mode("Sequential", config, 3)
        results["sequential"] = sequential_result
        await self.cleanup()

        # Test 2: Parallel processing (parallel_enabled=True)
        await self.setup()
        config["memory"]["memory_service"]["parallel_enabled"] = True

        parallel_result = await self._test_processing_mode("Parallel", config, 3)
        results["parallel"] = parallel_result
        await self.cleanup()

        return results

    async def _test_processing_mode(self, mode_name: str, config: Dict[str, Any], batch_count: int) -> StressTestResult:
        """Test a specific processing mode."""
        errors = []
        start_time = time.time()

        # Monitor connections in background
        monitor_task = asyncio.create_task(self._monitor_connections(60))

        try:
            memory_service = MemoryService(
                user=f"test_user_{mode_name.lower()}",
                agent=f"test_agent_{mode_name.lower()}",
                cfg=config
            )

            await memory_service.initialize()

            # Process batches sequentially to avoid interference
            successful = 0
            for batch_id in range(batch_count):
                try:
                    messages = [
                        [
                            {
                                "role": "user",
                                "content": f"Test message {i} in {mode_name} batch {batch_id}",
                                "metadata": {"batch_id": batch_id, "message_id": i, "mode": mode_name}
                            }
                        ]
                        for i in range(5)  # 5 message lists per batch
                    ]

                    result = await memory_service.add_batch(messages)
                    if result.get("status") == "success":
                        successful += 1

                except Exception as e:
                    errors.append(f"{mode_name} Batch {batch_id}: {str(e)}")

            await memory_service.close()

        except Exception as e:
            errors.append(f"{mode_name} MemoryService setup: {str(e)}")

        # Wait for monitoring to complete
        peak_connections = await monitor_task

        duration = time.time() - start_time
        final_connections = await self._get_connection_count()
        connection_leaks = final_connections - self.initial_connections

        return StressTestResult(
            scenario_name=f"{mode_name} Processing Mode",
            duration=duration,
            operations_completed=successful,
            operations_per_second=successful / duration,
            peak_connections=peak_connections,
            final_connections=final_connections,
            connection_leaks=connection_leaks,
            errors=errors,
            pool_stats=self.connection_manager.get_pool_statistics()
        )
        
    def print_results(self, results: List[StressTestResult]):
        """Print comprehensive test results."""
        print("\n" + "="*80)
        print("CONNECTION POOL STRESS TEST RESULTS")
        print("="*80)
        
        for result in results:
            print(f"\n📊 {result.scenario_name}")
            print(f"   Duration: {result.duration:.2f}s")
            print(f"   Operations: {result.operations_completed}")
            print(f"   Ops/sec: {result.operations_per_second:.2f}")
            print(f"   Peak connections: {result.peak_connections}")
            print(f"   Final connections: {result.final_connections}")
            
            if result.connection_leaks > 0:
                print(f"   ❌ Connection leaks: {result.connection_leaks}")
            else:
                print(f"   ✅ No connection leaks")
                
            if result.errors:
                print(f"   ❌ Errors: {len(result.errors)}")
                for error in result.errors[:3]:  # Show first 3 errors
                    print(f"      - {error}")
                if len(result.errors) > 3:
                    print(f"      ... and {len(result.errors) - 3} more")
            else:
                print(f"   ✅ No errors")
                
            if result.pool_stats:
                print(f"   📈 Pool stats: {result.pool_stats}")


async def main():
    """Run comprehensive connection pool stress tests."""
    print("🧪 Starting Connection Pool Stress Tests")
    
    tester = ConnectionPoolStressTester()
    results = []
    
    try:
        await tester.setup()
        
        # Test 1: Concurrent DatabaseService creation
        result1 = await tester.test_concurrent_database_service_creation(20)
        results.append(result1)
        await tester.cleanup()
        
        # Test 2: Concurrent PgaiStore operations
        await tester.setup()
        result2 = await tester.test_concurrent_pgai_store_operations(10)
        results.append(result2)
        await tester.cleanup()
        
        # Test 3: MemoryService parallel processing
        await tester.setup()
        result3 = await tester.test_memory_service_parallel_processing(5)
        results.append(result3)
        await tester.cleanup()

        # Test 4: Parallel vs Sequential comparison
        comparison_results = await tester.test_parallel_vs_sequential_comparison()
        results.extend(comparison_results.values())

        # Print comprehensive results
        tester.print_results(results)

        # Print comparison analysis
        if "sequential" in comparison_results and "parallel" in comparison_results:
            print("\n" + "="*80)
            print("PARALLEL VS SEQUENTIAL ANALYSIS")
            print("="*80)

            seq_result = comparison_results["sequential"]
            par_result = comparison_results["parallel"]

            print(f"📊 Performance Comparison:")
            print(f"   Sequential: {seq_result.operations_per_second:.2f} ops/sec, Peak: {seq_result.peak_connections} conn")
            print(f"   Parallel:   {par_result.operations_per_second:.2f} ops/sec, Peak: {par_result.peak_connections} conn")

            if par_result.operations_per_second > seq_result.operations_per_second:
                speedup = par_result.operations_per_second / seq_result.operations_per_second
                print(f"   ✅ Parallel is {speedup:.2f}x faster")
            else:
                slowdown = seq_result.operations_per_second / par_result.operations_per_second
                print(f"   ❌ Parallel is {slowdown:.2f}x slower")

            connection_overhead = par_result.peak_connections - seq_result.peak_connections
            if connection_overhead > 0:
                print(f"   ⚠️  Parallel uses {connection_overhead} more peak connections")
            else:
                print(f"   ✅ Parallel uses same or fewer connections")

            if par_result.connection_leaks > seq_result.connection_leaks:
                print(f"   ❌ Parallel has more connection leaks: {par_result.connection_leaks} vs {seq_result.connection_leaks}")
            else:
                print(f"   ✅ Parallel has same or fewer connection leaks")
        
        # Analysis and recommendations
        print("\n" + "="*80)
        print("ANALYSIS AND RECOMMENDATIONS")
        print("="*80)
        
        total_leaks = sum(r.connection_leaks for r in results)
        total_errors = sum(len(r.errors) for r in results)
        
        if total_leaks > 0:
            print(f"❌ Total connection leaks detected: {total_leaks}")
            print("   Recommendations:")
            print("   - Review connection cleanup in service destructors")
            print("   - Check for unclosed connections in error paths")
            print("   - Consider implementing connection timeout monitoring")
        else:
            print("✅ No connection leaks detected")
            
        if total_errors > 0:
            print(f"❌ Total errors: {total_errors}")
            print("   Recommendations:")
            print("   - Review error logs for patterns")
            print("   - Consider increasing connection pool size")
            print("   - Check for timeout issues")
        else:
            print("✅ No errors detected")
            
        # Performance analysis
        avg_ops_per_sec = sum(r.operations_per_second for r in results) / len(results)
        max_peak_connections = max(r.peak_connections for r in results)
        
        print(f"\n📈 Performance Summary:")
        print(f"   Average ops/sec: {avg_ops_per_sec:.2f}")
        print(f"   Peak connections: {max_peak_connections}")
        
        if max_peak_connections > 50:
            print("   ⚠️  High connection usage detected")
            print("   Consider optimizing connection pool configuration")
            
    except Exception as e:
        logger.error(f"Stress test failed: {e}")
        return False
        
    return True


if __name__ == "__main__":
    asyncio.run(main())
