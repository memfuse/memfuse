"""
Performance tests for connection pool optimization.

Tests the Phase 1 optimizations:
1. Lock-free connection pool access with read-write locks
2. Connection pool warmup for common databases
3. Fast path for existing pools vs slow path for new pools
"""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, patch, MagicMock
from typing import List, Dict, Any

from src.memfuse_core.services.global_connection_manager import (
    GlobalConnectionManager,
    get_global_connection_manager,
    warmup_global_connection_pools,
    AsyncRWLock
)


class TestAsyncRWLock:
    """Test the async read-write lock implementation."""
    
    @pytest.mark.asyncio
    async def test_multiple_readers(self):
        """Test that multiple readers can acquire lock simultaneously."""
        lock = AsyncRWLock()
        results = []
        
        async def reader_task(reader_id: int):
            async with lock.reader():
                results.append(f"reader_{reader_id}_start")
                await asyncio.sleep(0.1)  # Simulate work
                results.append(f"reader_{reader_id}_end")
        
        # Start multiple readers simultaneously
        tasks = [reader_task(i) for i in range(3)]
        await asyncio.gather(*tasks)
        
        # All readers should have started before any ended
        start_events = [r for r in results if r.endswith("_start")]
        end_events = [r for r in results if r.endswith("_end")]
        
        assert len(start_events) == 3
        assert len(end_events) == 3
        
        # All starts should come before all ends (parallel execution)
        first_end_index = results.index(end_events[0])
        last_start_index = max(results.index(start) for start in start_events)
        assert last_start_index < first_end_index
    
    @pytest.mark.asyncio
    async def test_writer_exclusivity(self):
        """Test that writers have exclusive access."""
        lock = AsyncRWLock()
        results = []
        
        async def writer_task(writer_id: int):
            async with lock.writer():
                results.append(f"writer_{writer_id}_start")
                await asyncio.sleep(0.1)  # Simulate work
                results.append(f"writer_{writer_id}_end")
        
        async def reader_task(reader_id: int):
            async with lock.reader():
                results.append(f"reader_{reader_id}_start")
                await asyncio.sleep(0.05)  # Shorter work
                results.append(f"reader_{reader_id}_end")
        
        # Start writer and reader simultaneously
        tasks = [writer_task(1), reader_task(1)]
        await asyncio.gather(*tasks)
        
        # Writer and reader should not overlap
        writer_start = results.index("writer_1_start")
        writer_end = results.index("writer_1_end")
        reader_start = results.index("reader_1_start")
        reader_end = results.index("reader_1_end")
        
        # Either writer completes before reader starts, or reader completes before writer starts
        assert (writer_end < reader_start) or (reader_end < writer_start)


class TestConnectionPoolOptimization:
    """Test connection pool performance optimizations."""
    
    @pytest.fixture
    def mock_pool(self):
        """Create a mock connection pool."""
        pool = AsyncMock()
        pool.closed = False
        pool.connection.return_value.__aenter__ = AsyncMock()
        pool.connection.return_value.__aexit__ = AsyncMock()
        
        # Mock connection for warmup testing
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()
        pool.connection.return_value.__aenter__.return_value = mock_conn
        
        return pool
    
    @pytest.fixture
    def manager(self):
        """Create a fresh connection manager for testing."""
        # Reset singleton
        GlobalConnectionManager._instance = None
        return GlobalConnectionManager()
    
    @pytest.mark.asyncio
    async def test_fast_path_performance(self, manager, mock_pool):
        """Test that fast path (read lock) is significantly faster than slow path."""
        db_url = "postgresql://test:test@localhost:5432/test"
        
        # Mock the pool creation to avoid actual database connections
        with patch('src.memfuse_core.services.global_connection_manager.AsyncConnectionPool') as mock_pool_class:
            mock_pool_class.return_value = mock_pool
            
            # First call creates the pool (slow path)
            start_time = time.perf_counter()
            pool1 = await manager.get_connection_pool(db_url)
            slow_path_time = time.perf_counter() - start_time
            
            # Subsequent calls use fast path
            fast_path_times = []
            for _ in range(10):
                start_time = time.perf_counter()
                pool2 = await manager.get_connection_pool(db_url)
                fast_path_time = time.perf_counter() - start_time
                fast_path_times.append(fast_path_time)
                
                assert pool1 is pool2  # Same pool instance
            
            avg_fast_path_time = sum(fast_path_times) / len(fast_path_times)
            
            # Fast path should be at least 10x faster than slow path
            assert avg_fast_path_time < slow_path_time / 10
            
            # Fast path should be very fast (< 1ms)
            assert avg_fast_path_time < 0.001
    
    @pytest.mark.asyncio
    async def test_concurrent_access_performance(self, manager, mock_pool):
        """Test performance under high concurrency."""
        db_url = "postgresql://test:test@localhost:5432/test"
        
        with patch('src.memfuse_core.services.global_connection_manager.AsyncConnectionPool') as mock_pool_class:
            mock_pool_class.return_value = mock_pool
            
            # Pre-create the pool
            await manager.get_connection_pool(db_url)
            
            # Test concurrent access
            async def concurrent_access():
                return await manager.get_connection_pool(db_url)
            
            # Run 50 concurrent requests
            start_time = time.perf_counter()
            tasks = [concurrent_access() for _ in range(50)]
            pools = await asyncio.gather(*tasks)
            total_time = time.perf_counter() - start_time
            
            # All should return the same pool
            assert all(pool is pools[0] for pool in pools)
            
            # Should complete quickly even under high concurrency
            assert total_time < 0.1  # 100ms for 50 concurrent requests
            
            # Average time per request should be very low
            avg_time_per_request = total_time / 50
            assert avg_time_per_request < 0.002  # 2ms per request
    
    @pytest.mark.asyncio
    async def test_warmup_functionality(self, manager, mock_pool):
        """Test connection pool warmup functionality."""
        db_urls = [
            "postgresql://test1:test@localhost:5432/test1",
            "postgresql://test2:test@localhost:5432/test2",
            "postgresql://test3:test@localhost:5432/test3"
        ]
        
        with patch('src.memfuse_core.services.global_connection_manager.AsyncConnectionPool') as mock_pool_class:
            mock_pool_class.return_value = mock_pool
            
            # Warmup should create pools for all URLs
            start_time = time.perf_counter()
            await manager.warmup_common_pools(db_urls)
            warmup_time = time.perf_counter() - start_time
            
            # Verify pools were created
            assert len(manager._pools) == 3
            for db_url in db_urls:
                assert db_url in manager._pools
            
            # Subsequent access should be fast (fast path)
            fast_access_times = []
            for db_url in db_urls:
                start_time = time.perf_counter()
                pool = await manager.get_connection_pool(db_url)
                access_time = time.perf_counter() - start_time
                fast_access_times.append(access_time)
                
                assert pool is mock_pool
            
            avg_fast_access_time = sum(fast_access_times) / len(fast_access_times)
            
            # Fast access should be much faster than warmup
            assert avg_fast_access_time < warmup_time / 10
            assert avg_fast_access_time < 0.001  # < 1ms
    
    @pytest.mark.asyncio
    async def test_global_warmup_function(self, mock_pool):
        """Test the global warmup function."""
        db_urls = ["postgresql://test:test@localhost:5432/test"]
        
        with patch('src.memfuse_core.services.global_connection_manager.AsyncConnectionPool') as mock_pool_class:
            mock_pool_class.return_value = mock_pool
            
            # Reset singleton to ensure clean state
            GlobalConnectionManager._instance = None
            
            # Test global warmup function
            await warmup_global_connection_pools(db_urls)
            
            # Verify manager was created and pools were warmed up
            manager = get_global_connection_manager()
            assert len(manager._pools) == 1
            assert db_urls[0] in manager._pools
    
    @pytest.mark.asyncio
    async def test_performance_regression_detection(self, manager, mock_pool):
        """Test that we can detect performance regressions."""
        db_url = "postgresql://test:test@localhost:5432/test"
        
        with patch('src.memfuse_core.services.global_connection_manager.AsyncConnectionPool') as mock_pool_class:
            mock_pool_class.return_value = mock_pool
            
            # Measure baseline performance (first access - slow path)
            start_time = time.perf_counter()
            await manager.get_connection_pool(db_url)
            baseline_time = time.perf_counter() - start_time
            
            # Measure optimized performance (subsequent access - fast path)
            times = []
            for _ in range(100):
                start_time = time.perf_counter()
                await manager.get_connection_pool(db_url)
                times.append(time.perf_counter() - start_time)
            
            avg_optimized_time = sum(times) / len(times)
            p95_optimized_time = sorted(times)[94]  # 95th percentile
            
            # Performance targets from optimization plan
            assert avg_optimized_time < 0.001  # < 1ms average
            assert p95_optimized_time < 0.002   # < 2ms 95th percentile
            assert avg_optimized_time < baseline_time / 20  # 20x improvement
            
            print(f"Baseline time: {baseline_time*1000:.2f}ms")
            print(f"Optimized avg time: {avg_optimized_time*1000:.2f}ms")
            print(f"Optimized p95 time: {p95_optimized_time*1000:.2f}ms")
            print(f"Improvement factor: {baseline_time/avg_optimized_time:.1f}x")


if __name__ == "__main__":
    # Run performance tests
    pytest.main([__file__, "-v", "-s"])
