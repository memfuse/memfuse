"""
Performance tests for query method optimizations.

This test verifies that the optimized error handling and unified query method
maintain good performance characteristics.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch
from typing import List

from memfuse_core.store.pgai_store.event_driven_store import EventDrivenPgaiStore
from memfuse_core.interfaces.chunk_store import ChunkData


class TestQueryMethodPerformance:
    """Performance tests for EventDrivenPgaiStore query method."""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration for performance testing."""
        return {
            "pgai": {
                "host": "localhost",
                "port": 5432,
                "database": "test_db",
                "user": "test_user",
                "password": "test_pass"
            }
        }

    @pytest.fixture
    def initialized_store(self, mock_config):
        """Create an initialized store for performance testing."""
        store = EventDrivenPgaiStore(config=mock_config, table_name="test_table")
        
        # Mock successful initialization
        mock_core_store = AsyncMock()
        mock_core_store.query.return_value = []
        mock_core_store.add.return_value = []
        mock_core_store.get.return_value = None
        mock_core_store.delete.return_value = True
        mock_core_store.update.return_value = True
        mock_core_store.list_chunks.return_value = []
        mock_core_store.count.return_value = 0
        
        store.initialized = True
        store.core_store = mock_core_store
        
        return store

    @pytest.fixture
    def failed_store(self, mock_config):
        """Create a store that fails initialization for error path testing."""
        store = EventDrivenPgaiStore(config=mock_config, table_name="test_table")
        store.initialized = False
        store.core_store = None
        return store

    @pytest.mark.asyncio
    async def test_query_method_performance_success_path(self, initialized_store):
        """Test query method performance on success path."""
        iterations = 100
        
        start_time = time.time()
        for _ in range(iterations):
            await initialized_store.query("test query", top_k=5)
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / iterations
        
        # Should average less than 1ms per call
        assert avg_time < 0.001, f"Query method too slow: {avg_time:.4f}s per call"
        
        print(f"Query method performance: {avg_time:.4f}s per call ({iterations} iterations)")

    @pytest.mark.asyncio
    async def test_query_method_performance_error_path(self, failed_store):
        """Test query method performance on error path."""
        iterations = 100
        
        with patch.object(failed_store, 'initialize', return_value=False):
            start_time = time.time()
            for _ in range(iterations):
                result = await failed_store.query("test query", top_k=5)
                assert result == []
            end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / iterations
        
        # Error path should still be fast (less than 2ms per call)
        assert avg_time < 0.002, f"Error path too slow: {avg_time:.4f}s per call"
        
        print(f"Error path performance: {avg_time:.4f}s per call ({iterations} iterations)")

    @pytest.mark.asyncio
    async def test_ensure_initialized_performance(self, initialized_store):
        """Test _ensure_initialized helper method performance."""
        iterations = 1000
        
        start_time = time.time()
        for _ in range(iterations):
            result = await initialized_store._ensure_initialized("test_operation")
            assert result is True
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / iterations
        
        # Should be very fast since store is already initialized
        assert avg_time < 0.0001, f"_ensure_initialized too slow: {avg_time:.6f}s per call"
        
        print(f"_ensure_initialized performance: {avg_time:.6f}s per call ({iterations} iterations)")

    @pytest.mark.asyncio
    async def test_all_methods_performance_comparison(self, initialized_store):
        """Compare performance of all optimized methods."""
        iterations = 50
        methods_to_test = [
            ("query", lambda: initialized_store.query("test", 5)),
            ("add", lambda: initialized_store.add([])),
            ("get", lambda: initialized_store.get("test_id")),
            ("delete", lambda: initialized_store.delete(["test_id"])),
            ("update", lambda: initialized_store.update([])),
            ("list_chunks", lambda: initialized_store.list_chunks()),
            ("count", lambda: initialized_store.count()),
        ]
        
        performance_results = {}
        
        for method_name, method_call in methods_to_test:
            start_time = time.time()
            for _ in range(iterations):
                await method_call()
            end_time = time.time()
            
            total_time = end_time - start_time
            avg_time = total_time / iterations
            performance_results[method_name] = avg_time
            
            # All methods should be fast
            assert avg_time < 0.002, f"{method_name} too slow: {avg_time:.4f}s per call"
        
        # Print performance summary
        print("\nMethod Performance Summary:")
        for method_name, avg_time in performance_results.items():
            print(f"  {method_name}: {avg_time:.4f}s per call")

    @pytest.mark.asyncio
    async def test_concurrent_query_performance(self, initialized_store):
        """Test query method performance under concurrent load."""
        concurrent_calls = 20
        calls_per_task = 10
        
        async def query_task():
            for _ in range(calls_per_task):
                await initialized_store.query("concurrent test", top_k=5)
        
        start_time = time.time()
        tasks = [query_task() for _ in range(concurrent_calls)]
        await asyncio.gather(*tasks)
        end_time = time.time()
        
        total_time = end_time - start_time
        total_calls = concurrent_calls * calls_per_task
        avg_time = total_time / total_calls
        
        # Should handle concurrent load well
        assert avg_time < 0.005, f"Concurrent performance too slow: {avg_time:.4f}s per call"
        
        print(f"Concurrent performance: {avg_time:.4f}s per call ({total_calls} total calls)")

    @pytest.mark.asyncio
    async def test_memory_usage_stability(self, initialized_store):
        """Test that repeated calls don't cause memory leaks."""
        import gc
        import sys
        
        # Force garbage collection
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Perform many operations
        for _ in range(100):
            await initialized_store.query("memory test", top_k=5)
            await initialized_store.add([])
            await initialized_store.get("test_id")
            await initialized_store.delete(["test_id"])
            await initialized_store.update([])
            await initialized_store.list_chunks()
            await initialized_store.count()
        
        # Force garbage collection again
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Object count should not grow significantly
        object_growth = final_objects - initial_objects
        assert object_growth < 100, f"Potential memory leak: {object_growth} new objects"
        
        print(f"Memory stability: {object_growth} new objects after 700 operations")

    @pytest.mark.asyncio
    async def test_error_handling_overhead(self, mock_config):
        """Test overhead of error handling in optimized code."""
        # Create two stores: one that will succeed, one that will fail
        success_store = EventDrivenPgaiStore(config=mock_config, table_name="success")
        success_store.initialized = True
        success_store.core_store = AsyncMock()
        success_store.core_store.query.return_value = []
        
        fail_store = EventDrivenPgaiStore(config=mock_config, table_name="fail")
        fail_store.initialized = False
        fail_store.core_store = None
        
        iterations = 100
        
        # Time success path
        start_time = time.time()
        for _ in range(iterations):
            await success_store.query("test", 5)
        success_time = time.time() - start_time
        
        # Time error path
        with patch.object(fail_store, 'initialize', return_value=False):
            start_time = time.time()
            for _ in range(iterations):
                await fail_store.query("test", 5)
            error_time = time.time() - start_time
        
        # Error path should not be more than 3x slower than success path
        overhead_ratio = error_time / success_time
        assert overhead_ratio < 3.0, f"Error handling overhead too high: {overhead_ratio:.2f}x"
        
        print(f"Error handling overhead: {overhead_ratio:.2f}x slower than success path")

    @pytest.mark.asyncio
    async def test_large_query_performance(self, initialized_store):
        """Test performance with large query strings."""
        # Create a large query string
        large_query = "space exploration " * 100  # ~1800 characters
        
        iterations = 50
        
        start_time = time.time()
        for _ in range(iterations):
            await initialized_store.query(large_query, top_k=10)
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / iterations
        
        # Should handle large queries efficiently
        assert avg_time < 0.005, f"Large query performance too slow: {avg_time:.4f}s per call"
        
        print(f"Large query performance: {avg_time:.4f}s per call (query length: {len(large_query)})")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
