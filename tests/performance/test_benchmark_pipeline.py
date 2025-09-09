"""
Performance benchmark tests for MemFuse pipeline components.

These tests measure performance characteristics of various components
under different load conditions and provide baseline metrics.
"""

import pytest
import asyncio
import time
import statistics
from typing import List, Dict, Any
from unittest.mock import AsyncMock

from src.memfuse_core.gateway.api_gateway import MemoryApiGateway
from src.memfuse_core.buffer.query_buffer import QueryBuffer
from src.memfuse_core.persistence.in_memory_store import InMemoryStore
from src.memfuse_core.persistence.adapters import make_retrieval_handler_from_store
from src.memfuse_core.models.core import Item
from src.memfuse_core.gateway.filter_cache import (
    get_regex_cache,
    get_content_cache,
    get_quality_cache
)
from src.memfuse_core.utils.global_config_manager import get_global_config_manager


class PerformanceMetrics:
    """Helper class to collect and analyze performance metrics."""
    
    def __init__(self):
        self.measurements: List[float] = []
    
    def add_measurement(self, duration: float):
        """Add a duration measurement in seconds."""
        self.measurements.append(duration)
    
    def get_stats(self) -> Dict[str, float]:
        """Get statistical summary of measurements."""
        if not self.measurements:
            return {}
        
        return {
            "count": len(self.measurements),
            "mean": statistics.mean(self.measurements),
            "median": statistics.median(self.measurements),
            "min": min(self.measurements),
            "max": max(self.measurements),
            "std_dev": statistics.stdev(self.measurements) if len(self.measurements) > 1 else 0,
            "p95": statistics.quantiles(self.measurements, n=20)[18] if len(self.measurements) > 1 else self.measurements[0],
            "p99": statistics.quantiles(self.measurements, n=100)[98] if len(self.measurements) > 1 else self.measurements[0]
        }


class MockHighPerformanceBufferService:
    """High-performance mock buffer service for benchmarking."""
    
    def __init__(self, result_count: int = 100):
        # Pre-generate results to avoid generation overhead during benchmarks
        self.results = [
            {
                "id": f"result_{i}",
                "content": f"This is test content number {i} with some additional text to make it realistic.",
                "relevance_score": 0.9 - (i * 0.001),  # Decreasing scores
                "metadata": {
                    "source": "benchmark_data",
                    "index": i,
                    "category": f"category_{i % 10}"
                }
            }
            for i in range(result_count)
        ]
    
    async def query(self, query: str, top_k: int = 5, **kwargs) -> Dict[str, Any]:
        """Ultra-fast mock query with minimal overhead."""
        return {
            "status": "success",
            "code": 200,
            "data": {
                "results": self.results[:top_k],
                "total": len(self.results)
            },
            "message": "ok",
            "errors": None
        }


@pytest.mark.performance
class TestGatewayPerformance:
    """Performance tests for Gateway component."""
    
    @pytest.fixture
    async def high_performance_gateway(self):
        """Create a gateway optimized for performance testing."""
        # Minimal configuration for maximum performance
        gcm = get_global_config_manager()
        await gcm.hot_reload({
            "gateway": {
                "pipeline": {
                    "inbound": [],
                    "outbound": [
                        {"name": "field_remover", "enabled": True}
                    ]
                },
                "debug": {
                    "enabled": False,  # Disable debug for performance
                    "include_durations": False
                }
            },
            "guardrail": {
                "enabled": True,
                "output": {"enabled": True, "remove_fields": ["metadata.internal"]}
            }
        })
        
        buffer_service = MockHighPerformanceBufferService(1000)
        return MemoryApiGateway(buffer_service=buffer_service, db_service=None)
    
    @pytest.mark.asyncio
    async def test_single_request_latency(self, high_performance_gateway):
        """Measure single request latency."""
        metrics = PerformanceMetrics()
        
        request_data = {
            "user_id": "perf_user",
            "agent_id": "perf_agent",
            "session_id": "perf_session",
            "query": "performance test query",
            "top_k": 10
        }
        
        # Warm up
        for _ in range(5):
            await high_performance_gateway.process_request(request_data)
        
        # Measure performance
        for _ in range(100):
            start_time = time.perf_counter()
            response = await high_performance_gateway.process_request(request_data)
            end_time = time.perf_counter()
            
            assert response["status"] == "success"
            metrics.add_measurement(end_time - start_time)
        
        stats = metrics.get_stats()
        print(f"\nSingle Request Latency Stats:")
        print(f"  Mean: {stats['mean']*1000:.2f}ms")
        print(f"  Median: {stats['median']*1000:.2f}ms")
        print(f"  P95: {stats['p95']*1000:.2f}ms")
        print(f"  P99: {stats['p99']*1000:.2f}ms")
        
        # Performance assertions (adjust based on your requirements)
        assert stats['mean'] < 0.1, f"Mean latency too high: {stats['mean']*1000:.2f}ms"
        assert stats['p95'] < 0.2, f"P95 latency too high: {stats['p95']*1000:.2f}ms"
    
    @pytest.mark.asyncio
    async def test_concurrent_request_throughput(self, high_performance_gateway):
        """Measure throughput under concurrent load."""
        async def make_request(request_id: int):
            request_data = {
                "user_id": f"user_{request_id}",
                "query": f"query_{request_id}",
                "top_k": 5
            }
            start_time = time.perf_counter()
            response = await high_performance_gateway.process_request(request_data)
            end_time = time.perf_counter()
            return end_time - start_time, response["status"] == "success"
        
        # Test different concurrency levels
        concurrency_levels = [1, 5, 10, 20, 50]
        
        for concurrency in concurrency_levels:
            print(f"\nTesting concurrency level: {concurrency}")
            
            start_time = time.perf_counter()
            tasks = [make_request(i) for i in range(concurrency)]
            results = await asyncio.gather(*tasks)
            end_time = time.perf_counter()
            
            total_time = end_time - start_time
            successful_requests = sum(1 for _, success in results if success)
            throughput = successful_requests / total_time
            
            latencies = [duration for duration, _ in results]
            avg_latency = statistics.mean(latencies)
            
            print(f"  Throughput: {throughput:.2f} requests/second")
            print(f"  Average latency: {avg_latency*1000:.2f}ms")
            print(f"  Success rate: {successful_requests/concurrency*100:.1f}%")
            
            # Performance assertions
            assert successful_requests == concurrency, "Some requests failed"
            assert throughput > concurrency * 0.5, f"Throughput too low: {throughput:.2f} req/s"


@pytest.mark.performance
class TestCachePerformance:
    """Performance tests for caching components."""
    
    @pytest.mark.asyncio
    async def test_regex_cache_performance(self):
        """Test regex pattern cache performance."""
        cache = get_regex_cache()
        cache.clear()
        
        patterns = [
            r"\b(password|secret|confidential)\b",
            r"\b\d{3}-\d{2}-\d{4}\b",  # SSN pattern
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email pattern
            r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b",  # Credit card pattern
            r"\b(?:\d{1,3}\.){3}\d{1,3}\b"  # IP address pattern
        ]
        
        metrics = PerformanceMetrics()
        
        # Test cache miss performance (compilation)
        for pattern in patterns:
            start_time = time.perf_counter()
            compiled_pattern = cache.get_pattern(pattern)
            end_time = time.perf_counter()
            
            assert compiled_pattern is not None
            metrics.add_measurement(end_time - start_time)
        
        miss_stats = metrics.get_stats()
        
        # Test cache hit performance
        hit_metrics = PerformanceMetrics()
        for _ in range(1000):
            for pattern in patterns:
                start_time = time.perf_counter()
                compiled_pattern = cache.get_pattern(pattern)
                end_time = time.perf_counter()
                
                hit_metrics.add_measurement(end_time - start_time)
        
        hit_stats = hit_metrics.get_stats()
        
        print(f"\nRegex Cache Performance:")
        print(f"  Cache miss (compilation) - Mean: {miss_stats['mean']*1000000:.2f}μs")
        print(f"  Cache hit - Mean: {hit_stats['mean']*1000000:.2f}μs")
        print(f"  Speedup: {miss_stats['mean']/hit_stats['mean']:.1f}x")
        
        # Cache hits should be significantly faster
        assert hit_stats['mean'] < miss_stats['mean'] * 0.1, "Cache not providing expected speedup"
    
    @pytest.mark.asyncio
    async def test_content_cache_performance(self):
        """Test content filter cache performance."""
        cache = get_content_cache()
        cache.clear()
        
        # Generate test content
        test_contents = [
            f"This is test content number {i} with various patterns and data."
            for i in range(100)
        ]
        
        config_hash = "test_config_hash"
        
        # Test cache miss performance
        miss_metrics = PerformanceMetrics()
        for content in test_contents:
            start_time = time.perf_counter()
            result = cache.get(content, config_hash)
            end_time = time.perf_counter()
            
            assert result is None  # Should be cache miss
            miss_metrics.add_measurement(end_time - start_time)
            
            # Store result in cache
            cache.set(content, config_hash, {"filtered": True, "score": 0.8})
        
        # Test cache hit performance
        hit_metrics = PerformanceMetrics()
        for _ in range(10):  # Multiple rounds
            for content in test_contents:
                start_time = time.perf_counter()
                result = cache.get(content, config_hash)
                end_time = time.perf_counter()
                
                assert result is not None  # Should be cache hit
                hit_metrics.add_measurement(end_time - start_time)
        
        miss_stats = miss_metrics.get_stats()
        hit_stats = hit_metrics.get_stats()
        
        print(f"\nContent Cache Performance:")
        print(f"  Cache miss - Mean: {miss_stats['mean']*1000000:.2f}μs")
        print(f"  Cache hit - Mean: {hit_stats['mean']*1000000:.2f}μs")
        
        # Cache hits should be faster than misses
        assert hit_stats['mean'] < miss_stats['mean'] * 2, "Cache hit not faster than miss"


@pytest.mark.performance
class TestQueryBufferPerformance:
    """Performance tests for QueryBuffer component."""
    
    @pytest.fixture
    async def performance_query_buffer(self):
        """Create QueryBuffer with large dataset for performance testing."""
        store = InMemoryStore()
        await store.initialize()
        
        # Add large number of items
        items = []
        for i in range(1000):
            item = Item(
                id=f"item_{i}",
                content=f"Performance test content {i} with additional text for realistic size",
                metadata={"category": f"cat_{i % 20}", "priority": i % 5}
            )
            items.append(item)
            await store.add(item)
        
        retrieval_handler = make_retrieval_handler_from_store(store)
        return QueryBuffer(retrieval_handler=retrieval_handler, max_size=100)
    
    @pytest.mark.asyncio
    async def test_query_performance_scaling(self, performance_query_buffer):
        """Test query performance with different result set sizes."""
        query_buffer = performance_query_buffer
        
        top_k_values = [1, 5, 10, 25, 50, 100]
        
        for top_k in top_k_values:
            metrics = PerformanceMetrics()
            
            # Warm up
            await query_buffer.query("performance", top_k=top_k)
            
            # Measure performance
            for i in range(50):
                start_time = time.perf_counter()
                results = await query_buffer.query(f"test query {i}", top_k=top_k)
                end_time = time.perf_counter()
                
                assert len(results) <= top_k
                metrics.add_measurement(end_time - start_time)
            
            stats = metrics.get_stats()
            print(f"\nQuery Performance (top_k={top_k}):")
            print(f"  Mean: {stats['mean']*1000:.2f}ms")
            print(f"  P95: {stats['p95']*1000:.2f}ms")
            
            # Performance should scale reasonably with result size
            assert stats['mean'] < 0.5, f"Query too slow for top_k={top_k}: {stats['mean']*1000:.2f}ms"
    
    @pytest.mark.asyncio
    async def test_cache_hit_performance(self, performance_query_buffer):
        """Test cache hit vs miss performance."""
        query_buffer = performance_query_buffer
        
        # Test cache miss
        miss_metrics = PerformanceMetrics()
        unique_queries = [f"unique query {i}" for i in range(50)]
        
        for query in unique_queries:
            start_time = time.perf_counter()
            results = await query_buffer.query(query, top_k=10)
            end_time = time.perf_counter()
            
            miss_metrics.add_measurement(end_time - start_time)
        
        # Test cache hit
        hit_metrics = PerformanceMetrics()
        for _ in range(5):  # Multiple rounds of same queries
            for query in unique_queries:
                start_time = time.perf_counter()
                results = await query_buffer.query(query, top_k=10)
                end_time = time.perf_counter()
                
                hit_metrics.add_measurement(end_time - start_time)
        
        miss_stats = miss_metrics.get_stats()
        hit_stats = hit_metrics.get_stats()
        
        print(f"\nQueryBuffer Cache Performance:")
        print(f"  Cache miss - Mean: {miss_stats['mean']*1000:.2f}ms")
        print(f"  Cache hit - Mean: {hit_stats['mean']*1000:.2f}ms")
        print(f"  Speedup: {miss_stats['mean']/hit_stats['mean']:.1f}x")
        
        # Cache hits should be significantly faster
        assert hit_stats['mean'] < miss_stats['mean'] * 0.5, "Cache not providing expected speedup"


if __name__ == "__main__":
    # Run performance tests with verbose output
    pytest.main([__file__, "-v", "-s", "-m", "performance"])
