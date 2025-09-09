"""
Performance regression tests for MemFuse components.

These tests establish performance baselines and detect regressions in critical
system components. They should be run regularly to ensure performance stability.
"""

import pytest
import asyncio
import time
import statistics
import psutil
import os
from typing import List, Dict, Any, Callable
from dataclasses import dataclass
from unittest.mock import AsyncMock

from src.memfuse_core.buffer.query_buffer import QueryBuffer
from src.memfuse_core.persistence.in_memory_store import InMemoryStore
from src.memfuse_core.persistence.adapters import make_retrieval_handler_from_store
from src.memfuse_core.models.core import Item, Query
from src.memfuse_core.gateway.filter_cache import (
    get_regex_cache,
    get_content_cache,
    get_quality_cache
)
from src.memfuse_core.gateway.semantic_validation import SemanticValidator


@dataclass
class PerformanceBaseline:
    """Performance baseline for regression testing."""
    operation: str
    max_duration_ms: float
    max_memory_mb: float
    min_throughput_ops_per_sec: float
    description: str


@dataclass
class PerformanceResult:
    """Result of a performance test."""
    operation: str
    duration_ms: float
    memory_mb: float
    throughput_ops_per_sec: float
    passed: bool
    details: Dict[str, Any]


class PerformanceRegressionTester:
    """Performance regression testing framework."""
    
    def __init__(self):
        self.baselines = {
            "regex_cache_hit": PerformanceBaseline(
                operation="regex_cache_hit",
                max_duration_ms=0.1,  # 0.1ms per operation
                max_memory_mb=10,
                min_throughput_ops_per_sec=10000,
                description="Regex pattern cache hit performance"
            ),
            "regex_cache_miss": PerformanceBaseline(
                operation="regex_cache_miss",
                max_duration_ms=50,  # 50ms per operation (compilation)
                max_memory_mb=20,
                min_throughput_ops_per_sec=20,
                description="Regex pattern cache miss performance"
            ),
            "content_cache_operations": PerformanceBaseline(
                operation="content_cache_operations",
                max_duration_ms=1.0,  # 1ms per operation
                max_memory_mb=50,
                min_throughput_ops_per_sec=1000,
                description="Content filter cache operations"
            ),
            "query_buffer_small": PerformanceBaseline(
                operation="query_buffer_small",
                max_duration_ms=10,  # 10ms per query
                max_memory_mb=100,
                min_throughput_ops_per_sec=100,
                description="QueryBuffer with small dataset"
            ),
            "query_buffer_large": PerformanceBaseline(
                operation="query_buffer_large",
                max_duration_ms=100,  # 100ms per query
                max_memory_mb=500,
                min_throughput_ops_per_sec=10,
                description="QueryBuffer with large dataset"
            ),
            "in_memory_store_add": PerformanceBaseline(
                operation="in_memory_store_add",
                max_duration_ms=5,  # 5ms per add
                max_memory_mb=200,
                min_throughput_ops_per_sec=200,
                description="InMemoryStore add operations"
            ),
            "in_memory_store_query": PerformanceBaseline(
                operation="in_memory_store_query",
                max_duration_ms=20,  # 20ms per query
                max_memory_mb=200,
                min_throughput_ops_per_sec=50,
                description="InMemoryStore query operations"
            ),
        }
    
    def measure_performance(self, operation_func: Callable, iterations: int = 100) -> Dict[str, Any]:
        """Measure performance of an operation."""
        process = psutil.Process(os.getpid())
        
        # Warm up
        for _ in range(min(10, iterations // 10)):
            operation_func()
        
        # Measure
        durations = []
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        start_time = time.perf_counter()
        for _ in range(iterations):
            op_start = time.perf_counter()
            operation_func()
            op_end = time.perf_counter()
            durations.append((op_end - op_start) * 1000)  # Convert to ms
        
        end_time = time.perf_counter()
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        
        total_duration = (end_time - start_time) * 1000  # ms
        memory_used = memory_after - memory_before
        throughput = iterations / (total_duration / 1000)  # ops per second
        
        return {
            "durations_ms": durations,
            "avg_duration_ms": statistics.mean(durations),
            "median_duration_ms": statistics.median(durations),
            "p95_duration_ms": statistics.quantiles(durations, n=20)[18],  # 95th percentile
            "p99_duration_ms": statistics.quantiles(durations, n=100)[98],  # 99th percentile
            "memory_used_mb": memory_used,
            "throughput_ops_per_sec": throughput,
            "total_duration_ms": total_duration,
            "iterations": iterations
        }
    
    async def measure_async_performance(self, operation_func: Callable, iterations: int = 100) -> Dict[str, Any]:
        """Measure performance of an async operation."""
        process = psutil.Process(os.getpid())
        
        # Warm up
        for _ in range(min(10, iterations // 10)):
            await operation_func()
        
        # Measure
        durations = []
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        start_time = time.perf_counter()
        for _ in range(iterations):
            op_start = time.perf_counter()
            await operation_func()
            op_end = time.perf_counter()
            durations.append((op_end - op_start) * 1000)  # Convert to ms
        
        end_time = time.perf_counter()
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        
        total_duration = (end_time - start_time) * 1000  # ms
        memory_used = memory_after - memory_before
        throughput = iterations / (total_duration / 1000)  # ops per second
        
        return {
            "durations_ms": durations,
            "avg_duration_ms": statistics.mean(durations),
            "median_duration_ms": statistics.median(durations),
            "p95_duration_ms": statistics.quantiles(durations, n=20)[18],  # 95th percentile
            "p99_duration_ms": statistics.quantiles(durations, n=100)[98],  # 99th percentile
            "memory_used_mb": memory_used,
            "throughput_ops_per_sec": throughput,
            "total_duration_ms": total_duration,
            "iterations": iterations
        }
    
    def check_regression(self, operation: str, results: Dict[str, Any]) -> PerformanceResult:
        """Check if performance results indicate a regression."""
        baseline = self.baselines.get(operation)
        if not baseline:
            return PerformanceResult(
                operation=operation,
                duration_ms=results["avg_duration_ms"],
                memory_mb=results["memory_used_mb"],
                throughput_ops_per_sec=results["throughput_ops_per_sec"],
                passed=True,  # No baseline, assume pass
                details={"warning": "No baseline defined for this operation"}
            )
        
        # Check against baseline
        duration_ok = results["avg_duration_ms"] <= baseline.max_duration_ms
        memory_ok = results["memory_used_mb"] <= baseline.max_memory_mb
        throughput_ok = results["throughput_ops_per_sec"] >= baseline.min_throughput_ops_per_sec
        
        passed = duration_ok and memory_ok and throughput_ok
        
        return PerformanceResult(
            operation=operation,
            duration_ms=results["avg_duration_ms"],
            memory_mb=results["memory_used_mb"],
            throughput_ops_per_sec=results["throughput_ops_per_sec"],
            passed=passed,
            details={
                "baseline": baseline,
                "duration_ok": duration_ok,
                "memory_ok": memory_ok,
                "throughput_ok": throughput_ok,
                "full_results": results
            }
        )


@pytest.fixture(scope="module")
def perf_tester():
    """Performance regression tester fixture."""
    return PerformanceRegressionTester()


@pytest.mark.performance
class TestCachePerformanceRegression:
    """Test cache performance regression."""
    
    def test_regex_cache_hit_performance(self, perf_tester):
        """Test regex cache hit performance."""
        cache = get_regex_cache()
        cache.clear()
        
        # Pre-populate cache
        pattern = r"\d{3}-\d{2}-\d{4}"
        cache.get_pattern(pattern)  # Cache miss to populate
        
        # Test cache hits
        def cache_hit_operation():
            return cache.get_pattern(pattern)
        
        results = perf_tester.measure_performance(cache_hit_operation, iterations=1000)
        regression_result = perf_tester.check_regression("regex_cache_hit", results)
        
        assert regression_result.passed, (
            f"Regex cache hit performance regression detected:\n"
            f"Duration: {regression_result.duration_ms:.3f}ms "
            f"(max: {regression_result.details['baseline'].max_duration_ms}ms)\n"
            f"Memory: {regression_result.memory_mb:.1f}MB "
            f"(max: {regression_result.details['baseline'].max_memory_mb}MB)\n"
            f"Throughput: {regression_result.throughput_ops_per_sec:.1f} ops/sec "
            f"(min: {regression_result.details['baseline'].min_throughput_ops_per_sec} ops/sec)"
        )
    
    def test_regex_cache_miss_performance(self, perf_tester):
        """Test regex cache miss performance."""
        cache = get_regex_cache()
        cache.clear()
        
        patterns = [f"pattern_{i}_\\d+_test" for i in range(100)]
        pattern_iter = iter(patterns)
        
        def cache_miss_operation():
            pattern = next(pattern_iter, "default_\\d+")
            return cache.get_pattern(pattern)
        
        results = perf_tester.measure_performance(cache_miss_operation, iterations=100)
        regression_result = perf_tester.check_regression("regex_cache_miss", results)
        
        assert regression_result.passed, (
            f"Regex cache miss performance regression detected:\n"
            f"Duration: {regression_result.duration_ms:.3f}ms "
            f"(max: {regression_result.details['baseline'].max_duration_ms}ms)\n"
            f"Memory: {regression_result.memory_mb:.1f}MB "
            f"(max: {regression_result.details['baseline'].max_memory_mb}MB)\n"
            f"Throughput: {regression_result.throughput_ops_per_sec:.1f} ops/sec "
            f"(min: {regression_result.details['baseline'].min_throughput_ops_per_sec} ops/sec)"
        )
    
    def test_content_cache_performance(self, perf_tester):
        """Test content cache performance."""
        cache = get_content_cache()
        cache.clear()
        
        contents = [f"Test content {i} with some additional text" for i in range(100)]
        config = {"test": "config"}
        content_iter = iter(contents)
        
        def cache_operation():
            content = next(content_iter, "default content")
            # Mix of cache and set operations
            if cache.get_cached_result(content, config) is None:
                cache.cache_result(content, config, {"result": "test"})
            return cache.get_cached_result(content, config)
        
        results = perf_tester.measure_performance(cache_operation, iterations=200)
        regression_result = perf_tester.check_regression("content_cache_operations", results)
        
        assert regression_result.passed, (
            f"Content cache performance regression detected:\n"
            f"Duration: {regression_result.duration_ms:.3f}ms "
            f"(max: {regression_result.details['baseline'].max_duration_ms}ms)\n"
            f"Memory: {regression_result.memory_mb:.1f}MB "
            f"(max: {regression_result.details['baseline'].max_memory_mb}MB)\n"
            f"Throughput: {regression_result.throughput_ops_per_sec:.1f} ops/sec "
            f"(min: {regression_result.details['baseline'].min_throughput_ops_per_sec} ops/sec)"
        )


@pytest.mark.performance
class TestQueryBufferPerformanceRegression:
    """Test QueryBuffer performance regression."""
    
    @pytest.mark.asyncio
    async def test_query_buffer_small_dataset_performance(self, perf_tester):
        """Test QueryBuffer performance with small dataset."""
        store = InMemoryStore()
        await store.initialize()
        
        # Add small dataset
        for i in range(50):
            item = Item(
                id=f"item_{i}",
                content=f"Test content {i}",
                metadata={"category": f"cat_{i % 5}"}
            )
            await store.add(item)
        
        retrieval_handler = make_retrieval_handler_from_store(store)
        query_buffer = QueryBuffer(retrieval_handler=retrieval_handler, max_size=20)
        
        queries = [f"test query {i}" for i in range(50)]
        query_iter = iter(queries)
        
        async def query_operation():
            query = next(query_iter, "default query")
            return await query_buffer.query(query, top_k=5)
        
        results = await perf_tester.measure_async_performance(query_operation, iterations=50)
        regression_result = perf_tester.check_regression("query_buffer_small", results)
        
        assert regression_result.passed, (
            f"QueryBuffer small dataset performance regression detected:\n"
            f"Duration: {regression_result.duration_ms:.3f}ms "
            f"(max: {regression_result.details['baseline'].max_duration_ms}ms)\n"
            f"Memory: {regression_result.memory_mb:.1f}MB "
            f"(max: {regression_result.details['baseline'].max_memory_mb}MB)\n"
            f"Throughput: {regression_result.throughput_ops_per_sec:.1f} ops/sec "
            f"(min: {regression_result.details['baseline'].min_throughput_ops_per_sec} ops/sec)"
        )

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_query_buffer_large_dataset_performance(self, perf_tester):
        """Test QueryBuffer performance with large dataset."""
        store = InMemoryStore()
        await store.initialize()

        # Add large dataset
        for i in range(500):
            item = Item(
                id=f"item_{i}",
                content=f"Large dataset test content {i} with additional text for realistic size",
                metadata={"category": f"cat_{i % 20}", "priority": i % 5}
            )
            await store.add(item)

        retrieval_handler = make_retrieval_handler_from_store(store)
        query_buffer = QueryBuffer(retrieval_handler=retrieval_handler, max_size=50)

        queries = [f"large dataset query {i}" for i in range(20)]
        query_iter = iter(queries)

        async def query_operation():
            query = next(query_iter, "default large query")
            return await query_buffer.query(query, top_k=10)

        results = await perf_tester.measure_async_performance(query_operation, iterations=20)
        regression_result = perf_tester.check_regression("query_buffer_large", results)

        assert regression_result.passed, (
            f"QueryBuffer large dataset performance regression detected:\n"
            f"Duration: {regression_result.duration_ms:.3f}ms "
            f"(max: {regression_result.details['baseline'].max_duration_ms}ms)\n"
            f"Memory: {regression_result.memory_mb:.1f}MB "
            f"(max: {regression_result.details['baseline'].max_memory_mb}MB)\n"
            f"Throughput: {regression_result.throughput_ops_per_sec:.1f} ops/sec "
            f"(min: {regression_result.details['baseline'].min_throughput_ops_per_sec} ops/sec)"
        )


@pytest.mark.performance
class TestInMemoryStorePerformanceRegression:
    """Test InMemoryStore performance regression."""

    @pytest.mark.asyncio
    async def test_in_memory_store_add_performance(self, perf_tester):
        """Test InMemoryStore add operation performance."""
        store = InMemoryStore()
        await store.initialize()

        items = [
            Item(
                id=f"perf_item_{i}",
                content=f"Performance test content {i}",
                metadata={"index": i, "category": f"cat_{i % 10}"}
            )
            for i in range(100)
        ]
        item_iter = iter(items)

        async def add_operation():
            item = next(item_iter, items[0])  # Fallback to first item
            return await store.add(item)

        results = await perf_tester.measure_async_performance(add_operation, iterations=100)
        regression_result = perf_tester.check_regression("in_memory_store_add", results)

        assert regression_result.passed, (
            f"InMemoryStore add performance regression detected:\n"
            f"Duration: {regression_result.duration_ms:.3f}ms "
            f"(max: {regression_result.details['baseline'].max_duration_ms}ms)\n"
            f"Memory: {regression_result.memory_mb:.1f}MB "
            f"(max: {regression_result.details['baseline'].max_memory_mb}MB)\n"
            f"Throughput: {regression_result.throughput_ops_per_sec:.1f} ops/sec "
            f"(min: {regression_result.details['baseline'].min_throughput_ops_per_sec} ops/sec)"
        )

    @pytest.mark.asyncio
    async def test_in_memory_store_query_performance(self, perf_tester):
        """Test InMemoryStore query operation performance."""
        store = InMemoryStore()
        await store.initialize()

        # Pre-populate store
        for i in range(200):
            item = Item(
                id=f"query_item_{i}",
                content=f"Query performance test content {i} with searchable terms",
                metadata={"index": i, "category": f"cat_{i % 15}"}
            )
            await store.add(item)

        queries = [
            Query(text=f"performance test {i}", metadata={"top_k": 5})
            for i in range(50)
        ]
        query_iter = iter(queries)

        async def query_operation():
            query = next(query_iter, queries[0])  # Fallback to first query
            return await store.query(query)

        results = await perf_tester.measure_async_performance(query_operation, iterations=50)
        regression_result = perf_tester.check_regression("in_memory_store_query", results)

        assert regression_result.passed, (
            f"InMemoryStore query performance regression detected:\n"
            f"Duration: {regression_result.duration_ms:.3f}ms "
            f"(max: {regression_result.details['baseline'].max_duration_ms}ms)\n"
            f"Memory: {regression_result.memory_mb:.1f}MB "
            f"(max: {regression_result.details['baseline'].max_memory_mb}MB)\n"
            f"Throughput: {regression_result.throughput_ops_per_sec:.1f} ops/sec "
            f"(min: {regression_result.details['baseline'].min_throughput_ops_per_sec} ops/sec)"
        )


@pytest.mark.performance
class TestSemanticValidationPerformanceRegression:
    """Test semantic validation performance regression."""

    @pytest.mark.asyncio
    async def test_semantic_validation_performance(self, perf_tester):
        """Test semantic validation performance."""
        validator = SemanticValidator()
        validator.enabled = True

        # Mock encoder for consistent performance
        from unittest.mock import MagicMock
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = [[0.1, 0.2, 0.3, 0.4, 0.5] * 77]  # 384 dimensions
        validator.encoder = mock_encoder

        contents = [
            f"Semantic validation test content {i} with various terms and phrases"
            for i in range(50)
        ]
        content_iter = iter(contents)

        async def validation_operation():
            content = next(content_iter, "default content")
            return await validator.validate_content(content)

        results = await perf_tester.measure_async_performance(validation_operation, iterations=50)

        # Custom baseline for semantic validation
        baseline = PerformanceBaseline(
            operation="semantic_validation",
            max_duration_ms=50,  # 50ms per validation
            max_memory_mb=100,
            min_throughput_ops_per_sec=20,
            description="Semantic validation performance"
        )

        # Check against baseline
        duration_ok = results["avg_duration_ms"] <= baseline.max_duration_ms
        memory_ok = results["memory_used_mb"] <= baseline.max_memory_mb
        throughput_ok = results["throughput_ops_per_sec"] >= baseline.min_throughput_ops_per_sec

        passed = duration_ok and memory_ok and throughput_ok

        assert passed, (
            f"Semantic validation performance regression detected:\n"
            f"Duration: {results['avg_duration_ms']:.3f}ms (max: {baseline.max_duration_ms}ms)\n"
            f"Memory: {results['memory_used_mb']:.1f}MB (max: {baseline.max_memory_mb}MB)\n"
            f"Throughput: {results['throughput_ops_per_sec']:.1f} ops/sec "
            f"(min: {baseline.min_throughput_ops_per_sec} ops/sec)"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "performance"])
