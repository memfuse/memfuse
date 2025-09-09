"""
Memory leak detection tests for MemFuse components.

These tests monitor memory usage patterns to detect potential memory leaks
in long-running operations and repeated usage scenarios.
"""

import pytest
import asyncio
import gc
import psutil
import os
import time
from typing import List, Dict, Any
from dataclasses import dataclass

from src.memfuse_core.buffer.query_buffer import QueryBuffer
from src.memfuse_core.persistence.in_memory_store import InMemoryStore
from src.memfuse_core.persistence.adapters import make_retrieval_handler_from_store
from src.memfuse_core.models.core import Item, Query
from src.memfuse_core.gateway.filter_cache import (
    get_regex_cache,
    get_content_cache,
    get_quality_cache
)


@dataclass
class MemorySnapshot:
    """Memory usage snapshot."""
    timestamp: float
    rss_mb: float
    vms_mb: float
    percent: float
    available_mb: float


class MemoryLeakDetector:
    """Memory leak detection utility."""
    
    def __init__(self, threshold_mb: float = 50.0, sample_interval: float = 0.1):
        self.threshold_mb = threshold_mb
        self.sample_interval = sample_interval
        self.process = psutil.Process(os.getpid())
        self.snapshots: List[MemorySnapshot] = []
    
    def take_snapshot(self) -> MemorySnapshot:
        """Take a memory usage snapshot."""
        memory_info = self.process.memory_info()
        memory_percent = self.process.memory_percent()
        virtual_memory = psutil.virtual_memory()
        
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            rss_mb=memory_info.rss / 1024 / 1024,
            vms_mb=memory_info.vms / 1024 / 1024,
            percent=memory_percent,
            available_mb=virtual_memory.available / 1024 / 1024
        )
        
        self.snapshots.append(snapshot)
        return snapshot
    
    def clear_snapshots(self):
        """Clear all snapshots."""
        self.snapshots.clear()
    
    def analyze_trend(self) -> Dict[str, Any]:
        """Analyze memory usage trend."""
        if len(self.snapshots) < 2:
            return {"error": "Not enough snapshots for analysis"}
        
        # Calculate trend
        first_snapshot = self.snapshots[0]
        last_snapshot = self.snapshots[-1]
        
        rss_growth = last_snapshot.rss_mb - first_snapshot.rss_mb
        vms_growth = last_snapshot.vms_mb - first_snapshot.vms_mb
        duration = last_snapshot.timestamp - first_snapshot.timestamp
        
        # Calculate growth rate (MB per second)
        rss_growth_rate = rss_growth / duration if duration > 0 else 0
        vms_growth_rate = vms_growth / duration if duration > 0 else 0
        
        # Detect potential leak
        leak_detected = (
            rss_growth > self.threshold_mb or
            rss_growth_rate > 1.0  # More than 1MB/sec growth
        )
        
        return {
            "snapshots_count": len(self.snapshots),
            "duration_seconds": duration,
            "rss_growth_mb": rss_growth,
            "vms_growth_mb": vms_growth,
            "rss_growth_rate_mb_per_sec": rss_growth_rate,
            "vms_growth_rate_mb_per_sec": vms_growth_rate,
            "leak_detected": leak_detected,
            "peak_rss_mb": max(s.rss_mb for s in self.snapshots),
            "peak_vms_mb": max(s.vms_mb for s in self.snapshots),
            "final_rss_mb": last_snapshot.rss_mb,
            "final_vms_mb": last_snapshot.vms_mb
        }
    
    async def monitor_async_operation(self, operation_func, iterations: int = 100):
        """Monitor memory usage during async operation."""
        self.clear_snapshots()
        
        # Initial snapshot
        self.take_snapshot()
        
        # Run operation with periodic monitoring
        for i in range(iterations):
            await operation_func()
            
            # Take snapshot every 10 iterations
            if i % 10 == 0:
                self.take_snapshot()
                await asyncio.sleep(self.sample_interval)
        
        # Final snapshot
        self.take_snapshot()
        
        # Force garbage collection
        gc.collect()
        await asyncio.sleep(0.1)
        
        # Post-GC snapshot
        self.take_snapshot()
        
        return self.analyze_trend()


@pytest.fixture(scope="function")
def memory_detector():
    """Memory leak detector fixture."""
    return MemoryLeakDetector(threshold_mb=30.0)


@pytest.mark.performance
class TestQueryBufferMemoryLeaks:
    """Test QueryBuffer for memory leaks."""
    
    @pytest.mark.asyncio
    async def test_query_buffer_repeated_queries_no_leak(self, memory_detector):
        """Test that repeated queries don't cause memory leaks."""
        store = InMemoryStore()
        await store.initialize()
        
        # Add test data
        for i in range(100):
            item = Item(
                id=f"leak_test_item_{i}",
                content=f"Memory leak test content {i}",
                metadata={"category": f"cat_{i % 10}"}
            )
            await store.add(item)
        
        retrieval_handler = make_retrieval_handler_from_store(store)
        query_buffer = QueryBuffer(retrieval_handler=retrieval_handler, max_size=20)
        
        # Test operation
        query_count = 0
        async def repeated_query():
            nonlocal query_count
            query = f"leak test query {query_count % 20}"  # Cycle through queries
            query_count += 1
            results = await query_buffer.query(query, top_k=5)
            return results
        
        # Monitor memory during repeated queries
        analysis = await memory_detector.monitor_async_operation(repeated_query, iterations=200)
        
        assert not analysis["leak_detected"], (
            f"Memory leak detected in QueryBuffer repeated queries:\n"
            f"RSS growth: {analysis['rss_growth_mb']:.2f}MB\n"
            f"Growth rate: {analysis['rss_growth_rate_mb_per_sec']:.3f}MB/sec\n"
            f"Peak RSS: {analysis['peak_rss_mb']:.2f}MB\n"
            f"Final RSS: {analysis['final_rss_mb']:.2f}MB"
        )
    
    @pytest.mark.asyncio
    async def test_query_buffer_cache_eviction_no_leak(self, memory_detector):
        """Test that cache eviction doesn't cause memory leaks."""
        store = InMemoryStore()
        await store.initialize()
        
        # Add test data
        for i in range(50):
            item = Item(
                id=f"eviction_test_item_{i}",
                content=f"Cache eviction test content {i}",
                metadata={"index": i}
            )
            await store.add(item)
        
        retrieval_handler = make_retrieval_handler_from_store(store)
        # Small cache size to force evictions
        query_buffer = QueryBuffer(retrieval_handler=retrieval_handler, max_size=5)
        
        # Test operation that forces cache evictions
        query_count = 0
        async def cache_eviction_query():
            nonlocal query_count
            # Use unique queries to force cache misses and evictions
            query = f"unique eviction query {query_count}"
            query_count += 1
            results = await query_buffer.query(query, top_k=3)
            return results
        
        # Monitor memory during cache evictions
        analysis = await memory_detector.monitor_async_operation(cache_eviction_query, iterations=100)
        
        assert not analysis["leak_detected"], (
            f"Memory leak detected in QueryBuffer cache eviction:\n"
            f"RSS growth: {analysis['rss_growth_mb']:.2f}MB\n"
            f"Growth rate: {analysis['rss_growth_rate_mb_per_sec']:.3f}MB/sec\n"
            f"Peak RSS: {analysis['peak_rss_mb']:.2f}MB\n"
            f"Final RSS: {analysis['final_rss_mb']:.2f}MB"
        )


@pytest.mark.performance
class TestInMemoryStoreMemoryLeaks:
    """Test InMemoryStore for memory leaks."""
    
    @pytest.mark.asyncio
    async def test_in_memory_store_repeated_adds_no_leak(self, memory_detector):
        """Test that repeated add operations don't cause memory leaks."""
        store = InMemoryStore()
        await store.initialize()
        
        # Test operation
        item_count = 0
        async def repeated_add():
            nonlocal item_count
            item = Item(
                id=f"repeated_add_item_{item_count}",
                content=f"Repeated add test content {item_count}",
                metadata={"count": item_count}
            )
            item_count += 1
            return await store.add(item)
        
        # Monitor memory during repeated adds
        analysis = await memory_detector.monitor_async_operation(repeated_add, iterations=200)
        
        # Note: This test expects some memory growth due to storing items
        # We check for excessive growth beyond expected storage
        expected_growth = (200 * 100) / 1024 / 1024  # Rough estimate: 200 items * ~100 bytes each
        excessive_growth = analysis["rss_growth_mb"] > (expected_growth * 3)  # Allow 3x overhead
        
        assert not excessive_growth, (
            f"Excessive memory growth detected in InMemoryStore repeated adds:\n"
            f"RSS growth: {analysis['rss_growth_mb']:.2f}MB (expected ~{expected_growth:.2f}MB)\n"
            f"Growth rate: {analysis['rss_growth_rate_mb_per_sec']:.3f}MB/sec\n"
            f"Peak RSS: {analysis['peak_rss_mb']:.2f}MB\n"
            f"Final RSS: {analysis['final_rss_mb']:.2f}MB"
        )
    
    @pytest.mark.asyncio
    async def test_in_memory_store_repeated_queries_no_leak(self, memory_detector):
        """Test that repeated query operations don't cause memory leaks."""
        store = InMemoryStore()
        await store.initialize()
        
        # Pre-populate store
        for i in range(100):
            item = Item(
                id=f"query_leak_test_item_{i}",
                content=f"Query leak test content {i} with searchable terms",
                metadata={"index": i}
            )
            await store.add(item)
        
        # Test operation
        query_count = 0
        async def repeated_query():
            nonlocal query_count
            query = Query(
                text=f"leak test {query_count % 10}",  # Cycle through queries
                metadata={"top_k": 5}
            )
            query_count += 1
            return await store.query(query)
        
        # Monitor memory during repeated queries
        analysis = await memory_detector.monitor_async_operation(repeated_query, iterations=150)
        
        assert not analysis["leak_detected"], (
            f"Memory leak detected in InMemoryStore repeated queries:\n"
            f"RSS growth: {analysis['rss_growth_mb']:.2f}MB\n"
            f"Growth rate: {analysis['rss_growth_rate_mb_per_sec']:.3f}MB/sec\n"
            f"Peak RSS: {analysis['peak_rss_mb']:.2f}MB\n"
            f"Final RSS: {analysis['final_rss_mb']:.2f}MB"
        )


@pytest.mark.performance
class TestCacheMemoryLeaks:
    """Test cache systems for memory leaks."""
    
    def test_regex_cache_repeated_operations_no_leak(self, memory_detector):
        """Test that repeated regex cache operations don't cause memory leaks."""
        cache = get_regex_cache()
        cache.clear()
        
        # Test operation
        pattern_count = 0
        def repeated_regex_operation():
            nonlocal pattern_count
            # Mix of cache hits and misses
            if pattern_count % 3 == 0:
                pattern = r"\d{3}-\d{2}-\d{4}"  # Repeated pattern (cache hit)
            else:
                pattern = f"pattern_{pattern_count}_\\d+"  # Unique pattern (cache miss)
            pattern_count += 1
            return cache.get_pattern(pattern)
        
        # Monitor memory (sync version)
        memory_detector.clear_snapshots()
        memory_detector.take_snapshot()
        
        for i in range(200):
            repeated_regex_operation()
            if i % 20 == 0:
                memory_detector.take_snapshot()
                time.sleep(0.01)
        
        memory_detector.take_snapshot()
        gc.collect()
        time.sleep(0.1)
        memory_detector.take_snapshot()
        
        analysis = memory_detector.analyze_trend()
        
        assert not analysis["leak_detected"], (
            f"Memory leak detected in regex cache operations:\n"
            f"RSS growth: {analysis['rss_growth_mb']:.2f}MB\n"
            f"Growth rate: {analysis['rss_growth_rate_mb_per_sec']:.3f}MB/sec\n"
            f"Peak RSS: {analysis['peak_rss_mb']:.2f}MB\n"
            f"Final RSS: {analysis['final_rss_mb']:.2f}MB"
        )
    
    def test_content_cache_repeated_operations_no_leak(self, memory_detector):
        """Test that repeated content cache operations don't cause memory leaks."""
        cache = get_content_cache()
        cache.clear()
        
        # Test operation
        content_count = 0
        def repeated_content_operation():
            nonlocal content_count
            content = f"Content cache test {content_count} with some additional text"
            config = {"test": "config", "count": content_count}
            
            # Mix of cache and set operations
            if cache.get_cached_result(content, config) is None:
                cache.cache_result(content, config, {"result": f"test_{content_count}"})
            
            content_count += 1
            return cache.get_cached_result(content, config)
        
        # Monitor memory (sync version)
        memory_detector.clear_snapshots()
        memory_detector.take_snapshot()
        
        for i in range(300):
            repeated_content_operation()
            if i % 30 == 0:
                memory_detector.take_snapshot()
                time.sleep(0.01)
        
        memory_detector.take_snapshot()
        gc.collect()
        time.sleep(0.1)
        memory_detector.take_snapshot()
        
        analysis = memory_detector.analyze_trend()
        
        assert not analysis["leak_detected"], (
            f"Memory leak detected in content cache operations:\n"
            f"RSS growth: {analysis['rss_growth_mb']:.2f}MB\n"
            f"Growth rate: {analysis['rss_growth_rate_mb_per_sec']:.3f}MB/sec\n"
            f"Peak RSS: {analysis['peak_rss_mb']:.2f}MB\n"
            f"Final RSS: {analysis['final_rss_mb']:.2f}MB"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "performance"])
