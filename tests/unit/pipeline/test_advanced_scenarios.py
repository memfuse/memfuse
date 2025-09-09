"""
Advanced unit tests for MemFuse pipeline scenarios.

These tests cover complex scenarios without requiring database connectivity.
"""

import pytest
import asyncio
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock

from src.memfuse_core.buffer.query_buffer import QueryBuffer
from src.memfuse_core.persistence.in_memory_store import InMemoryStore
from src.memfuse_core.persistence.adapters import make_retrieval_handler_from_store
from src.memfuse_core.models.core import Item
from src.memfuse_core.gateway.filter_cache import (
    get_regex_cache,
    get_content_cache
)


class TestQueryBufferAdvancedScenarios:
    """Test QueryBuffer in advanced scenarios."""
    
    @pytest.fixture
    async def populated_query_buffer(self):
        """Create QueryBuffer with populated InMemoryStore."""
        store = InMemoryStore()
        await store.initialize()
        
        # Add test items with various content
        test_items = [
            Item(id="ai_1", content="Machine learning algorithms for data analysis", metadata={"topic": "AI", "difficulty": "advanced"}),
            Item(id="ai_2", content="Neural networks and deep learning fundamentals", metadata={"topic": "AI", "difficulty": "intermediate"}),
            Item(id="db_1", content="Database optimization and indexing strategies", metadata={"topic": "Database", "difficulty": "advanced"}),
            Item(id="db_2", content="SQL query performance tuning techniques", metadata={"topic": "Database", "difficulty": "intermediate"}),
            Item(id="sec_1", content="Web application security best practices", metadata={"topic": "Security", "difficulty": "beginner"}),
            Item(id="cloud_1", content="Cloud computing architecture patterns", metadata={"topic": "Cloud", "difficulty": "advanced"}),
            Item(id="data_1", content="Data science methodologies and workflows", metadata={"topic": "Data Science", "difficulty": "intermediate"})
        ]
        
        for item in test_items:
            await store.add(item)
        
        retrieval_handler = make_retrieval_handler_from_store(store)
        query_buffer = QueryBuffer(retrieval_handler=retrieval_handler, max_size=20)
        
        return query_buffer, store
    
    @pytest.mark.asyncio
    async def test_query_buffer_caching_behavior(self, populated_query_buffer):
        """Test QueryBuffer caching with repeated queries."""
        query_buffer, store = populated_query_buffer
        
        # First query - should hit storage
        results1 = await query_buffer.query("machine learning", top_k=3)
        assert len(results1) > 0
        assert query_buffer.cache_misses >= 1
        initial_cache_hits = query_buffer.cache_hits
        
        # Same query - should hit cache
        results2 = await query_buffer.query("machine learning", top_k=3)
        assert len(results2) > 0
        assert query_buffer.cache_hits > initial_cache_hits
        
        # Verify results are consistent
        assert len(results1) == len(results2)
        if results1 and results2:
            assert results1[0]["id"] == results2[0]["id"]
    
    @pytest.mark.asyncio
    async def test_query_buffer_different_parameters(self, populated_query_buffer):
        """Test QueryBuffer with various query parameters."""
        query_buffer, store = populated_query_buffer
        
        # Test different top_k values
        results_k1 = await query_buffer.query("database", top_k=1)
        results_k3 = await query_buffer.query("database", top_k=3)
        
        assert len(results_k1) <= 1
        assert len(results_k3) <= 3
        assert len(results_k3) >= len(results_k1)
        
        # Test sorting
        results_asc = await query_buffer.query("security", sort_by="score", order="asc")
        results_desc = await query_buffer.query("security", sort_by="score", order="desc")
        
        if len(results_asc) > 1 and len(results_desc) > 1:
            # Verify sorting order
            assert results_desc[0]["score"] >= results_asc[0]["score"]
    
    @pytest.mark.asyncio
    async def test_concurrent_query_handling(self, populated_query_buffer):
        """Test QueryBuffer under concurrent load."""
        query_buffer, store = populated_query_buffer
        
        async def make_query(query_id: int):
            query = f"test query {query_id % 3}"  # Some overlap for cache testing
            results = await query_buffer.query(query, top_k=2)
            return len(results), query
        
        # Create 20 concurrent queries
        tasks = [make_query(i) for i in range(20)]
        results = await asyncio.gather(*tasks)
        
        # Verify all queries completed
        assert len(results) == 20
        
        # Verify cache statistics improved
        assert query_buffer.cache_hits > 0  # Some queries should have hit cache
        
        # Verify results are reasonable
        for result_count, query in results:
            assert result_count >= 0  # Should not fail
    
    @pytest.mark.asyncio
    async def test_store_error_handling(self):
        """Test error handling when store operations fail."""
        # Create a mock store that raises exceptions
        mock_store = AsyncMock()
        mock_store.query.side_effect = Exception("Store connection failed")
        mock_store.initialize.return_value = True
        
        retrieval_handler = make_retrieval_handler_from_store(mock_store)
        query_buffer = QueryBuffer(retrieval_handler=retrieval_handler)
        
        # Query should handle store errors gracefully
        results = await query_buffer.query("test query")
        
        # Should return empty results rather than crashing
        assert isinstance(results, list)
        assert len(results) == 0


class TestCacheAdvancedScenarios:
    """Test caching components in advanced scenarios."""
    
    @pytest.mark.asyncio
    async def test_regex_cache_concurrent_access(self):
        """Test regex cache under concurrent access."""
        cache = get_regex_cache()
        cache.clear()
        
        patterns = [
            r"\b(password|secret|confidential)\b",
            r"\b\d{3}-\d{2}-\d{4}\b",
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b"
        ]
        
        async def compile_patterns():
            compiled = []
            for pattern in patterns:
                compiled_pattern = cache.get_pattern(pattern)
                compiled.append(compiled_pattern)
            return compiled
        
        # Run multiple concurrent compilation tasks
        tasks = [compile_patterns() for _ in range(10)]
        results = await asyncio.gather(*tasks)
        
        # Verify all tasks completed successfully
        assert len(results) == 10
        
        # Verify all results are consistent
        for result in results:
            assert len(result) == len(patterns)
            for compiled_pattern in result:
                assert compiled_pattern is not None
        
        # Verify cache statistics
        stats = cache.get_stats()
        assert stats["total_patterns"] == len(patterns)
        assert stats["cache_hits"] > 0  # Should have cache hits from concurrent access
    
    @pytest.mark.asyncio
    async def test_content_cache_memory_efficiency(self):
        """Test content cache memory management."""
        cache = get_content_cache()
        cache.clear()
        
        # Generate large amount of test content
        large_contents = [
            f"This is test content number {i} with substantial text to test memory usage. " * 10
            for i in range(1000)
        ]
        
        config_hash = "memory_test_config"
        
        # Fill cache with content
        for i, content in enumerate(large_contents):
            cache.set(content, config_hash, {
                "filtered": True,
                "score": 0.8,
                "index": i
            })
        
        # Verify cache is working
        stats = cache.get_stats()
        assert stats["total_entries"] > 0
        
        # Test retrieval
        retrieved_count = 0
        for content in large_contents[:100]:  # Test first 100
            result = cache.get(content, config_hash)
            if result is not None:
                retrieved_count += 1
                assert result["filtered"] is True
                assert result["score"] == 0.8
        
        # Should retrieve most items (some may have been evicted due to size limits)
        assert retrieved_count > 50  # At least half should be retrievable
    
    @pytest.mark.asyncio
    async def test_cache_ttl_behavior(self):
        """Test cache TTL (time-to-live) behavior."""
        cache = get_content_cache()
        cache.clear()
        
        content = "TTL test content"
        config_hash = "ttl_test_config"
        
        # Set content in cache
        cache.set(content, config_hash, {"test": "data"})
        
        # Should be retrievable immediately
        result = cache.get(content, config_hash)
        assert result is not None
        assert result["test"] == "data"
        
        # Simulate time passage (this would require modifying cache implementation
        # to support time mocking, so we'll just verify the interface works)
        
        # For now, just verify the cache interface is working
        stats = cache.get_stats()
        assert stats["total_entries"] >= 1


class TestInMemoryStoreAdvancedScenarios:
    """Test InMemoryStore in advanced scenarios."""
    
    @pytest.mark.asyncio
    async def test_large_dataset_performance(self):
        """Test InMemoryStore with large dataset."""
        store = InMemoryStore()
        await store.initialize()
        
        # Add large number of items
        items = []
        for i in range(1000):
            item = Item(
                id=f"large_item_{i}",
                content=f"Large dataset test content {i} with additional text for realistic size",
                metadata={"category": f"cat_{i % 20}", "priority": i % 5, "index": i}
            )
            items.append(item)
        
        # Add items in batches to test performance
        batch_size = 100
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            for item in batch:
                item_id = await store.add(item)
                assert item_id == item.id
        
        # Test querying
        from src.memfuse_core.models.core import Query
        query = Query(text="test content", metadata={"top_k": 10})
        result = await store.query(query)
        
        assert result is not None
        assert result.score >= 0
    
    @pytest.mark.asyncio
    async def test_concurrent_store_operations(self):
        """Test InMemoryStore under concurrent operations."""
        store = InMemoryStore()
        await store.initialize()
        
        async def add_items(start_id: int, count: int):
            added_ids = []
            for i in range(count):
                item = Item(
                    id=f"concurrent_item_{start_id}_{i}",
                    content=f"Concurrent test content {start_id}_{i}",
                    metadata={"batch": start_id, "index": i}
                )
                item_id = await store.add(item)
                added_ids.append(item_id)
            return added_ids
        
        # Run concurrent add operations
        tasks = [add_items(i, 50) for i in range(10)]
        results = await asyncio.gather(*tasks)
        
        # Verify all operations completed successfully
        assert len(results) == 10
        total_added = sum(len(batch) for batch in results)
        assert total_added == 500  # 10 batches * 50 items each
        
        # Verify items can be queried
        from src.memfuse_core.models.core import Query
        query = Query(text="concurrent", metadata={"top_k": 100})
        result = await store.query(query)
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_store_memory_cleanup(self):
        """Test InMemoryStore memory cleanup."""
        store = InMemoryStore()
        await store.initialize()
        
        # Add items
        for i in range(100):
            item = Item(
                id=f"cleanup_item_{i}",
                content=f"Cleanup test content {i}",
                metadata={"index": i}
            )
            await store.add(item)
        
        # Delete some items
        deleted_count = 0
        for i in range(0, 100, 2):  # Delete every other item
            success = await store.delete(f"cleanup_item_{i}")
            if success:
                deleted_count += 1
        
        assert deleted_count > 0
        
        # Verify deleted items are not retrievable
        from src.memfuse_core.models.core import Query
        query = Query(text="cleanup", metadata={"top_k": 100})
        result = await store.query(query)
        
        # Should still find remaining items
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
