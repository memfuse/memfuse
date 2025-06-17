"""Tests for QueryBuffer in Buffer."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from memfuse_core.buffer.query_buffer import QueryBuffer
from memfuse_core.rag.chunk.base import ChunkData


@pytest.fixture
def sample_storage_results():
    """Fixture providing sample storage results."""
    return [
        {
            "id": "storage_1",
            "content": "Storage result 1",
            "score": 0.9,
            "type": "message",
            "created_at": "2024-01-01T10:00:00Z",
            "metadata": {"source": "storage"}
        },
        {
            "id": "storage_2", 
            "content": "Storage result 2",
            "score": 0.7,
            "type": "message",
            "created_at": "2024-01-01T09:00:00Z",
            "metadata": {"source": "storage"}
        }
    ]


@pytest.fixture
def mock_hybrid_buffer():
    """Fixture providing a mock HybridBuffer."""
    buffer = MagicMock()
    buffer._lock = asyncio.Lock()
    buffer.chunks = [
        ChunkData(content="Hybrid chunk 1", metadata={"strategy": "test"}),
        ChunkData(content="Hybrid chunk 2", metadata={"strategy": "test"})
    ]
    buffer.original_rounds = [
        [{"id": "hybrid_1", "role": "user", "content": "Hybrid message 1"}],
        [{"id": "hybrid_2", "role": "user", "content": "Hybrid message 2"}]
    ]
    return buffer


@pytest.fixture
def mock_retrieval_handler(sample_storage_results):
    """Fixture providing a mock retrieval handler."""
    return AsyncMock(return_value=sample_storage_results)


class TestQueryBufferInitialization:
    """Test cases for QueryBuffer initialization."""
    
    def test_default_initialization(self):
        """Test QueryBuffer initialization with default parameters."""
        buffer = QueryBuffer()
        
        assert buffer._max_size == 15
        assert buffer.cache_size == 100
        assert buffer.default_sort_by == "score"
        assert buffer.default_order == "desc"
        assert buffer.retrieval_handler is None
        assert buffer.query_cache == {}
        assert buffer._cache_order == []
        assert buffer._items == []
    
    def test_custom_initialization(self):
        """Test QueryBuffer initialization with custom parameters."""
        retrieval_handler = AsyncMock()
        buffer = QueryBuffer(
            retrieval_handler=retrieval_handler,
            max_size=20,
            cache_size=200,
            default_sort_by="timestamp",
            default_order="asc"
        )
        
        assert buffer.retrieval_handler == retrieval_handler
        assert buffer._max_size == 20
        assert buffer.cache_size == 200
        assert buffer.default_sort_by == "timestamp"
        assert buffer.default_order == "asc"


class TestQueryBufferBasicQuery:
    """Test cases for basic query functionality."""
    
    @pytest.mark.asyncio
    async def test_query_with_storage_only(self, mock_retrieval_handler, sample_storage_results):
        """Test query with storage results only."""
        buffer = QueryBuffer(retrieval_handler=mock_retrieval_handler)
        
        result = await buffer.query("test query")
        
        assert len(result) == 2
        assert result[0]["id"] == "storage_1"  # Higher score first (default desc)
        assert result[1]["id"] == "storage_2"
        mock_retrieval_handler.assert_called_once_with("test query", 30)  # 2 * max_size
    
    @pytest.mark.asyncio
    async def test_query_with_hybrid_buffer(self, mock_retrieval_handler, mock_hybrid_buffer):
        """Test query with HybridBuffer results."""
        buffer = QueryBuffer(retrieval_handler=mock_retrieval_handler)
        
        result = await buffer.query("chunk", hybrid_buffer=mock_hybrid_buffer)
        
        # Should include both storage and hybrid results
        assert len(result) > 2  # At least storage results + some hybrid results
        
        # Check that hybrid results are included
        hybrid_ids = [r["id"] for r in result if "hybrid_chunk" in r["id"]]
        assert len(hybrid_ids) > 0
    
    @pytest.mark.asyncio
    async def test_query_without_retrieval_handler(self, mock_hybrid_buffer):
        """Test query without retrieval handler."""
        buffer = QueryBuffer()
        
        result = await buffer.query("test", hybrid_buffer=mock_hybrid_buffer)
        
        # Should only return hybrid results
        assert all("hybrid_chunk" in r["id"] for r in result)
    
    @pytest.mark.asyncio
    async def test_query_with_custom_parameters(self, mock_retrieval_handler):
        """Test query with custom sort and limit parameters."""
        buffer = QueryBuffer(retrieval_handler=mock_retrieval_handler)
        
        result = await buffer.query(
            "test query",
            top_k=1,
            sort_by="timestamp",
            order="asc"
        )
        
        assert len(result) == 1
        assert result[0]["id"] == "storage_2"  # Earlier timestamp first (asc)


class TestQueryBufferSorting:
    """Test cases for sorting functionality."""
    
    @pytest.mark.asyncio
    async def test_sort_by_score_desc(self, mock_retrieval_handler):
        """Test sorting by score in descending order."""
        buffer = QueryBuffer(retrieval_handler=mock_retrieval_handler)
        
        result = await buffer.query("test", sort_by="score", order="desc")
        
        scores = [r["score"] for r in result]
        assert scores == sorted(scores, reverse=True)
    
    @pytest.mark.asyncio
    async def test_sort_by_score_asc(self, mock_retrieval_handler):
        """Test sorting by score in ascending order."""
        buffer = QueryBuffer(retrieval_handler=mock_retrieval_handler)
        
        result = await buffer.query("test", sort_by="score", order="asc")
        
        scores = [r["score"] for r in result]
        assert scores == sorted(scores)
    
    @pytest.mark.asyncio
    async def test_sort_by_timestamp_desc(self, mock_retrieval_handler):
        """Test sorting by timestamp in descending order."""
        buffer = QueryBuffer(retrieval_handler=mock_retrieval_handler)
        
        result = await buffer.query("test", sort_by="timestamp", order="desc")
        
        timestamps = [r["created_at"] for r in result]
        assert timestamps == sorted(timestamps, reverse=True)
    
    @pytest.mark.asyncio
    async def test_sort_by_timestamp_asc(self, mock_retrieval_handler):
        """Test sorting by timestamp in ascending order."""
        buffer = QueryBuffer(retrieval_handler=mock_retrieval_handler)
        
        result = await buffer.query("test", sort_by="timestamp", order="asc")
        
        timestamps = [r["created_at"] for r in result]
        assert timestamps == sorted(timestamps)


class TestQueryBufferCaching:
    """Test cases for caching functionality."""
    
    @pytest.mark.asyncio
    async def test_cache_hit(self, mock_retrieval_handler):
        """Test cache hit functionality."""
        buffer = QueryBuffer(retrieval_handler=mock_retrieval_handler)
        
        # First query (cache miss)
        result1 = await buffer.query("test query")
        assert buffer.cache_misses == 1
        assert buffer.cache_hits == 0
        
        # Second query (cache hit)
        result2 = await buffer.query("test query")
        assert buffer.cache_misses == 1
        assert buffer.cache_hits == 1
        
        # Results should be identical
        assert result1 == result2
        
        # Retrieval handler should only be called once
        mock_retrieval_handler.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cache_key_includes_parameters(self, mock_retrieval_handler):
        """Test that cache key includes sort parameters."""
        buffer = QueryBuffer(retrieval_handler=mock_retrieval_handler)
        
        # Same query with different parameters should be different cache entries
        await buffer.query("test", sort_by="score", order="desc")
        await buffer.query("test", sort_by="timestamp", order="asc")
        
        assert len(buffer.query_cache) == 2
        assert buffer.cache_misses == 2
        assert buffer.cache_hits == 0
    
    @pytest.mark.asyncio
    async def test_cache_lru_eviction(self, mock_retrieval_handler):
        """Test LRU cache eviction."""
        buffer = QueryBuffer(retrieval_handler=mock_retrieval_handler, cache_size=2)
        
        # Fill cache
        await buffer.query("query1")
        await buffer.query("query2")
        assert len(buffer.query_cache) == 2
        
        # Add third query (should evict oldest)
        await buffer.query("query3")
        assert len(buffer.query_cache) == 2
        assert "query1|score|desc|15" not in buffer.query_cache  # Oldest evicted
        assert "query2|score|desc|15" in buffer.query_cache
        assert "query3|score|desc|15" in buffer.query_cache
    
    @pytest.mark.asyncio
    async def test_cache_lru_update(self, mock_retrieval_handler):
        """Test LRU cache order update on access."""
        buffer = QueryBuffer(retrieval_handler=mock_retrieval_handler, cache_size=2)
        
        # Fill cache
        await buffer.query("query1")
        await buffer.query("query2")
        
        # Access first query again (should move to end)
        await buffer.query("query1")
        
        # Add third query (should evict query2, not query1)
        await buffer.query("query3")
        assert "query1|score|desc|15" in buffer.query_cache
        assert "query2|score|desc|15" not in buffer.query_cache
        assert "query3|score|desc|15" in buffer.query_cache


class TestQueryBufferHybridBufferQuery:
    """Test cases for HybridBuffer querying."""
    
    @pytest.mark.asyncio
    async def test_query_hybrid_buffer_text_matching(self, mock_hybrid_buffer):
        """Test querying HybridBuffer with text matching."""
        buffer = QueryBuffer()
        
        result = await buffer._query_hybrid_buffer("chunk", mock_hybrid_buffer, 10)
        
        assert len(result) == 2  # Both chunks contain "chunk"
        assert all("hybrid_chunk" in r["id"] for r in result)
        assert all(r["type"] == "chunk" for r in result)
        assert all("hybrid_buffer" in r["metadata"]["source"] for r in result)
    
    @pytest.mark.asyncio
    async def test_query_hybrid_buffer_no_matches(self, mock_hybrid_buffer):
        """Test querying HybridBuffer with no matches."""
        buffer = QueryBuffer()
        
        result = await buffer._query_hybrid_buffer("nonexistent", mock_hybrid_buffer, 10)
        
        assert len(result) == 0
    
    @pytest.mark.asyncio
    async def test_query_hybrid_buffer_limit(self, mock_hybrid_buffer):
        """Test querying HybridBuffer with result limit."""
        buffer = QueryBuffer()
        
        result = await buffer._query_hybrid_buffer("chunk", mock_hybrid_buffer, 1)
        
        assert len(result) == 1  # Limited to 1 result
    
    @pytest.mark.asyncio
    async def test_query_hybrid_buffer_error_handling(self):
        """Test error handling in HybridBuffer querying."""
        buffer = QueryBuffer()
        
        # Mock buffer that raises exception
        error_buffer = MagicMock()
        error_buffer._lock = asyncio.Lock()
        error_buffer.chunks = None  # Will cause error
        
        result = await buffer._query_hybrid_buffer("test", error_buffer, 10)

        assert result == []  # Should return empty list on error


class TestQueryBufferResultCombination:
    """Test cases for result combination functionality."""

    @pytest.mark.asyncio
    async def test_combine_and_sort_results_by_score(self, sample_storage_results):
        """Test combining and sorting results by score."""
        buffer = QueryBuffer()

        hybrid_results = [
            {"id": "hybrid_1", "score": 0.8, "created_at": "2024-01-01T11:00:00Z"},
            {"id": "hybrid_2", "score": 0.6, "created_at": "2024-01-01T08:00:00Z"}
        ]

        result = await buffer._combine_and_sort_results(
            sample_storage_results, hybrid_results, "score", "desc"
        )

        # Should be sorted by score descending
        scores = [r["score"] for r in result]
        assert scores == [0.9, 0.8, 0.7, 0.6]

    @pytest.mark.asyncio
    async def test_combine_and_sort_results_by_timestamp(self, sample_storage_results):
        """Test combining and sorting results by timestamp."""
        buffer = QueryBuffer()

        hybrid_results = [
            {"id": "hybrid_1", "score": 0.8, "created_at": "2024-01-01T11:00:00Z"},
            {"id": "hybrid_2", "score": 0.6, "created_at": "2024-01-01T08:00:00Z"}
        ]

        result = await buffer._combine_and_sort_results(
            sample_storage_results, hybrid_results, "timestamp", "desc"
        )

        # Should be sorted by timestamp descending
        timestamps = [r["created_at"] for r in result]
        expected = ["2024-01-01T11:00:00Z", "2024-01-01T10:00:00Z", "2024-01-01T09:00:00Z", "2024-01-01T08:00:00Z"]
        assert timestamps == expected

    @pytest.mark.asyncio
    async def test_combine_results_deduplication(self, sample_storage_results):
        """Test that duplicate results are removed."""
        buffer = QueryBuffer()

        # Create hybrid results with same ID as storage
        hybrid_results = [
            {"id": "storage_1", "score": 0.5, "created_at": "2024-01-01T11:00:00Z"}  # Duplicate ID
        ]

        result = await buffer._combine_and_sort_results(
            sample_storage_results, hybrid_results, "score", "desc"
        )

        # Should only have unique IDs
        ids = [r["id"] for r in result]
        assert len(ids) == len(set(ids))  # All unique
        assert len(result) == 2  # Only 2 unique results


class TestQueryBufferMetadata:
    """Test cases for metadata functionality."""

    @pytest.mark.asyncio
    async def test_get_buffer_metadata_with_hybrid_buffer(self, mock_hybrid_buffer):
        """Test getting buffer metadata with HybridBuffer."""
        buffer = QueryBuffer()

        metadata = await buffer.get_buffer_metadata(mock_hybrid_buffer)

        assert metadata["buffer_messages_available"] is True
        assert metadata["buffer_messages_count"] == 2  # Two rounds with one message each
        assert metadata["buffer_chunks_count"] == 2
        assert metadata["sort_by"] == "score"
        assert metadata["order"] == "desc"

    @pytest.mark.asyncio
    async def test_get_buffer_metadata_without_hybrid_buffer(self):
        """Test getting buffer metadata without HybridBuffer."""
        buffer = QueryBuffer()

        metadata = await buffer.get_buffer_metadata()

        assert metadata["buffer_messages_available"] is False
        assert metadata["buffer_messages_count"] == 0
        assert metadata["sort_by"] == "score"
        assert metadata["order"] == "desc"

    @pytest.mark.asyncio
    async def test_get_buffer_metadata_error_handling(self):
        """Test error handling in metadata retrieval."""
        buffer = QueryBuffer()

        # Mock buffer that raises exception
        error_buffer = MagicMock()
        error_buffer._lock = asyncio.Lock()
        error_buffer.original_rounds = None  # Will cause error

        metadata = await buffer.get_buffer_metadata(error_buffer)

        # Should return default metadata on error
        assert metadata["buffer_messages_available"] is False
        assert metadata["buffer_messages_count"] == 0


class TestQueryBufferUtilityMethods:
    """Test cases for utility methods."""

    @pytest.mark.asyncio
    async def test_get_items(self, mock_retrieval_handler):
        """Test getting items from buffer."""
        buffer = QueryBuffer(retrieval_handler=mock_retrieval_handler)

        # Query to populate items
        await buffer.query("test")

        items = await buffer.get_items()

        assert len(items) > 0
        assert isinstance(items, list)

    @pytest.mark.asyncio
    async def test_clear_buffer(self, mock_retrieval_handler):
        """Test clearing buffer."""
        buffer = QueryBuffer(retrieval_handler=mock_retrieval_handler)

        # Query to populate items
        await buffer.query("test")
        assert len(buffer._items) > 0

        await buffer.clear()

        assert len(buffer._items) == 0

    @pytest.mark.asyncio
    async def test_clear_cache(self, mock_retrieval_handler):
        """Test clearing cache."""
        buffer = QueryBuffer(retrieval_handler=mock_retrieval_handler)

        # Query to populate cache
        await buffer.query("test")
        assert len(buffer.query_cache) > 0

        await buffer.clear_cache()

        assert len(buffer.query_cache) == 0
        assert len(buffer._cache_order) == 0

    def test_properties(self, mock_retrieval_handler):
        """Test buffer properties."""
        buffer = QueryBuffer(retrieval_handler=mock_retrieval_handler, max_size=20)

        assert buffer.max_size == 20
        assert isinstance(buffer.items, list)

    def test_get_stats(self, mock_retrieval_handler):
        """Test getting buffer statistics."""
        buffer = QueryBuffer(retrieval_handler=mock_retrieval_handler, max_size=20, cache_size=50)

        stats = buffer.get_stats()

        assert stats["size"] == 0  # No items initially
        assert stats["max_size"] == 20
        assert stats["cache_size"] == 50
        assert stats["cache_entries"] == 0
        assert stats["total_queries"] == 0
        assert stats["cache_hits"] == 0
        assert stats["cache_misses"] == 0
        assert stats["cache_hit_rate"] == "0.0%"
        assert stats["total_hybrid_results"] == 0
        assert stats["total_storage_results"] == 0
        assert stats["default_sort_by"] == "score"
        assert stats["default_order"] == "desc"
        assert stats["has_retrieval_handler"] is True


class TestQueryBufferErrorHandling:
    """Test cases for error handling."""

    @pytest.mark.asyncio
    async def test_query_error_handling(self):
        """Test error handling during query."""
        # Mock retrieval handler that raises exception
        error_handler = AsyncMock(side_effect=Exception("Retrieval failed"))
        buffer = QueryBuffer(retrieval_handler=error_handler)

        result = await buffer.query("test")

        assert result == []  # Should return empty list on error

    @pytest.mark.asyncio
    async def test_concurrent_queries(self, mock_retrieval_handler):
        """Test concurrent query handling."""
        buffer = QueryBuffer(retrieval_handler=mock_retrieval_handler)

        # Run multiple concurrent queries
        tasks = [buffer.query(f"query_{i}") for i in range(10)]
        results = await asyncio.gather(*tasks)

        # All queries should complete successfully
        assert len(results) == 10
        assert all(isinstance(result, list) for result in results)

        # Statistics should be consistent
        assert buffer.total_queries >= 10
        assert isinstance(buffer.get_stats(), dict)


class TestQueryBufferIntegration:
    """Integration tests for QueryBuffer."""

    @pytest.mark.asyncio
    async def test_full_query_workflow(self, mock_retrieval_handler, mock_hybrid_buffer):
        """Test complete query workflow with all components."""
        buffer = QueryBuffer(
            retrieval_handler=mock_retrieval_handler,
            max_size=10,
            default_sort_by="score",
            default_order="desc"
        )

        # First query (cache miss)
        result1 = await buffer.query(
            "test query",
            top_k=5,
            sort_by="score",
            order="desc",
            hybrid_buffer=mock_hybrid_buffer
        )

        assert len(result1) <= 5  # Respects top_k limit
        assert buffer.cache_misses == 1

        # Second identical query (cache hit)
        result2 = await buffer.query(
            "test query",
            top_k=5,
            sort_by="score",
            order="desc",
            hybrid_buffer=mock_hybrid_buffer
        )

        assert result1 == result2
        assert buffer.cache_hits == 1

        # Get metadata
        metadata = await buffer.get_buffer_metadata(mock_hybrid_buffer)
        assert metadata["buffer_messages_available"] is True

        # Get stats
        stats = buffer.get_stats()
        assert stats["total_queries"] == 2
        assert stats["cache_hit_rate"] == "50.0%"
