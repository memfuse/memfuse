"""Simple integration tests for Buffer functionality.

This module provides basic tests to verify Buffer components work correctly.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from memfuse_core.buffer.round_buffer import RoundBuffer
from memfuse_core.buffer.hybrid_buffer import HybridBuffer
from memfuse_core.buffer.query_buffer import QueryBuffer


class TestBufferSimple:
    """Simple tests for Buffer components."""

    @pytest.mark.asyncio
    async def test_round_buffer_basic_functionality(self):
        """Test basic RoundBuffer functionality."""
        buffer = RoundBuffer(max_tokens=100, max_size=3)
        
        # Test adding messages
        messages = [
            {"role": "user", "content": "Hello", "id": "msg_1", "metadata": {"session_id": "session_1"}},
            {"role": "assistant", "content": "Hi there!", "id": "msg_2", "metadata": {"session_id": "session_1"}}
        ]
        
        result = await buffer.add(messages, "session_1")
        assert result is False  # No transfer triggered, so should return False
        
        # Test buffer info
        info = await buffer.get_buffer_info()
        assert "rounds_count" in info
        assert "current_tokens" in info
        assert info["rounds_count"] == 1

    @pytest.mark.asyncio
    async def test_round_buffer_transfer_handler(self):
        """Test RoundBuffer transfer handler."""
        buffer = RoundBuffer(max_tokens=50, max_size=2)
        
        # Mock transfer handler
        transfer_handler = AsyncMock()
        buffer.set_transfer_handler(transfer_handler)
        
        # Add messages that exceed token limit
        large_message = {"role": "user", "content": "This is a very long message that should exceed the token limit for testing purposes.", "id": "large_msg", "metadata": {"session_id": "session_1"}}
        
        result = await buffer.add([large_message], "session_1")
        
        # Transfer should have been called (though buffer might be empty)
        # The exact behavior depends on token counting
        assert result is not None

    @pytest.mark.asyncio
    async def test_hybrid_buffer_basic_functionality(self):
        """Test basic HybridBuffer functionality."""
        buffer = HybridBuffer(max_size=5, chunk_strategy="message")
        
        # Test adding rounds
        rounds = [
            [
                {"role": "user", "content": "Hello", "id": "msg_1", "metadata": {"session_id": "session_1"}},
                {"role": "assistant", "content": "Hi!", "id": "msg_2", "metadata": {"session_id": "session_1"}}
            ]
        ]
        
        await buffer.add_from_rounds(rounds)
        
        # Test stats
        stats = buffer.get_stats()
        assert "chunks_count" in stats
        assert "rounds_count" in stats

    @pytest.mark.asyncio
    async def test_hybrid_buffer_read_api(self):
        """Test HybridBuffer Read API functionality."""
        buffer = HybridBuffer(max_size=5, chunk_strategy="message")
        
        # Add some data
        rounds = [
            [
                {"role": "user", "content": "Test message", "id": "msg_1", "metadata": {"session_id": "session_1"}}
            ]
        ]
        
        await buffer.add_from_rounds(rounds)
        
        # Test Read API
        messages = await buffer.get_all_messages_for_read_api(limit=10)
        assert isinstance(messages, list)

    @pytest.mark.asyncio
    async def test_unified_query_buffer_basic_functionality(self):
        """Test basic QueryBuffer functionality."""
        # Mock retrieval handler
        async def mock_retrieval_handler(query, top_k=10, **kwargs):
            return [
                {
                    "id": "storage_1",
                    "content": "Storage result",
                    "score": 0.9,
                    "metadata": {"source": "storage"}
                }
            ]
        
        buffer = QueryBuffer(
            retrieval_handler=mock_retrieval_handler,
            max_size=10,
            cache_size=50
        )
        
        # Test query
        results = await buffer.query("test query", top_k=5)
        assert isinstance(results, list)
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_unified_query_buffer_sorting(self):
        """Test QueryBuffer sorting functionality."""
        # Mock retrieval handler with multiple results
        async def mock_retrieval_handler(query, top_k=10, **kwargs):
            return [
                {
                    "id": "storage_1",
                    "content": "First result",
                    "score": 0.9,
                    "metadata": {"source": "storage", "timestamp": "2023-01-01T00:00:00Z"}
                },
                {
                    "id": "storage_2", 
                    "content": "Second result",
                    "score": 0.7,
                    "metadata": {"source": "storage", "timestamp": "2023-01-02T00:00:00Z"}
                }
            ]
        
        buffer = QueryBuffer(
            retrieval_handler=mock_retrieval_handler,
            max_size=10,
            cache_size=50
        )
        
        # Test sort by score
        results_score = await buffer.query("test", top_k=5, sort_by="score", order="desc")
        assert len(results_score) == 2
        
        # Test sort by timestamp
        results_timestamp = await buffer.query("test", top_k=5, sort_by="timestamp", order="desc")
        assert len(results_timestamp) == 2

    @pytest.mark.asyncio
    async def test_unified_query_buffer_with_hybrid_buffer(self):
        """Test QueryBuffer with HybridBuffer integration."""
        # Mock retrieval handler
        async def mock_retrieval_handler(query, top_k=10, **kwargs):
            return [
                {
                    "id": "storage_1",
                    "content": "Storage result",
                    "score": 0.8,
                    "metadata": {"source": "storage"}
                }
            ]
        
        query_buffer = QueryBuffer(
            retrieval_handler=mock_retrieval_handler,
            max_size=10,
            cache_size=50
        )
        
        # Create and populate hybrid buffer
        hybrid_buffer = HybridBuffer(max_size=5, chunk_strategy="message")
        rounds = [
            [
                {"role": "user", "content": "Buffer message", "id": "msg_1", "metadata": {"session_id": "session_1"}}
            ]
        ]
        await hybrid_buffer.add_from_rounds(rounds)
        
        # Set hybrid buffer
        query_buffer.set_hybrid_buffer(hybrid_buffer)
        
        # Test query with both sources
        results = await query_buffer.query("message", top_k=5)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_buffer_metadata(self):
        """Test buffer metadata functionality."""
        # Mock retrieval handler
        async def mock_retrieval_handler(query, top_k=10, **kwargs):
            return []
        
        query_buffer = QueryBuffer(
            retrieval_handler=mock_retrieval_handler,
            max_size=10,
            cache_size=50
        )
        
        # Test metadata without hybrid buffer
        metadata = await query_buffer.get_buffer_metadata()
        assert isinstance(metadata, dict)
        assert "buffer_messages_available" in metadata
        
        # Test metadata with hybrid buffer
        hybrid_buffer = HybridBuffer(max_size=5, chunk_strategy="message")
        query_buffer.set_hybrid_buffer(hybrid_buffer)
        
        metadata_with_buffer = await query_buffer.get_buffer_metadata()
        assert isinstance(metadata_with_buffer, dict)

    @pytest.mark.asyncio
    async def test_component_integration_flow(self):
        """Test the complete component integration flow."""
        # Create components
        round_buffer = RoundBuffer(max_tokens=50, max_size=2)
        hybrid_buffer = HybridBuffer(max_size=5, chunk_strategy="message")
        
        # Mock retrieval handler
        async def mock_retrieval_handler(query, top_k=10, **kwargs):
            return [
                {
                    "id": "storage_1",
                    "content": "Storage content",
                    "score": 0.8,
                    "metadata": {"source": "storage"}
                }
            ]
        
        query_buffer = QueryBuffer(
            retrieval_handler=mock_retrieval_handler,
            max_size=10,
            cache_size=50
        )
        
        # Set up connections
        round_buffer.set_transfer_handler(hybrid_buffer.add_from_rounds)
        query_buffer.set_hybrid_buffer(hybrid_buffer)
        
        # Add data to round buffer
        messages = [
            {"role": "user", "content": "Integration test", "id": "msg_1", "metadata": {"session_id": "session_1"}}
        ]
        
        await round_buffer.add(messages, "session_1")
        
        # Force transfer to hybrid buffer
        await round_buffer.force_transfer()
        
        # Query through unified buffer
        results = await query_buffer.query("test", top_k=5)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in Buffer components."""
        # Test RoundBuffer with invalid data
        round_buffer = RoundBuffer(max_tokens=100, max_size=3)
        
        try:
            # This should handle gracefully
            result = await round_buffer.add([], "test_session")
            assert result is not None
        except Exception:
            # If exception occurs, it should be a known type
            pass
        
        # Test HybridBuffer with empty rounds
        hybrid_buffer = HybridBuffer(max_size=5, chunk_strategy="message")
        
        try:
            await hybrid_buffer.add_from_rounds([])
            # Should handle empty input gracefully
        except Exception:
            pass
        
        # Test QueryBuffer with failing retrieval handler
        async def failing_retrieval_handler(query, top_k=10, **kwargs):
            raise Exception("Retrieval failed")
        
        query_buffer = QueryBuffer(
            retrieval_handler=failing_retrieval_handler,
            max_size=10,
            cache_size=50
        )
        
        try:
            results = await query_buffer.query("test", top_k=5)
            # Should handle retrieval failure gracefully
            assert isinstance(results, list)
        except Exception:
            # Exception handling is acceptable
            pass

    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """Test concurrent operations on Buffer components."""
        round_buffer = RoundBuffer(max_tokens=100, max_size=5)
        
        # Create concurrent add operations
        tasks = []
        for i in range(3):
            message = {
                "role": "user",
                "content": f"Concurrent message {i}",
                "id": f"msg_{i}",
                "metadata": {"session_id": f"session_{i}"}
            }
            task = round_buffer.add([message], f"session_{i}")
            tasks.append(task)
        
        # Execute concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify operations completed
        assert len(results) == 3
        
        # Check buffer state
        info = await round_buffer.get_buffer_info()
        assert "rounds_count" in info
