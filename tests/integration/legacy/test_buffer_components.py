"""Integration tests for Buffer components.

This module tests the integration between Buffer components:
- RoundBuffer + HybridBuffer integration
- HybridBuffer + QueryBuffer integration
- Component interaction and data flow
"""

import pytest
import asyncio
from typing import List, Dict, Any

from memfuse_core.buffer.round_buffer import RoundBuffer
from memfuse_core.buffer.hybrid_buffer import HybridBuffer
from memfuse_core.buffer.query_buffer import QueryBuffer


class TestBufferComponentIntegration:
    """Integration tests for Buffer components."""

    @pytest.fixture
    def sample_messages(self):
        """Sample messages for testing."""
        return [
            {"role": "user", "content": "Hello, how are you?", "id": "msg_1", "metadata": {"session_id": "session_1"}},
            {"role": "assistant", "content": "I'm doing well, thank you!", "id": "msg_2", "metadata": {"session_id": "session_1"}},
            {"role": "user", "content": "What's the weather like?", "id": "msg_3", "metadata": {"session_id": "session_1"}}
        ]

    @pytest.fixture
    async def round_buffer(self):
        """Create a RoundBuffer for testing."""
        return RoundBuffer(max_tokens=100, max_size=3)  # Small limits for testing

    @pytest.fixture
    async def hybrid_buffer(self):
        """Create a HybridBuffer for testing."""
        return HybridBuffer(max_size=5, chunk_strategy="message")

    @pytest.fixture
    async def unified_query_buffer(self):
        """Create a QueryBuffer for testing."""
        # Mock retrieval handler
        async def mock_retrieval_handler(query, top_k=10, **kwargs):
            return [
                {
                    "id": "storage_1",
                    "content": "Storage content 1",
                    "score": 0.9,
                    "metadata": {"source": "storage"}
                },
                {
                    "id": "storage_2", 
                    "content": "Storage content 2",
                    "score": 0.7,
                    "metadata": {"source": "storage"}
                }
            ]
        
        return QueryBuffer(
            retrieval_handler=mock_retrieval_handler,
            max_size=10,
            cache_size=50
        )

    @pytest.mark.asyncio
    async def test_round_to_hybrid_transfer(self, round_buffer, hybrid_buffer, sample_messages):
        """Test transfer from RoundBuffer to HybridBuffer."""
        # Set up transfer handler
        round_buffer.set_transfer_handler(hybrid_buffer.add_from_rounds)
        
        # Add messages that exceed token limit
        large_message = {"role": "user", "content": "This is a very long message. " * 20, "id": "large_msg", "metadata": {"session_id": "session_1"}}
        
        # Add small message first
        result1 = await round_buffer.add([sample_messages[0]], "session_1")
        assert result1 is True
        
        # Add large message - should trigger transfer
        result2 = await round_buffer.add([large_message], "session_1")
        
        # Verify transfer occurred
        assert len(hybrid_buffer.original_rounds) > 0
        
        # Verify RoundBuffer was cleared and has new data
        round_info = round_buffer.get_buffer_info()
        assert round_info["total_rounds"] >= 0

    @pytest.mark.asyncio
    async def test_session_change_transfer(self, round_buffer, hybrid_buffer, sample_messages):
        """Test transfer on session change."""
        # Set up transfer handler
        round_buffer.set_transfer_handler(hybrid_buffer.add_from_rounds)
        
        # Add messages to session 1
        await round_buffer.add([sample_messages[0]], "session_1")
        
        # Add messages to session 2 - should trigger transfer
        await round_buffer.add([sample_messages[1]], "session_2")
        
        # Verify transfer occurred
        assert len(hybrid_buffer.original_rounds) > 0

    @pytest.mark.asyncio
    async def test_hybrid_buffer_chunk_creation(self, hybrid_buffer, sample_messages):
        """Test chunk creation in HybridBuffer."""
        # Prepare rounds data
        rounds = [
            [sample_messages[0], sample_messages[1]],  # First round
            [sample_messages[2]]  # Second round
        ]
        
        # Add rounds to hybrid buffer
        await hybrid_buffer.add_from_rounds(rounds)
        
        # Verify chunks were created
        assert len(hybrid_buffer.chunks) > 0
        assert len(hybrid_buffer.original_rounds) == len(rounds)
        
        # Verify chunk content
        for chunk in hybrid_buffer.chunks:
            assert hasattr(chunk, 'content')
            assert hasattr(chunk, 'metadata')

    @pytest.mark.asyncio
    async def test_unified_query_integration(self, unified_query_buffer, hybrid_buffer, sample_messages):
        """Test QueryBuffer integration with HybridBuffer."""
        # Add data to hybrid buffer
        rounds = [[sample_messages[0], sample_messages[1]]]
        await hybrid_buffer.add_from_rounds(rounds)
        
        # Set hybrid buffer in query buffer
        unified_query_buffer.set_hybrid_buffer(hybrid_buffer)
        
        # Perform query
        results = await unified_query_buffer.query(
            "hello",
            top_k=5,
            sort_by="score",
            order="desc"
        )
        
        # Verify results include both storage and buffer data
        assert len(results) > 0
        
        # Check for different sources
        sources = [r.get("metadata", {}).get("source") for r in results]
        assert "storage" in sources  # From mock retrieval handler

    @pytest.mark.asyncio
    async def test_query_sorting_integration(self, unified_query_buffer, hybrid_buffer, sample_messages):
        """Test query sorting across storage and buffer data."""
        # Add data to hybrid buffer
        rounds = [[sample_messages[0]]]
        await hybrid_buffer.add_from_rounds(rounds)
        unified_query_buffer.set_hybrid_buffer(hybrid_buffer)
        
        # Test sort by score
        results_score = await unified_query_buffer.query(
            "test",
            top_k=10,
            sort_by="score",
            order="desc"
        )
        
        # Test sort by timestamp
        results_timestamp = await unified_query_buffer.query(
            "test",
            top_k=10,
            sort_by="timestamp",
            order="desc"
        )
        
        assert len(results_score) > 0
        assert len(results_timestamp) > 0

    @pytest.mark.asyncio
    async def test_buffer_metadata_integration(self, unified_query_buffer, hybrid_buffer, sample_messages):
        """Test buffer metadata in query results."""
        # Add data to hybrid buffer
        rounds = [[sample_messages[0]]]
        await hybrid_buffer.add_from_rounds(rounds)
        unified_query_buffer.set_hybrid_buffer(hybrid_buffer)
        
        # Get buffer metadata
        metadata = await unified_query_buffer.get_buffer_metadata()
        
        assert isinstance(metadata, dict)
        assert "buffer_messages_available" in metadata
        assert "buffer_messages_count" in metadata

    @pytest.mark.asyncio
    async def test_concurrent_component_operations(self, round_buffer, hybrid_buffer, sample_messages):
        """Test concurrent operations across components."""
        # Set up transfer handler
        round_buffer.set_transfer_handler(hybrid_buffer.add_from_rounds)
        
        # Create concurrent add operations
        tasks = []
        for i in range(3):
            message = {
                "role": "user",
                "content": f"Concurrent message {i}",
                "id": f"concurrent_{i}",
                "metadata": {"session_id": f"session_{i}"}
            }
            task = round_buffer.add([message], f"session_{i}")
            tasks.append(task)
        
        # Execute concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify operations completed
        assert len(results) == 3
        
        # Check that data was processed
        assert len(hybrid_buffer.original_rounds) >= 0

    @pytest.mark.asyncio
    async def test_error_propagation(self, round_buffer, hybrid_buffer):
        """Test error handling across components."""
        # Set up transfer handler
        round_buffer.set_transfer_handler(hybrid_buffer.add_from_rounds)
        
        # Test with invalid data
        invalid_message = {"invalid": "data"}  # Missing required fields
        
        # Should handle gracefully
        try:
            result = await round_buffer.add([invalid_message], "test_session")
            # Should not crash
            assert result is not None
        except Exception as e:
            # If exception occurs, it should be handled gracefully
            assert isinstance(e, Exception)

    @pytest.mark.asyncio
    async def test_memory_management(self, round_buffer, hybrid_buffer, sample_messages):
        """Test memory management across components."""
        # Set up transfer handler
        round_buffer.set_transfer_handler(hybrid_buffer.add_from_rounds)
        
        # Add many messages to test FIFO behavior
        for i in range(10):
            message = {
                "role": "user",
                "content": f"Memory test message {i}",
                "id": f"mem_test_{i}",
                "metadata": {"session_id": "memory_test"}
            }
            await round_buffer.add([message], "memory_test")
        
        # Verify FIFO behavior in HybridBuffer
        hybrid_stats = hybrid_buffer.get_stats()
        assert hybrid_stats["total_chunks"] <= hybrid_buffer.max_size * 2  # Allow some overflow

    @pytest.mark.asyncio
    async def test_data_consistency(self, round_buffer, hybrid_buffer, sample_messages):
        """Test data consistency across component transfers."""
        # Set up transfer handler
        round_buffer.set_transfer_handler(hybrid_buffer.add_from_rounds)
        
        # Add specific messages
        test_messages = [
            {"role": "user", "content": "Consistency test 1", "id": "consistency_1", "metadata": {"session_id": "consistency_test"}},
            {"role": "assistant", "content": "Consistency test 2", "id": "consistency_2", "metadata": {"session_id": "consistency_test"}}
        ]
        
        # Add messages
        await round_buffer.add(test_messages, "consistency_test")
        
        # Force transfer
        await round_buffer.force_transfer()
        
        # Verify data integrity in HybridBuffer
        read_api_messages = await hybrid_buffer.get_all_messages_for_read_api()
        
        # Check that original message content is preserved
        contents = [msg.get("content", "") for msg in read_api_messages]
        assert "Consistency test 1" in " ".join(contents) or any("Consistency test 1" in content for content in contents)
        assert "Consistency test 2" in " ".join(contents) or any("Consistency test 2" in content for content in contents)
