"""Integration tests for Buffer system."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from memfuse_core.buffer.round_buffer import RoundBuffer
from memfuse_core.buffer.hybrid_buffer import HybridBuffer
from memfuse_core.buffer.query_buffer import QueryBuffer
from memfuse_core.utils.token_counter import TokenCounter


@pytest.fixture
def sample_messages():
    """Fixture providing sample messages for testing."""
    return [
        {
            "id": "msg_1",
            "role": "user",
            "content": "Hello, how are you today?",
            "created_at": "2024-01-01T10:00:00Z",
            "metadata": {"session_id": "session_1"}
        },
        {
            "id": "msg_2",
            "role": "assistant", 
            "content": "I'm doing well, thank you for asking!",
            "created_at": "2024-01-01T10:01:00Z",
            "metadata": {"session_id": "session_1"}
        }
    ]


@pytest.fixture
def mock_storage_handlers():
    """Fixture providing mock storage handlers."""
    return {
        "sqlite": AsyncMock(),
        "qdrant": AsyncMock()
    }


@pytest.fixture
def mock_retrieval_handler():
    """Fixture providing mock retrieval handler."""
    return AsyncMock(return_value=[
        {
            "id": "storage_1",
            "content": "Storage result",
            "score": 0.8,
            "type": "message",
            "created_at": "2024-01-01T09:00:00Z",
            "metadata": {"source": "storage"}
        }
    ])


class TestBufferIntegration:
    """Integration tests for complete Buffer system."""
    
    @pytest.mark.asyncio
    async def test_complete_buffer_workflow(self, sample_messages, mock_storage_handlers, mock_retrieval_handler):
        """Test complete Buffer workflow from RoundBuffer to query."""
        # Initialize components
        round_buffer = RoundBuffer(max_tokens=100, max_size=2)
        hybrid_buffer = HybridBuffer(max_size=3)
        query_buffer = QueryBuffer(retrieval_handler=mock_retrieval_handler)
        

        
        # Set up transfer handler
        round_buffer.set_transfer_handler(hybrid_buffer.add_from_rounds)
        
        # Mock chunk strategy and embedding model
        with patch.object(hybrid_buffer, '_load_chunk_strategy') as mock_load_strategy:
            with patch.object(hybrid_buffer, '_load_embedding_model') as mock_load_embedding:
                # Mock the loaded components
                mock_strategy = MagicMock()
                mock_strategy.create_chunks = AsyncMock(return_value=[
                    MagicMock(content="Test chunk", metadata={"strategy": "test"})
                ])
                hybrid_buffer.chunk_strategy = mock_strategy
                hybrid_buffer.embedding_model = AsyncMock(return_value=[0.1] * 384)
                
                # Step 1: Add messages to RoundBuffer
                await round_buffer.add(sample_messages, "session_1")
                assert len(round_buffer.rounds) == 1
                assert round_buffer.current_session_id == "session_1"
                
                # Step 2: Trigger transfer by adding more messages
                more_messages = [
                    {
                        "id": "msg_3",
                        "role": "user",
                        "content": "What's the weather like?",
                        "metadata": {"session_id": "session_1"}
                    }
                ]
                
                # Mock high token count to trigger transfer
                with patch.object(round_buffer.token_counter, 'count_message_tokens', return_value=150):
                    await round_buffer.add(more_messages, "session_1")
                
                # Verify transfer occurred
                assert len(hybrid_buffer.original_rounds) > 0
                assert len(hybrid_buffer.chunks) > 0
                assert len(hybrid_buffer.embeddings) > 0
                
                # Step 3: Query the system
                results = await query_buffer.query(
                    "test query",
                    top_k=5,
                    sort_by="score",
                    order="desc",
                    hybrid_buffer=hybrid_buffer
                )
                
                # Verify query results
                assert len(results) > 0
                assert any(r["id"] == "storage_1" for r in results)  # Storage result included
                
                # Step 4: Test Read API functionality
                round_messages = await round_buffer.get_all_messages_for_read_api()
                hybrid_messages = await hybrid_buffer.get_all_messages_for_read_api()
                
                # Verify Read API results
                assert isinstance(round_messages, list)
                assert isinstance(hybrid_messages, list)
                
                # Step 5: Test buffer info
                round_info = await round_buffer.get_buffer_info()
                query_metadata = await query_buffer.get_buffer_metadata(hybrid_buffer)
                
                assert round_info["buffer_type"] == "round_buffer"
                assert query_metadata["buffer_messages_available"] is True
    
    @pytest.mark.asyncio
    async def test_session_change_workflow(self, sample_messages, mock_storage_handlers):
        """Test session change triggering transfer workflow."""
        round_buffer = RoundBuffer(max_tokens=1000)  # High token limit
        hybrid_buffer = HybridBuffer()
        
        # Set up transfer handler
        round_buffer.set_transfer_handler(hybrid_buffer.add_from_rounds)
        
        # Mock components
        with patch.object(hybrid_buffer, '_load_chunk_strategy'):
            with patch.object(hybrid_buffer, '_load_embedding_model'):
                mock_strategy = MagicMock()
                mock_strategy.create_chunks = AsyncMock(return_value=[
                    MagicMock(content="Test chunk", metadata={})
                ])
                hybrid_buffer.chunk_strategy = mock_strategy
                hybrid_buffer.embedding_model = AsyncMock(return_value=[0.1] * 384)
                
                # Add messages for session_1
                await round_buffer.add(sample_messages, "session_1")
                assert round_buffer.current_session_id == "session_1"
                assert len(round_buffer.rounds) == 1
                
                # Add messages for session_2 (should trigger transfer)
                session2_messages = [
                    {
                        "id": "msg_s2_1",
                        "role": "user",
                        "content": "New session message",
                        "metadata": {"session_id": "session_2"}
                    }
                ]
                
                await round_buffer.add(session2_messages, "session_2")
                
                # Verify session change triggered transfer
                assert round_buffer.current_session_id == "session_2"
                assert len(round_buffer.rounds) == 1  # Only new session data
                assert round_buffer.total_session_changes == 1
                assert len(hybrid_buffer.original_rounds) > 0  # Data transferred
    
    @pytest.mark.asyncio
    async def test_fifo_behavior_across_buffers(self, mock_storage_handlers):
        """Test FIFO behavior across buffer components."""
        round_buffer = RoundBuffer(max_tokens=50, max_size=2)
        hybrid_buffer = HybridBuffer(max_size=2)
        
        # Set up transfer handler
        round_buffer.set_transfer_handler(hybrid_buffer.add_from_rounds)
        
        # Mock components
        with patch.object(hybrid_buffer, '_load_chunk_strategy'):
            with patch.object(hybrid_buffer, '_load_embedding_model'):
                mock_strategy = MagicMock()
                mock_strategy.create_chunks = AsyncMock(return_value=[
                    MagicMock(content="Test chunk", metadata={})
                ])
                hybrid_buffer.chunk_strategy = mock_strategy
                hybrid_buffer.embedding_model = AsyncMock(return_value=[0.1] * 384)
                
                # Mock token counter to return high values
                with patch.object(round_buffer.token_counter, 'count_message_tokens', return_value=60):
                    # Add multiple rounds to trigger FIFO in both buffers
                    for i in range(5):
                        messages = [
                            {
                                "id": f"msg_{i}",
                                "role": "user",
                                "content": f"Message {i}",
                                "metadata": {"session_id": "session_1"}
                            }
                        ]
                        await round_buffer.add(messages, "session_1")
                
                # Verify FIFO behavior
                assert len(round_buffer.rounds) <= round_buffer.max_size
                assert len(hybrid_buffer.chunks) <= hybrid_buffer.max_size
                assert hybrid_buffer.total_fifo_removals > 0
    
    @pytest.mark.asyncio
    async def test_storage_flush_workflow(self, sample_messages, mock_storage_handlers):
        """Test storage flush workflow."""
        hybrid_buffer = HybridBuffer()
        
        # Mock components
        with patch.object(hybrid_buffer, '_load_chunk_strategy'):
            with patch.object(hybrid_buffer, '_load_embedding_model'):
                mock_strategy = MagicMock()
                mock_strategy.create_chunks = AsyncMock(return_value=[
                    MagicMock(content="Test chunk", metadata={})
                ])
                hybrid_buffer.chunk_strategy = mock_strategy
                hybrid_buffer.embedding_model = AsyncMock(return_value=[0.1] * 384)
                
                # Add rounds to buffer
                await hybrid_buffer.add_from_rounds([sample_messages])
                
                # Verify data is in buffer
                assert len(hybrid_buffer.original_rounds) > 0
                assert len(hybrid_buffer.chunks) > 0
                
                # Flush to storage
                result = await hybrid_buffer.flush_to_storage()
                
                # Verify flush succeeded
                assert result is True
                assert len(hybrid_buffer.original_rounds) == 0  # Cleared after flush
                assert hybrid_buffer.total_flushes == 1
                
                # Verify storage handlers were called
                mock_storage_handlers["sqlite"].assert_called_once()
                mock_storage_handlers["qdrant"].assert_called_once()
    
    @pytest.mark.asyncio
    async def test_query_caching_workflow(self, mock_retrieval_handler):
        """Test query caching workflow."""
        query_buffer = QueryBuffer(retrieval_handler=mock_retrieval_handler, cache_size=3)
        
        # First query (cache miss)
        result1 = await query_buffer.query("test query 1")
        assert query_buffer.cache_misses == 1
        assert query_buffer.cache_hits == 0
        
        # Same query (cache hit)
        result2 = await query_buffer.query("test query 1")
        assert query_buffer.cache_misses == 1
        assert query_buffer.cache_hits == 1
        assert result1 == result2
        
        # Different queries to fill cache
        await query_buffer.query("test query 2")
        await query_buffer.query("test query 3")
        assert len(query_buffer.query_cache) == 3
        
        # Add one more query (should evict oldest)
        await query_buffer.query("test query 4")
        assert len(query_buffer.query_cache) == 3
        
        # Verify LRU eviction
        cache_keys = list(query_buffer.query_cache.keys())
        assert "test query 1|score|desc|15" not in cache_keys  # Oldest evicted
    
    @pytest.mark.asyncio
    async def test_error_handling_workflow(self, sample_messages):
        """Test error handling across buffer components."""
        round_buffer = RoundBuffer()
        hybrid_buffer = HybridBuffer()
        
        # Set up transfer handler that fails
        failing_handler = AsyncMock(side_effect=Exception("Transfer failed"))
        round_buffer.set_transfer_handler(failing_handler)
        
        # Mock token counter to trigger transfer
        with patch.object(round_buffer.token_counter, 'count_message_tokens', return_value=1000):
            await round_buffer.add(sample_messages, "session_1")
        
        # Verify data is preserved on transfer failure
        assert len(round_buffer.rounds) == 1  # Data not lost
        assert round_buffer.total_transfers == 0  # Transfer not counted as successful
        
        # Test HybridBuffer error handling
        with patch.object(hybrid_buffer, '_load_chunk_strategy', side_effect=Exception("Strategy load failed")):
            await hybrid_buffer.add_from_rounds([sample_messages])
            
            # Should still add original rounds even if chunking fails
            assert len(hybrid_buffer.original_rounds) > 0
    
    @pytest.mark.asyncio
    async def test_concurrent_operations_workflow(self, mock_storage_handlers):
        """Test concurrent operations across buffer components."""
        round_buffer = RoundBuffer(max_tokens=1000)
        hybrid_buffer = HybridBuffer()
        query_buffer = QueryBuffer()
        
        # Set up components
        round_buffer.set_transfer_handler(hybrid_buffer.add_from_rounds)
        
        # Mock components
        with patch.object(hybrid_buffer, '_load_chunk_strategy'):
            with patch.object(hybrid_buffer, '_load_embedding_model'):
                mock_strategy = MagicMock()
                mock_strategy.create_chunks = AsyncMock(return_value=[
                    MagicMock(content="Test chunk", metadata={})
                ])
                hybrid_buffer.chunk_strategy = mock_strategy
                hybrid_buffer.embedding_model = AsyncMock(return_value=[0.1] * 384)
                
                # Run concurrent operations
                tasks = []
                
                # Concurrent adds to RoundBuffer
                for i in range(10):
                    messages = [
                        {
                            "id": f"msg_{i}",
                            "role": "user",
                            "content": f"Message {i}",
                            "metadata": {"session_id": f"session_{i % 3}"}
                        }
                    ]
                    tasks.append(round_buffer.add(messages, f"session_{i % 3}"))
                
                # Concurrent queries
                for i in range(5):
                    tasks.append(query_buffer.query(f"query {i}"))
                
                # Concurrent flushes
                for i in range(3):
                    tasks.append(hybrid_buffer.flush_to_storage())
                
                # Execute all tasks concurrently
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Verify no exceptions occurred
                exceptions = [r for r in results if isinstance(r, Exception)]
                assert len(exceptions) == 0
                
                # Verify system state is consistent
                assert isinstance(round_buffer.get_stats(), dict)
                assert isinstance(hybrid_buffer.get_stats(), dict)
                assert isinstance(query_buffer.get_stats(), dict)


class TestBufferPerformance:
    """Performance tests for Buffer system."""
    
    @pytest.mark.asyncio
    async def test_token_counting_performance(self):
        """Test token counting performance."""
        counter = TokenCounter()
        
        # Test with various text sizes
        texts = [
            "Short text",
            "Medium length text with more words and content",
            "Very long text " * 100,  # Very long text
            "Mixed 中文 and English text with various characters"
        ]
        
        for text in texts:
            tokens = counter.count_tokens(text)
            assert tokens > 0
            assert isinstance(tokens, int)
        
        # Test with message lists
        messages = [
            {"role": "user", "content": text}
            for text in texts
        ]
        
        total_tokens = counter.count_message_tokens(messages)
        assert total_tokens > 0
        assert isinstance(total_tokens, int)
    
    @pytest.mark.asyncio
    async def test_buffer_throughput(self, mock_storage_handlers):
        """Test buffer throughput with many operations."""
        round_buffer = RoundBuffer(max_tokens=500, max_size=10)
        hybrid_buffer = HybridBuffer(max_size=20)
        
        round_buffer.set_transfer_handler(hybrid_buffer.add_from_rounds)
        
        # Mock components for performance
        with patch.object(hybrid_buffer, '_load_chunk_strategy'):
            with patch.object(hybrid_buffer, '_load_embedding_model'):
                mock_strategy = MagicMock()
                mock_strategy.create_chunks = AsyncMock(return_value=[
                    MagicMock(content="Test chunk", metadata={})
                ])
                hybrid_buffer.chunk_strategy = mock_strategy
                hybrid_buffer.embedding_model = AsyncMock(return_value=[0.1] * 384)
                
                # Add many messages
                for i in range(100):
                    messages = [
                        {
                            "id": f"msg_{i}",
                            "role": "user",
                            "content": f"Performance test message {i}",
                            "metadata": {"session_id": f"session_{i % 10}"}
                        }
                    ]
                    await round_buffer.add(messages, f"session_{i % 10}")
                
                # Verify system handled the load
                assert round_buffer.total_rounds_added == 100
                assert hybrid_buffer.total_rounds_received > 0
                
                # Test query performance
                query_buffer = QueryBuffer()
                
                for i in range(50):
                    results = await query_buffer.query(f"query {i}", hybrid_buffer=hybrid_buffer)
                    assert isinstance(results, list)
                
                assert query_buffer.total_queries == 50
