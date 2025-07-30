"""Unit tests for HybridBuffer.get_messages_by_session method.

This module tests the newly added get_messages_by_session method in HybridBuffer
to ensure it returns consistent format and handles session filtering correctly.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from typing import Dict, Any, List

from memfuse_core.buffer.hybrid_buffer import HybridBuffer
from memfuse_core.interfaces import MessageList


class TestHybridBufferGetMessagesBySession:
    """Test cases for HybridBuffer.get_messages_by_session method."""

    @pytest.fixture
    def sample_rounds(self) -> List[MessageList]:
        """Sample message rounds for testing."""
        return [
            [
                {
                    "id": "msg_1",
                    "role": "user",
                    "content": "Hello from session 1",
                    "created_at": "2024-01-01T10:00:00Z",
                    "updated_at": "2024-01-01T10:00:00Z",
                    "metadata": {"session_id": "session_1", "user_id": "user_1"}
                },
                {
                    "id": "msg_2",
                    "role": "assistant",
                    "content": "Hi there from session 1",
                    "created_at": "2024-01-01T10:01:00Z",
                    "updated_at": "2024-01-01T10:01:00Z",
                    "metadata": {"session_id": "session_1", "user_id": "user_1"}
                }
            ],
            [
                {
                    "id": "msg_3",
                    "role": "user",
                    "content": "Hello from session 2",
                    "created_at": "2024-01-01T11:00:00Z",
                    "updated_at": "2024-01-01T11:00:00Z",
                    "metadata": {"session_id": "session_2", "user_id": "user_2"}
                }
            ],
            [
                {
                    "id": "msg_4",
                    "role": "user",
                    "content": "Another message from session 1",
                    "created_at": "2024-01-01T12:00:00Z",
                    "updated_at": "2024-01-01T12:00:00Z",
                    "metadata": {"session_id": "session_1", "user_id": "user_1"}
                }
            ]
        ]

    @pytest.fixture
    def mock_chunk_strategy(self):
        """Mock chunk strategy."""
        strategy = MagicMock()
        strategy.create_chunks.return_value = []
        return strategy

    @pytest.fixture
    def mock_embedding_model(self):
        """Mock embedding model."""
        model = MagicMock()
        model.encode.return_value = [0.1, 0.2, 0.3]
        return model

    @pytest.mark.asyncio
    async def test_get_messages_by_session_basic(self, sample_rounds, mock_chunk_strategy, mock_embedding_model):
        """Test basic functionality of get_messages_by_session."""
        buffer = HybridBuffer()
        buffer.chunk_strategy = mock_chunk_strategy
        buffer.embedding_model = mock_embedding_model
        
        # Add sample data
        await buffer.add_from_rounds(sample_rounds)
        
        # Get messages for session_1
        result = await buffer.get_messages_by_session("session_1")
        
        # Should return 3 messages from session_1 (msg_1, msg_2, msg_4)
        assert len(result) == 3
        assert all(isinstance(msg, dict) for msg in result)
        
        # Check that all messages belong to session_1
        for msg in result:
            assert msg["metadata"]["session_id"] == "session_1"
            assert msg["metadata"]["source"] == "hybrid_buffer"

    @pytest.mark.asyncio
    async def test_get_messages_by_session_filtering(self, sample_rounds, mock_chunk_strategy, mock_embedding_model):
        """Test session filtering works correctly."""
        buffer = HybridBuffer()
        buffer.chunk_strategy = mock_chunk_strategy
        buffer.embedding_model = mock_embedding_model
        
        await buffer.add_from_rounds(sample_rounds)
        
        # Get messages for session_2
        result = await buffer.get_messages_by_session("session_2")
        
        # Should return only 1 message from session_2
        assert len(result) == 1
        assert result[0]["id"] == "msg_3"
        assert result[0]["content"] == "Hello from session 2"
        assert result[0]["metadata"]["session_id"] == "session_2"

    @pytest.mark.asyncio
    async def test_get_messages_by_session_nonexistent(self, sample_rounds, mock_chunk_strategy, mock_embedding_model):
        """Test querying for non-existent session."""
        buffer = HybridBuffer()
        buffer.chunk_strategy = mock_chunk_strategy
        buffer.embedding_model = mock_embedding_model
        
        await buffer.add_from_rounds(sample_rounds)
        
        # Get messages for non-existent session
        result = await buffer.get_messages_by_session("nonexistent_session")
        
        # Should return empty list
        assert len(result) == 0
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_get_messages_by_session_empty_buffer(self, mock_chunk_strategy, mock_embedding_model):
        """Test querying empty buffer."""
        buffer = HybridBuffer()
        buffer.chunk_strategy = mock_chunk_strategy
        buffer.embedding_model = mock_embedding_model
        
        # Query empty buffer
        result = await buffer.get_messages_by_session("any_session")
        
        # Should return empty list
        assert len(result) == 0
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_get_messages_by_session_sorting(self, mock_chunk_strategy, mock_embedding_model):
        """Test sorting functionality."""
        buffer = HybridBuffer()
        buffer.chunk_strategy = mock_chunk_strategy
        buffer.embedding_model = mock_embedding_model
        
        # Create messages with different timestamps
        rounds = [
            [
                {
                    "id": "msg_1",
                    "role": "user",
                    "content": "First message",
                    "created_at": "2024-01-01T10:00:00Z",
                    "metadata": {"session_id": "test_session"}
                },
                {
                    "id": "msg_2",
                    "role": "user",
                    "content": "Second message",
                    "created_at": "2024-01-01T12:00:00Z",
                    "metadata": {"session_id": "test_session"}
                },
                {
                    "id": "msg_3",
                    "role": "user",
                    "content": "Third message",
                    "created_at": "2024-01-01T11:00:00Z",
                    "metadata": {"session_id": "test_session"}
                }
            ]
        ]
        
        await buffer.add_from_rounds(rounds)
        
        # Test descending order (default)
        result_desc = await buffer.get_messages_by_session("test_session", sort_by="created_at", order="desc")
        assert len(result_desc) == 3
        assert result_desc[0]["content"] == "Second message"  # Latest first
        assert result_desc[1]["content"] == "Third message"
        assert result_desc[2]["content"] == "First message"
        
        # Test ascending order
        result_asc = await buffer.get_messages_by_session("test_session", sort_by="created_at", order="asc")
        assert len(result_asc) == 3
        assert result_asc[0]["content"] == "First message"  # Earliest first
        assert result_asc[1]["content"] == "Third message"
        assert result_asc[2]["content"] == "Second message"

    @pytest.mark.asyncio
    async def test_get_messages_by_session_limit(self, sample_rounds, mock_chunk_strategy, mock_embedding_model):
        """Test limit functionality."""
        buffer = HybridBuffer()
        buffer.chunk_strategy = mock_chunk_strategy
        buffer.embedding_model = mock_embedding_model
        
        await buffer.add_from_rounds(sample_rounds)
        
        # Get messages with limit
        result = await buffer.get_messages_by_session("session_1", limit=2)
        
        # Should return only 2 messages even though session_1 has 3
        assert len(result) == 2
        assert all(msg["metadata"]["session_id"] == "session_1" for msg in result)

    @pytest.mark.asyncio
    async def test_get_messages_by_session_sort_by_id(self, mock_chunk_strategy, mock_embedding_model):
        """Test sorting by ID."""
        buffer = HybridBuffer()
        buffer.chunk_strategy = mock_chunk_strategy
        buffer.embedding_model = mock_embedding_model
        
        rounds = [
            [
                {
                    "id": "msg_c",
                    "role": "user",
                    "content": "Message C",
                    "metadata": {"session_id": "test_session"}
                },
                {
                    "id": "msg_a",
                    "role": "user",
                    "content": "Message A",
                    "metadata": {"session_id": "test_session"}
                },
                {
                    "id": "msg_b",
                    "role": "user",
                    "content": "Message B",
                    "metadata": {"session_id": "test_session"}
                }
            ]
        ]
        
        await buffer.add_from_rounds(rounds)
        
        # Test sorting by ID in ascending order
        result = await buffer.get_messages_by_session("test_session", sort_by="id", order="asc")
        assert len(result) == 3
        assert result[0]["id"] == "msg_a"
        assert result[1]["id"] == "msg_b"
        assert result[2]["id"] == "msg_c"

    @pytest.mark.asyncio
    async def test_get_messages_by_session_session_id_fallback(self, mock_chunk_strategy, mock_embedding_model):
        """Test fallback to message-level session_id when not in metadata."""
        buffer = HybridBuffer()
        buffer.chunk_strategy = mock_chunk_strategy
        buffer.embedding_model = mock_embedding_model
        
        # Message with session_id at top level instead of metadata
        rounds = [
            [
                {
                    "id": "msg_1",
                    "role": "user",
                    "content": "Message with top-level session_id",
                    "session_id": "fallback_session",
                    "metadata": {}
                }
            ]
        ]
        
        await buffer.add_from_rounds(rounds)
        
        result = await buffer.get_messages_by_session("fallback_session")
        assert len(result) == 1
        assert result[0]["content"] == "Message with top-level session_id"

    @pytest.mark.asyncio
    async def test_get_messages_by_session_api_format(self, sample_rounds, mock_chunk_strategy, mock_embedding_model):
        """Test that returned messages are in proper API format."""
        buffer = HybridBuffer()
        buffer.chunk_strategy = mock_chunk_strategy
        buffer.embedding_model = mock_embedding_model
        
        await buffer.add_from_rounds(sample_rounds)
        
        result = await buffer.get_messages_by_session("session_1")
        
        # Check API format for each message
        for msg in result:
            assert "id" in msg
            assert "role" in msg
            assert "content" in msg
            assert "created_at" in msg
            assert "updated_at" in msg
            assert "metadata" in msg
            assert msg["metadata"]["source"] == "hybrid_buffer"
            assert isinstance(msg, dict)
