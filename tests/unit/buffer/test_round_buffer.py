"""Tests for RoundBuffer in Buffer."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from memfuse_core.buffer.round_buffer import RoundBuffer


@pytest.fixture
def sample_messages():
    """Fixture providing sample messages for testing."""
    return [
        {
            "id": "msg_1",
            "role": "user",
            "content": "Hello, how are you?",
            "metadata": {"session_id": "session_1"}
        },
        {
            "id": "msg_2", 
            "role": "assistant",
            "content": "I'm doing well, thank you!",
            "metadata": {"session_id": "session_1"}
        }
    ]


@pytest.fixture
def large_messages():
    """Fixture providing large messages that exceed token limits."""
    return [
        {
            "id": "msg_large",
            "role": "user",
            "content": "This is a very long message " * 100,  # Very long content
            "metadata": {"session_id": "session_1"}
        }
    ]


@pytest.fixture
def mock_token_counter():
    """Fixture providing a mock token counter."""
    with patch('memfuse_core.buffer.round_buffer.get_token_counter') as mock_get_counter:
        mock_counter = MagicMock()
        mock_counter.count_message_tokens.return_value = 10  # Default 10 tokens per message
        mock_get_counter.return_value = mock_counter
        yield mock_counter


class TestRoundBufferInitialization:
    """Test cases for RoundBuffer initialization."""
    
    def test_default_initialization(self):
        """Test RoundBuffer initialization with default parameters."""
        buffer = RoundBuffer()
        
        assert buffer.max_tokens == 800
        assert buffer.max_size == 5
        assert buffer.token_model == "gpt-4o-mini"
        assert buffer.rounds == []
        assert buffer.current_tokens == 0
        assert buffer.current_session_id is None
        assert buffer.transfer_handler is None
    
    def test_custom_initialization(self):
        """Test RoundBuffer initialization with custom parameters."""
        buffer = RoundBuffer(max_tokens=1000, max_size=10, token_model="gpt-4")
        
        assert buffer.max_tokens == 1000
        assert buffer.max_size == 10
        assert buffer.token_model == "gpt-4"
    
    def test_set_transfer_handler(self):
        """Test setting transfer handler."""
        buffer = RoundBuffer()
        handler = AsyncMock()
        
        buffer.set_transfer_handler(handler)
        
        assert buffer.transfer_handler == handler


class TestRoundBufferBasicOperations:
    """Test cases for basic RoundBuffer operations."""
    
    @pytest.mark.asyncio
    async def test_add_messages_basic(self, sample_messages, mock_token_counter):
        """Test adding messages to buffer."""
        buffer = RoundBuffer(max_tokens=100)
        
        result = await buffer.add(sample_messages, "session_1")
        
        assert result is False  # No transfer triggered
        assert len(buffer.rounds) == 1
        assert buffer.rounds[0] == sample_messages
        assert buffer.current_tokens == 10  # Mock returns 10 tokens
        assert buffer.current_session_id == "session_1"
        assert buffer.total_rounds_added == 1
    
    @pytest.mark.asyncio
    async def test_add_empty_messages(self, mock_token_counter):
        """Test adding empty message list."""
        buffer = RoundBuffer()
        
        result = await buffer.add([], "session_1")
        
        assert result is False
        assert len(buffer.rounds) == 0
        assert buffer.current_tokens == 0
    
    @pytest.mark.asyncio
    async def test_extract_session_id_from_messages(self, mock_token_counter):
        """Test extracting session_id from message metadata."""
        buffer = RoundBuffer()
        messages = [
            {"role": "user", "content": "Hello", "metadata": {"session_id": "auto_session"}}
        ]
        
        await buffer.add(messages)
        
        assert buffer.current_session_id == "auto_session"
    
    @pytest.mark.asyncio
    async def test_add_messages_without_session_id(self, mock_token_counter):
        """Test adding messages without session_id."""
        buffer = RoundBuffer()
        messages = [{"role": "user", "content": "Hello"}]
        
        await buffer.add(messages)
        
        assert buffer.current_session_id is None
        assert len(buffer.rounds) == 1


class TestRoundBufferTokenLimitTrigger:
    """Test cases for token limit triggering."""
    
    @pytest.mark.asyncio
    async def test_token_limit_triggers_transfer(self, sample_messages, mock_token_counter):
        """Test that exceeding token limit triggers transfer."""
        buffer = RoundBuffer(max_tokens=50)
        transfer_handler = AsyncMock()
        buffer.set_transfer_handler(transfer_handler)
        
        # Mock token counter to return high token count
        mock_token_counter.count_message_tokens.return_value = 60
        
        result = await buffer.add(sample_messages, "session_1")
        
        assert result is False  # Transfer happens before adding
        transfer_handler.assert_called_once()
        assert len(buffer.rounds) == 1  # New message added after transfer
        assert buffer.current_tokens == 60
        assert buffer.total_transfers == 1
    
    @pytest.mark.asyncio
    async def test_token_limit_with_existing_data(self, sample_messages, mock_token_counter):
        """Test token limit with existing data in buffer."""
        buffer = RoundBuffer(max_tokens=50)
        transfer_handler = AsyncMock()
        buffer.set_transfer_handler(transfer_handler)
        
        # Add first message (10 tokens)
        await buffer.add(sample_messages, "session_1")
        assert buffer.current_tokens == 10
        
        # Add second message that would exceed limit (10 + 45 > 50)
        mock_token_counter.count_message_tokens.return_value = 45
        await buffer.add(sample_messages, "session_1")
        
        transfer_handler.assert_called_once()
        assert buffer.current_tokens == 45  # Only new message after transfer
        assert buffer.total_transfers == 1


class TestRoundBufferSessionChangeTrigger:
    """Test cases for session change triggering."""
    
    @pytest.mark.asyncio
    async def test_session_change_triggers_transfer(self, sample_messages, mock_token_counter):
        """Test that session change triggers transfer."""
        buffer = RoundBuffer()
        transfer_handler = AsyncMock()
        buffer.set_transfer_handler(transfer_handler)
        
        # Add messages for session_1
        await buffer.add(sample_messages, "session_1")
        assert buffer.current_session_id == "session_1"
        assert len(buffer.rounds) == 1
        
        # Add messages for session_2 (should trigger transfer)
        new_messages = [{"role": "user", "content": "New session", "metadata": {"session_id": "session_2"}}]
        await buffer.add(new_messages, "session_2")
        
        transfer_handler.assert_called_once()
        assert buffer.current_session_id == "session_2"
        assert len(buffer.rounds) == 1  # Only new session data
        assert buffer.total_session_changes == 1
    
    @pytest.mark.asyncio
    async def test_session_change_from_none(self, sample_messages, mock_token_counter):
        """Test session change from None to actual session."""
        buffer = RoundBuffer()
        transfer_handler = AsyncMock()
        buffer.set_transfer_handler(transfer_handler)
        
        # Add messages without session_id
        messages_no_session = [{"role": "user", "content": "Hello"}]
        await buffer.add(messages_no_session)
        assert buffer.current_session_id is None
        
        # Add messages with session_id (should not trigger transfer)
        await buffer.add(sample_messages, "session_1")
        
        transfer_handler.assert_not_called()  # No transfer for None -> session_1
        assert buffer.current_session_id == "session_1"
        assert len(buffer.rounds) == 2


class TestRoundBufferSizeLimitTrigger:
    """Test cases for size limit triggering."""
    
    @pytest.mark.asyncio
    async def test_size_limit_triggers_transfer(self, sample_messages, mock_token_counter):
        """Test that exceeding size limit triggers transfer."""
        buffer = RoundBuffer(max_size=2, max_tokens=1000)  # High token limit
        transfer_handler = AsyncMock()
        buffer.set_transfer_handler(transfer_handler)
        
        # Add messages up to size limit
        await buffer.add(sample_messages, "session_1")
        await buffer.add(sample_messages, "session_1")
        assert len(buffer.rounds) == 2
        
        # Add one more (should trigger transfer)
        await buffer.add(sample_messages, "session_1")
        
        transfer_handler.assert_called_once()
        assert len(buffer.rounds) == 1  # Only new message after transfer
        assert buffer.total_transfers == 1


class TestRoundBufferReadAPI:
    """Test cases for Read API functionality."""
    
    @pytest.mark.asyncio
    async def test_get_all_messages_for_read_api(self, mock_token_counter):
        """Test getting all messages for Read API."""
        buffer = RoundBuffer()
        
        # Add multiple rounds
        messages1 = [{"id": "1", "role": "user", "content": "Hello", "created_at": "2024-01-01T10:00:00Z"}]
        messages2 = [{"id": "2", "role": "assistant", "content": "Hi", "created_at": "2024-01-01T10:01:00Z"}]
        
        await buffer.add(messages1, "session_1")
        await buffer.add(messages2, "session_1")
        
        result = await buffer.get_all_messages_for_read_api()
        
        assert len(result) == 2
        assert all("source" in msg["metadata"] for msg in result)
        assert all(msg["metadata"]["source"] == "round_buffer" for msg in result)
    
    @pytest.mark.asyncio
    async def test_get_messages_with_limit(self, mock_token_counter):
        """Test getting messages with limit."""
        buffer = RoundBuffer()
        
        # Add multiple messages
        for i in range(5):
            messages = [{"id": f"{i}", "role": "user", "content": f"Message {i}"}]
            await buffer.add(messages, "session_1")
        
        result = await buffer.get_all_messages_for_read_api(limit=3)
        
        assert len(result) == 3
    
    @pytest.mark.asyncio
    async def test_get_messages_with_sorting(self, mock_token_counter):
        """Test getting messages with sorting."""
        buffer = RoundBuffer()
        
        # Add messages with timestamps
        messages1 = [{"id": "1", "role": "user", "content": "First", "created_at": "2024-01-01T10:00:00Z"}]
        messages2 = [{"id": "2", "role": "user", "content": "Second", "created_at": "2024-01-01T09:00:00Z"}]
        
        await buffer.add(messages1, "session_1")
        await buffer.add(messages2, "session_1")
        
        # Test descending order (default)
        result_desc = await buffer.get_all_messages_for_read_api(sort_by="timestamp", order="desc")
        assert result_desc[0]["id"] == "1"  # Later timestamp first
        
        # Test ascending order
        result_asc = await buffer.get_all_messages_for_read_api(sort_by="timestamp", order="asc")
        assert result_asc[0]["id"] == "2"  # Earlier timestamp first


class TestRoundBufferBufferInfo:
    """Test cases for buffer info functionality."""
    
    @pytest.mark.asyncio
    async def test_get_buffer_info_empty(self):
        """Test getting buffer info when empty."""
        buffer = RoundBuffer()
        
        info = await buffer.get_buffer_info()
        
        assert info["messages_available"] is False
        assert info["messages_count"] == 0
        assert info["rounds_count"] == 0
        assert info["current_tokens"] == 0
        assert info["max_tokens"] == 800
        assert info["current_session_id"] is None
        assert info["buffer_type"] == "round_buffer"
    
    @pytest.mark.asyncio
    async def test_get_buffer_info_with_data(self, sample_messages, mock_token_counter):
        """Test getting buffer info with data."""
        buffer = RoundBuffer()
        
        await buffer.add(sample_messages, "session_1")
        
        info = await buffer.get_buffer_info()
        
        assert info["messages_available"] is True
        assert info["messages_count"] == 2  # Two messages in sample_messages
        assert info["rounds_count"] == 1
        assert info["current_tokens"] == 10
        assert info["current_session_id"] == "session_1"


class TestRoundBufferUtilityMethods:
    """Test cases for utility methods."""
    
    @pytest.mark.asyncio
    async def test_force_transfer(self, sample_messages, mock_token_counter):
        """Test force transfer functionality."""
        buffer = RoundBuffer()
        transfer_handler = AsyncMock()
        buffer.set_transfer_handler(transfer_handler)
        
        await buffer.add(sample_messages, "session_1")
        
        result = await buffer.force_transfer()
        
        assert result is True
        transfer_handler.assert_called_once()
        assert len(buffer.rounds) == 0
        assert buffer.current_tokens == 0
    
    @pytest.mark.asyncio
    async def test_force_transfer_empty_buffer(self):
        """Test force transfer on empty buffer."""
        buffer = RoundBuffer()
        transfer_handler = AsyncMock()
        buffer.set_transfer_handler(transfer_handler)
        
        result = await buffer.force_transfer()
        
        assert result is True
        transfer_handler.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_force_transfer_without_handler(self, sample_messages, mock_token_counter):
        """Test force transfer without transfer handler."""
        buffer = RoundBuffer()
        
        await buffer.add(sample_messages, "session_1")
        
        result = await buffer.force_transfer()
        
        assert result is True  # Should still succeed
        assert len(buffer.rounds) == 0  # Data should be cleared
    
    @pytest.mark.asyncio
    async def test_clear_buffer(self, sample_messages, mock_token_counter):
        """Test clearing buffer."""
        buffer = RoundBuffer()
        
        await buffer.add(sample_messages, "session_1")
        assert len(buffer.rounds) == 1
        
        await buffer.clear()
        
        assert len(buffer.rounds) == 0
        assert buffer.current_tokens == 0
        assert buffer.current_session_id is None
    
    def test_get_stats(self, mock_token_counter):
        """Test getting buffer statistics."""
        buffer = RoundBuffer(max_tokens=1000, max_size=10)
        
        stats = buffer.get_stats()
        
        assert stats["rounds_count"] == 0
        assert stats["current_tokens"] == 0
        assert stats["max_tokens"] == 1000
        assert stats["max_size"] == 10
        assert stats["current_session_id"] is None
        assert stats["total_rounds_added"] == 0
        assert stats["total_transfers"] == 0
        assert stats["total_session_changes"] == 0
        assert stats["has_transfer_handler"] is False
        assert stats["token_model"] == "gpt-4o-mini"


class TestRoundBufferErrorHandling:
    """Test cases for error handling."""
    
    @pytest.mark.asyncio
    async def test_transfer_handler_error(self, sample_messages, mock_token_counter):
        """Test handling transfer handler errors."""
        buffer = RoundBuffer(max_tokens=50)
        transfer_handler = AsyncMock(side_effect=Exception("Transfer failed"))
        buffer.set_transfer_handler(transfer_handler)
        
        # Mock high token count to trigger transfer
        mock_token_counter.count_message_tokens.return_value = 60
        
        await buffer.add(sample_messages, "session_1")
        
        # Buffer should not be cleared on transfer failure
        assert len(buffer.rounds) == 1  # Original data preserved
        assert buffer.total_transfers == 0  # Transfer not counted as successful
    
    @pytest.mark.asyncio
    async def test_concurrent_access(self, sample_messages, mock_token_counter):
        """Test concurrent access to buffer."""
        buffer = RoundBuffer()
        
        # Simulate concurrent adds
        tasks = []
        for i in range(10):
            messages = [{"role": "user", "content": f"Message {i}", "metadata": {"session_id": f"session_{i}"}}]
            tasks.append(buffer.add(messages, f"session_{i}"))
        
        await asyncio.gather(*tasks)
        
        # Should handle concurrent access gracefully
        assert buffer.total_rounds_added >= 1  # At least some messages added
        assert isinstance(buffer.get_stats(), dict)  # Stats should be accessible
