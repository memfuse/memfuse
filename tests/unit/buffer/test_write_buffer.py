"""Unit tests for WriteBuffer implementation."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from memfuse_core.buffer.write_buffer import WriteBuffer


class TestWriteBuffer:
    """Test cases for WriteBuffer functionality."""
    
    @pytest.fixture
    def write_buffer_config(self):
        """Configuration for WriteBuffer testing."""
        return {
            'round_buffer': {
                'max_tokens': 100,
                'max_size': 2,
                'token_model': 'gpt-4o-mini'
            },
            'hybrid_buffer': {
                'max_size': 2,
                'chunk_strategy': 'message',
                'embedding_model': 'all-MiniLM-L6-v2'
            }
        }
    
    @pytest.fixture
    def mock_handlers(self):
        """Mock storage handlers."""
        sqlite_handler = AsyncMock()
        qdrant_handler = AsyncMock()
        return sqlite_handler, qdrant_handler
    
    @pytest.fixture
    def write_buffer(self, write_buffer_config, mock_handlers):
        """WriteBuffer instance for testing."""
        sqlite_handler, qdrant_handler = mock_handlers
        return WriteBuffer(
            config=write_buffer_config,
            sqlite_handler=sqlite_handler,
            qdrant_handler=qdrant_handler
        )
    
    def test_initialization(self, write_buffer):
        """Test WriteBuffer initialization."""
        assert write_buffer is not None
        assert write_buffer.round_buffer is not None
        assert write_buffer.hybrid_buffer is not None
        assert write_buffer.total_writes == 0
        assert write_buffer.total_transfers == 0
    
    def test_component_access(self, write_buffer):
        """Test access to internal components."""
        round_buffer = write_buffer.get_round_buffer()
        hybrid_buffer = write_buffer.get_hybrid_buffer()
        
        assert round_buffer is not None
        assert hybrid_buffer is not None
        assert round_buffer is write_buffer.round_buffer
        assert hybrid_buffer is write_buffer.hybrid_buffer
    
    @pytest.mark.asyncio
    async def test_add_single_message(self, write_buffer):
        """Test adding a single message list."""
        messages = [{"role": "user", "content": "test message"}]
        
        # Mock the round_buffer.add method
        write_buffer.round_buffer.add = AsyncMock(return_value=False)
        
        result = await write_buffer.add(messages, session_id="test_session")
        
        assert result["status"] == "success"
        assert result["transfer_triggered"] is False
        assert result["total_writes"] == 1
        assert result["total_transfers"] == 0
        
        # Verify round_buffer.add was called
        write_buffer.round_buffer.add.assert_called_once_with(messages, "test_session")
    
    @pytest.mark.asyncio
    async def test_add_with_transfer_trigger(self, write_buffer):
        """Test adding messages that trigger transfer."""
        messages = [{"role": "user", "content": "test message"}]
        
        # Mock the round_buffer.add method to return True (transfer triggered)
        write_buffer.round_buffer.add = AsyncMock(return_value=True)
        
        result = await write_buffer.add(messages, session_id="test_session")
        
        assert result["status"] == "success"
        assert result["transfer_triggered"] is True
        assert result["total_writes"] == 1
        assert result["total_transfers"] == 1
    
    @pytest.mark.asyncio
    async def test_add_batch(self, write_buffer):
        """Test adding a batch of message lists."""
        message_batch = [
            [{"role": "user", "content": "message 1"}],
            [{"role": "assistant", "content": "response 1"}],
            [{"role": "user", "content": "message 2"}]
        ]
        
        # Mock the add method
        original_add = write_buffer.add
        write_buffer.add = AsyncMock(side_effect=[
            {"status": "success", "transfer_triggered": False, "total_writes": 1, "total_transfers": 0},
            {"status": "success", "transfer_triggered": True, "total_writes": 2, "total_transfers": 1},
            {"status": "success", "transfer_triggered": False, "total_writes": 3, "total_transfers": 1}
        ])
        
        result = await write_buffer.add_batch(message_batch, session_id="test_session")
        
        assert result["status"] == "success"
        assert result["batch_size"] == 3
        assert result["processed"] == 3
        assert result["total_transfers"] == 1
        assert len(result["results"]) == 3
        
        # Verify add was called for each message list
        assert write_buffer.add.call_count == 3
    
    @pytest.mark.asyncio
    async def test_add_batch_empty(self, write_buffer):
        """Test adding empty batch."""
        result = await write_buffer.add_batch([], session_id="test_session")
        
        assert result["status"] == "success"
        assert result["message"] == "No message lists to add"
    
    @pytest.mark.asyncio
    async def test_flush_all(self, write_buffer):
        """Test flushing all buffers."""
        # Mock the internal methods
        write_buffer.round_buffer.rounds = [{"test": "data"}]
        write_buffer.round_buffer._transfer_and_clear = AsyncMock()
        write_buffer.hybrid_buffer.flush_to_storage = AsyncMock()
        
        result = await write_buffer.flush_all()
        
        assert result["status"] == "success"
        assert "flushed successfully" in result["message"]
        
        # Verify methods were called
        write_buffer.round_buffer._transfer_and_clear.assert_called_once_with("manual_flush")
        write_buffer.hybrid_buffer.flush_to_storage.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_flush_all_error(self, write_buffer):
        """Test flush_all error handling."""
        # Mock the internal methods to raise an exception
        write_buffer.round_buffer.rounds = [{"test": "data"}]
        write_buffer.round_buffer._transfer_and_clear = AsyncMock(side_effect=Exception("Test error"))
        
        result = await write_buffer.flush_all()
        
        assert result["status"] == "error"
        assert "Test error" in result["message"]
    
    def test_get_stats(self, write_buffer):
        """Test getting comprehensive statistics."""
        # Mock the stats methods
        write_buffer.round_buffer.get_stats = MagicMock(return_value={"rounds": 2})
        write_buffer.hybrid_buffer.get_stats = MagicMock(return_value={"chunks": 3})
        
        # Set some statistics
        write_buffer.total_writes = 5
        write_buffer.total_transfers = 2
        
        stats = write_buffer.get_stats()
        
        assert "write_buffer" in stats
        assert stats["write_buffer"]["total_writes"] == 5
        assert stats["write_buffer"]["total_transfers"] == 2
        assert stats["write_buffer"]["round_buffer"]["rounds"] == 2
        assert stats["write_buffer"]["hybrid_buffer"]["chunks"] == 3
    
    def test_is_empty(self, write_buffer):
        """Test checking if buffers are empty."""
        # Mock empty buffers
        write_buffer.round_buffer.rounds = []
        write_buffer.hybrid_buffer.chunks = []
        
        assert write_buffer.is_empty() is True
        
        # Mock non-empty buffers
        write_buffer.round_buffer.rounds = [{"test": "data"}]
        
        assert write_buffer.is_empty() is False
    
    @pytest.mark.asyncio
    async def test_clear_all(self, write_buffer):
        """Test clearing all buffers."""
        # Set up some data
        write_buffer.round_buffer.rounds = [{"test": "data"}]
        write_buffer.round_buffer.current_tokens = 100
        write_buffer.round_buffer.current_session_id = "test"
        write_buffer.hybrid_buffer.chunks = [{"test": "chunk"}]
        write_buffer.hybrid_buffer.embeddings = [{"test": "embedding"}]
        write_buffer.hybrid_buffer.original_rounds = [{"test": "round"}]
        write_buffer.total_writes = 5
        write_buffer.total_transfers = 2
        
        result = await write_buffer.clear_all()
        
        assert result["status"] == "success"
        assert "cleared" in result["message"]
        
        # Verify everything was cleared
        assert len(write_buffer.round_buffer.rounds) == 0
        assert write_buffer.round_buffer.current_tokens == 0
        assert write_buffer.round_buffer.current_session_id is None
        assert len(write_buffer.hybrid_buffer.chunks) == 0
        assert len(write_buffer.hybrid_buffer.embeddings) == 0
        assert len(write_buffer.hybrid_buffer.original_rounds) == 0
        assert write_buffer.total_writes == 0
        assert write_buffer.total_transfers == 0
    
    @pytest.mark.asyncio
    async def test_clear_all_error(self, write_buffer):
        """Test clear_all error handling."""
        # Mock an attribute that doesn't exist to cause an error
        write_buffer.round_buffer.rounds = None
        
        result = await write_buffer.clear_all()
        
        assert result["status"] == "error"
        assert "Clear failed" in result["message"]


if __name__ == "__main__":
    pytest.main([__file__])
