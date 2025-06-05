"""Unit tests for MessageChunkStrategy."""

import pytest
from memfuse_core.rag.chunk import MessageChunkStrategy, ChunkData


class TestMessageChunkStrategy:
    """Test cases for MessageChunkStrategy."""
    
    @pytest.fixture
    def strategy(self):
        """Create a MessageChunkStrategy instance for testing."""
        return MessageChunkStrategy()
    
    @pytest.mark.asyncio
    async def test_create_chunks_empty_input(self, strategy):
        """Test create_chunks with empty input."""
        result = await strategy.create_chunks([])
        assert result == []
    
    @pytest.mark.asyncio
    async def test_create_chunks_single_message_list(self, strategy):
        """Test create_chunks with a single MessageList."""
        message_batch_list = [
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ]
        ]
        
        result = await strategy.create_chunks(message_batch_list)
        
        assert len(result) == 1
        assert isinstance(result[0], ChunkData)
        
        chunk = result[0]
        assert "Hello" in chunk.content
        assert "Hi there!" in chunk.content
        assert chunk.metadata["strategy"] == "message"
        assert chunk.metadata["message_count"] == 2
        assert chunk.metadata["batch_index"] == 0
        assert chunk.metadata["roles"] == ["user", "assistant"]
    
    @pytest.mark.asyncio
    async def test_create_chunks_multiple_message_lists(self, strategy):
        """Test create_chunks with multiple MessageLists."""
        message_batch_list = [
            [
                {"role": "user", "content": "First conversation"},
                {"role": "assistant", "content": "First response"}
            ],
            [
                {"role": "user", "content": "Second conversation"},
                {"role": "assistant", "content": "Second response"}
            ]
        ]
        
        result = await strategy.create_chunks(message_batch_list)
        
        assert len(result) == 2
        
        # Check first chunk
        chunk1 = result[0]
        assert "First conversation" in chunk1.content
        assert "First response" in chunk1.content
        assert chunk1.metadata["batch_index"] == 0
        
        # Check second chunk
        chunk2 = result[1]
        assert "Second conversation" in chunk2.content
        assert "Second response" in chunk2.content
        assert chunk2.metadata["batch_index"] == 1
    
    @pytest.mark.asyncio
    async def test_create_chunks_empty_message_list(self, strategy):
        """Test create_chunks with empty MessageList."""
        message_batch_list = [
            [],  # Empty MessageList
            [
                {"role": "user", "content": "Valid message"}
            ]
        ]
        
        result = await strategy.create_chunks(message_batch_list)
        
        # Should skip empty MessageList
        assert len(result) == 1
        assert "Valid message" in result[0].content
    
    @pytest.mark.asyncio
    async def test_create_chunks_with_metadata(self, strategy):
        """Test create_chunks preserves message metadata."""
        message_batch_list = [
            [
                {
                    "role": "user", 
                    "content": "Test message",
                    "metadata": {"session_id": "test_session", "timestamp": "2024-01-01"}
                }
            ]
        ]
        
        result = await strategy.create_chunks(message_batch_list)
        
        assert len(result) == 1
        chunk = result[0]
        assert chunk.metadata["strategy"] == "message"
        assert chunk.metadata["message_count"] == 1
        # Original message metadata is not directly copied to chunk metadata
        # but the content should be preserved
        assert "Test message" in chunk.content
    
    @pytest.mark.asyncio
    async def test_create_chunks_with_missing_role(self, strategy):
        """Test create_chunks handles missing role gracefully."""
        message_batch_list = [
            [
                {"content": "Message without role"},
                {"role": "assistant", "content": "Response with role"}
            ]
        ]
        
        result = await strategy.create_chunks(message_batch_list)
        
        assert len(result) == 1
        chunk = result[0]
        assert chunk.metadata["roles"] == ["unknown", "assistant"]
    
    @pytest.mark.asyncio
    async def test_create_chunks_content_combination(self, strategy):
        """Test that messages are properly combined in chunks."""
        message_batch_list = [
            [
                {"role": "user", "content": "Line 1"},
                {"role": "assistant", "content": "Line 2"},
                {"role": "user", "content": "Line 3"}
            ]
        ]
        
        result = await strategy.create_chunks(message_batch_list)
        
        assert len(result) == 1
        chunk = result[0]
        
        # Check that content is properly formatted
        lines = chunk.content.split('\n')
        assert len(lines) == 3
        assert "user: Line 1" in chunk.content
        assert "assistant: Line 2" in chunk.content
        assert "user: Line 3" in chunk.content
    
    @pytest.mark.asyncio
    async def test_create_chunks_dict_content(self, strategy):
        """Test create_chunks handles dict content."""
        message_batch_list = [
            [
                {
                    "role": "user", 
                    "content": {"text": "Dict content message"}
                }
            ]
        ]
        
        result = await strategy.create_chunks(message_batch_list)
        
        assert len(result) == 1
        chunk = result[0]
        assert "Dict content message" in chunk.content
    
    @pytest.mark.asyncio
    async def test_create_chunks_none_content(self, strategy):
        """Test create_chunks handles None content."""
        message_batch_list = [
            [
                {"role": "user", "content": None},
                {"role": "assistant", "content": "Valid content"}
            ]
        ]
        
        result = await strategy.create_chunks(message_batch_list)
        
        assert len(result) == 1
        chunk = result[0]
        assert "Valid content" in chunk.content
        # None content should be handled gracefully
    
    def test_combine_messages_method(self, strategy):
        """Test the _combine_messages helper method."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        
        result = strategy._combine_messages(messages)
        
        assert "user: Hello" in result
        assert "assistant: Hi there!" in result
        assert result.count('\n') == 1  # One newline between messages
    
    def test_combine_messages_empty_list(self, strategy):
        """Test _combine_messages with empty list."""
        result = strategy._combine_messages([])
        assert result == ""
    
    def test_combine_messages_single_message(self, strategy):
        """Test _combine_messages with single message."""
        messages = [{"role": "user", "content": "Single message"}]
        result = strategy._combine_messages(messages)
        assert result == "user: Single message"

    @pytest.mark.asyncio
    async def test_chunk_ids_are_unique(self, strategy):
        """Test that each chunk gets a unique ID."""
        message_batch_list = [
            [{"role": "user", "content": "Message 1"}],
            [{"role": "user", "content": "Message 2"}],
            [{"role": "user", "content": "Message 3"}]
        ]

        result = await strategy.create_chunks(message_batch_list)

        chunk_ids = [chunk.chunk_id for chunk in result]
        assert len(chunk_ids) == len(set(chunk_ids))  # All IDs should be unique
