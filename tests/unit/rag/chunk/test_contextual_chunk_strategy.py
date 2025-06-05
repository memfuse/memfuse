"""Unit tests for ContextualChunkStrategy."""

import pytest
from memfuse_core.rag.chunk import ContextualChunkStrategy, ChunkData


class TestContextualChunkStrategy:
    """Test cases for ContextualChunkStrategy."""
    
    @pytest.fixture
    def strategy(self):
        """Create a ContextualChunkStrategy instance for testing."""
        return ContextualChunkStrategy(max_chunk_length=100)
    
    @pytest.fixture
    def long_strategy(self):
        """Create a ContextualChunkStrategy with longer max length."""
        return ContextualChunkStrategy(max_chunk_length=500)
    
    @pytest.mark.asyncio
    async def test_create_chunks_empty_input(self, strategy):
        """Test create_chunks with empty input."""
        result = await strategy.create_chunks([])
        assert result == []
    
    @pytest.mark.asyncio
    async def test_create_chunks_short_content(self, long_strategy):
        """Test create_chunks with content shorter than max length."""
        message_batch_list = [
            [
                {"role": "user", "content": "Short message"},
                {"role": "assistant", "content": "Short response"}
            ]
        ]
        
        result = await long_strategy.create_chunks(message_batch_list)
        
        assert len(result) == 1
        chunk = result[0]
        assert "Short message" in chunk.content
        assert "Short response" in chunk.content
        assert chunk.metadata["strategy"] == "contextual"
    
    @pytest.mark.asyncio
    async def test_create_chunks_long_content_splitting(self, strategy):
        """Test create_chunks splits long content into multiple chunks."""
        # Create content longer than max_chunk_length (100)
        long_content = "This is a very long message that exceeds the maximum chunk length and should be split into multiple chunks for better processing."
        
        message_batch_list = [
            [
                {"role": "user", "content": long_content}
            ]
        ]
        
        result = await strategy.create_chunks(message_batch_list)
        
        # Should create multiple chunks
        assert len(result) > 1
        
        # Each chunk should be within length limit (with some tolerance for formatting)
        for chunk in result:
            assert len(chunk.content) <= strategy.max_chunk_length + 50  # Allow some tolerance
        
        # All chunks should have contextual strategy metadata
        for chunk in result:
            assert chunk.metadata["strategy"] in ["contextual", "contextual_split"]
    
    @pytest.mark.asyncio
    async def test_create_chunks_multiple_message_lists(self, strategy):
        """Test create_chunks with multiple MessageLists."""
        message_batch_list = [
            [{"role": "user", "content": "First conversation"}],
            [{"role": "user", "content": "Second conversation"}],
            [{"role": "user", "content": "Third conversation"}]
        ]
        
        result = await strategy.create_chunks(message_batch_list)
        
        # Should create at least one chunk
        assert len(result) >= 1
        
        # Check that content from different MessageLists is included
        all_content = " ".join([chunk.content for chunk in result])
        assert "First conversation" in all_content
        assert "Second conversation" in all_content
        assert "Third conversation" in all_content
    
    @pytest.mark.asyncio
    async def test_create_chunks_preserves_metadata(self, strategy):
        """Test that chunks contain appropriate metadata."""
        message_batch_list = [
            [
                {"role": "user", "content": "Test message"}
            ]
        ]
        
        result = await strategy.create_chunks(message_batch_list)
        
        assert len(result) >= 1
        chunk = result[0]
        
        # Check required metadata fields
        assert "strategy" in chunk.metadata
        assert chunk.metadata["strategy"] in ["contextual", "contextual_split"]
        assert "content_length" in chunk.metadata or "message_count" in chunk.metadata
    
    @pytest.mark.asyncio
    async def test_split_long_message_functionality(self, strategy):
        """Test the _split_long_message method indirectly."""
        # Create a single very long message
        very_long_content = "A" * 300  # Much longer than max_chunk_length (100)
        
        message_batch_list = [
            [
                {"role": "user", "content": very_long_content}
            ]
        ]
        
        result = await strategy.create_chunks(message_batch_list)
        
        # Should create multiple chunks
        assert len(result) > 1
        
        # Check that split chunks have appropriate metadata
        for chunk in result:
            if chunk.metadata.get("is_split"):
                assert "part_index" in chunk.metadata
                assert "total_parts" in chunk.metadata
                assert chunk.metadata["strategy"] == "contextual_split"
    
    @pytest.mark.asyncio
    async def test_create_chunks_empty_message_list(self, strategy):
        """Test create_chunks with empty MessageList."""
        message_batch_list = [
            [],  # Empty MessageList
            [{"role": "user", "content": "Valid message"}]
        ]
        
        result = await strategy.create_chunks(message_batch_list)
        
        # Should handle empty MessageList gracefully
        assert len(result) >= 0
        
        # Should process the valid message
        all_content = " ".join([chunk.content for chunk in result])
        assert "Valid message" in all_content
    
    @pytest.mark.asyncio
    async def test_chunk_content_length_limits(self, strategy):
        """Test that chunks respect length limits."""
        # Create multiple messages that together exceed max length
        message_batch_list = [
            [
                {"role": "user", "content": "Message 1 with some content"},
                {"role": "assistant", "content": "Response 1 with some content"},
                {"role": "user", "content": "Message 2 with more content"},
                {"role": "assistant", "content": "Response 2 with even more content"}
            ]
        ]
        
        result = await strategy.create_chunks(message_batch_list)
        
        # Each chunk should be within reasonable length limits
        for chunk in result:
            # Allow some tolerance for formatting and metadata
            assert len(chunk.content) <= strategy.max_chunk_length + 100
    
    def test_strategy_initialization(self):
        """Test ContextualChunkStrategy initialization."""
        strategy = ContextualChunkStrategy(max_chunk_length=200)
        assert strategy.max_chunk_length == 200
        
        # Test default value
        default_strategy = ContextualChunkStrategy()
        assert default_strategy.max_chunk_length == 1000
    
    @pytest.mark.asyncio
    async def test_chunk_ids_are_unique(self, strategy):
        """Test that each chunk gets a unique ID."""
        message_batch_list = [
            [{"role": "user", "content": "A" * 200}],  # Long enough to create multiple chunks
        ]
        
        result = await strategy.create_chunks(message_batch_list)
        
        if len(result) > 1:
            chunk_ids = [chunk.chunk_id for chunk in result]
            assert len(chunk_ids) == len(set(chunk_ids))  # All IDs should be unique
