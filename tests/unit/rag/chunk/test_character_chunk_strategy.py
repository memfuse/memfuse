"""Unit tests for CharacterChunkStrategy."""

import pytest
from memfuse_core.rag.chunk import CharacterChunkStrategy, ChunkData


class TestCharacterChunkStrategy:
    """Test cases for CharacterChunkStrategy."""
    
    @pytest.fixture
    def strategy(self):
        """Create a CharacterChunkStrategy instance for testing."""
        return CharacterChunkStrategy(max_chunk_length=50, overlap_length=10)
    
    @pytest.fixture
    def no_overlap_strategy(self):
        """Create a CharacterChunkStrategy without overlap."""
        return CharacterChunkStrategy(max_chunk_length=50, overlap_length=0)
    
    @pytest.mark.asyncio
    async def test_create_chunks_empty_input(self, strategy):
        """Test create_chunks with empty input."""
        result = await strategy.create_chunks([])
        assert result == []
    
    @pytest.mark.asyncio
    async def test_create_chunks_short_content(self, strategy):
        """Test create_chunks with content shorter than max length."""
        message_batch_list = [
            [
                {"role": "user", "content": "Short"},
                {"role": "assistant", "content": "Reply"}
            ]
        ]
        
        result = await strategy.create_chunks(message_batch_list)
        
        assert len(result) == 1
        chunk = result[0]
        assert "Short" in chunk.content
        assert "Reply" in chunk.content
        assert chunk.metadata["strategy"] == "character"
        assert chunk.metadata["chunk_index"] == 0
        assert not chunk.metadata["has_overlap"]
    
    @pytest.mark.asyncio
    async def test_create_chunks_long_content_with_splitting(self, strategy):
        """Test create_chunks splits long content into multiple chunks."""
        # Create content longer than max_chunk_length (50)
        long_content = "This is a very long message that definitely exceeds the maximum chunk length and should be split into multiple chunks."
        
        message_batch_list = [
            [
                {"role": "user", "content": long_content}
            ]
        ]
        
        result = await strategy.create_chunks(message_batch_list)
        
        # Should create multiple chunks
        assert len(result) > 1
        
        # Each chunk should be within length limit
        for chunk in result:
            assert len(chunk.content) <= strategy.max_chunk_length
        
        # All chunks should have character strategy metadata
        for chunk in result:
            assert chunk.metadata["strategy"] == "character"
        
        # Check overlap metadata
        assert not result[0].metadata["has_overlap"]  # First chunk has no overlap
        for chunk in result[1:]:
            assert chunk.metadata["has_overlap"]  # Subsequent chunks have overlap
    
    @pytest.mark.asyncio
    async def test_create_chunks_with_overlap(self, strategy):
        """Test that overlap is correctly implemented."""
        # Create content that will be split
        content = "A" * 100  # 100 characters, will be split into chunks of 50
        
        message_batch_list = [
            [
                {"role": "user", "content": content}
            ]
        ]
        
        result = await strategy.create_chunks(message_batch_list)
        
        assert len(result) >= 2
        
        # Check overlap between consecutive chunks
        for i in range(1, len(result)):
            current_chunk = result[i]
            assert current_chunk.metadata["has_overlap"]
            assert current_chunk.metadata["overlap_length"] == strategy.overlap_length
    
    @pytest.mark.asyncio
    async def test_create_chunks_without_overlap(self, no_overlap_strategy):
        """Test create_chunks without overlap."""
        content = "A" * 100  # 100 characters
        
        message_batch_list = [
            [
                {"role": "user", "content": content}
            ]
        ]
        
        result = await no_overlap_strategy.create_chunks(message_batch_list)
        
        assert len(result) == 2  # 100 chars / 50 chars per chunk
        
        # No chunks should have overlap
        for chunk in result:
            assert not chunk.metadata["has_overlap"]
            assert chunk.metadata["overlap_length"] == 0
    
    @pytest.mark.asyncio
    async def test_create_chunks_multiple_message_lists(self, strategy):
        """Test create_chunks with multiple MessageLists."""
        message_batch_list = [
            [{"role": "user", "content": "First conversation message"}],
            [{"role": "user", "content": "Second conversation message"}],
            [{"role": "user", "content": "Third conversation message"}]
        ]
        
        result = await strategy.create_chunks(message_batch_list)
        
        # Should create at least one chunk
        assert len(result) >= 1
        
        # Check that content from all MessageLists is included
        all_content = "".join([chunk.content for chunk in result])
        assert "First conversation" in all_content
        assert "Second conversation" in all_content
        assert "Third conversation" in all_content
    
    @pytest.mark.asyncio
    async def test_create_chunks_metadata_completeness(self, strategy):
        """Test that chunks contain all required metadata."""
        message_batch_list = [
            [
                {"role": "user", "content": "Test message for metadata"}
            ]
        ]
        
        result = await strategy.create_chunks(message_batch_list)
        
        assert len(result) >= 1
        
        for i, chunk in enumerate(result):
            metadata = chunk.metadata
            
            # Check required metadata fields
            assert metadata["strategy"] == "character"
            assert metadata["chunk_index"] == i
            assert "start_position" in metadata
            assert "end_position" in metadata
            assert "content_length" in metadata
            assert "has_overlap" in metadata
            assert "overlap_length" in metadata
            assert metadata["source"] == "character_split"
            assert "total_messages" in metadata
            assert "message_metadata" in metadata
    
    @pytest.mark.asyncio
    async def test_word_boundary_breaking(self, strategy):
        """Test that the strategy tries to break at word boundaries."""
        # Create content with clear word boundaries
        content = "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10"
        
        message_batch_list = [
            [
                {"role": "user", "content": content}
            ]
        ]
        
        result = await strategy.create_chunks(message_batch_list)
        
        if len(result) > 1:
            # Check that chunks don't break in the middle of words (when possible)
            for chunk in result[:-1]:  # Exclude last chunk
                # Should not end with a partial word (unless forced to)
                if not chunk.content.endswith(' '):
                    # If it doesn't end with space, it should be at max length
                    assert len(chunk.content) == strategy.max_chunk_length
    
    def test_find_optimal_break_point_method(self, strategy):
        """Test the _find_optimal_break_point helper method."""
        # Test sentence boundary
        text = "First sentence. Second sentence continues here."
        break_point = strategy._find_optimal_break_point(text, 20)
        assert break_point == 16  # After "First sentence."
        
        # Test word boundary
        text = "word1 word2 word3 word4"
        break_point = strategy._find_optimal_break_point(text, 15)
        assert text[break_point-1] == ' ' or break_point == 15
        
        # Test max length fallback
        text = "verylongwordwithoutspaces"
        break_point = strategy._find_optimal_break_point(text, 10)
        assert break_point == 10
    
    def test_strategy_initialization(self):
        """Test CharacterChunkStrategy initialization."""
        strategy = CharacterChunkStrategy(max_chunk_length=200, overlap_length=50)
        assert strategy.max_chunk_length == 200
        assert strategy.overlap_length == 50
        
        # Test default values
        default_strategy = CharacterChunkStrategy()
        assert default_strategy.max_chunk_length == 1000
        assert default_strategy.overlap_length == 100
    
    @pytest.mark.asyncio
    async def test_chunk_ids_are_unique(self, strategy):
        """Test that each chunk gets a unique ID."""
        # Create content long enough to generate multiple chunks
        long_content = "A" * 200
        message_batch_list = [
            [{"role": "user", "content": long_content}]
        ]
        
        result = await strategy.create_chunks(message_batch_list)
        
        if len(result) > 1:
            chunk_ids = [chunk.chunk_id for chunk in result]
            assert len(chunk_ids) == len(set(chunk_ids))  # All IDs should be unique
    
    @pytest.mark.asyncio
    async def test_empty_message_list_handling(self, strategy):
        """Test handling of empty MessageLists."""
        message_batch_list = [
            [],  # Empty MessageList
            [{"role": "user", "content": "Valid content"}]
        ]
        
        result = await strategy.create_chunks(message_batch_list)
        
        # Should handle empty MessageList gracefully
        assert len(result) >= 0
        
        # Should process the valid content
        all_content = "".join([chunk.content for chunk in result])
        assert "Valid content" in all_content
