"""Unit tests for chunk base classes."""

import pytest
from unittest.mock import patch
import uuid

from memfuse_core.rag.chunk.base import ChunkData, ChunkStrategy


class TestChunkData:
    """Test cases for ChunkData class."""
    
    def test_chunk_data_creation_with_defaults(self):
        """Test ChunkData creation with default values."""
        content = "This is test content"
        chunk = ChunkData(content=content)
        
        assert chunk.content == content
        assert chunk.chunk_id is not None
        assert isinstance(chunk.chunk_id, str)
        assert len(chunk.chunk_id) > 0
        assert chunk.metadata == {}
    
    def test_chunk_data_creation_with_custom_values(self):
        """Test ChunkData creation with custom values."""
        content = "Custom test content"
        chunk_id = "custom_chunk_123"
        metadata = {"strategy": "test", "index": 1}
        
        chunk = ChunkData(
            content=content,
            chunk_id=chunk_id,
            metadata=metadata
        )
        
        assert chunk.content == content
        assert chunk.chunk_id == chunk_id
        assert chunk.metadata == metadata
    
    def test_chunk_data_auto_id_generation(self):
        """Test that chunk IDs are automatically generated when not provided."""
        chunk1 = ChunkData("Content 1")
        chunk2 = ChunkData("Content 2")
        
        assert chunk1.chunk_id != chunk2.chunk_id
        assert len(chunk1.chunk_id) > 0
        assert len(chunk2.chunk_id) > 0
    
    @patch('memfuse_core.rag.chunk.base.uuid.uuid4')
    def test_chunk_id_generation_uses_uuid(self, mock_uuid):
        """Test that chunk ID generation uses UUID."""
        mock_uuid.return_value.hex = "test_uuid_hex"
        
        chunk = ChunkData("Test content")
        
        mock_uuid.assert_called_once()
        assert "test_uuid_hex" in chunk.chunk_id
    
    def test_chunk_data_string_representation(self):
        """Test string representation of ChunkData."""
        chunk = ChunkData(
            content="Test content",
            chunk_id="test_123",
            metadata={"test": True}
        )
        
        str_repr = str(chunk)
        assert "test_123" in str_repr
        assert "Test content" in str_repr
    
    def test_chunk_data_equality(self):
        """Test ChunkData equality comparison."""
        chunk1 = ChunkData("Same content", "same_id", {"key": "value"})
        chunk2 = ChunkData("Same content", "same_id", {"key": "value"})
        chunk3 = ChunkData("Different content", "same_id", {"key": "value"})
        
        assert chunk1 == chunk2
        assert chunk1 != chunk3
    
    def test_chunk_data_metadata_immutability(self):
        """Test that metadata can be safely modified."""
        original_metadata = {"key": "value"}
        chunk = ChunkData("Content", metadata=original_metadata)
        
        # Modify the chunk's metadata
        chunk.metadata["new_key"] = "new_value"
        
        # Original metadata should not be affected
        assert "new_key" not in original_metadata
        assert chunk.metadata["new_key"] == "new_value"


class TestChunkStrategy:
    """Test cases for ChunkStrategy abstract base class."""
    
    def test_chunk_strategy_is_abstract(self):
        """Test that ChunkStrategy cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ChunkStrategy()
    
    def test_chunk_strategy_subclass_must_implement_create_chunks(self):
        """Test that subclasses must implement create_chunks method."""
        
        class IncompleteStrategy(ChunkStrategy):
            pass
        
        with pytest.raises(TypeError):
            IncompleteStrategy()
    
    def test_chunk_strategy_subclass_with_implementation(self):
        """Test that proper subclass can be instantiated."""
        
        class ValidStrategy(ChunkStrategy):
            async def create_chunks(self, message_batch_list):
                return []
        
        strategy = ValidStrategy()
        assert isinstance(strategy, ChunkStrategy)
    
    @pytest.mark.asyncio
    async def test_chunk_strategy_create_chunks_signature(self):
        """Test that create_chunks has correct signature."""
        
        class TestStrategy(ChunkStrategy):
            async def create_chunks(self, message_batch_list):
                return [ChunkData("test")]
        
        strategy = TestStrategy()
        result = await strategy.create_chunks([])
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], ChunkData)
    
    def test_chunk_strategy_extract_message_content_method(self):
        """Test the _extract_message_content helper method."""
        
        class TestStrategy(ChunkStrategy):
            async def create_chunks(self, message_batch_list):
                return []
        
        strategy = TestStrategy()
        
        # Test with string content
        message1 = {"content": "Simple string content"}
        assert strategy._extract_message_content(message1) == "Simple string content"
        
        # Test with dict content
        message2 = {"content": {"text": "Dict content"}}
        assert strategy._extract_message_content(message2) == "Dict content"
        
        # Test with missing content
        message3 = {"role": "user"}
        assert strategy._extract_message_content(message3) == ""
        
        # Test with None content
        message4 = {"content": None}
        assert strategy._extract_message_content(message4) == ""
