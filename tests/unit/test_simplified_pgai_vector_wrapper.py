"""Unit tests for the simplified PgaiVectorWrapper implementation."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any, Optional
import numpy as np

from src.memfuse_core.store.pgai_vector_wrapper import PgaiVectorWrapper
from src.memfuse_core.store.pgai_store import PgaiStore
from src.memfuse_core.rag.chunk.base import ChunkData
from src.memfuse_core.rag.encode.base import EncoderBase
from src.memfuse_core.models import Query


class MockEncoder(EncoderBase):
    """Mock encoder for testing."""

    def __init__(self):
        self.model_name = "mock-encoder"

    async def encode_text(self, text: str) -> np.ndarray:
        """Mock encoding that returns dummy vector."""
        return np.array([0.1, 0.2, 0.3])

    async def encode_texts(self, texts: List[str]) -> List[np.ndarray]:
        """Mock encoding that returns dummy vectors."""
        return [np.array([0.1, 0.2, 0.3]) for _ in texts]


class TestSimplifiedPgaiVectorWrapper:
    """Test cases for the simplified PgaiVectorWrapper."""
    
    @pytest.fixture
    def mock_pgai_store(self):
        """Create a mock PgaiStore."""
        store = AsyncMock(spec=PgaiStore)
        store.initialized = True
        return store
    
    @pytest.fixture
    def mock_encoder(self):
        """Create a mock encoder."""
        return MockEncoder()
    
    @pytest.fixture
    def wrapper(self, mock_pgai_store, mock_encoder):
        """Create a PgaiVectorWrapper instance with mocks."""
        return PgaiVectorWrapper(mock_pgai_store, mock_encoder, cache_size=100)
    
    @pytest.mark.asyncio
    async def test_initialization(self, wrapper, mock_pgai_store):
        """Test wrapper initialization."""
        # Test first initialization
        result = await wrapper.initialize()
        assert result is True
        assert wrapper.initialized is True
        
        # Test already initialized
        result = await wrapper.initialize()
        assert result is True
        
        # Test with uninitialized pgai store
        mock_pgai_store.initialized = False
        wrapper.initialized = False
        result = await wrapper.initialize()
        assert result is True
        mock_pgai_store.initialize.assert_called_once()
    
    def test_getattr_forwarding(self, wrapper, mock_pgai_store):
        """Test dynamic method forwarding via __getattr__."""
        # Test forwarding to existing method
        mock_pgai_store.some_method = MagicMock(return_value="test_result")
        result = wrapper.some_method()
        assert result == "test_result"
        mock_pgai_store.some_method.assert_called_once()
        
        # Test forwarding non-existent method
        with pytest.raises(AttributeError):
            wrapper.non_existent_method()
    
    @pytest.mark.asyncio
    async def test_add_items_data_transformation(self, wrapper, mock_pgai_store):
        """Test add_items method with data transformation."""
        # Setup mock
        mock_pgai_store.add.return_value = ["id1", "id2"]
        
        # Test data
        items = [
            {"id": "id1", "content": "content1", "metadata": {"key": "value1"}},
            {"id": "id2", "content": "content2", "metadata": {"key": "value2"}}
        ]
        
        # Execute
        result = await wrapper.add_items(items)
        
        # Verify
        assert result == ["id1", "id2"]
        mock_pgai_store.add.assert_called_once()
        
        # Check that items were converted to ChunkData
        call_args = mock_pgai_store.add.call_args[0][0]
        assert len(call_args) == 2
        assert all(isinstance(chunk, ChunkData) for chunk in call_args)
        assert call_args[0].chunk_id == "id1"
        assert call_args[0].content == "content1"
        assert call_args[0].metadata == {"key": "value1"}
    
    @pytest.mark.asyncio
    async def test_search_with_filtering(self, wrapper, mock_pgai_store):
        """Test search method with filtering and data transformation."""
        # Setup mock chunks
        mock_chunks = [
            ChunkData(chunk_id="id1", content="content1", metadata={"type": "doc"}),
            ChunkData(chunk_id="id2", content="content2", metadata={"type": "note"}),
            ChunkData(chunk_id="id3", content="content3", metadata={"type": "doc"})
        ]
        mock_pgai_store.query.return_value = mock_chunks
        
        # Test without filters
        result = await wrapper.search("test query", top_k=3)
        assert len(result) == 3
        assert result[0]["id"] == "id1"
        assert result[0]["content"] == "content1"
        assert result[0]["score"] == 1.0
        
        # Test with filters
        result = await wrapper.search("test query", top_k=3, filters={"type": "doc"})
        assert len(result) == 2
        assert all(item["metadata"]["type"] == "doc" for item in result)
        
        # Verify Query object was created
        mock_pgai_store.query.assert_called()
        call_args = mock_pgai_store.query.call_args[0]
        assert isinstance(call_args[0], Query)
        assert call_args[0].text == "test query"
    
    @pytest.mark.asyncio
    async def test_get_item(self, wrapper, mock_pgai_store):
        """Test get_item method."""
        # Test successful retrieval
        mock_chunk = ChunkData(chunk_id="id1", content="content1", metadata={"key": "value"})
        mock_pgai_store.read.return_value = [mock_chunk]
        
        result = await wrapper.get_item("id1")
        assert result is not None
        assert result["id"] == "id1"
        assert result["content"] == "content1"
        assert result["metadata"] == {"key": "value"}
        
        # Test item not found
        mock_pgai_store.read.return_value = [None]
        result = await wrapper.get_item("nonexistent")
        assert result is None
        
        # Test empty result
        mock_pgai_store.read.return_value = []
        result = await wrapper.get_item("empty")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_update_item(self, wrapper, mock_pgai_store):
        """Test update_item method with data transformation."""
        mock_pgai_store.update.return_value = True
        
        item_data = {"content": "updated content", "metadata": {"updated": True}}
        result = await wrapper.update_item("id1", item_data)
        
        assert result is True
        mock_pgai_store.update.assert_called_once()
        
        # Check ChunkData conversion
        call_args = mock_pgai_store.update.call_args[0]
        assert call_args[0] == "id1"
        assert isinstance(call_args[1], ChunkData)
        assert call_args[1].content == "updated content"
        assert call_args[1].metadata == {"updated": True}
    
    @pytest.mark.asyncio
    async def test_get_stats_transformation(self, wrapper, mock_pgai_store):
        """Test get_stats method with format transformation."""
        mock_stats = {
            "total_chunks": 100,
            "by_session": {"session1": 50},
            "by_user": {"user1": 30},
            "by_strategy": {"strategy1": 20},
            "storage_size": "10MB"
        }
        mock_pgai_store.get_chunks_stats.return_value = mock_stats
        
        result = await wrapper.get_stats()
        
        # Verify transformation to VectorStore format
        assert result["total_items"] == 100
        assert result["by_session"] == {"session1": 50}
        assert result["by_user"] == {"user1": 30}
        assert result["by_strategy"] == {"strategy1": 20}
        assert result["storage_size"] == "10MB"
        assert result["backend"] == "pgai"
    
    @pytest.mark.asyncio
    async def test_search_by_embedding(self, wrapper, mock_pgai_store):
        """Test search_by_embedding method."""
        mock_results = [
            {"id": "id1", "content": "content1", "metadata": {"key": "value"}, "distance": 0.1},
            {"id": "id2", "content": "content2", "metadata": {}, "distance": 0.3}
        ]
        mock_pgai_store.search_similar.return_value = mock_results
        
        embedding = [0.1, 0.2, 0.3]
        result = await wrapper.search_by_embedding(embedding, top_k=2)
        
        assert len(result) == 2
        assert result[0]["id"] == "id1"
        assert result[0]["score"] == 0.9  # 1.0 - 0.1
        assert result[1]["score"] == 0.7  # 1.0 - 0.3
        
        mock_pgai_store.search_similar.assert_called_once_with(embedding, 2)
    
    @pytest.mark.asyncio
    async def test_get_items_by_session(self, wrapper, mock_pgai_store):
        """Test get_items_by_session method."""
        mock_chunks = [
            ChunkData(chunk_id="id1", content="content1", metadata={"session": "s1"}),
            ChunkData(chunk_id="id2", content="content2", metadata={"session": "s1"})
        ]
        mock_pgai_store.get_chunks_by_session.return_value = mock_chunks
        
        result = await wrapper.get_items_by_session("s1")
        
        assert len(result) == 2
        assert result[0]["id"] == "id1"
        assert result[1]["id"] == "id2"
        mock_pgai_store.get_chunks_by_session.assert_called_once_with("s1")
    
    @pytest.mark.asyncio
    async def test_get_items_by_user(self, wrapper, mock_pgai_store):
        """Test get_items_by_user method."""
        mock_chunks = [
            ChunkData(chunk_id="id1", content="content1", metadata={"user": "u1"})
        ]
        mock_pgai_store.get_chunks_by_user.return_value = mock_chunks
        
        result = await wrapper.get_items_by_user("u1")
        
        assert len(result) == 1
        assert result[0]["id"] == "id1"
        mock_pgai_store.get_chunks_by_user.assert_called_once_with("u1")
    
    @pytest.mark.asyncio
    async def test_close_and_cleanup(self, wrapper, mock_pgai_store):
        """Test close method and cleanup."""
        wrapper.initialized = True
        
        await wrapper.close()
        
        assert wrapper.initialized is False
        mock_pgai_store.close.assert_called_once()
        
        # Test __del__ method
        wrapper.initialized = True
        with patch('src.memfuse_core.store.pgai_vector_wrapper.logger') as mock_logger:
            wrapper.__del__()
            mock_logger.debug.assert_called_once()
    
    def test_compatibility_attributes(self, wrapper, mock_encoder):
        """Test that compatibility attributes are preserved."""
        assert wrapper.encoder == mock_encoder
        assert wrapper.cache_size == 100
        assert hasattr(wrapper, 'pgai_store')
        assert hasattr(wrapper, 'initialized')


if __name__ == "__main__":
    pytest.main([__file__])
