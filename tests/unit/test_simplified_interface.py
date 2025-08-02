"""Test simplified storage interface solution."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from src.memfuse_core.hierarchy.storage import StoreBackendAdapter
from src.memfuse_core.hierarchy.core import StorageType
from src.memfuse_core.rag.chunk.base import ChunkData


class TestSimplifiedStorageInterface:
    """Test the simplified storage interface solution."""

    @pytest.fixture
    def sample_chunk_data(self):
        """Create sample ChunkData for testing."""
        return ChunkData(
            chunk_id="test-chunk-1",
            content="This is test content",
            metadata={"source": "test", "type": "chunk"}
        )

    @pytest.fixture
    def mock_vector_store(self):
        """Create mock vector store."""
        store = Mock()
        store.add = AsyncMock(return_value=["chunk-id-1"])
        return store

    @pytest.fixture
    def mock_keyword_store(self):
        """Create mock keyword store."""
        store = Mock()
        store.add = AsyncMock(return_value=["chunk-id-1"])
        return store

    @pytest.fixture
    def mock_graph_store(self):
        """Create mock graph store."""
        store = Mock()
        store.add = AsyncMock(return_value=["chunk-id-1"])
        return store

    @pytest.fixture
    def mock_database_store(self):
        """Create mock database store."""
        store = Mock()
        store.add = Mock(return_value="chunk-id-1")
        return store

    @pytest.mark.asyncio
    async def test_vector_store_write(self, mock_vector_store, sample_chunk_data):
        """Test writing to vector store using unified interface."""
        adapter = StoreBackendAdapter(mock_vector_store, StorageType.VECTOR)
        adapter.initialized = True  # Skip initialization
        
        result = await adapter.write(sample_chunk_data)
        
        assert result == "chunk-id-1"
        mock_vector_store.add.assert_called_once()
        # Verify it was called with List[ChunkData]
        call_args = mock_vector_store.add.call_args[0][0]
        assert isinstance(call_args, list)
        assert len(call_args) == 1
        assert isinstance(call_args[0], ChunkData)

    @pytest.mark.asyncio
    async def test_keyword_store_write(self, mock_keyword_store, sample_chunk_data):
        """Test writing to keyword store using unified interface."""
        adapter = StoreBackendAdapter(mock_keyword_store, StorageType.KEYWORD)
        adapter.initialized = True  # Skip initialization
        
        result = await adapter.write(sample_chunk_data)
        
        assert result == "chunk-id-1"
        mock_keyword_store.add.assert_called_once()
        # Verify it was called with List[ChunkData]
        call_args = mock_keyword_store.add.call_args[0][0]
        assert isinstance(call_args, list)
        assert len(call_args) == 1
        assert isinstance(call_args[0], ChunkData)

    @pytest.mark.asyncio
    async def test_graph_store_write(self, mock_graph_store, sample_chunk_data):
        """Test writing to graph store using unified interface."""
        adapter = StoreBackendAdapter(mock_graph_store, StorageType.GRAPH)
        adapter.initialized = True  # Skip initialization
        
        result = await adapter.write(sample_chunk_data)
        
        assert result == "chunk-id-1"
        mock_graph_store.add.assert_called_once()
        # Verify it was called with List[ChunkData]
        call_args = mock_graph_store.add.call_args[0][0]
        assert isinstance(call_args, list)
        assert len(call_args) == 1
        assert isinstance(call_args[0], ChunkData)

    @pytest.mark.asyncio
    async def test_database_write(self, mock_database_store, sample_chunk_data):
        """Test writing to database using unified interface."""
        adapter = StoreBackendAdapter(mock_database_store, StorageType.SQL)
        adapter.initialized = True  # Skip initialization
        
        result = await adapter.write(sample_chunk_data)
        
        assert result == "chunk-id-1"
        mock_database_store.add.assert_called_once()
        # Verify it was called with (table_name, dict_data)
        call_args = mock_database_store.add.call_args[0]
        assert len(call_args) == 2
        assert isinstance(call_args[0], str)  # table name
        assert isinstance(call_args[1], dict)  # data dict

    @pytest.mark.asyncio
    async def test_list_input_handling(self, mock_vector_store):
        """Test handling of list input data."""
        chunk1 = ChunkData(chunk_id="1", content="Content 1")
        chunk2 = ChunkData(chunk_id="2", content="Content 2")
        
        adapter = StoreBackendAdapter(mock_vector_store, StorageType.VECTOR)
        adapter.initialized = True
        
        result = await adapter.write([chunk1, chunk2])
        
        assert result == "chunk-id-1"
        mock_vector_store.add.assert_called_once()
        # Should pass the list as-is to vector store
        call_args = mock_vector_store.add.call_args[0][0]
        assert isinstance(call_args, list)
        assert len(call_args) == 2

    @pytest.mark.asyncio
    async def test_dict_input_conversion(self, mock_vector_store):
        """Test conversion of dict input to ChunkData."""
        dict_data = {
            "content": "Test content",
            "metadata": {"source": "test"}
        }
        
        adapter = StoreBackendAdapter(mock_vector_store, StorageType.VECTOR)
        adapter.initialized = True
        
        result = await adapter.write(dict_data)
        
        assert result == "chunk-id-1"
        mock_vector_store.add.assert_called_once()
        # Should convert dict to ChunkData
        call_args = mock_vector_store.add.call_args[0][0]
        assert isinstance(call_args, list)
        assert len(call_args) == 1
        assert isinstance(call_args[0], ChunkData)
        assert call_args[0].content == "Test content"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
