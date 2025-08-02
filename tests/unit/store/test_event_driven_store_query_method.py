"""
Test EventDrivenPgaiStore query method and error handling.

This test verifies that the EventDrivenPgaiStore correctly handles:
1. Query method implementation
2. Error handling when initialization fails
3. Proper delegation to core store
4. Graceful degradation when pgai dependencies are missing
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any, Optional

from memfuse_core.store.pgai_store.event_driven_store import EventDrivenPgaiStore
from memfuse_core.interfaces.chunk_store import ChunkData


class TestEventDrivenPgaiStoreQueryMethod:
    """Test the query method and error handling in EventDrivenPgaiStore."""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing."""
        return {
            "pgai": {
                "host": "localhost",
                "port": 5432,
                "database": "test_db",
                "user": "test_user",
                "password": "test_pass"
            }
        }

    @pytest.fixture
    def event_driven_store(self, mock_config):
        """Create EventDrivenPgaiStore instance for testing."""
        return EventDrivenPgaiStore(config=mock_config, table_name="test_table")

    @pytest.mark.asyncio
    async def test_query_method_exists(self, event_driven_store):
        """Test that query method exists and is callable."""
        assert hasattr(event_driven_store, 'query')
        assert callable(getattr(event_driven_store, 'query'))

    @pytest.mark.asyncio
    async def test_query_with_uninitialized_store(self, event_driven_store):
        """Test query method when store is not initialized."""
        # Ensure store is not initialized
        event_driven_store.initialized = False
        event_driven_store.core_store = None
        
        # Mock initialize to fail
        with patch.object(event_driven_store, 'initialize', return_value=False):
            result = await event_driven_store.query("test query", top_k=5)
            
        assert result == []

    @pytest.mark.asyncio
    async def test_query_with_none_core_store(self, event_driven_store):
        """Test query method when core_store is None."""
        event_driven_store.initialized = True
        event_driven_store.core_store = None
        
        result = await event_driven_store.query("test query", top_k=5)
        assert result == []

    @pytest.mark.asyncio
    async def test_query_delegates_to_core_store(self, event_driven_store):
        """Test that query method properly delegates to core store."""
        # Mock core store
        mock_core_store = AsyncMock()
        mock_chunks = [
            ChunkData(
                chunk_id="test_1",
                content="Test content 1",
                metadata={"source": "test"}
            ),
            ChunkData(
                chunk_id="test_2", 
                content="Test content 2",
                metadata={"source": "test"}
            )
        ]
        mock_core_store.query.return_value = mock_chunks
        
        event_driven_store.initialized = True
        event_driven_store.core_store = mock_core_store
        
        result = await event_driven_store.query("test query", top_k=5)
        
        # Verify delegation
        mock_core_store.query.assert_called_once_with("test query", 5)
        assert result == mock_chunks

    @pytest.mark.asyncio
    async def test_ensure_initialized_helper_method(self, event_driven_store):
        """Test the _ensure_initialized helper method."""
        # Test when not initialized and initialization fails
        event_driven_store.initialized = False
        with patch.object(event_driven_store, 'initialize', return_value=False):
            result = await event_driven_store._ensure_initialized("test_operation")
            assert result is False

        # Test when not initialized but initialization succeeds
        event_driven_store.initialized = False
        event_driven_store.core_store = Mock()
        with patch.object(event_driven_store, 'initialize', return_value=True):
            event_driven_store.initialized = True  # Simulate successful init
            result = await event_driven_store._ensure_initialized("test_operation")
            assert result is True

        # Test when already initialized but core_store is None
        event_driven_store.initialized = True
        event_driven_store.core_store = None
        result = await event_driven_store._ensure_initialized("test_operation")
        assert result is False

        # Test when already initialized and core_store exists
        event_driven_store.initialized = True
        event_driven_store.core_store = Mock()
        result = await event_driven_store._ensure_initialized("test_operation")
        assert result is True

    @pytest.mark.asyncio
    async def test_add_method_error_handling(self, event_driven_store):
        """Test add method error handling."""
        # Test with failed initialization
        event_driven_store.initialized = False
        with patch.object(event_driven_store, 'initialize', return_value=False):
            result = await event_driven_store.add([])
            assert result == []

    @pytest.mark.asyncio
    async def test_get_method_error_handling(self, event_driven_store):
        """Test get method error handling."""
        # Test with failed initialization
        event_driven_store.initialized = False
        with patch.object(event_driven_store, 'initialize', return_value=False):
            result = await event_driven_store.get("test_id")
            assert result is None

    @pytest.mark.asyncio
    async def test_delete_method_error_handling(self, event_driven_store):
        """Test delete method error handling."""
        # Test with failed initialization
        event_driven_store.initialized = False
        with patch.object(event_driven_store, 'initialize', return_value=False):
            result = await event_driven_store.delete(["test_id"])
            assert result is False

    @pytest.mark.asyncio
    async def test_update_method_error_handling(self, event_driven_store):
        """Test update method error handling."""
        # Test with failed initialization
        event_driven_store.initialized = False
        with patch.object(event_driven_store, 'initialize', return_value=False):
            result = await event_driven_store.update([])
            assert result is False

    @pytest.mark.asyncio
    async def test_list_chunks_method_error_handling(self, event_driven_store):
        """Test list_chunks method error handling."""
        # Test with failed initialization
        event_driven_store.initialized = False
        with patch.object(event_driven_store, 'initialize', return_value=False):
            result = await event_driven_store.list_chunks()
            assert result == []

    @pytest.mark.asyncio
    async def test_count_method_error_handling(self, event_driven_store):
        """Test count method error handling."""
        # Test with failed initialization
        event_driven_store.initialized = False
        with patch.object(event_driven_store, 'initialize', return_value=False):
            result = await event_driven_store.count()
            assert result == 0

    @pytest.mark.asyncio
    async def test_no_search_method(self, event_driven_store):
        """Test that search method does not exist (unified to use query)."""
        assert not hasattr(event_driven_store, 'search')

    @pytest.mark.asyncio
    async def test_query_method_signature(self, event_driven_store):
        """Test query method has correct signature."""
        import inspect
        sig = inspect.signature(event_driven_store.query)
        params = list(sig.parameters.keys())
        
        # Should have 'query' and 'top_k' parameters
        assert 'query' in params
        assert 'top_k' in params
        
        # Check default value for top_k
        assert sig.parameters['top_k'].default == 5

    def test_properties_error_handling(self, event_driven_store):
        """Test property access when core_store is None."""
        event_driven_store.core_store = None
        
        assert event_driven_store.pool is None
        assert event_driven_store.encoder is None
        
        # Test encoder setter
        event_driven_store.encoder = "test_encoder"  # Should not raise error


if __name__ == "__main__":
    pytest.main([__file__])
