"""
Integration test for query method fixes.

This test verifies the end-to-end functionality after fixing:
1. EventDrivenPgaiStore query method implementation
2. Unified usage of query method instead of search
3. Proper error handling when pgai dependencies are missing
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
import tempfile
import os

from memfuse_core.store.pgai_store.event_driven_store import EventDrivenPgaiStore
from memfuse_core.rag.retrieve.hybrid import HybridRetrieval
from memfuse_core.interfaces.chunk_store import ChunkData


class TestQueryMethodIntegration:
    """Integration tests for query method functionality."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def mock_config(self, temp_data_dir):
        """Mock configuration for integration testing."""
        return {
            "data_dir": temp_data_dir,
            "pgai": {
                "host": "localhost",
                "port": 5432,
                "database": "test_db",
                "user": "test_user",
                "password": "test_pass"
            },
            "store": {
                "vector": {
                    "connection_pool_size": 5,
                    "timeout": 30.0,
                    "batch_size": 100
                }
            }
        }

    @pytest.mark.asyncio
    async def test_event_driven_store_query_method_exists(self, mock_config):
        """Test that EventDrivenPgaiStore has query method."""
        store = EventDrivenPgaiStore(config=mock_config, table_name="test_table")
        
        # Verify query method exists
        assert hasattr(store, 'query')
        assert callable(getattr(store, 'query'))
        
        # Verify search method does not exist (unified to query)
        assert not hasattr(store, 'search')

    @pytest.mark.asyncio
    async def test_event_driven_store_graceful_degradation(self, mock_config):
        """Test that EventDrivenPgaiStore gracefully handles missing pgai dependencies."""
        store = EventDrivenPgaiStore(config=mock_config, table_name="test_table")
        
        # When pgai dependencies are missing, query should return empty results
        result = await store.query("test query", top_k=5)
        assert result == []
        
        # Other methods should also handle gracefully
        add_result = await store.add([])
        assert add_result == []
        
        get_result = await store.get("test_id")
        assert get_result is None
        
        delete_result = await store.delete(["test_id"])
        assert delete_result is False
        
        update_result = await store.update([])
        assert update_result is False
        
        list_result = await store.list_chunks()
        assert list_result == []
        
        count_result = await store.count()
        assert count_result == 0

    @pytest.mark.asyncio
    async def test_hybrid_retrieval_with_event_driven_store(self, mock_config, temp_data_dir):
        """Test HybridRetrieval works with EventDrivenPgaiStore query method."""
        # Create mock stores
        vector_store = EventDrivenPgaiStore(config=mock_config, table_name="test_vector")
        
        # Mock keyword store
        keyword_store = Mock()
        keyword_store.search = AsyncMock(return_value=[])
        
        # Create HybridRetrieval instance
        hybrid = HybridRetrieval(
            vector_store=vector_store,
            keyword_store=keyword_store,
            config=mock_config
        )
        
        # Test that it can call query method without errors
        with patch.object(vector_store, 'query', return_value=[]):
            results = await hybrid.retrieve("test query", top_k=5)
            assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_query_method_with_mock_core_store(self, mock_config):
        """Test query method with mocked core store."""
        store = EventDrivenPgaiStore(config=mock_config, table_name="test_table")
        
        # Mock successful initialization
        mock_core_store = AsyncMock()
        mock_chunks = [
            ChunkData(
                chunk_id="test_1",
                content="Test content about Mars exploration",
                metadata={"source": "test", "timestamp": "2024-01-01"}
            ),
            ChunkData(
                chunk_id="test_2",
                content="Test content about space challenges",
                metadata={"source": "test", "timestamp": "2024-01-02"}
            )
        ]
        mock_core_store.query.return_value = mock_chunks
        
        # Set up store state
        store.initialized = True
        store.core_store = mock_core_store
        
        # Test query
        result = await store.query("Mars exploration", top_k=5)
        
        # Verify results
        assert len(result) == 2
        assert result[0].chunk_id == "test_1"
        assert result[1].chunk_id == "test_2"
        
        # Verify core store was called correctly
        mock_core_store.query.assert_called_once_with("Mars exploration", 5)

    @pytest.mark.asyncio
    async def test_error_handling_consistency(self, mock_config):
        """Test that all methods handle errors consistently."""
        store = EventDrivenPgaiStore(config=mock_config, table_name="test_table")
        
        # Ensure store is not initialized
        store.initialized = False
        store.core_store = None
        
        # Mock initialize to fail
        with patch.object(store, 'initialize', return_value=False):
            # Test all methods return appropriate default values
            query_result = await store.query("test", top_k=5)
            assert query_result == []
            
            add_result = await store.add([])
            assert add_result == []
            
            get_result = await store.get("test_id")
            assert get_result is None
            
            delete_result = await store.delete(["test_id"])
            assert delete_result is False
            
            update_result = await store.update([])
            assert update_result is False
            
            list_result = await store.list_chunks(limit=10, offset=0)
            assert list_result == []
            
            count_result = await store.count()
            assert count_result == 0

    @pytest.mark.asyncio
    async def test_ensure_initialized_helper_integration(self, mock_config):
        """Test _ensure_initialized helper method in integration context."""
        store = EventDrivenPgaiStore(config=mock_config, table_name="test_table")
        
        # Test various scenarios
        scenarios = [
            ("query", []),
            ("add", []),
            ("get", None),
            ("delete", False),
            ("update", False),
            ("list_chunks", []),
            ("count", 0)
        ]
        
        for operation, expected_default in scenarios:
            # Reset store state
            store.initialized = False
            store.core_store = None
            
            # Mock initialize to fail
            with patch.object(store, 'initialize', return_value=False):
                result = await store._ensure_initialized(operation)
                assert result is False

    def test_interface_compliance(self, mock_config):
        """Test that EventDrivenPgaiStore complies with expected interface."""
        store = EventDrivenPgaiStore(config=mock_config, table_name="test_table")
        
        # Check required methods exist
        required_methods = ['query', 'add', 'get', 'delete', 'update', 'list_chunks', 'count']
        for method_name in required_methods:
            assert hasattr(store, method_name), f"Missing method: {method_name}"
            assert callable(getattr(store, method_name)), f"Method not callable: {method_name}"
        
        # Check that search method does not exist (unified to query)
        assert not hasattr(store, 'search'), "search method should not exist, use query instead"
        
        # Check properties
        assert hasattr(store, 'pool')
        assert hasattr(store, 'encoder')

    @pytest.mark.asyncio
    async def test_performance_with_error_handling(self, mock_config):
        """Test that error handling doesn't significantly impact performance."""
        store = EventDrivenPgaiStore(config=mock_config, table_name="test_table")
        
        # Set up successful state
        mock_core_store = AsyncMock()
        mock_core_store.query.return_value = []
        store.initialized = True
        store.core_store = mock_core_store
        
        # Time multiple calls
        import time
        start_time = time.time()
        
        for _ in range(10):
            await store.query("test query", top_k=5)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should complete quickly (less than 1 second for 10 calls)
        assert total_time < 1.0, f"Error handling overhead too high: {total_time}s"


if __name__ == "__main__":
    pytest.main([__file__])
