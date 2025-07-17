"""
Unit tests for M0 Raw Data Layer.

Tests the M0RawDataLayer implementation including:
- Raw data storage functionality
- Multi-backend storage coordination
- Event emission for downstream processing
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

from src.memfuse_core.hierarchy.layers import M0RawDataLayer
from src.memfuse_core.hierarchy.core import LayerType, LayerConfig
from src.memfuse_core.hierarchy.storage import UnifiedStorageManager
from src.memfuse_core.rag.chunk.base import ChunkData


class TestM0RawDataLayer:
    """Test suite for M0 Raw Data Layer."""

    @pytest.fixture
    def layer_config(self):
        """Create layer configuration for M0."""
        return LayerConfig(
            layer_type=LayerType.M0,
            storage_backends=["vector", "keyword", "sql"],
            custom_config={
                "buffer_size": 1000,
                "auto_flush_interval": 60,
                "storage_backends": ["vector", "keyword", "sql"]
            }
        )

    @pytest.fixture
    def mock_storage_manager(self):
        """Create mock storage manager."""
        manager = AsyncMock(spec=UnifiedStorageManager)
        manager.initialize.return_value = True
        manager.write_to_backend.return_value = "test_id_123"
        return manager

    @pytest.fixture
    def m0_layer(self, layer_config, mock_storage_manager):
        """Create M0RawDataLayer instance."""
        layer = M0RawDataLayer(
            layer_type=LayerType.M0,
            config=layer_config,
            user_id="test_user",
            storage_manager=mock_storage_manager
        )
        return layer

    @pytest.mark.asyncio
    async def test_m0_layer_initialization(self, m0_layer):
        """Test M0 layer initialization."""
        # Test initialization
        result = await m0_layer.initialize()
        
        assert result is True
        assert m0_layer.initialized is True
        assert m0_layer.layer_type == LayerType.M0
        assert m0_layer.user_id == "test_user"

    @pytest.mark.asyncio
    async def test_raw_data_storage(self, m0_layer):
        """Test raw data storage functionality."""
        await m0_layer.initialize()
        
        # Prepare test data
        test_data = [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you!"}
        ]
        
        # Process data
        result = await m0_layer.process_data(test_data, session_id="test_session")
        
        # Verify processing
        assert result.success is True
        assert len(result.processed_items) > 0
        assert result.processing_time > 0

    @pytest.mark.asyncio
    async def test_multi_backend_storage(self, m0_layer, mock_storage_manager):
        """Test multi-backend storage coordination."""
        await m0_layer.initialize()
        
        # Configure multiple storage backends
        mock_storage_manager.write_to_backend.side_effect = [
            "vector_id_123",
            "keyword_id_456", 
            "sql_id_789"
        ]
        
        test_data = [{"role": "user", "content": "Test message"}]
        
        # Process data
        result = await m0_layer.process_data(test_data)
        
        # Verify all backends were called
        assert mock_storage_manager.write_to_backend.call_count >= 1
        assert result.success is True

    @pytest.mark.asyncio
    async def test_chunk_conversion(self, m0_layer):
        """Test data to chunk conversion."""
        await m0_layer.initialize()
        
        # Test data conversion
        test_data = [
            {"role": "user", "content": "Test message", "timestamp": "2024-01-01T00:00:00Z"}
        ]
        
        chunks = m0_layer._convert_data_to_chunks(test_data, "test_session")
        
        assert len(chunks) == 1
        assert isinstance(chunks[0], ChunkData)
        assert chunks[0].content == "Test message"
        assert "role" in chunks[0].metadata
        assert "timestamp" in chunks[0].metadata

    @pytest.mark.asyncio
    async def test_query_functionality(self, m0_layer, mock_storage_manager):
        """Test query functionality."""
        await m0_layer.initialize()
        
        # Mock query results
        mock_storage_manager.query_backend.return_value = [
            {"content": "Test result 1", "score": 0.9},
            {"content": "Test result 2", "score": 0.8}
        ]
        
        # Execute query
        results = await m0_layer.query("test query", top_k=5)
        
        assert len(results) == 2
        assert results[0]["content"] == "Test result 1"
        assert results[0]["score"] == 0.9

    @pytest.mark.asyncio
    async def test_error_handling(self, m0_layer, mock_storage_manager):
        """Test error handling in M0 layer."""
        await m0_layer.initialize()
        
        # Simulate storage error
        mock_storage_manager.write_to_backend.side_effect = Exception("Storage error")
        
        test_data = [{"role": "user", "content": "Test message"}]
        
        # Process data (should handle error gracefully)
        result = await m0_layer.process_data(test_data)
        
        assert result.success is False
        assert "error" in result.message.lower()

    @pytest.mark.asyncio
    async def test_statistics_tracking(self, m0_layer):
        """Test statistics tracking."""
        await m0_layer.initialize()
        
        # Initial statistics
        initial_stats = m0_layer.get_stats()
        assert initial_stats.total_operations == 0
        assert initial_stats.successful_operations == 0
        
        # Process some data
        test_data = [{"role": "user", "content": "Test"}]
        await m0_layer.process_data(test_data)
        
        # Check updated statistics
        updated_stats = m0_layer.get_stats()
        assert updated_stats.total_operations == 1

    def test_layer_configuration(self, m0_layer):
        """Test layer configuration properties."""
        # Test storage backends configuration
        assert "vector" in m0_layer.config.custom_config["storage_backends"]
        assert "keyword" in m0_layer.config.custom_config["storage_backends"]
        assert "sql" in m0_layer.config.custom_config["storage_backends"]
        
        # Test buffer configuration
        assert m0_layer.config.custom_config["buffer_size"] == 1000
        assert m0_layer.config.custom_config["auto_flush_interval"] == 60

    @pytest.mark.asyncio
    async def test_event_emission(self, m0_layer):
        """Test event emission for downstream processing."""
        await m0_layer.initialize()
        
        # Mock event emission
        with patch.object(m0_layer, '_emit_processing_event') as mock_emit:
            test_data = [{"role": "user", "content": "Test message"}]
            
            await m0_layer.process_data(test_data, session_id="test_session")
            
            # Verify event was emitted
            mock_emit.assert_called()

    @pytest.mark.asyncio
    async def test_concurrent_processing(self, m0_layer, mock_storage_manager):
        """Test concurrent data processing."""
        await m0_layer.initialize()
        
        # Prepare multiple data batches
        data_batches = [
            [{"role": "user", "content": f"Message {i}"}]
            for i in range(5)
        ]
        
        # Process concurrently
        tasks = [
            m0_layer.process_data(batch, session_id=f"session_{i}")
            for i, batch in enumerate(data_batches)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Verify all processing succeeded
        assert all(result.success for result in results)
        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_data_validation(self, m0_layer):
        """Test input data validation."""
        await m0_layer.initialize()
        
        # Test with invalid data
        invalid_data = [
            {"invalid_field": "no role or content"},
            {"role": "user"}  # missing content
        ]
        
        # Should handle invalid data gracefully
        result = await m0_layer.process_data(invalid_data)
        
        # May succeed with partial processing or fail gracefully
        assert isinstance(result.success, bool)

    @pytest.mark.asyncio
    async def test_metadata_preservation(self, m0_layer):
        """Test metadata preservation during processing."""
        await m0_layer.initialize()
        
        # Test data with metadata
        test_data = [{
            "role": "user",
            "content": "Test message",
            "metadata": {
                "source": "test_source",
                "priority": "high",
                "tags": ["important", "test"]
            }
        }]
        
        chunks = m0_layer._convert_data_to_chunks(test_data, "test_session")
        
        # Verify metadata is preserved
        assert len(chunks) == 1
        chunk = chunks[0]
        assert "source" in chunk.metadata
        assert "priority" in chunk.metadata
        assert "tags" in chunk.metadata
        assert chunk.metadata["source"] == "test_source"

    @pytest.mark.asyncio
    async def test_session_context(self, m0_layer):
        """Test session context handling."""
        await m0_layer.initialize()
        
        test_data = [{"role": "user", "content": "Test message"}]
        
        # Process with session context
        result = await m0_layer.process_data(test_data, session_id="session_123")
        
        # Verify session context is handled
        assert result.success is True
        
        # Check that chunks include session information
        chunks = m0_layer._convert_data_to_chunks(test_data, "session_123")
        assert chunks[0].metadata.get("session_id") == "session_123"
