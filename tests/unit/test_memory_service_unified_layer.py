"""
Unit tests for MemoryService with Unified Memory Layer integration.

Tests the integration between MemoryService and UnifiedMemoryLayer,
ensuring proper L0/L1/L2 parallel processing functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List

from src.memfuse_core.services.memory_service import MemoryService
from src.memfuse_core.interfaces.unified_memory_layer import WriteResult, UnifiedResult, LayerStatus
from src.memfuse_core.interfaces.message_interface import MessageBatchList


class TestMemoryServiceUnifiedLayer:
    """Test MemoryService integration with Unified Memory Layer."""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration with unified layer enabled."""
        return {
            "data_dir": "test_data",
            "memory": {
                "memory_service": {
                    "parallel_enabled": True
                },
                "l0_enabled": True,
                "l1_enabled": True,
                "l2_enabled": True
            },
            "store": {
                "multi_path": {
                    "use_graph": False
                }
            },
            "buffer": {
                "hybrid_buffer": {
                    "chunk_strategy": "message"
                }
            }
        }
    
    @pytest.fixture
    def mock_config_disabled(self):
        """Mock configuration with unified layer disabled."""
        return {
            "data_dir": "test_data",
            "memory": {
                "memory_service": {
                    "parallel_enabled": False
                }
            },
            "store": {
                "multi_path": {
                    "use_graph": False
                }
            }
        }
    
    @pytest.fixture
    def sample_message_batch_list(self):
        """Sample message batch list for testing."""
        return [
            [
                {"id": "msg1", "content": "Hello world", "role": "user"},
                {"id": "msg2", "content": "How are you?", "role": "assistant"}
            ],
            [
                {"id": "msg3", "content": "I'm fine, thanks", "role": "user"}
            ]
        ]
    
    @pytest.mark.asyncio
    async def test_unified_layer_enabled_detection(self, mock_config):
        """Test that MemoryService correctly detects when unified layer should be enabled."""
        with patch('src.memfuse_core.services.database_service.DatabaseService') as mock_db_service:
            # Mock database service
            mock_db = Mock()
            mock_db.get_or_create_user_by_name.return_value = "user123"
            mock_db.get_or_create_agent_by_name.return_value = "agent456"
            mock_db_service.get_instance.return_value = mock_db
            
            # Create MemoryService with unified layer enabled config
            memory_service = MemoryService(cfg=mock_config, user="test_user")
            
            # Check that unified layer is enabled
            assert memory_service.use_unified_layer is True
            assert memory_service.unified_memory_layer is None  # Not initialized yet
    
    @pytest.mark.asyncio
    async def test_unified_layer_disabled_detection(self, mock_config_disabled):
        """Test that MemoryService correctly detects when unified layer should be disabled."""
        with patch('src.memfuse_core.services.database_service.DatabaseService') as mock_db_service:
            # Mock database service
            mock_db = Mock()
            mock_db.get_or_create_user_by_name.return_value = "user123"
            mock_db.get_or_create_agent_by_name.return_value = "agent456"
            mock_db_service.get_instance.return_value = mock_db
            
            # Create MemoryService with unified layer disabled config
            memory_service = MemoryService(cfg=mock_config_disabled, user="test_user")
            
            # Check that unified layer is disabled
            assert memory_service.use_unified_layer is False
            assert memory_service.unified_memory_layer is None
    
    @pytest.mark.asyncio
    async def test_unified_layer_initialization(self, mock_config):
        """Test that unified layer is properly initialized during MemoryService.initialize()."""
        with patch('src.memfuse_core.services.database_service.DatabaseService') as mock_db_service, \
             patch('src.memfuse_core.store.factory.StoreFactory') as mock_store_factory, \
             patch('src.memfuse_core.services.memory_service.UnifiedMemoryLayerImpl') as mock_unified_impl:
            
            # Mock database service
            mock_db = Mock()
            mock_db.get_or_create_user_by_name.return_value = "user123"
            mock_db.get_or_create_agent_by_name.return_value = "agent456"
            mock_db_service.get_instance.return_value = mock_db
            
            # Mock store factory
            mock_store_factory.create_vector_store = AsyncMock(return_value=Mock())
            mock_store_factory.create_keyword_store = AsyncMock(return_value=Mock())
            mock_store_factory.create_multi_path_retrieval = AsyncMock(return_value=Mock())
            
            # Mock unified memory layer implementation
            mock_unified_layer = Mock()
            mock_unified_layer.initialize = AsyncMock(return_value=True)
            mock_unified_impl.return_value = mock_unified_layer
            
            # Create and initialize MemoryService
            memory_service = MemoryService(cfg=mock_config, user="test_user")
            await memory_service.initialize()
            
            # Verify unified layer was created and initialized
            assert memory_service.use_unified_layer is True
            assert memory_service.unified_memory_layer is not None
            mock_unified_layer.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_add_batch_with_unified_layer(self, mock_config, sample_message_batch_list):
        """Test add_batch method uses unified layer when enabled."""
        with patch('src.memfuse_core.services.database_service.DatabaseService') as mock_db_service:
            # Mock database service
            mock_db = Mock()
            mock_db.get_or_create_user_by_name.return_value = "user123"
            mock_db.get_or_create_agent_by_name.return_value = "agent456"
            mock_db_service.get_instance.return_value = mock_db
            
            # Create MemoryService
            memory_service = MemoryService(cfg=mock_config, user="test_user")
            
            # Mock unified memory layer
            mock_unified_layer = Mock()
            mock_write_result = WriteResult(
                success=True,
                message="Successfully processed through unified layer",
                layer_results={"L0": {"processed_count": 3}, "L1": {"processed_count": 3}, "L2": {"processed_count": 3}},
                metadata={"processing_method": "unified_parallel"}
            )
            mock_unified_layer.write_parallel = AsyncMock(return_value=mock_write_result)
            memory_service.unified_memory_layer = mock_unified_layer
            
            # Mock session preparation
            memory_service._prepare_session_and_round = AsyncMock(return_value=("session123", "round456"))
            memory_service._store_original_messages_with_round = AsyncMock(return_value=["msg1", "msg2", "msg3"])
            
            # Call add_batch
            result = await memory_service.add_batch(sample_message_batch_list)
            
            # Verify unified layer was called
            mock_unified_layer.write_parallel.assert_called_once()
            call_args = mock_unified_layer.write_parallel.call_args
            assert call_args[1]["message_batch_list"] == sample_message_batch_list
            assert call_args[1]["session_id"] == "session123"
            
            # Verify result
            assert result["status"] == "success"
            assert "unified_parallel" in str(result)
    
    @pytest.mark.asyncio
    async def test_add_batch_fallback_to_traditional(self, mock_config_disabled, sample_message_batch_list):
        """Test add_batch method falls back to traditional method when unified layer is disabled."""
        with patch('src.memfuse_core.services.database_service.DatabaseService') as mock_db_service:
            # Mock database service
            mock_db = Mock()
            mock_db.get_or_create_user_by_name.return_value = "user123"
            mock_db.get_or_create_agent_by_name.return_value = "agent456"
            mock_db_service.get_instance.return_value = mock_db
            
            # Create MemoryService with unified layer disabled
            memory_service = MemoryService(cfg=mock_config_disabled, user="test_user")
            
            # Mock traditional processing method
            memory_service._process_with_traditional_method = AsyncMock(return_value={
                "status": "success",
                "message": "Processed with traditional method",
                "data": ["msg1", "msg2", "msg3"]
            })
            
            # Call add_batch
            result = await memory_service.add_batch(sample_message_batch_list)
            
            # Verify traditional method was called
            memory_service._process_with_traditional_method.assert_called_once_with(
                sample_message_batch_list
            )
            
            # Verify result
            assert result["status"] == "success"
            assert "traditional method" in result["message"]
    
    @pytest.mark.asyncio
    async def test_unified_layer_error_handling(self, mock_config, sample_message_batch_list):
        """Test error handling when unified layer processing fails."""
        with patch('src.memfuse_core.services.database_service.DatabaseService') as mock_db_service:
            # Mock database service
            mock_db = Mock()
            mock_db.get_or_create_user_by_name.return_value = "user123"
            mock_db.get_or_create_agent_by_name.return_value = "agent456"
            mock_db_service.get_instance.return_value = mock_db
            
            # Create MemoryService
            memory_service = MemoryService(cfg=mock_config, user="test_user")
            
            # Mock unified memory layer with failure
            mock_unified_layer = Mock()
            mock_write_result = WriteResult(
                success=False,
                message="Processing failed due to L1 layer error",
                layer_results={"L0": {"processed_count": 3}, "L1": {"error": "Connection timeout"}},
                metadata={}
            )
            mock_unified_layer.write_parallel = AsyncMock(return_value=mock_write_result)
            memory_service.unified_memory_layer = mock_unified_layer
            
            # Mock session preparation
            memory_service._prepare_session_and_round = AsyncMock(return_value=("session123", "round456"))
            memory_service._store_original_messages_with_round = AsyncMock(return_value=["msg1", "msg2", "msg3"])
            
            # Call add_batch
            result = await memory_service.add_batch(sample_message_batch_list)
            
            # Verify error handling
            assert result["status"] == "error"
            assert "Processing failed due to L1 layer error" in result["message"]


if __name__ == "__main__":
    pytest.main([__file__])
