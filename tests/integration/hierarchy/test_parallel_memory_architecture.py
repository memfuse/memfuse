"""
Integration tests for parallel memory architecture.

Tests the M0/M1/M2 parallel processing architecture to ensure it works
according to the documented design.
"""

import pytest
import asyncio
import tempfile
import os
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch

from src.memfuse_core.hierarchy.memory_layer_impl import MemoryLayerImpl
from src.memfuse_core.hierarchy.parallel_manager import ParallelMemoryLayerManager
from src.memfuse_core.hierarchy.manager import MemoryHierarchyManager
from src.memfuse_core.hierarchy.types import WriteStrategy, ParallelWriteResult
from src.memfuse_core.interfaces.memory_layer import MemoryLayerConfig, WriteResult
from src.memfuse_core.interfaces.message_interface import MessageBatchList
from src.memfuse_core.utils.config import ConfigManager


class TestParallelMemoryArchitecture:
    """Test suite for parallel memory architecture implementation."""

    @pytest.fixture
    async def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def config_manager(self):
        """Create config manager with test configuration."""
        config_manager = ConfigManager()
        test_config = {
            "memory": {
                "processing": {
                    "strategy": "parallel",
                    "enable_fallback": True,
                    "max_concurrent_layers": 3
                },
                "layers": {
                    "m0": {"enabled": True, "priority": 1},
                    "m1": {"enabled": True, "priority": 2},
                    "m2": {"enabled": True, "priority": 3}
                },
                "memory_service": {
                    "parallel_enabled": True,
                    "parallel_strategy": "parallel"
                }
            }
        }
        config_manager.set_config(test_config)
        return config_manager

    @pytest.fixture
    def memory_layer_config(self):
        """Create memory layer configuration."""
        return MemoryLayerConfig(
            m0_enabled=True,
            m1_enabled=True,
            m2_enabled=True,
            parallel_strategy="parallel",
            enable_fallback=True,
            timeout_per_layer=30.0,
            max_retries=3
        )

    @pytest.fixture
    async def memory_layer_impl(self, config_manager, memory_layer_config, temp_dir):
        """Create MemoryLayerImpl instance for testing."""
        memory_layer = MemoryLayerImpl(
            user_id="test_user",
            config_manager=config_manager,
            config=memory_layer_config
        )
        
        # Mock the hierarchy manager initialization to avoid database dependencies
        with patch.object(memory_layer, '_initialize_hierarchy_manager') as mock_init:
            mock_hierarchy = AsyncMock(spec=MemoryHierarchyManager)
            mock_hierarchy.initialized = True
            mock_hierarchy.layers = {}
            mock_init.return_value = mock_hierarchy
            memory_layer.hierarchy_manager = mock_hierarchy
            
            # Mock parallel manager
            mock_parallel = AsyncMock(spec=ParallelMemoryLayerManager)
            memory_layer.parallel_manager = mock_parallel
            
            await memory_layer.initialize()
            yield memory_layer

    @pytest.mark.asyncio
    async def test_memory_layer_initialization(self, memory_layer_impl):
        """Test that MemoryLayerImpl initializes correctly with parallel processing."""
        assert memory_layer_impl.initialized
        assert memory_layer_impl.parallel_manager is not None
        assert memory_layer_impl.hierarchy_manager is not None
        assert memory_layer_impl.config.parallel_strategy == "parallel"

    @pytest.mark.asyncio
    async def test_parallel_write_strategy(self, memory_layer_impl):
        """Test parallel write strategy execution."""
        # Prepare test data
        message_batch_list = [
            {"role": "user", "content": "Test message 1"},
            {"role": "assistant", "content": "Test response 1"}
        ]
        
        # Mock successful parallel write result
        mock_result = ParallelWriteResult(
            success=True,
            layer_results={
                "M0": MagicMock(success=True, processed_items=["item1"]),
                "M1": MagicMock(success=True, processed_items=["item2"]),
                "M2": MagicMock(success=True, processed_items=["item3"])
            },
            total_processing_time=1.5,
            strategy_used=WriteStrategy.PARALLEL,
            total_processed=3
        )
        
        memory_layer_impl.parallel_manager.write_data.return_value = mock_result
        
        # Execute parallel write
        result = await memory_layer_impl.write_parallel(
            message_batch_list=message_batch_list,
            session_id="test_session"
        )
        
        # Verify results
        assert isinstance(result, WriteResult)
        assert result.success
        assert "Successfully processed" in result.message
        assert len(result.layer_results) == 3
        assert "operation_time" in result.metadata
        
        # Verify parallel manager was called with correct strategy
        memory_layer_impl.parallel_manager.write_data.assert_called_once()
        call_args = memory_layer_impl.parallel_manager.write_data.call_args
        assert call_args[1]["strategy"] == WriteStrategy.PARALLEL

    @pytest.mark.asyncio
    async def test_sequential_fallback(self, memory_layer_impl):
        """Test fallback to sequential processing when parallel fails."""
        # Configure for sequential strategy
        memory_layer_impl.config.parallel_strategy = "sequential"
        
        message_batch_list = [{"role": "user", "content": "Test message"}]
        
        # Mock sequential write result
        mock_result = ParallelWriteResult(
            success=True,
            layer_results={"M0": MagicMock(success=True)},
            total_processing_time=1.0,
            strategy_used=WriteStrategy.SEQUENTIAL,
            total_processed=1
        )
        
        memory_layer_impl.parallel_manager.write_data.return_value = mock_result
        
        # Execute write
        result = await memory_layer_impl.write_parallel(
            message_batch_list=message_batch_list
        )
        
        # Verify sequential strategy was used
        assert result.success
        call_args = memory_layer_impl.parallel_manager.write_data.call_args
        assert call_args[1]["strategy"] == WriteStrategy.SEQUENTIAL

    @pytest.mark.asyncio
    async def test_error_handling_and_statistics(self, memory_layer_impl):
        """Test error handling and statistics tracking."""
        # Mock failed write
        memory_layer_impl.parallel_manager.write_data.side_effect = Exception("Test error")
        
        message_batch_list = [{"role": "user", "content": "Test message"}]
        
        # Execute write (should handle error gracefully)
        result = await memory_layer_impl.write_parallel(
            message_batch_list=message_batch_list
        )
        
        # Verify error handling
        assert not result.success
        assert "Write operation failed" in result.message
        assert memory_layer_impl.failed_operations == 1
        assert memory_layer_impl.total_operations == 1

    @pytest.mark.asyncio
    async def test_layer_status_tracking(self, memory_layer_impl):
        """Test that layer status is properly tracked."""
        # Check initial status
        assert memory_layer_impl.layer_status["M0"] == "INACTIVE"
        assert memory_layer_impl.layer_status["M1"] == "INACTIVE"
        assert memory_layer_impl.layer_status["M2"] == "INACTIVE"
        
        # The status should be updated during actual layer initialization
        # This test verifies the structure is in place

    def test_configuration_validation(self, config_manager):
        """Test that configuration is properly validated."""
        config = config_manager.get_config()
        
        # Verify parallel processing is enabled
        assert config["memory"]["memory_service"]["parallel_enabled"] is True
        assert config["memory"]["processing"]["strategy"] == "parallel"
        
        # Verify all layers are enabled
        layers = config["memory"]["layers"]
        assert layers["m0"]["enabled"] is True
        assert layers["m1"]["enabled"] is True
        assert layers["m2"]["enabled"] is True


class TestParallelManagerIntegration:
    """Test ParallelMemoryLayerManager integration."""

    @pytest.fixture
    def mock_hierarchy_manager(self):
        """Create mock hierarchy manager."""
        manager = AsyncMock(spec=MemoryHierarchyManager)
        manager.initialized = True
        
        # Mock layers
        mock_m0 = AsyncMock()
        mock_m0.initialized = True
        mock_m1 = AsyncMock()
        mock_m1.initialized = True
        mock_m2 = AsyncMock()
        mock_m2.initialized = True
        
        manager.layers = {
            "M0": mock_m0,
            "M1": mock_m1,
            "M2": mock_m2
        }
        
        return manager

    @pytest.mark.asyncio
    async def test_parallel_manager_strategies(self, mock_hierarchy_manager):
        """Test different write strategies in ParallelMemoryLayerManager."""
        parallel_manager = ParallelMemoryLayerManager(
            hierarchy_manager=mock_hierarchy_manager,
            config={}
        )
        
        test_data = {"content": "test data"}
        
        # Test parallel strategy
        with patch.object(parallel_manager, '_write_parallel') as mock_parallel:
            mock_parallel.return_value = ParallelWriteResult(
                success=True, layer_results={}, total_processing_time=1.0,
                strategy_used=WriteStrategy.PARALLEL
            )
            
            result = await parallel_manager.write_data(
                data=test_data,
                strategy=WriteStrategy.PARALLEL
            )
            
            assert result.success
            assert result.strategy_used == WriteStrategy.PARALLEL
            mock_parallel.assert_called_once()

    @pytest.mark.asyncio
    async def test_layer_coordination(self, mock_hierarchy_manager):
        """Test coordination between different memory layers."""
        parallel_manager = ParallelMemoryLayerManager(
            hierarchy_manager=mock_hierarchy_manager,
            config={}
        )
        
        # Mock successful processing for all layers
        for layer in mock_hierarchy_manager.layers.values():
            layer.process_data.return_value = MagicMock(
                success=True,
                processed_items=["test_item"],
                processing_time=0.5
            )
        
        test_data = {"content": "test data"}
        
        # Execute parallel write
        result = await parallel_manager._write_parallel(test_data)
        
        # Verify all layers were called
        assert result.success
        for layer in mock_hierarchy_manager.layers.values():
            layer.process_data.assert_called_once_with(test_data, None)
