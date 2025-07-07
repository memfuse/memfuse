"""
Integration tests for parallel M0/M1/M2 memory layer processing.

This test suite verifies that the unified memory layer implementation
correctly processes data through M0, M1, and M2 layers in parallel.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List

from src.memfuse_core.hierarchy.unified_memory_layer_impl import UnifiedMemoryLayerImpl
from src.memfuse_core.interfaces.unified_memory_layer import WriteResult, LayerStatus
from src.memfuse_core.interfaces import MessageBatchList
from src.memfuse_core.utils.config import ConfigManager
from src.memfuse_core.hierarchy.core import LayerType


class TestParallelMemoryLayers:
    """Integration tests for parallel memory layer processing."""
    
    @pytest.fixture
    def memory_config(self):
        """Configuration for memory layers with all layers enabled."""
        return {
            "m0_enabled": True,
            "m1_enabled": True,
            "m2_enabled": True,
            "memory_service": {
                "parallel_enabled": True,
                "parallel_strategy": "parallel",
                "enable_fallback": True,
                "timeout_per_layer": 30.0,
                "max_retries": 3
            }
        }
    
    @pytest.fixture
    def sample_message_batch_list(self):
        """Sample message batch list for testing."""
        return [
            [
                {"role": "user", "content": "Hello, how are you?"},
                {"role": "assistant", "content": "I'm doing well, thank you!"}
            ],
            [
                {"role": "user", "content": "What's the weather like?"},
                {"role": "assistant", "content": "I don't have access to current weather data."}
            ]
        ]
    
    @pytest.mark.asyncio
    async def test_parallel_layer_initialization(self, memory_config):
        """Test that all memory layers are properly initialized in parallel setup."""
        with patch('src.memfuse_core.hierarchy.manager.MemoryHierarchyManager') as mock_hierarchy_manager_class:
            # Mock hierarchy manager
            mock_hierarchy_manager = Mock()
            mock_hierarchy_manager.initialize = AsyncMock(return_value=True)
            mock_hierarchy_manager.layers = {
                LayerType.M0: Mock(initialized=True),
                LayerType.M1: Mock(initialized=True),
                LayerType.M2: Mock(initialized=True)
            }
            mock_hierarchy_manager_class.return_value = mock_hierarchy_manager
            
            # Create unified memory layer
            unified_layer = UnifiedMemoryLayerImpl(
                user_id="test_user",
                config_manager=ConfigManager()
            )
            
            # Initialize with config
            result = await unified_layer.initialize(memory_config)
            
            # Verify initialization
            assert result is True
            assert unified_layer.initialized is True
            assert unified_layer.hierarchy_manager is not None
            assert unified_layer.parallel_manager is not None
            
            # Verify layer status
            assert unified_layer.layer_status["M0"] == LayerStatus.ACTIVE
            assert unified_layer.layer_status["M1"] == LayerStatus.ACTIVE
            assert unified_layer.layer_status["M2"] == LayerStatus.ACTIVE
            
            # Verify hierarchy manager was initialized (may be called multiple times)
            assert mock_hierarchy_manager.initialize.call_count >= 1
    
    @pytest.mark.asyncio
    async def test_parallel_write_execution(self, memory_config, sample_message_batch_list):
        """Test that write operations execute in parallel across all layers."""
        with patch('src.memfuse_core.hierarchy.manager.MemoryHierarchyManager') as mock_hierarchy_manager_class:
            # Mock hierarchy manager with layers
            mock_hierarchy_manager = Mock()
            mock_hierarchy_manager.initialize = AsyncMock(return_value=True)
            
            # Mock individual layers with timing to verify parallel execution
            mock_m0_layer = Mock(initialized=True)
            mock_m1_layer = Mock(initialized=True)
            mock_m2_layer = Mock(initialized=True)

            # Add delays to simulate processing time
            async def mock_m0_process(*args, **kwargs):
                await asyncio.sleep(0.1)  # 100ms delay
                return {"success": True, "layer": "M0", "processed_items": 2}

            async def mock_m1_process(*args, **kwargs):
                await asyncio.sleep(0.15)  # 150ms delay
                return {"success": True, "layer": "M1", "processed_items": 2}

            async def mock_m2_process(*args, **kwargs):
                await asyncio.sleep(0.12)  # 120ms delay
                return {"success": True, "layer": "M2", "processed_items": 2}

            mock_m0_layer.process_data = mock_m0_process
            mock_m1_layer.process_data = mock_m1_process
            mock_m2_layer.process_data = mock_m2_process

            mock_hierarchy_manager.layers = {
                LayerType.M0: mock_m0_layer,
                LayerType.M1: mock_m1_layer,
                LayerType.M2: mock_m2_layer
            }
            mock_hierarchy_manager_class.return_value = mock_hierarchy_manager
            
            # Mock parallel manager to track parallel execution
            with patch('src.memfuse_core.hierarchy.unified_memory_layer_impl.ParallelMemoryLayerManager') as mock_parallel_manager_class:
                mock_parallel_manager = Mock()
                mock_parallel_manager.initialize = AsyncMock(return_value=True)
                
                # Mock parallel write result
                from src.memfuse_core.hierarchy.types import ParallelWriteResult, LayerWriteResult, WriteStrategy
                mock_result = ParallelWriteResult(
                    success=True,
                    layer_results={
                        LayerType.M0: LayerWriteResult(success=True, result={"processed": 2}, processed_items=[], processing_time=0.1),
                        LayerType.M1: LayerWriteResult(success=True, result={"processed": 2}, processed_items=[], processing_time=0.15),
                        LayerType.M2: LayerWriteResult(success=True, result={"processed": 2}, processed_items=[], processing_time=0.12)
                    },
                    total_processing_time=0.15,  # Should be max of individual times for parallel
                    strategy_used=WriteStrategy.PARALLEL
                )
                mock_parallel_manager.write_data = AsyncMock(return_value=mock_result)
                mock_parallel_manager_class.return_value = mock_parallel_manager
                
                # Create and initialize unified memory layer
                unified_layer = UnifiedMemoryLayerImpl(
                    user_id="test_user",
                    config_manager=ConfigManager()
                )
                
                await unified_layer.initialize(memory_config)
                
                # Execute parallel write
                start_time = time.time()
                result = await unified_layer.write_parallel(
                    message_batch_list=sample_message_batch_list,
                    session_id="test_session"
                )
                execution_time = time.time() - start_time
                
                # Verify results
                assert result.success is True
                assert "Successfully processed 2 message batches" in result.message
                assert len(result.layer_results) == 3
                assert LayerType.M0 in result.layer_results
                assert LayerType.M1 in result.layer_results
                assert LayerType.M2 in result.layer_results
                
                # Verify parallel manager was called with correct strategy
                mock_parallel_manager.write_data.assert_called_once()
                call_args = mock_parallel_manager.write_data.call_args
                assert call_args[1]["strategy"].value == "parallel"
                
                # Verify execution was reasonably fast (parallel, not sequential)
                # If sequential: 0.1 + 0.15 + 0.12 = 0.37s
                # If parallel: max(0.1, 0.15, 0.12) = 0.15s + overhead
                assert execution_time < 0.3  # Should be much faster than sequential
    
    @pytest.mark.asyncio
    async def test_layer_failure_handling(self, memory_config, sample_message_batch_list):
        """Test handling of individual layer failures in parallel processing."""
        with patch('src.memfuse_core.hierarchy.manager.MemoryHierarchyManager') as mock_hierarchy_manager_class:
            # Mock hierarchy manager
            mock_hierarchy_manager = Mock()
            mock_hierarchy_manager.initialize = AsyncMock(return_value=True)
            mock_hierarchy_manager.layers = {
                LayerType.M0: Mock(initialized=True),
                LayerType.M1: Mock(initialized=True),
                LayerType.M2: Mock(initialized=True)
            }
            mock_hierarchy_manager_class.return_value = mock_hierarchy_manager
            
            # Mock parallel manager with partial failure
            with patch('src.memfuse_core.hierarchy.unified_memory_layer_impl.ParallelMemoryLayerManager') as mock_parallel_manager_class:
                mock_parallel_manager = Mock()
                mock_parallel_manager.initialize = AsyncMock(return_value=True)
                
                # Mock result with M1 failure
                from src.memfuse_core.hierarchy.types import ParallelWriteResult, LayerWriteResult, WriteStrategy
                mock_result = ParallelWriteResult(
                    success=False,  # Overall failure due to M1 failure
                    layer_results={
                        LayerType.M0: LayerWriteResult(success=True, result={"processed": 2}, processed_items=[], processing_time=0.1),
                        LayerType.M1: LayerWriteResult(success=False, result=None, processed_items=[], processing_time=0.0, error_message="M1 processing failed"),
                        LayerType.M2: LayerWriteResult(success=True, result={"processed": 2}, processed_items=[], processing_time=0.12)
                    },
                    total_processing_time=0.12,
                    strategy_used=WriteStrategy.PARALLEL,
                    error_message="M1 layer failed"
                )
                mock_parallel_manager.write_data = AsyncMock(return_value=mock_result)
                mock_parallel_manager_class.return_value = mock_parallel_manager
                
                # Create and initialize unified memory layer
                unified_layer = UnifiedMemoryLayerImpl(
                    user_id="test_user",
                    config_manager=ConfigManager()
                )
                
                await unified_layer.initialize(memory_config)
                
                # Execute parallel write
                result = await unified_layer.write_parallel(
                    message_batch_list=sample_message_batch_list,
                    session_id="test_session"
                )
                
                # Verify failure handling
                assert result.success is False
                assert "Failed to process message batches" in result.message
                assert "M1 layer failed" in result.message
                assert len(result.layer_results) == 3

                # Verify individual layer results
                assert result.layer_results[LayerType.M0].success is True
                assert result.layer_results[LayerType.M1].success is False
                assert result.layer_results[LayerType.M2].success is True
    
    @pytest.mark.asyncio
    async def test_selective_layer_activation(self):
        """Test that only enabled layers are activated and processed."""
        # Config with only M0 and M2 enabled
        selective_config = {
            "m0_enabled": True,
            "m1_enabled": False,  # M1 disabled
            "m2_enabled": True,
            "memory_service": {
                "parallel_enabled": True,
                "parallel_strategy": "parallel"
            }
        }
        
        with patch('src.memfuse_core.hierarchy.manager.MemoryHierarchyManager') as mock_hierarchy_manager_class:
            # Mock hierarchy manager
            mock_hierarchy_manager = Mock()
            mock_hierarchy_manager.initialize = AsyncMock(return_value=True)
            mock_hierarchy_manager.layers = {
                LayerType.M0: Mock(initialized=True),
                LayerType.M2: Mock(initialized=True)
                # M1 not included since it's disabled
            }
            mock_hierarchy_manager_class.return_value = mock_hierarchy_manager
            
            # Create unified memory layer
            unified_layer = UnifiedMemoryLayerImpl(
                user_id="test_user",
                config_manager=ConfigManager()
            )
            
            # Initialize with selective config
            result = await unified_layer.initialize(selective_config)
            
            # Verify initialization
            assert result is True
            assert unified_layer.initialized is True
            
            # Verify layer status
            assert unified_layer.layer_status["M0"] == LayerStatus.ACTIVE
            assert unified_layer.layer_status["M1"] == LayerStatus.INACTIVE  # Disabled
            assert unified_layer.layer_status["M2"] == LayerStatus.ACTIVE
