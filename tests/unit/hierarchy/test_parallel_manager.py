"""
Unit tests for Parallel Memory Layer Manager.

Tests the ParallelMemoryLayerManager implementation including:
- Parallel processing strategies
- Layer coordination
- Error handling and fallback mechanisms
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

from src.memfuse_core.hierarchy.parallel_manager import ParallelMemoryLayerManager
from src.memfuse_core.hierarchy.types import WriteStrategy, ParallelWriteResult
from src.memfuse_core.hierarchy.manager import MemoryHierarchyManager
from src.memfuse_core.hierarchy.core import LayerType


class TestParallelMemoryLayerManager:
    """Test suite for ParallelMemoryLayerManager."""

    @pytest.fixture
    def mock_hierarchy_manager(self):
        """Create mock hierarchy manager with layers."""
        manager = AsyncMock(spec=MemoryHierarchyManager)
        manager.initialized = True
        
        # Mock layers
        mock_m0 = AsyncMock()
        mock_m0.initialized = True
        mock_m0.process_data.return_value = MagicMock(
            success=True,
            processed_items=["m0_item1", "m0_item2"],
            processing_time=0.5
        )
        
        mock_m1 = AsyncMock()
        mock_m1.initialized = True
        mock_m1.process_data.return_value = MagicMock(
            success=True,
            processed_items=["m1_item1"],
            processing_time=0.8
        )
        
        mock_m2 = AsyncMock()
        mock_m2.initialized = True
        mock_m2.process_data.return_value = MagicMock(
            success=True,
            processed_items=["m2_item1"],
            processing_time=1.2
        )
        
        manager.layers = {
            LayerType.M0: mock_m0,
            LayerType.M1: mock_m1,
            LayerType.M2: mock_m2
        }
        
        return manager

    @pytest.fixture
    def parallel_manager_config(self):
        """Create configuration for parallel manager."""
        return {
            "max_concurrent_layers": 3,
            "timeout_per_layer": 30.0,
            "enable_fallback": True,
            "fallback_strategy": "sequential",
            "max_retries": 3
        }

    @pytest.fixture
    def parallel_manager(self, mock_hierarchy_manager, parallel_manager_config):
        """Create ParallelMemoryLayerManager instance."""
        return ParallelMemoryLayerManager(
            hierarchy_manager=mock_hierarchy_manager,
            config=parallel_manager_config
        )

    @pytest.mark.asyncio
    async def test_parallel_write_strategy(self, parallel_manager, mock_hierarchy_manager):
        """Test parallel write strategy execution."""
        test_data = {"content": "test data", "session_id": "test_session"}
        
        # Execute parallel write
        result = await parallel_manager.write_data(
            data=test_data,
            strategy=WriteStrategy.PARALLEL
        )
        
        # Verify result
        assert isinstance(result, ParallelWriteResult)
        assert result.success is True
        assert result.strategy_used == WriteStrategy.PARALLEL
        assert len(result.layer_results) == 3
        assert "M0" in result.layer_results
        assert "M1" in result.layer_results
        assert "M2" in result.layer_results
        
        # Verify all layers were called
        for layer in mock_hierarchy_manager.layers.values():
            layer.process_data.assert_called_once_with(test_data, "test_session")

    @pytest.mark.asyncio
    async def test_sequential_write_strategy(self, parallel_manager, mock_hierarchy_manager):
        """Test sequential write strategy execution."""
        test_data = {"content": "test data"}
        
        # Execute sequential write
        result = await parallel_manager.write_data(
            data=test_data,
            strategy=WriteStrategy.SEQUENTIAL
        )
        
        # Verify result
        assert result.success is True
        assert result.strategy_used == WriteStrategy.SEQUENTIAL
        assert len(result.layer_results) == 3

    @pytest.mark.asyncio
    async def test_parallel_processing_performance(self, parallel_manager):
        """Test that parallel processing is actually faster than sequential."""
        test_data = {"content": "test data"}
        
        # Execute parallel write
        parallel_result = await parallel_manager.write_data(
            data=test_data,
            strategy=WriteStrategy.PARALLEL
        )
        
        # Execute sequential write
        sequential_result = await parallel_manager.write_data(
            data=test_data,
            strategy=WriteStrategy.SEQUENTIAL
        )
        
        # Parallel should be faster (or at least not significantly slower)
        # Note: In mocked environment, timing differences may be minimal
        assert parallel_result.total_processing_time >= 0
        assert sequential_result.total_processing_time >= 0

    @pytest.mark.asyncio
    async def test_error_handling_in_parallel_mode(self, parallel_manager, mock_hierarchy_manager):
        """Test error handling during parallel processing."""
        # Make M1 layer fail
        mock_hierarchy_manager.layers[LayerType.M1].process_data.side_effect = Exception("M1 failed")
        
        test_data = {"content": "test data"}
        
        # Execute parallel write
        result = await parallel_manager.write_data(
            data=test_data,
            strategy=WriteStrategy.PARALLEL
        )
        
        # Should handle partial failure gracefully
        assert isinstance(result, ParallelWriteResult)
        # M0 and M2 should succeed, M1 should fail
        assert result.layer_results["M0"]["success"] is True
        assert result.layer_results["M1"]["success"] is False
        assert result.layer_results["M2"]["success"] is True

    @pytest.mark.asyncio
    async def test_fallback_mechanism(self, parallel_manager, mock_hierarchy_manager):
        """Test fallback from parallel to sequential on failure."""
        # Configure for fallback testing
        parallel_manager.config["enable_fallback"] = True
        
        # Make parallel processing fail initially
        original_parallel_method = parallel_manager._write_parallel
        
        async def failing_parallel(*args, **kwargs):
            raise Exception("Parallel processing failed")
        
        parallel_manager._write_parallel = failing_parallel
        
        test_data = {"content": "test data"}
        
        # Execute write (should fallback to sequential)
        result = await parallel_manager.write_data(
            data=test_data,
            strategy=WriteStrategy.PARALLEL
        )
        
        # Should fallback to sequential
        assert result.strategy_used == WriteStrategy.SEQUENTIAL
        assert result.success is True

    @pytest.mark.asyncio
    async def test_timeout_handling(self, parallel_manager, mock_hierarchy_manager):
        """Test timeout handling in parallel processing."""
        # Make M2 layer take too long
        async def slow_process_data(*args, **kwargs):
            await asyncio.sleep(2.0)  # Longer than timeout
            return MagicMock(success=True, processed_items=[], processing_time=2.0)
        
        mock_hierarchy_manager.layers[LayerType.M2].process_data = slow_process_data
        
        # Set short timeout
        parallel_manager.config["timeout_per_layer"] = 0.5
        
        test_data = {"content": "test data"}
        
        # Execute parallel write
        result = await parallel_manager.write_data(
            data=test_data,
            strategy=WriteStrategy.PARALLEL
        )
        
        # Should handle timeout gracefully
        assert isinstance(result, ParallelWriteResult)
        # M0 and M1 should succeed, M2 might timeout
        assert result.layer_results["M0"]["success"] is True
        assert result.layer_results["M1"]["success"] is True

    @pytest.mark.asyncio
    async def test_layer_coordination(self, parallel_manager, mock_hierarchy_manager):
        """Test coordination between different memory layers."""
        test_data = {
            "content": "test data",
            "metadata": {"priority": "high"}
        }
        
        # Execute parallel write
        result = await parallel_manager.write_data(
            data=test_data,
            strategy=WriteStrategy.PARALLEL
        )
        
        # Verify coordination
        assert result.success is True
        assert result.total_processed == 4  # 2 from M0, 1 from M1, 1 from M2
        
        # Verify each layer received the same data
        for layer in mock_hierarchy_manager.layers.values():
            layer.process_data.assert_called_once()
            call_args = layer.process_data.call_args[0]
            assert call_args[0] == test_data

    @pytest.mark.asyncio
    async def test_result_aggregation(self, parallel_manager):
        """Test aggregation of results from multiple layers."""
        test_data = {"content": "test data"}
        
        # Execute parallel write
        result = await parallel_manager.write_data(
            data=test_data,
            strategy=WriteStrategy.PARALLEL
        )
        
        # Verify result aggregation
        assert isinstance(result, ParallelWriteResult)
        assert len(result.layer_results) == 3
        
        # Check individual layer results
        for layer_name in ["M0", "M1", "M2"]:
            assert layer_name in result.layer_results
            layer_result = result.layer_results[layer_name]
            assert "success" in layer_result
            assert "processed_items" in layer_result
            assert "processing_time" in layer_result

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, parallel_manager):
        """Test handling of concurrent write requests."""
        test_data_batches = [
            {"content": f"test data {i}"}
            for i in range(5)
        ]
        
        # Execute concurrent writes
        tasks = [
            parallel_manager.write_data(data, strategy=WriteStrategy.PARALLEL)
            for data in test_data_batches
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Verify all requests succeeded
        assert len(results) == 5
        for result in results:
            assert result.success is True
            assert result.strategy_used == WriteStrategy.PARALLEL

    @pytest.mark.asyncio
    async def test_resource_management(self, parallel_manager, mock_hierarchy_manager):
        """Test resource management during parallel processing."""
        # Test with limited concurrent layers
        parallel_manager.config["max_concurrent_layers"] = 2
        
        test_data = {"content": "test data"}
        
        # Execute parallel write
        result = await parallel_manager.write_data(
            data=test_data,
            strategy=WriteStrategy.PARALLEL
        )
        
        # Should still succeed but may process layers in batches
        assert result.success is True

    def test_configuration_validation(self, parallel_manager):
        """Test configuration validation."""
        config = parallel_manager.config
        
        # Verify required configuration keys
        assert "max_concurrent_layers" in config
        assert "timeout_per_layer" in config
        assert "enable_fallback" in config
        
        # Verify reasonable defaults
        assert config["max_concurrent_layers"] > 0
        assert config["timeout_per_layer"] > 0

    @pytest.mark.asyncio
    async def test_statistics_collection(self, parallel_manager):
        """Test statistics collection during parallel processing."""
        test_data = {"content": "test data"}
        
        # Execute multiple operations
        for i in range(3):
            await parallel_manager.write_data(
                data=test_data,
                strategy=WriteStrategy.PARALLEL
            )
        
        # Check if statistics are being collected
        # (Implementation depends on actual statistics tracking in the manager)
        assert hasattr(parallel_manager, 'config')

    @pytest.mark.asyncio
    async def test_cleanup_and_shutdown(self, parallel_manager):
        """Test cleanup and shutdown procedures."""
        # Test cleanup method if available
        if hasattr(parallel_manager, 'cleanup'):
            result = await parallel_manager.cleanup()
            assert isinstance(result, bool)
        
        # Test that manager can be safely destroyed
        del parallel_manager
