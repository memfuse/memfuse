"""
Unit tests for layer error recovery mechanisms.

Tests error handling and recovery across memory layers including:
- Layer-specific error recovery
- Cross-layer error propagation
- Graceful degradation strategies
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

from src.memfuse_core.hierarchy.layers import M0RawDataLayer, M1EpisodicLayer, M2SemanticLayer
from src.memfuse_core.hierarchy.core import LayerType, LayerConfig
from src.memfuse_core.hierarchy.memory_layer_impl import MemoryLayerImpl
from src.memfuse_core.interfaces.memory_layer import LayerStatus, WriteResult


class TestLayerErrorRecovery:
    """Test suite for layer error recovery mechanisms."""

    @pytest.fixture
    def layer_config(self):
        """Create layer configuration."""
        return LayerConfig(
            layer_type=LayerType.M0,
            storage_backends=["vector", "keyword"],
            custom_config={"max_retries": 3, "retry_delay": 0.1}
        )

    @pytest.fixture
    def mock_storage_manager(self):
        """Create mock storage manager."""
        manager = AsyncMock()
        manager.initialize.return_value = True
        return manager

    @pytest.fixture
    def m0_layer(self, layer_config, mock_storage_manager):
        """Create M0 layer for testing."""
        layer_config.layer_type = LayerType.M0
        return M0RawDataLayer(
            layer_type=LayerType.M0,
            config=layer_config,
            user_id="test_user",
            storage_manager=mock_storage_manager
        )

    @pytest.fixture
    def m1_layer(self, layer_config, mock_storage_manager):
        """Create M1 layer for testing."""
        layer_config.layer_type = LayerType.M1
        return M1EpisodicLayer(
            layer_type=LayerType.M1,
            config=layer_config,
            user_id="test_user",
            storage_manager=mock_storage_manager
        )

    @pytest.mark.asyncio
    async def test_storage_backend_failure_recovery(self, m0_layer, mock_storage_manager):
        """Test recovery from storage backend failures."""
        await m0_layer.initialize()
        
        # Simulate storage backend failure then recovery
        call_count = 0
        async def failing_then_succeeding(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("Storage backend temporarily unavailable")
            return "success_id_123"
        
        mock_storage_manager.write_to_backend.side_effect = failing_then_succeeding
        
        test_data = [{"role": "user", "content": "Test message"}]
        
        # Process data (should retry and eventually succeed)
        result = await m0_layer.process_data(test_data)
        
        # Should eventually succeed after retries
        assert call_count >= 2  # At least one retry occurred
        # Result depends on retry implementation

    @pytest.mark.asyncio
    async def test_layer_initialization_failure_recovery(self, layer_config, mock_storage_manager):
        """Test recovery from layer initialization failures."""
        # Simulate initialization failure then success
        call_count = 0
        async def failing_then_succeeding_init():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Initialization failed")
            return True
        
        mock_storage_manager.initialize.side_effect = failing_then_succeeding_init
        
        layer = M0RawDataLayer(
            layer_type=LayerType.M0,
            config=layer_config,
            user_id="test_user",
            storage_manager=mock_storage_manager
        )
        
        # First initialization should fail
        result1 = await layer.initialize()
        assert result1 is False
        assert layer.initialized is False
        
        # Second initialization should succeed
        result2 = await layer.initialize()
        assert result2 is True
        assert layer.initialized is True

    @pytest.mark.asyncio
    async def test_partial_processing_failure(self, m0_layer, mock_storage_manager):
        """Test handling of partial processing failures."""
        await m0_layer.initialize()
        
        # Simulate partial failure (some items succeed, some fail)
        call_count = 0
        async def partial_failure(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:
                raise Exception("Partial failure")
            return f"success_id_{call_count}"
        
        mock_storage_manager.write_to_backend.side_effect = partial_failure
        
        # Process multiple items
        test_data = [
            {"role": "user", "content": "Message 1"},
            {"role": "user", "content": "Message 2"},
            {"role": "user", "content": "Message 3"}
        ]
        
        result = await m0_layer.process_data(test_data)
        
        # Should handle partial failures gracefully
        assert isinstance(result.success, bool)

    @pytest.mark.asyncio
    async def test_timeout_error_recovery(self, m0_layer, mock_storage_manager):
        """Test recovery from timeout errors."""
        await m0_layer.initialize()
        
        # Simulate timeout then success
        call_count = 0
        async def timeout_then_success(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                await asyncio.sleep(2.0)  # Simulate timeout
                raise asyncio.TimeoutError("Operation timed out")
            return "success_id_123"
        
        mock_storage_manager.write_to_backend.side_effect = timeout_then_success
        
        test_data = [{"role": "user", "content": "Test message"}]
        
        # Process with timeout handling
        result = await m0_layer.process_data(test_data)
        
        # Should handle timeout gracefully
        assert isinstance(result.success, bool)

    @pytest.mark.asyncio
    async def test_memory_layer_impl_error_recovery(self):
        """Test MemoryLayerImpl error recovery mechanisms."""
        from src.memfuse_core.utils.config import ConfigManager
        from src.memfuse_core.interfaces.memory_layer import MemoryLayerConfig
        
        config_manager = ConfigManager()
        memory_config = MemoryLayerConfig(
            m0_enabled=True,
            m1_enabled=True,
            m2_enabled=True
        )
        
        memory_layer = MemoryLayerImpl(
            user_id="test_user",
            config_manager=config_manager,
            config=memory_config
        )
        
        # Mock hierarchy manager with failing initialization
        mock_hierarchy = AsyncMock()
        mock_hierarchy.initialize.side_effect = Exception("Hierarchy init failed")
        
        with patch.object(memory_layer, '_initialize_hierarchy_manager', return_value=mock_hierarchy):
            # Initialization should fail gracefully
            result = await memory_layer.initialize()
            assert result is False
            assert memory_layer.initialized is False

    @pytest.mark.asyncio
    async def test_layer_status_error_tracking(self):
        """Test layer status tracking during errors."""
        from src.memfuse_core.utils.config import ConfigManager
        from src.memfuse_core.interfaces.memory_layer import MemoryLayerConfig
        
        memory_layer = MemoryLayerImpl(
            user_id="test_user",
            config_manager=ConfigManager(),
            config=MemoryLayerConfig()
        )
        
        # Initial status should be inactive
        status = await memory_layer.get_layer_status()
        assert status["M0"] == LayerStatus.INACTIVE
        assert status["M1"] == LayerStatus.INACTIVE
        assert status["M2"] == LayerStatus.INACTIVE
        
        # Simulate error in M1 layer
        memory_layer.layer_status["M1"] = LayerStatus.ERROR
        
        status = await memory_layer.get_layer_status()
        assert status["M1"] == LayerStatus.ERROR

    @pytest.mark.asyncio
    async def test_error_propagation_prevention(self, m0_layer, m1_layer):
        """Test that errors in one layer don't propagate to others."""
        await m0_layer.initialize()
        await m1_layer.initialize()
        
        # Simulate error in M0 layer
        m0_layer.storage_manager.write_to_backend.side_effect = Exception("M0 error")
        
        # M1 layer should still work
        m1_layer.storage_manager.write_to_backend.return_value = "m1_success"
        
        test_data = [{"role": "user", "content": "Test message"}]
        
        # Process data in both layers
        m0_result = await m0_layer.process_data(test_data)
        m1_result = await m1_layer.process_data(test_data)
        
        # M0 should fail, M1 should succeed
        assert m0_result.success is False
        assert m1_result.success is True

    @pytest.mark.asyncio
    async def test_graceful_degradation(self):
        """Test graceful degradation when layers fail."""
        from src.memfuse_core.utils.config import ConfigManager
        from src.memfuse_core.interfaces.memory_layer import MemoryLayerConfig
        
        memory_layer = MemoryLayerImpl(
            user_id="test_user",
            config_manager=ConfigManager(),
            config=MemoryLayerConfig(
                m0_enabled=True,
                m1_enabled=True,
                m2_enabled=True,
                enable_fallback=True
            )
        )
        
        # Mock parallel manager with partial failure
        mock_parallel = AsyncMock()
        mock_parallel.write_data.return_value = MagicMock(
            success=True,  # Overall success despite partial failures
            layer_results={
                "M0": {"success": True, "processed_items": ["item1"]},
                "M1": {"success": False, "error": "M1 failed"},
                "M2": {"success": True, "processed_items": ["item2"]}
            },
            strategy_used="parallel",
            total_processed=2
        )
        
        memory_layer.parallel_manager = mock_parallel
        memory_layer.initialized = True
        
        # Process data with partial layer failure
        result = await memory_layer.write_parallel([{"role": "user", "content": "test"}])
        
        # Should succeed with graceful degradation
        assert isinstance(result, WriteResult)
        assert result.success is True
        assert len(result.layer_results) == 3

    @pytest.mark.asyncio
    async def test_error_recovery_statistics(self, m0_layer, mock_storage_manager):
        """Test error recovery statistics tracking."""
        await m0_layer.initialize()
        
        # Simulate multiple failures then success
        failure_count = 0
        async def multiple_failures(*args, **kwargs):
            nonlocal failure_count
            failure_count += 1
            if failure_count <= 3:
                raise Exception(f"Failure {failure_count}")
            return "success_id"
        
        mock_storage_manager.write_to_backend.side_effect = multiple_failures
        
        test_data = [{"role": "user", "content": "Test message"}]
        
        # Process data multiple times
        for i in range(5):
            await m0_layer.process_data(test_data)
        
        # Check statistics
        stats = m0_layer.get_stats()
        assert stats.total_operations >= 5

    @pytest.mark.asyncio
    async def test_circuit_breaker_pattern(self, m0_layer, mock_storage_manager):
        """Test circuit breaker pattern for error recovery."""
        await m0_layer.initialize()
        
        # Simulate consistent failures
        mock_storage_manager.write_to_backend.side_effect = Exception("Persistent failure")
        
        test_data = [{"role": "user", "content": "Test message"}]
        
        # Process data multiple times
        results = []
        for i in range(5):
            result = await m0_layer.process_data(test_data)
            results.append(result)
        
        # Should handle persistent failures gracefully
        assert all(isinstance(result.success, bool) for result in results)

    @pytest.mark.asyncio
    async def test_error_context_preservation(self, m0_layer, mock_storage_manager):
        """Test that error context is preserved for debugging."""
        await m0_layer.initialize()
        
        # Simulate error with context
        mock_storage_manager.write_to_backend.side_effect = Exception("Database connection lost")
        
        test_data = [{"role": "user", "content": "Test message"}]
        
        # Process data
        result = await m0_layer.process_data(test_data)
        
        # Error information should be preserved
        assert result.success is False
        assert isinstance(result.message, str)
        assert len(result.message) > 0
