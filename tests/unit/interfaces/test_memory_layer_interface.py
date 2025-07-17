"""
Unit tests for MemoryLayer interface abstraction.

Tests the enhanced MemoryLayer interface and its implementation
to ensure proper Service Layer and Memory Layer decoupling.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from typing import Dict, Any, List

from src.memfuse_core.interfaces.memory_layer import (
    MemoryLayer,
    MemoryLayerConfig,
    WriteResult,
    QueryResult,
    LayerStatus
)
from src.memfuse_core.hierarchy.memory_layer_impl import MemoryLayerImpl
from src.memfuse_core.utils.config import ConfigManager


class TestMemoryLayerInterface:
    """Test suite for MemoryLayer interface abstraction."""

    def test_memory_layer_interface_completeness(self):
        """Test that MemoryLayer interface has all required methods."""
        # Check that all abstract methods are defined
        abstract_methods = MemoryLayer.__abstractmethods__
        
        expected_methods = {
            'initialize',
            'write_parallel',
            'query',
            'get_layer_status',
            'get_statistics',
            'health_check',
            'reset_layer',
            'cleanup'
        }
        
        assert abstract_methods == expected_methods, \
            f"Missing methods: {expected_methods - abstract_methods}"

    def test_write_result_structure(self):
        """Test WriteResult data structure."""
        result = WriteResult(
            success=True,
            message="Test message",
            layer_results={"M0": {"success": True}},
            metadata={"operation_time": 1.5}
        )
        
        assert result.success is True
        assert result.message == "Test message"
        assert "M0" in result.layer_results
        assert "operation_time" in result.metadata
        
        # Test to_dict conversion
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict["success"] is True

    def test_query_result_structure(self):
        """Test QueryResult data structure."""
        test_results = [
            {"content": "Test result 1", "score": 0.9},
            {"content": "Test result 2", "score": 0.8}
        ]
        
        result = QueryResult(
            results=test_results,
            query="test query",
            layer_sources={"M0": 1, "M1": 1},
            total_count=2,
            metadata={"query_time": 0.5}
        )
        
        assert len(result.results) == 2
        assert result.query == "test query"
        assert result.total_count == 2
        assert "M0" in result.layer_sources
        
        # Test to_dict conversion
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert len(result_dict["results"]) == 2

    def test_memory_layer_config(self):
        """Test MemoryLayerConfig structure and validation."""
        config = MemoryLayerConfig(
            m0_enabled=True,
            m1_enabled=True,
            m2_enabled=False,
            parallel_strategy="parallel",
            enable_fallback=True,
            timeout_per_layer=30.0,
            max_retries=3
        )
        
        assert config.m0_enabled is True
        assert config.m1_enabled is True
        assert config.m2_enabled is False
        assert config.parallel_strategy == "parallel"
        assert config.enable_fallback is True
        
        # Test to_dict conversion
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict["m0_enabled"] is True
        
        # Test from_dict creation
        new_config = MemoryLayerConfig.from_dict(config_dict)
        assert new_config.m0_enabled == config.m0_enabled
        assert new_config.parallel_strategy == config.parallel_strategy

    def test_layer_status_enum(self):
        """Test LayerStatus enum values."""
        assert LayerStatus.ACTIVE.value == "active"
        assert LayerStatus.INACTIVE.value == "inactive"
        assert LayerStatus.ERROR.value == "error"
        assert LayerStatus.INITIALIZING.value == "initializing"


class TestMemoryLayerImplementation:
    """Test MemoryLayerImpl implementation of the interface."""

    @pytest.fixture
    def config_manager(self):
        """Create mock config manager."""
        config_manager = ConfigManager()
        test_config = {
            "memory": {
                "processing": {"strategy": "parallel"},
                "layers": {
                    "m0": {"enabled": True},
                    "m1": {"enabled": True},
                    "m2": {"enabled": True}
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
            parallel_strategy="parallel"
        )

    @pytest.fixture
    async def memory_layer_impl(self, config_manager, memory_layer_config):
        """Create MemoryLayerImpl instance."""
        impl = MemoryLayerImpl(
            user_id="test_user",
            config_manager=config_manager,
            config=memory_layer_config
        )
        
        # Mock hierarchy manager to avoid database dependencies
        impl.hierarchy_manager = AsyncMock()
        impl.parallel_manager = AsyncMock()
        
        return impl

    @pytest.mark.asyncio
    async def test_interface_implementation(self, memory_layer_impl):
        """Test that MemoryLayerImpl properly implements the interface."""
        # Verify it's an instance of MemoryLayer
        assert isinstance(memory_layer_impl, MemoryLayer)
        
        # Test all interface methods exist
        assert hasattr(memory_layer_impl, 'initialize')
        assert hasattr(memory_layer_impl, 'write_parallel')
        assert hasattr(memory_layer_impl, 'query')
        assert hasattr(memory_layer_impl, 'get_layer_status')
        assert hasattr(memory_layer_impl, 'get_statistics')
        assert hasattr(memory_layer_impl, 'health_check')
        assert hasattr(memory_layer_impl, 'reset_layer')
        assert hasattr(memory_layer_impl, 'cleanup')

    @pytest.mark.asyncio
    async def test_get_statistics_method(self, memory_layer_impl):
        """Test get_statistics method implementation."""
        # Set some test statistics
        memory_layer_impl.total_operations = 10
        memory_layer_impl.successful_operations = 8
        memory_layer_impl.failed_operations = 2
        
        stats = await memory_layer_impl.get_statistics()
        
        assert isinstance(stats, dict)
        assert stats["total_operations"] == 10
        assert stats["successful_operations"] == 8
        assert stats["failed_operations"] == 2
        assert stats["success_rate"] == 0.8
        assert "layer_status" in stats
        assert "initialized" in stats

    @pytest.mark.asyncio
    async def test_health_check_method(self, memory_layer_impl):
        """Test health_check method implementation."""
        health = await memory_layer_impl.health_check()
        
        assert isinstance(health, dict)
        assert "overall_status" in health
        assert "layers" in health
        assert "components" in health
        
        # Check layer health information
        assert "M0" in health["layers"]
        assert "M1" in health["layers"]
        assert "M2" in health["layers"]
        
        # Check component health
        assert "hierarchy_manager" in health["components"]
        assert "parallel_manager" in health["components"]

    @pytest.mark.asyncio
    async def test_reset_layer_method(self, memory_layer_impl):
        """Test reset_layer method implementation."""
        # Test valid layer reset
        result = await memory_layer_impl.reset_layer("M0")
        assert isinstance(result, bool)
        
        # Test invalid layer reset
        result = await memory_layer_impl.reset_layer("INVALID")
        assert result is False

    @pytest.mark.asyncio
    async def test_cleanup_method(self, memory_layer_impl):
        """Test cleanup method implementation."""
        # Initialize first
        memory_layer_impl.initialized = True
        
        # Test cleanup
        result = await memory_layer_impl.cleanup()
        assert isinstance(result, bool)
        assert memory_layer_impl.initialized is False
        
        # Verify all layers are set to inactive
        for status in memory_layer_impl.layer_status.values():
            assert status == LayerStatus.INACTIVE


class TestServiceLayerDecoupling:
    """Test Service Layer and Memory Layer decoupling."""

    def test_memory_service_dependency(self):
        """Test that MemoryService only depends on MemoryLayer interface."""
        from src.memfuse_core.services.memory_service import MemoryService
        
        # Check imports - should only import MemoryLayer interface
        import inspect
        source = inspect.getsource(MemoryService)
        
        # Should import MemoryLayer interface
        assert "MemoryLayer" in source
        
        # Should not directly import specific layer implementations
        assert "M0RawDataLayer" not in source
        assert "M1EpisodicLayer" not in source
        assert "M2SemanticLayer" not in source

    @pytest.mark.asyncio
    async def test_interface_abstraction_level(self):
        """Test that interface provides proper abstraction level."""
        # The interface should hide M0/M1/M2 complexity from service layer
        
        # Create mock implementation
        mock_memory_layer = AsyncMock(spec=MemoryLayer)
        
        # Mock write_parallel to return proper result
        mock_result = WriteResult(
            success=True,
            message="Success",
            layer_results={"M0": {"success": True}, "M1": {"success": True}},
            metadata={"operation_time": 1.0}
        )
        mock_memory_layer.write_parallel.return_value = mock_result
        
        # Test that service layer can work with interface
        result = await mock_memory_layer.write_parallel([{"role": "user", "content": "test"}])
        
        assert isinstance(result, WriteResult)
        assert result.success is True
        assert "M0" in result.layer_results
        assert "M1" in result.layer_results

    def test_configuration_abstraction(self):
        """Test that configuration is properly abstracted."""
        config = MemoryLayerConfig()
        
        # Configuration should provide high-level controls
        assert hasattr(config, 'm0_enabled')
        assert hasattr(config, 'm1_enabled')
        assert hasattr(config, 'm2_enabled')
        assert hasattr(config, 'parallel_strategy')
        assert hasattr(config, 'enable_fallback')
        
        # Should not expose low-level implementation details
        assert not hasattr(config, 'database_connection')
        assert not hasattr(config, 'storage_backends')

    def test_error_handling_abstraction(self):
        """Test that error handling is properly abstracted."""
        # WriteResult should provide high-level error information
        error_result = WriteResult(
            success=False,
            message="Operation failed",
            layer_results={"M0": {"success": False, "error": "Database error"}},
            metadata={"error_type": "storage_error"}
        )
        
        # Service layer should be able to handle errors without knowing specifics
        assert error_result.success is False
        assert "failed" in error_result.message.lower()
        assert "error_type" in error_result.metadata
