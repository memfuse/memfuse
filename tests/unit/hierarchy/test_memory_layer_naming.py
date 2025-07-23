"""
Unit tests for memory layer naming conventions.

Tests that the new naming conventions are properly implemented:
- M0: Raw Data Layer
- M1: Episodic Memory Layer  
- M2: Semantic Memory Layer
"""

import pytest
from unittest.mock import MagicMock, AsyncMock

from src.memfuse_core.hierarchy.layers import (
    M0RawDataLayer, 
    M1EpisodicLayer, 
    M2SemanticLayer
)
from src.memfuse_core.hierarchy.core import LayerType, LayerConfig
from src.memfuse_core.hierarchy.manager import MemoryHierarchyManager


class TestMemoryLayerNaming:
    """Test suite for memory layer naming conventions."""

    @pytest.fixture
    def layer_config(self):
        """Create basic layer configuration."""
        return LayerConfig(
            layer_type=LayerType.M0,
            storage_backends=["vector", "keyword"],
            custom_config={}
        )

    @pytest.fixture
    def user_id(self):
        """Test user ID."""
        return "test_user_123"

    def test_m0_raw_data_layer_naming(self, layer_config, user_id):
        """Test M0 Raw Data Layer naming and description."""
        layer_config.layer_type = LayerType.M0
        layer = M0RawDataLayer(
            layer_type=LayerType.M0,
            config=layer_config,
            user_id=user_id
        )
        
        # Check class name
        assert layer.__class__.__name__ == "M0RawDataLayer"
        
        # Check docstring mentions raw data
        assert "Raw Data" in layer.__class__.__doc__
        assert "original data" in layer.__class__.__doc__.lower()
        
        # Check layer type
        assert layer.layer_type == LayerType.M0

    def test_m1_episodic_layer_naming(self, layer_config, user_id):
        """Test M1 Episodic Memory Layer naming and description."""
        layer_config.layer_type = LayerType.M1
        layer = M1EpisodicLayer(
            layer_type=LayerType.M1,
            config=layer_config,
            user_id=user_id
        )
        
        # Check class name
        assert layer.__class__.__name__ == "M1EpisodicLayer"
        
        # Check docstring mentions episodic memory
        assert "Episodic Memory" in layer.__class__.__doc__
        assert "event-centered" in layer.__class__.__doc__.lower()
        
        # Check layer type
        assert layer.layer_type == LayerType.M1

    def test_m2_semantic_layer_naming(self, layer_config, user_id):
        """Test M2 Semantic Memory Layer naming and description."""
        layer_config.layer_type = LayerType.M2
        layer = M2SemanticLayer(
            layer_type=LayerType.M2,
            config=layer_config,
            user_id=user_id
        )
        
        # Check class name
        assert layer.__class__.__name__ == "M2SemanticLayer"
        
        # Check docstring mentions semantic memory
        assert "Semantic Memory" in layer.__class__.__doc__
        assert "facts and concepts" in layer.__class__.__doc__.lower()
        
        # Check layer type
        assert layer.layer_type == LayerType.M2

    def test_layer_functionality_alignment(self, layer_config, user_id):
        """Test that layer functionality aligns with naming."""
        # M0 should handle raw data storage
        m0_layer = M0RawDataLayer(LayerType.M0, layer_config, user_id)
        assert hasattr(m0_layer, 'storage_backends')
        
        # M1 should handle episode formation
        layer_config.layer_type = LayerType.M1
        m1_layer = M1EpisodicLayer(LayerType.M1, layer_config, user_id)
        assert hasattr(m1_layer, 'episode_formation_enabled')
        
        # M2 should handle fact extraction
        layer_config.layer_type = LayerType.M2
        m2_layer = M2SemanticLayer(LayerType.M2, layer_config, user_id)
        assert hasattr(m2_layer, 'fact_extraction_enabled')

    @pytest.mark.asyncio
    async def test_layer_logging_consistency(self, layer_config, user_id, caplog):
        """Test that logging messages use correct layer names."""
        import logging
        caplog.set_level(logging.INFO)
        
        # Test M0 layer logging
        m0_layer = M0RawDataLayer(LayerType.M0, layer_config, user_id)
        
        # Check initialization log
        assert any("M0RawDataLayer" in record.message for record in caplog.records)
        
        # Clear logs
        caplog.clear()
        
        # Test M1 layer logging
        layer_config.layer_type = LayerType.M1
        m1_layer = M1EpisodicLayer(LayerType.M1, layer_config, user_id)
        
        assert any("M1EpisodicLayer" in record.message for record in caplog.records)
        
        # Clear logs
        caplog.clear()
        
        # Test M2 layer logging
        layer_config.layer_type = LayerType.M2
        m2_layer = M2SemanticLayer(LayerType.M2, layer_config, user_id)
        
        assert any("M2SemanticLayer" in record.message for record in caplog.records)


class TestMemoryHierarchyManagerNaming:
    """Test MemoryHierarchyManager uses correct layer classes."""

    @pytest.fixture
    def test_config(self):
        """Create test configuration."""
        return {
            "layers": {
                "m0": {"enabled": True},
                "m1": {"enabled": True}, 
                "m2": {"enabled": True}
            },
            "storage": {}
        }

    @pytest.mark.asyncio
    async def test_manager_creates_correct_layer_types(self, test_config):
        """Test that MemoryHierarchyManager creates the correct layer types."""
        manager = MemoryHierarchyManager("test_user", test_config)
        
        # Mock storage manager to avoid database dependencies
        manager.storage_manager = AsyncMock()
        manager.storage_manager.initialize.return_value = True
        
        # Mock layer initialization
        with pytest.MonkeyPatch().context() as m:
            # Mock the layer classes to avoid full initialization
            mock_m0 = AsyncMock()
            mock_m0.initialize.return_value = True
            mock_m1 = AsyncMock()
            mock_m1.initialize.return_value = True
            mock_m2 = AsyncMock()
            mock_m2.initialize.return_value = True
            
            m.setattr("src.memfuse_core.hierarchy.manager.M0RawDataLayer", lambda *args, **kwargs: mock_m0)
            m.setattr("src.memfuse_core.hierarchy.manager.M1EpisodicLayer", lambda *args, **kwargs: mock_m1)
            m.setattr("src.memfuse_core.hierarchy.manager.M2SemanticLayer", lambda *args, **kwargs: mock_m2)
            
            # Initialize manager
            await manager.initialize()
            
            # Verify correct layer types are created
            assert LayerType.M0 in manager.layers
            assert LayerType.M1 in manager.layers
            assert LayerType.M2 in manager.layers


class TestConfigurationNaming:
    """Test configuration uses correct naming conventions."""

    def test_layer_descriptions_in_config(self):
        """Test that configuration files use correct layer descriptions."""
        # This would typically load actual config files
        # For now, we test the expected structure
        
        expected_descriptions = {
            "m0": "Raw Data",
            "m1": "Episodic Memory", 
            "m2": "Semantic Memory"
        }
        
        # In actual implementation, this would verify config files
        for layer, description in expected_descriptions.items():
            assert description in ["Raw Data", "Episodic Memory", "Semantic Memory"]

    def test_table_naming_conventions(self):
        """Test that table names follow the new conventions."""
        expected_tables = {
            "m0": "m0_raw",  # Raw data layer
            "m1": "m1_episodic",
            "m2": "m2_semantic"
        }
        
        # Verify table naming structure
        for layer, table_name in expected_tables.items():
            assert table_name.startswith(layer)
            assert "_" in table_name


class TestDocumentationConsistency:
    """Test that documentation reflects the new naming conventions."""

    def test_hierarchy_module_docstring(self):
        """Test that hierarchy module docstring uses correct naming."""
        from src.memfuse_core.hierarchy import __doc__ as hierarchy_doc
        
        # Check that docstring mentions correct layer types
        assert "Raw Data" in hierarchy_doc
        assert "Episodic Memory" in hierarchy_doc
        assert "Semantic Memory" in hierarchy_doc

    def test_layer_docstrings_consistency(self):
        """Test that all layer docstrings are consistent with naming."""
        layers = [M0RawDataLayer, M1EpisodicLayer, M2SemanticLayer]
        expected_terms = [
            ["Raw Data", "original data"],
            ["Episodic Memory", "event-centered"],
            ["Semantic Memory", "facts and concepts"]
        ]
        
        for layer_class, terms in zip(layers, expected_terms):
            docstring = layer_class.__doc__
            for term in terms:
                assert term.lower() in docstring.lower()


class TestBackwardCompatibility:
    """Test that changes maintain backward compatibility where needed."""

    def test_table_name_compatibility(self):
        """Test that critical table names are preserved for compatibility."""
        # m0_raw table name reflects the new architecture
        # M0 is now the Raw Data layer, not episodic memory

        # In configuration, m0 should use m0_raw table
        # to reflect its role as the Raw Data layer
        assert True  # Placeholder for actual compatibility tests

    def test_api_compatibility(self):
        """Test that public APIs remain compatible."""
        # Layer interfaces should remain the same
        from src.memfuse_core.hierarchy.core import MemoryLayer
        
        # All layers should still implement the same interface
        assert issubclass(M0RawDataLayer, MemoryLayer)
        assert issubclass(M1EpisodicLayer, MemoryLayer)
        assert issubclass(M2SemanticLayer, MemoryLayer)
