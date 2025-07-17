"""
Unit tests for configuration file structure optimization.

Tests that configuration files are properly structured and aligned
with the new memory layer naming conventions and architecture.
"""

import pytest
import yaml
import os
from pathlib import Path
from typing import Dict, Any

from src.memfuse_core.utils.config import ConfigManager


class TestConfigurationOptimization:
    """Test suite for configuration optimization."""

    @pytest.fixture
    def config_dir(self):
        """Get configuration directory path."""
        return Path(__file__).parent.parent.parent.parent / "config"

    @pytest.fixture
    def config_manager(self):
        """Create config manager for testing."""
        return ConfigManager()

    def test_main_config_structure(self, config_dir):
        """Test main configuration file structure."""
        config_file = config_dir / "config.yaml"
        assert config_file.exists(), "Main config file should exist"
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check that defaults include all required modules
        assert "defaults" in config
        defaults = config["defaults"]
        
        required_modules = [
            "server", "store", "embedding", "buffer", 
            "retrieval", "memory", "database"
        ]
        
        for module in required_modules:
            assert any(module in str(default) for default in defaults), \
                f"Module {module} should be in defaults"

    def test_memory_config_structure(self, config_dir):
        """Test memory configuration structure."""
        memory_config_file = config_dir / "memory" / "default.yaml"
        assert memory_config_file.exists(), "Memory config file should exist"
        
        with open(memory_config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Test core architecture settings
        assert "processing" in config
        assert "layers" in config
        assert "memory_service" in config
        
        # Test processing configuration
        processing = config["processing"]
        assert processing["strategy"] in ["parallel", "sequential"]
        assert isinstance(processing["enable_fallback"], bool)
        assert processing["max_concurrent_layers"] > 0
        
        # Test layer configuration
        layers = config["layers"]
        assert "m0" in layers
        assert "m1" in layers
        assert "m2" in layers
        
        # Test each layer has required fields
        for layer_name in ["m0", "m1", "m2"]:
            layer = layers[layer_name]
            assert "enabled" in layer
            assert "priority" in layer
            assert "storage" in layer

    def test_layer_naming_consistency(self, config_dir):
        """Test that layer naming is consistent across configurations."""
        # Test memory config
        memory_config_file = config_dir / "memory" / "default.yaml"
        with open(memory_config_file, 'r') as f:
            memory_config = yaml.safe_load(f)
        
        # Test pgai config
        pgai_config_file = config_dir / "store" / "pgai.yaml"
        with open(pgai_config_file, 'r') as f:
            pgai_config = yaml.safe_load(f)
        
        # Check layer consistency between configs
        memory_layers = set(memory_config["layers"].keys())
        pgai_memory_layers = set(pgai_config["memory_layers"].keys())
        
        # M0 and M1 should be consistent
        assert "m0" in memory_layers
        assert "m1" in memory_layers
        assert "m0" in pgai_memory_layers
        assert "m1" in pgai_memory_layers

    def test_table_naming_conventions(self, config_dir):
        """Test table naming conventions."""
        pgai_config_file = config_dir / "store" / "pgai.yaml"
        with open(pgai_config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        tables = config["tables"]
        
        # Test M0 table naming (compatibility preserved)
        assert tables["m0"]["messages"] == "m0_episodic"
        
        # Test M1 table naming (new episodic memory layer)
        assert tables["m1"]["episodes"] == "m1_episodic"
        
        # Test M2 table naming (semantic memory layer)
        if "m2" in tables:
            assert tables["m2"]["facts"] == "m2_semantic"

    def test_parallel_processing_configuration(self, config_dir):
        """Test parallel processing configuration."""
        memory_config_file = config_dir / "memory" / "default.yaml"
        with open(memory_config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Test memory service parallel configuration
        memory_service = config["memory_service"]
        assert memory_service["parallel_enabled"] is True
        assert memory_service["parallel_strategy"] == "parallel"
        
        # Test processing strategy
        processing = config["processing"]
        assert processing["strategy"] == "parallel"
        assert processing["enable_fallback"] is True

    def test_pgai_integration_configuration(self, config_dir):
        """Test PgAI integration configuration."""
        pgai_config_file = config_dir / "store" / "pgai.yaml"
        with open(pgai_config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Test global PgAI settings
        assert config["enabled"] is True
        assert config["auto_embedding"] is True
        
        # Test memory layer PgAI settings
        memory_layers = config["memory_layers"]
        
        for layer_name in ["m0", "m1"]:
            if layer_name in memory_layers:
                layer = memory_layers[layer_name]
                pgai_settings = layer["pgai"]
                
                assert pgai_settings["auto_embedding"] is True
                assert pgai_settings["immediate_trigger"] is True
                assert "embedding_model" in pgai_settings
                assert "embedding_dimensions" in pgai_settings

    def test_configuration_validation_with_config_manager(self, config_manager):
        """Test configuration validation using ConfigManager."""
        # Load configuration
        config = config_manager.get_config()
        
        # Test that memory configuration is loaded
        assert "memory" in config
        memory_config = config["memory"]
        
        # Test layer configuration
        assert "layers" in memory_config
        layers = memory_config["layers"]
        
        # Test M0 layer (Raw Data)
        assert "m0" in layers
        m0_config = layers["m0"]
        assert m0_config["enabled"] is True
        assert m0_config["priority"] == 1
        
        # Test M1 layer (Episodic Memory)
        assert "m1" in layers
        m1_config = layers["m1"]
        assert m1_config["enabled"] is True
        assert m1_config["priority"] == 2
        
        # Test M2 layer (Semantic Memory)
        assert "m2" in layers
        m2_config = layers["m2"]
        assert m2_config["enabled"] is True
        assert m2_config["priority"] == 3

    def test_layer_functionality_alignment(self, config_dir):
        """Test that layer configuration aligns with functionality."""
        memory_config_file = config_dir / "memory" / "default.yaml"
        with open(memory_config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        layers = config["layers"]
        
        # M0 (Raw Data) should have storage backends
        m0_config = layers["m0"]
        assert "storage_backends" in m0_config
        assert "vector" in m0_config["storage_backends"]
        
        # M1 (Episodic Memory) should have episode formation settings
        m1_config = layers["m1"]
        assert "episode_formation_enabled" in m1_config
        
        # M2 (Semantic Memory) should have fact extraction settings
        m2_config = layers["m2"]
        assert "fact_extraction_enabled" in m2_config

    def test_backward_compatibility_preservation(self, config_dir):
        """Test that backward compatibility is preserved where needed."""
        pgai_config_file = config_dir / "store" / "pgai.yaml"
        with open(pgai_config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # M0 should still use m0_episodic table for compatibility
        tables = config["tables"]
        assert tables["m0"]["messages"] == "m0_episodic"
        
        # Memory layer configuration should preserve table name
        memory_layers = config["memory_layers"]
        assert memory_layers["m0"]["table_name"] == "m0_episodic"

    def test_configuration_documentation(self, config_dir):
        """Test that configuration files have proper documentation."""
        memory_config_file = config_dir / "memory" / "default.yaml"
        
        with open(memory_config_file, 'r') as f:
            content = f.read()
        
        # Check for updated architecture documentation
        assert "Raw Data Layer" in content
        assert "Episodic Memory Layer" in content
        assert "Semantic Memory Layer" in content
        
        # Check for processing strategy documentation
        assert "parallel" in content
        assert "sequential" in content

    def test_environment_specific_configurations(self, config_dir):
        """Test environment-specific configuration handling."""
        # Test that default configurations exist
        assert (config_dir / "memory" / "default.yaml").exists()
        assert (config_dir / "store" / "default.yaml").exists()
        assert (config_dir / "store" / "pgai.yaml").exists()
        
        # Test configuration structure allows for environment overrides
        memory_config_file = config_dir / "memory" / "default.yaml"
        with open(memory_config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Configuration should be structured to allow easy overrides
        assert isinstance(config, dict)
        assert "processing" in config
        assert "layers" in config


class TestConfigurationIntegration:
    """Test configuration integration with actual components."""

    def test_config_manager_integration(self):
        """Test ConfigManager integration with optimized configurations."""
        config_manager = ConfigManager()
        config = config_manager.get_config()
        
        # Test that memory configuration is properly loaded
        assert "memory" in config
        memory_config = config["memory"]
        
        # Test parallel processing configuration
        assert memory_config["processing"]["strategy"] == "parallel"
        assert memory_config["memory_service"]["parallel_enabled"] is True
        
        # Test layer configuration
        layers = memory_config["layers"]
        assert all(layer in layers for layer in ["m0", "m1", "m2"])

    def test_configuration_validation_rules(self):
        """Test configuration validation rules."""
        # Test that required fields are present
        config_manager = ConfigManager()
        config = config_manager.get_config()
        
        memory_config = config["memory"]
        
        # Test processing configuration validation
        processing = memory_config["processing"]
        assert processing["strategy"] in ["parallel", "sequential"]
        assert isinstance(processing["enable_fallback"], bool)
        assert processing["max_concurrent_layers"] > 0
        
        # Test layer configuration validation
        for layer_name in ["m0", "m1", "m2"]:
            layer = memory_config["layers"][layer_name]
            assert isinstance(layer["enabled"], bool)
            assert isinstance(layer["priority"], int)
            assert layer["priority"] > 0
