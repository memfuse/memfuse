"""Tests for M3 configuration management."""

import pytest
import os
import tempfile
import yaml
from unittest.mock import patch

from src.memfuse_core.m3.config import M3Config, M3ConfigManager, get_m3_config


class TestM3Config:
    """Test cases for M3Config."""

    def test_default_config(self):
        """Test default configuration values."""
        config = M3Config()
        
        assert config.workflow_reuse_threshold == 0.9
        assert config.max_workflow_reuse_candidates == 5
        assert config.default_agent_timeout == 300
        assert config.max_agent_retries == 3
        assert config.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.embedding_dim == 384
        assert config.enable_workflow_reuse is True
        assert config.enable_lesson_learning is True
        assert isinstance(config.agent_configs, dict)

    def test_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "workflow_reuse_threshold": 0.8,
            "max_workflow_reuse_candidates": 10,
            "default_agent_timeout": 600,
            "embedding_model": "custom-model",
            "enable_workflow_reuse": False
        }
        
        config = M3Config.from_dict(config_dict)
        
        assert config.workflow_reuse_threshold == 0.8
        assert config.max_workflow_reuse_candidates == 10
        assert config.default_agent_timeout == 600
        assert config.embedding_model == "custom-model"
        assert config.enable_workflow_reuse is False
        # Defaults should still be set for unspecified values
        assert config.max_agent_retries == 3

    def test_from_yaml(self):
        """Test creating config from YAML file."""
        config_data = {
            "m3": {
                "workflow_reuse_threshold": 0.7,
                "max_workflow_reuse_candidates": 8,
                "embedding_model": "yaml-model"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            yaml_path = f.name
        
        try:
            config = M3Config.from_yaml(yaml_path)
            
            assert config.workflow_reuse_threshold == 0.7
            assert config.max_workflow_reuse_candidates == 8
            assert config.embedding_model == "yaml-model"
            # Defaults should still be set
            assert config.max_agent_retries == 3
        finally:
            os.unlink(yaml_path)

    def test_from_yaml_missing_file(self):
        """Test creating config from missing YAML file returns default."""
        config = M3Config.from_yaml("nonexistent.yaml")
        
        # Should return default config
        assert config.workflow_reuse_threshold == 0.9
        assert config.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"

    def test_from_env(self):
        """Test creating config from environment variables."""
        env_vars = {
            "M3_WORKFLOW_REUSE_THRESHOLD": "0.85",
            "M3_AGENT_TIMEOUT": "450",
            "M3_EMBEDDING_MODEL": "env-model",
            "M3_ENABLE_WORKFLOW_REUSE": "false",
            "M3_ENABLE_LESSON_LEARNING": "true"
        }
        
        with patch.dict(os.environ, env_vars):
            config = M3Config.from_env()
            
            assert config.workflow_reuse_threshold == 0.85
            assert config.default_agent_timeout == 450
            assert config.embedding_model == "env-model"
            assert config.enable_workflow_reuse is False
            assert config.enable_lesson_learning is True

    def test_from_env_invalid_values(self):
        """Test creating config from environment with invalid values."""
        env_vars = {
            "M3_WORKFLOW_REUSE_THRESHOLD": "invalid",
            "M3_AGENT_TIMEOUT": "not_a_number",
            "M3_ENABLE_WORKFLOW_REUSE": "maybe"
        }
        
        with patch.dict(os.environ, env_vars):
            config = M3Config.from_env()
            
            # Should fall back to defaults for invalid values
            assert config.workflow_reuse_threshold == 0.9
            assert config.default_agent_timeout == 300
            assert config.enable_workflow_reuse is True

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = M3Config(
            workflow_reuse_threshold=0.8,
            embedding_model="test-model"
        )
        
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict["workflow_reuse_threshold"] == 0.8
        assert config_dict["embedding_model"] == "test-model"
        assert "agent_configs" in config_dict


class TestM3ConfigManager:
    """Test cases for M3ConfigManager."""

    def test_singleton_pattern(self):
        """Test that M3ConfigManager follows singleton pattern."""
        manager1 = M3ConfigManager.get_instance()
        manager2 = M3ConfigManager.get_instance()
        
        assert manager1 is manager2

    def test_get_config_returns_m3config(self):
        """Test that get_config returns M3Config instance."""
        manager = M3ConfigManager()
        config = manager.get_config()
        
        assert isinstance(config, M3Config)

    def test_reload_config(self):
        """Test config reloading."""
        manager = M3ConfigManager()
        
        # Get initial config
        config1 = manager.get_config()
        
        # Reload config
        config2 = manager.reload_config()
        
        assert isinstance(config2, M3Config)
        # Should be a fresh instance
        assert config2 is manager._config

    def test_update_config(self):
        """Test updating configuration values."""
        manager = M3ConfigManager()
        
        # Update some values
        manager.update_config(
            workflow_reuse_threshold=0.75,
            max_workflow_reuse_candidates=15
        )
        
        config = manager.get_config()
        assert config.workflow_reuse_threshold == 0.75
        assert config.max_workflow_reuse_candidates == 15

    def test_update_config_invalid_key(self):
        """Test updating config with invalid key."""
        manager = M3ConfigManager()
        
        # Should not raise exception, just log warning
        manager.update_config(invalid_key="value")
        
        # Config should still be valid
        config = manager.get_config()
        assert isinstance(config, M3Config)

    def test_load_config_with_yaml_file(self):
        """Test loading config when YAML file exists."""
        config_data = {
            "m3": {
                "workflow_reuse_threshold": 0.6,
                "embedding_model": "file-model"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            yaml_path = f.name
        
        try:
            manager = M3ConfigManager()
            manager._config_paths = [yaml_path]  # Override config paths
            
            config = manager._load_config()
            
            assert config.workflow_reuse_threshold == 0.6
            assert config.embedding_model == "file-model"
        finally:
            os.unlink(yaml_path)

    def test_load_config_with_env_override(self):
        """Test loading config with environment variable override."""
        config_data = {
            "m3": {
                "workflow_reuse_threshold": 0.6,
                "embedding_model": "file-model"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            yaml_path = f.name
        
        env_vars = {
            "M3_WORKFLOW_REUSE_THRESHOLD": "0.95",
            "M3_EMBEDDING_MODEL": "env-override-model"
        }
        
        try:
            with patch.dict(os.environ, env_vars):
                manager = M3ConfigManager()
                manager._config_paths = [yaml_path]
                
                config = manager._load_config()
                
                # Environment should override file values
                assert config.workflow_reuse_threshold == 0.95
                assert config.embedding_model == "env-override-model"
        finally:
            os.unlink(yaml_path)


def test_get_m3_config():
    """Test global get_m3_config function."""
    config = get_m3_config()
    
    assert isinstance(config, M3Config)
    
    # Should return same instance on subsequent calls
    config2 = get_m3_config()
    assert config is config2