"""Unit tests for unified configuration architecture."""

import pytest
from unittest.mock import patch, MagicMock
from typing import Dict, Any

from src.memfuse_core.hierarchy.storage import UnifiedStorageManager
from src.memfuse_core.hierarchy.core import StorageType
from src.memfuse_core.utils.config import ConfigManager


class TestUnifiedConfiguration:
    """Test cases for unified configuration architecture."""
    
    @pytest.fixture
    def mock_unified_config(self):
        """Mock unified configuration with pgai settings."""
        return {
            "store": {
                "backend": "pgai",
                "buffer_size": 10,
                "cache_size": 100,
                "top_k": 5,
                "pgai": {
                    "enabled": True,
                    "vectorizer_worker_enabled": True,
                    "auto_embedding": True,
                    "embedding": {
                        "model": "text-embedding-3-small",
                        "dimensions": 1536,
                        "batch_size": 100
                    },
                    "database": {
                        "host": "localhost",
                        "port": 5432,
                        "pool_size": 10
                    },
                    "storage_backends": {
                        "vector": {
                            "connection_pool_size": 5,
                            "timeout": 30.0,
                            "batch_size": 100,
                            "backend": "pgai"
                        },
                        "keyword": {
                            "connection_pool_size": 3,
                            "timeout": 15.0,
                            "backend": "sqlite"
                        },
                        "sql": {
                            "connection_pool_size": 5,
                            "timeout": 30.0,
                            "backend": "pgai"
                        }
                    }
                }
            },
            "memory": {
                "storage": {
                    "vector": {
                        "connection_pool_size": 5,
                        "timeout": 30.0,
                        "batch_size": 100
                    },
                    "keyword": {
                        "connection_pool_size": 3,
                        "timeout": 15.0,
                        "backend": "sqlite"
                    }
                }
            }
        }
    
    @pytest.fixture
    def storage_manager(self, mock_unified_config):
        """Create UnifiedStorageManager with mock config."""
        with patch('src.memfuse_core.utils.config.config_manager') as mock_config_manager:
            mock_config_manager.get_config.return_value = mock_unified_config
            
            manager = UnifiedStorageManager(
                user_id="test-user",
                config=mock_unified_config["memory"]["storage"]
            )
            return manager
    
    def test_get_unified_backend_config_vector_storage(self, storage_manager, mock_unified_config):
        """Test unified config for vector storage."""
        layer_config = {"custom_setting": "value"}
        
        with patch('src.memfuse_core.utils.config.config_manager') as mock_config_manager:
            mock_config_manager.get_config.return_value = mock_unified_config
            
            unified_config = storage_manager._get_unified_backend_config(
                StorageType.VECTOR, 
                layer_config
            )
            
            # Should inherit pgai backend from store config
            assert unified_config["backend"] == "pgai"
            
            # Should include pgai-specific vector settings
            assert unified_config["connection_pool_size"] == 5
            assert unified_config["timeout"] == 30.0
            assert unified_config["batch_size"] == 100
            
            # Should include layer-specific overrides
            assert unified_config["custom_setting"] == "value"
    
    def test_get_unified_backend_config_keyword_storage(self, storage_manager, mock_unified_config):
        """Test unified config for keyword storage."""
        layer_config = {}
        
        with patch('src.memfuse_core.utils.config.config_manager') as mock_config_manager:
            mock_config_manager.get_config.return_value = mock_unified_config
            
            unified_config = storage_manager._get_unified_backend_config(
                StorageType.KEYWORD, 
                layer_config
            )
            
            # Should use sqlite backend for keyword storage
            assert unified_config["backend"] == "sqlite"
            assert unified_config["connection_pool_size"] == 3
            assert unified_config["timeout"] == 15.0
    
    def test_get_unified_backend_config_sql_storage(self, storage_manager, mock_unified_config):
        """Test unified config for SQL storage."""
        layer_config = {}
        
        with patch('src.memfuse_core.utils.config.config_manager') as mock_config_manager:
            mock_config_manager.get_config.return_value = mock_unified_config
            
            unified_config = storage_manager._get_unified_backend_config(
                StorageType.SQL, 
                layer_config
            )
            
            # Should inherit pgai backend for SQL storage
            assert unified_config["backend"] == "pgai"
            assert unified_config["connection_pool_size"] == 5
            assert unified_config["timeout"] == 30.0
    
    def test_configuration_hierarchy_precedence(self, storage_manager, mock_unified_config):
        """Test that configuration hierarchy works correctly."""
        # Layer config should override store config but keep pgai backend for storage-specific config
        layer_config = {
            "connection_pool_size": 99,
            "custom_layer_setting": "layer_value"
        }

        with patch('src.memfuse_core.utils.config.config_manager') as mock_config_manager:
            mock_config_manager.get_config.return_value = mock_unified_config

            unified_config = storage_manager._get_unified_backend_config(
                StorageType.VECTOR,
                layer_config
            )

            # Layer config should take precedence
            assert unified_config["connection_pool_size"] == 99
            assert unified_config["custom_layer_setting"] == "layer_value"

            # Store config should still be present for non-overridden values
            assert unified_config["backend"] == "pgai"  # From store config
            assert unified_config["timeout"] == 30.0  # From store pgai vector config
            assert unified_config["batch_size"] == 100  # From store pgai vector config
    
    def test_fallback_to_default_backend(self, storage_manager):
        """Test fallback to default backend when pgai not configured."""
        minimal_config = {
            "store": {
                "backend": "qdrant",
                "buffer_size": 10
            },
            "memory": {
                "storage": {}
            }
        }
        
        layer_config = {}
        
        with patch('src.memfuse_core.utils.config.config_manager') as mock_config_manager:
            mock_config_manager.get_config.return_value = minimal_config
            
            unified_config = storage_manager._get_unified_backend_config(
                StorageType.VECTOR, 
                layer_config
            )
            
            # Should fallback to qdrant backend
            assert unified_config["backend"] == "qdrant"
            assert unified_config["buffer_size"] == 10
    
    def test_pgai_specific_configuration_loading(self, storage_manager, mock_unified_config):
        """Test that pgai-specific configuration is loaded correctly."""
        layer_config = {}
        
        with patch('src.memfuse_core.utils.config.config_manager') as mock_config_manager:
            mock_config_manager.get_config.return_value = mock_unified_config
            
            unified_config = storage_manager._get_unified_backend_config(
                StorageType.VECTOR, 
                layer_config
            )
            
            # Should include pgai-specific settings
            assert unified_config["backend"] == "pgai"
            assert "enabled" in unified_config
            assert "vectorizer_worker_enabled" in unified_config
            assert "auto_embedding" in unified_config
            
            # Should include pgai vector-specific settings
            assert unified_config["connection_pool_size"] == 5
            assert unified_config["timeout"] == 30.0
            assert unified_config["batch_size"] == 100
    
    def test_configuration_debug_logging(self, storage_manager, mock_unified_config):
        """Test that configuration debug logging works."""
        layer_config = {"test": "value"}
        
        with patch('src.memfuse_core.utils.config.config_manager') as mock_config_manager:
            mock_config_manager.get_config.return_value = mock_unified_config
            
            with patch('src.memfuse_core.hierarchy.storage.logger') as mock_logger:
                unified_config = storage_manager._get_unified_backend_config(
                    StorageType.VECTOR, 
                    layer_config
                )
                
                # Should log debug information
                mock_logger.debug.assert_called()
                debug_call = mock_logger.debug.call_args[0][0]
                assert "Unified config for vector" in debug_call
    
    def test_missing_storage_type_config(self, storage_manager, mock_unified_config):
        """Test handling of missing storage type configuration."""
        # Remove vector-specific config
        config_without_vector = mock_unified_config.copy()
        config_without_vector["store"]["pgai"]["storage_backends"].pop("vector", None)
        
        layer_config = {}
        
        with patch('src.memfuse_core.utils.config.config_manager') as mock_config_manager:
            mock_config_manager.get_config.return_value = config_without_vector
            
            unified_config = storage_manager._get_unified_backend_config(
                StorageType.VECTOR, 
                layer_config
            )
            
            # Should still work with general pgai config
            assert unified_config["backend"] == "pgai"
            assert "enabled" in unified_config
            
            # Should not have vector-specific settings
            assert "connection_pool_size" not in unified_config or unified_config["connection_pool_size"] != 5


if __name__ == "__main__":
    pytest.main([__file__])
