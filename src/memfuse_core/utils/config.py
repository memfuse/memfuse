"""Configuration management for MemFuse server.

This module provides a simple configuration system for MemFuse.
It uses Hydra's DictConfig directly without dataclass definitions,
allowing for more flexible configuration.

It also includes utilities for accessing configuration values.
"""

from loguru import logger
from typing import Dict, Any, Optional
from omegaconf import OmegaConf
import dotenv

from ..models.core import StoreBackend

dotenv.load_dotenv()


def _get_config() -> Dict[str, Any]:
    """Get the configuration from the cache or load it.

    Returns:
        Configuration dictionary
    """
    return config_manager.get_config()


class ConfigManager:
    """Configuration manager for MemFuse server.

    This class provides a singleton instance for accessing the configuration.
    It expects the configuration to be set from outside, typically from the
    Hydra-decorated main function.
    """

    _instance = None
    _cfg = None

    def __new__(cls):
        """Create a singleton instance."""
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the configuration manager."""
        # Only initialize once
        if ConfigManager._cfg is None:
            # Initialize with empty config
            ConfigManager._cfg = {}

    def set_config(self, cfg: Any):
        """Set the configuration.

        Args:
            cfg: Configuration object (DictConfig or dict)
        """
        if hasattr(cfg, 'to_container'):
            # If it's a DictConfig, convert it to a dict
            ConfigManager._cfg = OmegaConf.to_container(cfg, resolve=True)
        else:
            # Otherwise, use it as is
            ConfigManager._cfg = cfg

        logger.info("Configuration set successfully")

    def get_config(self):
        """Get the configuration.

        Returns:
            Configuration dictionary
        """
        return ConfigManager._cfg

    def get_store_backend(self):
        """Get the store backend from configuration.

        Returns:
            Store backend
        """
        config = self.get_config()
        backend_str = config.get("store", {}).get("backend", "qdrant")
        try:
            return StoreBackend(backend_str)
        except ValueError:
            logger.warning(f"Unknown store backend: {backend_str}")
            logger.warning(
                f"Using default store backend: {StoreBackend.QDRANT.value}")
            return StoreBackend.QDRANT

    def to_dict(self):
        """Convert configuration to dictionary.

        Returns:
            Configuration dictionary
        """
        return self.get_config()


# Helper functions for accessing configuration values

def get_top_k() -> int:
    """Get the top_k value from configuration.

    Returns:
        The top_k value
    """
    config = _get_config()
    return config.get("store", {}).get("top_k", 5)


def get_similarity_threshold() -> float:
    """Get the similarity threshold value from configuration.

    Returns:
        The similarity threshold value
    """
    config = _get_config()
    return config.get("store", {}).get("similarity_threshold", 0.3)


def get_embedding_dim(model_name: Optional[str] = None) -> int:
    """Get the embedding dimension from configuration.

    Args:
        model_name: Name of the embedding model (optional)

    Returns:
        The embedding dimension
    """
    config = _get_config()
    embedding_dim = config.get("embedding", {}).get("dimension", 384)

    # Check if model-specific dimension is available
    if model_name is not None:
        # Get model-specific dimension from config
        models = config.get("embedding", {}).get("models", {})
        if model_name in models:
            model_config = models[model_name]
            if "dimension" in model_config:
                model_dim = model_config["dimension"]
                if isinstance(model_dim, int) and model_dim > 0:
                    return model_dim

    return embedding_dim


def get_graph_relation() -> str:
    """Get the default graph relation value from configuration.

    Returns:
        The default graph relation value
    """
    config = _get_config()
    return config.get("store", {}).get("graph_store", {}).get("default_relation", "RELATED_TO")


def get_edge_weight() -> float:
    """Get the default edge weight value from configuration.

    Returns:
        The default edge weight value
    """
    config = _get_config()
    return config.get("store", {}).get("graph_store", {}).get("default_edge_weight", 1.0)


def get_vector_items_file() -> str:
    """Get the vector items file path from configuration.

    Returns:
        The vector items file path
    """
    config = _get_config()
    return config.get("store", {}).get("file_paths", {}).get("vector_items_file", "items.json")


def get_vector_embeddings_file() -> str:
    """Get the vector embeddings file path from configuration.

    Returns:
        The vector embeddings file path
    """
    config = _get_config()
    return config.get("store", {}).get("file_paths", {}).get("vector_embeddings_file", "embeddings.json")


def get_graph_nodes_file() -> str:
    """Get the graph nodes file path from configuration.

    Returns:
        The graph nodes file path
    """
    config = _get_config()
    return config.get("store", {}).get("file_paths", {}).get("graph_nodes_file", "nodes.json")


def get_graph_edges_file() -> str:
    """Get the graph edges file path from configuration.

    Returns:
        The graph edges file path
    """
    config = _get_config()
    return config.get("store", {}).get("file_paths", {}).get("graph_edges_file", "edges.json")


def get_api_key_header() -> str:
    """Get the API key header from configuration.

    Returns:
        The API key header
    """
    config = _get_config()
    return config.get("server", {}).get("api_key_header", "X-API-Key")


# Create a singleton instance of the configuration manager
config_manager = ConfigManager()
