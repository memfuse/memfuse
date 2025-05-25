"""Configuration constants for MemFuse.

This module contains constants and default configuration values used throughout
the MemFuse framework. These constants define default behavior when not overridden
by user configuration.
"""

from typing import Dict

# Embedding model configuration
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_DIM = 384

# Model-specific prefixes for embedding models
MODEL_PREFIXES: Dict[str, str] = {
    "all-MiniLM-L6-v2": "",
    "openai": "",
}

# Query parameters
DEFAULT_TOP_K = 5
DEFAULT_SIMILARITY_THRESHOLD = 0.3

# Graph defaults
DEFAULT_GRAPH_RELATION = "RELATED_TO"
DEFAULT_EDGE_WEIGHT = 1.0

# Storage file paths
VECTOR_ITEMS_FILE = "items.json"
VECTOR_EMBEDDINGS_FILE = "embeddings.npy"
GRAPH_NODES_FILE = "nodes.graphml"
GRAPH_EDGES_FILE = "edges.graphml"

# API configuration
DEFAULT_API_KEY_HEADER = "X-API-Key"
