"""Store implementations for MemFuse server."""

# Export main classes
from .base import StoreBase
from .factory import StoreFactory

# Export store implementations
from .vector_store.base import VectorStore
from .graph_store.base import GraphStore
from .keyword_store.base import KeywordStore

# Export retrieval implementations
from ..rag.base import BaseRetrieval


# Export specific implementations
from .vector_store.qdrant_store import QdrantVectorStore
from .vector_store.numpy_store import NumpyVectorStore
from .vector_store.sqlite_store import SQLiteVectorStore
from .graph_store.graphml_store import GraphMLStore
from .keyword_store.sqlite_store import SQLiteKeywordStore

# Define __all__ to control what is imported with "from memfuse_core.server.storage import *"
__all__ = [
    # Base classes
    'StoreBase',
    'StoreFactory',

    # Store types
    'VectorStore',
    'GraphStore',
    'KeywordStore',

    # Retrieval types
    'BaseRetrieval',

    # Compatibility layer
    'get_vector_store',
    'get_graph_store',
    'get_keyword_store',
    'get_multi_path_retrieval',
    'query_memory',

    # Vector store implementations
    'NumpyVectorStore',
    'QdrantVectorStore',
    'SQLiteVectorStore',

    # Graph store implementations
    'GraphMLStore',

    # Keyword store implementations
    'SQLiteKeywordStore',
]
