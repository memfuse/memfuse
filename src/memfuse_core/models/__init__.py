"""Models package for MemFuse.

This module provides a unified interface to all data models used in the MemFuse framework.
It re-exports classes from the core, api, schema, interfaces, and config modules.
"""

# Core models and types
from .core import (
    # Base data models
    Item,
    Node,
    Edge,
    Query,
    EmbeddingItem,
    QueryResult,
    RetrievalResult,
    Message,
    
    # API response models
    ErrorDetail,
    ApiResponse,
    
    # Type definitions
    StoreBackend,
    StoreType
)

# API request models
from .api import (
    # User API models
    UserCreate,
    UserUpdate,
    
    # Agent API models
    AgentCreate,
    AgentUpdate,
    
    # Session API models
    SessionCreate,
    SessionUpdate,
    
    # API Key models
    ApiKeyCreate,
    
    # Memory API models
    MemoryInit,
    MemoryQuery,
    MessageAdd,
    MessageRead,
    MessageUpdate,
    MessageDelete,
    
    # Knowledge API models
    KnowledgeAdd,
    KnowledgeRead,
    KnowledgeUpdate,
    KnowledgeDelete
)

# Database schema models
from .schema import (
    BaseRecord,
    User,
    Agent,
    Session,
    MessageRecord,
    KnowledgeRecord,
    ApiKey
)

# Interface definitions
from .interfaces import MemoryInterface, BufferInterface

# Configuration constants
from .config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_TOP_K,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_GRAPH_RELATION,
    DEFAULT_EDGE_WEIGHT,
    VECTOR_ITEMS_FILE,
    VECTOR_EMBEDDINGS_FILE,
    GRAPH_NODES_FILE,
    GRAPH_EDGES_FILE,
    DEFAULT_API_KEY_HEADER,
    MODEL_PREFIXES
)

__all__ = [
    # Core models
    "Item", "Node", "Edge", "Query", "EmbeddingItem", "QueryResult", 
    "RetrievalResult", "Message", "ErrorDetail", "ApiResponse",
    
    # Type definitions
    "StoreBackend", "StoreType",
    
    # API request models - User
    "UserCreate", "UserUpdate",
    
    # API request models - Agent
    "AgentCreate", "AgentUpdate",
    
    # API request models - Session
    "SessionCreate", "SessionUpdate",
    
    # API request models - API Key
    "ApiKeyCreate",
    
    # API request models - Memory
    "MemoryInit", "MemoryQuery", "MessageAdd", "MessageRead",
    "MessageUpdate", "MessageDelete",
    
    # API request models - Knowledge
    "KnowledgeAdd", "KnowledgeRead", "KnowledgeUpdate", "KnowledgeDelete",
    
    # Schema models
    "BaseRecord", "User", "Agent", "Session", "MessageRecord", 
    "KnowledgeRecord", "ApiKey",
    
    # Interface definitions
    "MemoryInterface", "BufferInterface",
    
    # Configuration constants
    "DEFAULT_EMBEDDING_MODEL", "DEFAULT_EMBEDDING_DIM", "DEFAULT_TOP_K",
    "DEFAULT_SIMILARITY_THRESHOLD", "DEFAULT_GRAPH_RELATION", "DEFAULT_EDGE_WEIGHT",
    "VECTOR_ITEMS_FILE", "VECTOR_EMBEDDINGS_FILE", "GRAPH_NODES_FILE",
    "GRAPH_EDGES_FILE", "DEFAULT_API_KEY_HEADER", "MODEL_PREFIXES"
]