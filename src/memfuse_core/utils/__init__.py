"""Utility functions for MemFuse server.

This module includes utilities from both the original utils and common directories."""

from .validation import (
    validate_user_exists,
    validate_agent_exists,
    validate_session_exists,
    validate_user_by_name_exists,
    validate_agent_by_name_exists,
    validate_session_by_name_exists,
    validate_user_name_available,
    validate_agent_name_available,
    validate_session_name_available,
    # New convenience functions that raise exceptions
    ensure_user_exists,
    ensure_user_by_name_exists,
    ensure_user_name_available,
    ensure_agent_exists,
    ensure_agent_by_name_exists,
    ensure_agent_name_available,
    ensure_session_exists,
    ensure_session_by_name_exists,
    ensure_session_name_available,
    raise_api_error,
)
from .error_handling import handle_api_errors
from .serialization import convert_numpy_types, prepare_response_data
from ..models.core import Message, ErrorDetail, ApiResponse
from ..models.core import StoreBackend, StoreType

# Configuration utilities
from .config import (
    config_manager,
    ConfigManager,
    get_top_k,
    get_similarity_threshold,
    get_embedding_dim,
    get_graph_relation,
    get_edge_weight,
    get_vector_items_file,
    get_vector_embeddings_file,
    get_graph_nodes_file,
    get_graph_edges_file,
    get_api_key_header
)

from .embeddings import (
    cosine_similarity,
    cosine_similarity_matrix,
    normalize_embeddings,
    normalize_embedding,
    EmbeddingCache,
)
from .performance_monitor import PerformanceMonitor
from .cache import Cache
from .buffer import Buffer
from .token_counter import (
    TokenCounter,
    get_token_counter,
    count_tokens,
    count_message_tokens
)
from .version import (
    get_project_version,
    get_version_info
)

__all__ = [
    # From validation
    "validate_user_exists",
    "validate_agent_exists",
    "validate_session_exists",
    "validate_user_by_name_exists",
    "validate_agent_by_name_exists",
    "validate_session_by_name_exists",
    "validate_user_name_available",
    "validate_agent_name_available",
    "validate_session_name_available",
    # New convenience functions that raise exceptions
    "ensure_user_exists",
    "ensure_user_by_name_exists",
    "ensure_user_name_available",
    "ensure_agent_exists",
    "ensure_agent_by_name_exists",
    "ensure_agent_name_available",
    "ensure_session_exists",
    "ensure_session_by_name_exists",
    "ensure_session_name_available",
    "raise_api_error",
    # From error_handling
    "handle_api_errors",
    # From serialization
    "convert_numpy_types",
    "prepare_response_data",
    # From embeddings
    "EmbeddingCache",
    # From models (previously in common)
    "Message",
    "ErrorDetail",
    "ApiResponse",
    # From types (previously in common)
    "StoreBackend",
    "StoreType",
    # From embeddings (previously in common/utils)
    "cosine_similarity",
    "cosine_similarity_matrix",
    "normalize_embeddings",
    "normalize_embedding",
    # From performance_monitor (previously in common/utils)
    "PerformanceMonitor",
    # From cache and buffer (previously in models/common)
    "Cache",
    "Buffer",
    # From token_counter
    "TokenCounter",
    "get_token_counter",
    "count_tokens",
    "count_message_tokens",
    # From version
    "get_project_version",
    "get_version_info",
]
