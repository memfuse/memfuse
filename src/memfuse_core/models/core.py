"""Core model classes and types for MemFuse.

This module contains the core data models and type definitions used throughout
the MemFuse framework, including base classes for items, nodes, edges, and queries.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
from pydantic import BaseModel
from enum import Enum


# Type definitions
class StoreBackend(str, Enum):
    """Store backend types."""

    NUMPY = "numpy"
    QDRANT = "qdrant"
    SQLITE = "sqlite"
    PGVECTOR = "pgvector"
    IGRAPH = "igraph"
    NEO4J = "neo4j"
    HYBRID = "hybrid"


class StoreType(str, Enum):
    """Store types."""

    VECTOR = "vector"
    GRAPH = "graph"
    KEYWORD = "keyword"


# Base model classes
@dataclass
class Item:
    """Base class for all items stored in MemFuse."""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Node(Item):
    """Node in the graph store."""
    type: str = "node"


@dataclass
class Edge:
    """Edge in the graph store."""
    id: str
    source_id: str
    target_id: str
    relation: str = "RELATED_TO"
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Query:
    """Query for retrieving items from stores."""
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingItem(Item):
    """Item with an embedding vector."""
    embedding: Optional[np.ndarray] = None


@dataclass
class QueryResult:
    """Result of a query operation."""
    id: str
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    store_type: Optional[str] = None  # Will be set to a StoreType value


@dataclass
class RetrievalResult:
    """Combined result of retrieval operations."""
    results: List[QueryResult]
    content: str


class Message(BaseModel):
    """Message model."""

    role: str
    content: str


class ErrorDetail(BaseModel):
    """Error detail model."""

    field: str
    message: str


class ApiResponse(BaseModel):
    """API response model."""

    status: str
    code: int
    data: Optional[Dict[str, Any]] = None
    message: str
    errors: Optional[List[ErrorDetail]] = None

    @classmethod
    def success(cls, data: Optional[Dict[str, Any]] = None, message: str = "Success") -> "ApiResponse":
        """Create a success response."""
        # Process DictConfig objects
        if data is not None:
            # Check if any value is a DictConfig
            from omegaconf import DictConfig
            if any(isinstance(v, DictConfig) for v in data.values()):
                # Create a new dictionary with DictConfig converted to native containers
                processed_data = {}
                for k, v in data.items():
                    if isinstance(v, DictConfig):
                        from omegaconf import OmegaConf
                        processed_data[k] = OmegaConf.to_container(
                            v, resolve=True)
                    else:
                        processed_data[k] = v
                data = processed_data

        return cls(
            status="success",
            code=200,
            data=data,
            message=message,
            errors=None,
        )

    @classmethod
    def error(cls, message: str, code: int = 500, errors: Optional[List[ErrorDetail]] = None) -> "ApiResponse":
        """Create an error response."""
        if errors is None:
            errors = [ErrorDetail(field="general", message=message)]

        return cls(
            status="error",
            code=code,
            data=None,
            message=message,
            errors=errors,
        )
