"""Core model classes and types for MemFuse.

This module contains the core data models and type definitions used throughout
the MemFuse framework, including base classes for items, nodes, edges, and queries.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal
from datetime import datetime
import numpy as np
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum


# Type definitions
class M2Status(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class M2FactStatus(str, Enum):
    ACTIVE = "active"
    DEPRECATED = "deprecated"


class StoreBackend(str, Enum):
    """Store backend types."""

    NUMPY = "numpy"
    QDRANT = "qdrant"
    SQLITE = "sqlite"
    PGVECTOR = "pgvector"
    PGAI = "pgai"
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

    role: Literal["user", "assistant", "system"] = Field(
        ..., description="Message role - must be 'user', 'assistant', or 'system'"
    )
    content: str = Field(..., min_length=1, description="Message content - cannot be empty")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional message metadata")


class Chunk(BaseModel):
    """M1 chunk model for structured data handling."""
    
    chunk_id: str = Field(..., description="UUID of the chunk")
    content: str = Field(..., description="Text content of the chunk")
    token_count: int = Field(..., description="Number of tokens in the chunk")
    user_id: str = Field(..., description="UUID of the user who owns this chunk")
    session_id: Optional[str] = Field(None, description="Session ID associated with this chunk")
    created_at: datetime = Field(..., description="When the chunk was created")
    updated_at: Optional[datetime] = Field(None, description="When the chunk was last updated")
    m2_status: M2Status = Field(default=M2Status.PENDING, description="M2 processing status")
    
    # Additional metadata fields that might be useful for M2 processing
    chunking_strategy: Optional[str] = Field(None, description="Strategy used to create this chunk")
    m0_raw_ids: List[str] = Field(default_factory=list, description="Source message IDs")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional chunk metadata")


class Fact(BaseModel):
    """M2 semantic fact model for structured knowledge extraction."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    fact_id: str = Field(default_factory=lambda: str(__import__('uuid').uuid4()), description="UUID of the fact")
    text: str = Field(..., description="Text content of the semantic fact")
    hash: Optional[str] = Field(None, description="Unique hash for idempotency and duplicate detection")
    embedding: Optional[np.ndarray] = Field(None, description="384-dimensional embedding vector")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")
    status: M2FactStatus = Field(default=M2FactStatus.ACTIVE, description="Fact status (active or deprecated)")
    chunk_ids: List[str] = Field(default_factory=list, description="List of chunk UUIDs this fact was extracted from")
    user_id: str = Field(..., description="UUID of the user who owns this fact")
    policy_version: str = Field(default="v1.0", description="Policy version used for fact extraction")
    created_at: datetime = Field(default_factory=lambda: datetime.now(), description="When the fact was created")
    updated_at: Optional[datetime] = Field(None, description="When the fact was last updated")
    embedding_generated_at: Optional[datetime] = Field(None, description="When the embedding was generated")
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", description="Model used for embedding generation")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional fact metadata")


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
    def success(cls, data: Optional[Dict[str, Any]] = None, message: str = "Success", code: int = 200) -> "ApiResponse":
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
            code=code,
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
