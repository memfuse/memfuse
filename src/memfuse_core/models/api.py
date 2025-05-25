"""API models for MemFuse server.

This module contains the request and response models used in the MemFuse API,
providing a unified interface for API interactions.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel

from .core import Message, ErrorDetail, ApiResponse, StoreType


# User API models
class UserCreate(BaseModel):
    """Request model for creating a user."""
    name: str
    description: Optional[str] = None


class UserUpdate(BaseModel):
    """Request model for updating a user."""
    name: Optional[str] = None
    description: Optional[str] = None


# Agent API models
class AgentCreate(BaseModel):
    """Request model for creating an agent."""
    name: str
    description: Optional[str] = None


class AgentUpdate(BaseModel):
    """Request model for updating an agent."""
    name: Optional[str] = None
    description: Optional[str] = None


# Session API models
class SessionCreate(BaseModel):
    """Request model for creating a session."""
    user_id: str
    agent_id: str
    name: Optional[str] = None  # Session name, will be auto-generated if not provided


class SessionUpdate(BaseModel):
    """Request model for updating a session."""
    name: Optional[str] = None  # New session name


# API Key models
class ApiKeyCreate(BaseModel):
    """Request model for creating an API key."""
    user_id: str  # User ID for whom the API key is created
    name: Optional[str] = None  # Name of the API key
    key: Optional[str] = None  # Optional custom key, will be auto-generated if not provided
    permissions: Optional[Dict[str, Any]] = None  # Optional permissions dictionary
    expires_at: Optional[str] = None  # Optional expiration date


# Memory API models
class MemoryInit(BaseModel):
    """Request model for initializing memory."""
    user: str
    agent: Optional[str] = None
    session: Optional[str] = None


class MemoryQuery(BaseModel):
    """Request model for querying memory.

    This model defines the parameters for querying memory across sessions for a user.
    If session_id is provided, results will be tagged with scope="in_session" or
    scope="cross_session" depending on whether they belong to the specified session.
    If session_id is not provided, all results will have scope=null.
    """
    query: str
    session_id: Optional[str] = None
    agent_id: Optional[str] = None
    top_k: int = 5
    store_type: Optional[StoreType] = None
    include_messages: bool = True
    include_knowledge: bool = True


class MessageAdd(BaseModel):
    """Request model for adding messages."""
    messages: List[Message]


class MessageRead(BaseModel):
    """Request model for reading messages."""
    message_ids: List[str]


class MessageUpdate(BaseModel):
    """Request model for updating messages."""
    message_ids: List[str]
    new_messages: List[Message]


class MessageDelete(BaseModel):
    """Request model for deleting messages."""
    message_ids: List[str]


# Knowledge API models
class KnowledgeAdd(BaseModel):
    """Request model for adding knowledge."""
    knowledge: List[str]


class KnowledgeRead(BaseModel):
    """Request model for reading knowledge."""
    knowledge_ids: List[str]


class KnowledgeUpdate(BaseModel):
    """Request model for updating knowledge."""
    knowledge_ids: List[str]
    new_knowledge: List[str]


class KnowledgeDelete(BaseModel):
    """Request model for deleting knowledge."""
    knowledge_ids: List[str]
