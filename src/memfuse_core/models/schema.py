"""Database schema definitions for MemFuse.

This module contains Pydantic models that represent database entities in the MemFuse system.
These models are used for data validation, serialization, and documentation.
"""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class BaseRecord(BaseModel):
    """Base record schema with common fields."""
    
    id: str = Field(..., description="Unique identifier")
    created_at: str = Field(..., description="Creation timestamp")


class User(BaseRecord):
    """User entity schema."""
    
    name: str = Field(..., description="User name")
    description: Optional[str] = Field(None, description="User description")
    updated_at: str = Field(..., description="Last update timestamp")


class Agent(BaseRecord):
    """Agent entity schema."""
    
    name: str = Field(..., description="Agent name")
    description: Optional[str] = Field(None, description="Agent description")
    updated_at: str = Field(..., description="Last update timestamp")


class Session(BaseRecord):
    """Session entity schema."""
    
    name: str = Field(..., description="Session name")
    user_id: str = Field(..., description="User ID reference")
    agent_id: str = Field(..., description="Agent ID reference")
    updated_at: str = Field(..., description="Last update timestamp")


class MessageRecord(BaseRecord):
    """Message record entity schema."""
    
    session_id: str = Field(..., description="Session ID reference")
    role: str = Field(..., description="Message role (user/assistant)")
    content: str = Field(..., description="Message content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class KnowledgeRecord(BaseRecord):
    """Knowledge record entity schema."""
    
    user_id: str = Field(..., description="User ID reference")
    content: str = Field(..., description="Knowledge content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ApiKey(BaseRecord):
    """API key entity schema."""
    
    user_id: str = Field(..., description="User ID reference")
    key: str = Field(..., description="API key value")
    name: Optional[str] = Field(None, description="API key name")
    permissions: Dict[str, Any] = Field(default_factory=dict, description="Permission settings")
    expires_at: Optional[str] = Field(None, description="Expiration timestamp")
