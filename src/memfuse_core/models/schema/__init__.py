"""Database schema management for MemFuse.

This module provides unified database schema definitions for M0 and M1 tables.
All database schema definitions should be managed through this module.

Also re-exports the original schema classes for compatibility.
"""

# New schema management system
from .base import BaseSchema
from .m0_raw import M0RawSchema
from .m1_episodic import M1EpisodicSchema
from .manager import SchemaManager

# Re-export original schema classes for compatibility
# Import from the parent schema.py file
import sys
from pathlib import Path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

try:
    from schema import (
        BaseRecord,
        User,
        Agent,
        Session,
        MessageRecord,
        KnowledgeRecord,
        ApiKey
    )
except ImportError:
    # Fallback: import directly
    import importlib.util
    schema_file = parent_dir / "schema.py"
    spec = importlib.util.spec_from_file_location("schema", schema_file)
    schema_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(schema_module)

    BaseRecord = schema_module.BaseRecord
    User = schema_module.User
    Agent = schema_module.Agent
    Session = schema_module.Session
    MessageRecord = schema_module.MessageRecord
    KnowledgeRecord = schema_module.KnowledgeRecord
    ApiKey = schema_module.ApiKey

__all__ = [
    # New schema management
    "BaseSchema",
    "M0RawSchema",
    "M1EpisodicSchema",
    "SchemaManager",

    # Original schema classes
    "BaseRecord",
    "User",
    "Agent",
    "Session",
    "MessageRecord",
    "KnowledgeRecord",
    "ApiKey"
]
