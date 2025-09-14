"""M3 migration management."""

from .migration_manager import (
    M3MigrationManager,
    initialize_m3_schema,
    get_m3_migration_status,
    validate_m3_schema
)

__all__ = [
    "M3MigrationManager",
    "initialize_m3_schema", 
    "get_m3_migration_status",
    "validate_m3_schema"
]