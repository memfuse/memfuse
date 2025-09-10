"""Services for MemFuse server.

This package initializes with a minimal import surface to avoid heavy side-effects
at import time (e.g., optional dependencies like qdrant_client). Submodules should
be imported directly where needed to enable lazy loading.
"""

from .base_service import BaseService, ServiceRegistry
from .app_service import AppService, get_app_service
from .logging_service import LoggingService, get_logging_service
from .database_service import DatabaseService

__all__ = [
    # Base classes
    "BaseService",
    "ServiceRegistry",

    # Core services
    "AppService",
    "get_app_service",
    "LoggingService",
    "get_logging_service",
    "DatabaseService",
]
