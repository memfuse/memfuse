"""Server package for MemFuse."""

# Optional server imports - only import if dependencies are available
__all__ = []

try:
    from .server import create_app, main
    __all__.extend(["create_app", "main"])
except ImportError:
    # Server dependencies not available, skip server imports
    pass
