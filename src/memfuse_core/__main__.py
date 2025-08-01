"""
Entry point for running the MemFuse server as a module.
This allows running the server with `python -m memfuse_core`.
"""

from .server import main

if __name__ == "__main__":
    main()
