#!/usr/bin/env python3
"""
Uvicorn worker launcher for scaling tests.
Starts uvicorn programmatically with configurable worker count.
"""

import argparse
import asyncio
import uvicorn
import sys
import os
from pathlib import Path

# Ensure memfuse_core can be imported
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

def main():
    parser = argparse.ArgumentParser(description="Start uvicorn with configurable workers")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8010, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")

    args = parser.parse_args()

    print(f"Starting uvicorn with {args.workers} workers on {args.host}:{args.port}")

    # Use uvicorn.run which handles the event loop setup properly
    # Only pass workers if > 1 to avoid the multiprocessing issue
    if args.workers > 1:
        uvicorn.run(
            "memfuse_core.server:create_app",
            host=args.host,
            port=args.port,
            workers=args.workers,
            factory=True,
        )
    else:
        # For single worker, don't use workers parameter
        uvicorn.run(
            "memfuse_core.server:create_app",
            host=args.host,
            port=args.port,
            factory=True,
        )

if __name__ == "__main__":
    main()