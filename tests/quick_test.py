#!/usr/bin/env python3
"""Quick test runner for chunking functionality."""

import subprocess
import sys


def main():
    """Run quick chunking tests."""
    print("🧪 Running Quick Chunking Tests")
    print("=" * 40)
    
    cmd = [
        "python", "-m", "pytest",
        "tests/unit/rag/chunk/",
        "-v",
        "-m", "chunking and not slow",
        "--tb=short"
    ]
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("\n✅ Quick chunking tests passed!")
    else:
        print("\n❌ Quick chunking tests failed!")
    
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
