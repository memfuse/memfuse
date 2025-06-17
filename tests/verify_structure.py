#!/usr/bin/env python3
"""Verify test structure and configuration."""

import os
import sys
from pathlib import Path


def check_file_exists(file_path, description=""):
    """Check if a file exists."""
    if os.path.exists(file_path):
        print(f"✅ {description or file_path}")
        return True
    else:
        print(f"❌ {description or file_path} - MISSING")
        return False


def check_directory_structure():
    """Check that the test directory structure is correct."""
    print("🔍 Checking test directory structure...")
    
    required_dirs = [
        "tests/",
        "tests/unit/",
        "tests/unit/rag/",
        "tests/unit/rag/chunk/",
        "tests/integration/",
        "tests/e2e/",
    ]
    
    all_good = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            print(f"✅ Directory: {dir_path}")
        else:
            print(f"❌ Directory: {dir_path} - MISSING")
            all_good = False
    
    return all_good


def check_test_files():
    """Check that required test files exist."""
    print("\n🔍 Checking test files...")
    
    required_files = [
        ("tests/__init__.py", "Test package init"),
        ("tests/conftest.py", "Pytest configuration"),
        ("tests/README.md", "Test documentation"),
        ("pytest.ini", "Pytest configuration file"),
        ("tests/unit/rag/chunk/test_base.py", "ChunkData/ChunkStrategy tests"),
        ("tests/unit/rag/chunk/test_message_chunk_strategy.py", "MessageChunkStrategy tests"),
        ("tests/unit/rag/chunk/test_contextual_chunk_strategy.py", "ContextualChunkStrategy tests"),
        ("tests/unit/rag/chunk/test_character_chunk_strategy.py", "CharacterChunkStrategy tests"),
        ("tests/integration/test_chunking_integration.py", "Chunking integration tests"),
        ("tests/e2e/test_chunking.py", "E2E chunking tests"),
    ]
    
    all_good = True
    for file_path, description in required_files:
        if not check_file_exists(file_path, description):
            all_good = False
    
    return all_good


def check_dependencies():
    """Check if test dependencies are available."""
    print("\n🔍 Checking test dependencies...")

    # Try to run a simple pytest command to check if it works
    try:
        import subprocess
        result = subprocess.run(
            ["python", "-m", "pytest", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            print("✅ pytest is working")

            # Check for asyncio support
            result2 = subprocess.run(
                ["python", "-c", "import pytest_asyncio; print('pytest-asyncio available')"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result2.returncode == 0:
                print("✅ pytest-asyncio is available")
            else:
                print("⚠️  pytest-asyncio may not be available in current environment")
                print("   Try: poetry run python tests/verify_structure.py")

            return True
        else:
            print("❌ pytest is not working properly")
            return False
    except Exception as e:
        print(f"❌ Error checking dependencies: {e}")
        return False


def main():
    """Main verification function."""
    print("🧪 MemFuse Test Structure Verification")
    print("=" * 50)
    
    checks = [
        ("Directory Structure", check_directory_structure),
        ("Test Files", check_test_files),
        ("Dependencies", check_dependencies),
    ]
    
    all_passed = True
    results = []
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
            if not result:
                all_passed = False
        except Exception as e:
            print(f"❌ {check_name} - ERROR: {e}")
            results.append((check_name, False))
            all_passed = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Verification Summary:")
    
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {check_name}: {status}")
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All checks passed! Test structure is ready.")
        print("\nNext steps:")
        print("  1. Run: make test-quick")
        print("  2. Run: make test-chunking")
        print("  3. Start MemFuse server and run: make test-e2e")
        return 0
    else:
        print("💥 Some checks failed! Please fix the issues above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
