#!/usr/bin/env python3
"""
Run all contextual chunking related tests
Organized test runner for contextual functionality
"""

import subprocess
import sys
import os
from pathlib import Path


def run_test(test_path, description):
    """Run a single test and return success status"""
    print(f"\n{'='*60}")
    print(f"🧪 Running: {description}")
    print(f"📁 Path: {test_path}")
    print('='*60)
    
    try:
        result = subprocess.run(
            [sys.executable, test_path],
            cwd=Path(__file__).parent.parent,
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            print(f"✅ {description} - PASSED")
            return True
        else:
            print(f"❌ {description} - FAILED (exit code: {result.returncode})")
            return False
            
    except Exception as e:
        print(f"❌ {description} - ERROR: {e}")
        return False


def main():
    """Run all contextual chunking tests"""
    print("🚀 Starting Contextual Chunking Test Suite")
    print(f"📁 Working directory: {os.getcwd()}")
    
    # Define test suite
    tests = [
        {
            "path": "tests/unit/rag/chunk/test_contextual_strategy.py",
            "description": "Unit Tests - Contextual Strategy Core Logic"
        },
        {
            "path": "tests/integration/llm/test_contextual_llm_integration.py", 
            "description": "Integration Tests - LLM & Contextual Strategy"
        },
        {
            "path": "tests/integration/config/test_contextual_config.py",
            "description": "Integration Tests - Contextual Configuration"
        },
        {
            "path": "tests/utils/verify_qdrant_data.py",
            "description": "Utility - Qdrant Data Verification"
        }
    ]
    
    # Run tests
    results = []
    for test in tests:
        success = run_test(test["path"], test["description"])
        results.append({
            "test": test["description"],
            "success": success
        })
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 TEST SUITE SUMMARY")
    print('='*60)
    
    passed = sum(1 for r in results if r["success"])
    total = len(results)
    
    for result in results:
        status = "✅ PASSED" if result["success"] else "❌ FAILED"
        print(f"{status} - {result['test']}")
    
    print(f"\n📈 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All contextual chunking tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the output above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
