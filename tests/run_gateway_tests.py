#!/usr/bin/env python3
"""Simple Gateway test runner."""

import subprocess
import sys

def run_tests():
    """Run Gateway tests."""
    print("🧪 Running Gateway Tests")
    print("=" * 50)
    
    test_commands = [
        ("Unit Tests", ["poetry", "run", "pytest", "tests/unit/gateway/", "-v"]),
        ("Integration Tests", ["poetry", "run", "pytest", "tests/integration/test_gateway_sync_only.py", "-v"]),
    ]
    
    results = []
    for name, command in test_commands:
        print(f"\n🔄 Running {name}...")
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                print(f"✅ {name} - PASSED")
                results.append(True)
            else:
                print(f"❌ {name} - FAILED")
                if result.stderr:
                    print(f"Error: {result.stderr}")
                results.append(False)
        except Exception as e:
            print(f"💥 {name} - ERROR: {e}")
            results.append(False)
    
    passed = sum(results)
    total = len(results)
    print(f"\n📊 Results: {passed}/{total} test suites passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(run_tests())
