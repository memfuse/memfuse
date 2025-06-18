#!/usr/bin/env python3
"""Test runner script for MemFuse test suite."""

import argparse
import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description=""):
    """Run a command and handle the result."""
    print(f"\n🔄 {description}")
    print(f"Running: {' '.join(cmd)}")
    print("-" * 50)
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode == 0:
        print(f"✅ {description} - PASSED")
    else:
        print(f"❌ {description} - FAILED")
        return False
    
    return True


def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = ['pytest', 'pytest-asyncio', 'aiohttp']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print(f"Install with: pip install {' '.join(missing_packages)}")
        return False
    
    return True


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="MemFuse Test Runner")
    parser.add_argument(
        "--type", 
        choices=["unit", "integration", "e2e", "all"], 
        default="all",
        help="Type of tests to run"
    )
    parser.add_argument(
        "--coverage", 
        action="store_true",
        help="Run tests with coverage report"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--markers", "-m",
        help="Run tests with specific markers (e.g., 'chunking', 'not slow')"
    )
    parser.add_argument(
        "--file", "-f",
        help="Run specific test file"
    )
    parser.add_argument(
        "--parallel", "-p",
        action="store_true",
        help="Run tests in parallel (requires pytest-xdist)"
    )
    
    args = parser.parse_args()
    
    print("🧪 MemFuse Test Runner")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Build pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add verbosity
    if args.verbose:
        cmd.append("-v")
    
    # Add coverage
    if args.coverage:
        cmd.extend([
            "--cov=src/memfuse_core",
            "--cov-report=html",
            "--cov-report=term-missing"
        ])
    
    # Add parallel execution
    if args.parallel:
        cmd.extend(["-n", "auto"])
    
    # Add markers
    if args.markers:
        cmd.extend(["-m", args.markers])
    
    # Add specific file or test type
    if args.file:
        cmd.append(args.file)
    elif args.type == "unit":
        cmd.extend(["tests/unit/", "-m", "unit"])
    elif args.type == "integration":
        cmd.extend(["tests/integration/", "-m", "integration"])
    elif args.type == "e2e":
        cmd.extend(["tests/e2e/", "-m", "e2e"])
    elif args.type == "all":
        cmd.append("tests/")
    
    # Run the tests
    success = run_command(cmd, f"Running {args.type} tests")
    
    if args.coverage and success:
        print("\n📊 Coverage report generated in htmlcov/index.html")
    
    # Summary
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests completed successfully!")
        sys.exit(0)
    else:
        print("💥 Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
