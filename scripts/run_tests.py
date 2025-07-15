#!/usr/bin/env python
"""
Run MemFuse test layers in order and abort on first failure.

Usage:
    python scripts/run_tests.py            # run all layers
    python scripts/run_tests.py smoke e2e  # run up to e2e
    poetry run python scripts/run_tests.py smoke  # recommended with Poetry
"""
from __future__ import annotations
import sys
import subprocess
from pathlib import Path

# ---- config ---------------------------------------------------------------
# Map layers to their specific directories to avoid scanning everything
LAYER_CONFIG = {
    "smoke": "tests/smoke",
    # "unit": "tests/unit", 
    "contract": "tests/contract",
    "integration": "tests/integration",
    "retrieval": "tests/retrieval",
    "e2e": "tests/e2e",
    "perf": "tests/perf",
    "slow": "tests"  # slow tests might be everywhere, so scan all
}

LAYER_ORDER = list(LAYER_CONFIG.keys())
PYTEST_CMD  = [sys.executable, "-m", "pytest"]        # respects your venv
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# ---------------------------------------------------------------------------


def run_layer(marker: str) -> int:
    """Run pytest for one marker, return its exit code."""
    print(f"\n\033[1;34m▶▶  Running {marker} layer …\033[0m")
    
    # Get the specific directory for this layer
    test_dir = LAYER_CONFIG.get(marker, "tests")
    test_path = PROJECT_ROOT / test_dir
    
    # Check if the test directory exists
    if not test_path.exists():
        print(f"\033[1;33m⚠  Test directory {test_dir} does not exist, skipping {marker} layer\033[0m")
        return 0
    
    # Special handling for integration tests - ensure database is started
    if marker == "integration":
        print("\n\033[1;36m🔧  Starting database for integration tests...\033[0m")
        
        # Start database using memfuse_launcher.py
        db_cmd = [
            sys.executable, "scripts/memfuse_launcher.py", 
            "--start-db", "--optimize-db", "--background"
        ]
        
        print(f"Running: {' '.join(db_cmd)}")
        result = subprocess.run(db_cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
        
        if result.returncode != 0:
            print(f"\033[1;31m✖  Failed to start database: {result.stderr}\033[0m")
            return result.returncode
        
        print("\033[1;32m✓  Database started successfully\033[0m")
        
        # Wait a moment for database to be fully ready
        import time
        time.sleep(3)
    
    # For layers with specific directories, we can run without markers
    # For layers that might be everywhere, use markers
    if marker == "slow":
        # slow tests might be anywhere, so use marker
        cmd = PYTEST_CMD + [
            "-m", marker,
            "--tb=short",
            str(test_path)
        ]
    elif marker == "integration":
        # integration layer - exclude legacy tests
        cmd = PYTEST_CMD + [
            "--tb=short",
            "--ignore=" + str(test_path / "legacy"),
            str(test_path)
        ]
    else:
        # specific directory layers - run all tests in that directory
        cmd = PYTEST_CMD + [
            "--tb=short",
            str(test_path)
        ]
    
    print(f"Running: {' '.join(cmd)}")
    
    # Run the command and capture output for better error handling
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Print output for transparency
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    # Check if any tests were actually run
    if "collected 0 items" in result.stdout:
        print(f"\033[1;33m⚠  No tests found in {test_dir}\033[0m")
        return 0  # Don't fail if no tests exist for this layer
    
    return result.returncode


def main(argv: list[str]) -> None:
    # Optional CLI arg: run only up to a given layer
    if argv and argv[0] not in LAYER_ORDER:
        print(f"\033[1;31m✖  Unknown layer: {argv[0]}\033[0m")
        print(f"Available layers: {', '.join(LAYER_ORDER)}")
        sys.exit(1)
    
    max_layer_idx = (
        LAYER_ORDER.index(argv[0]) if argv else len(LAYER_ORDER) - 1
    )

    for marker in LAYER_ORDER[: max_layer_idx + 1]:
        rc = run_layer(marker)
        if rc != 0:
            print(f"\n\033[1;31m✖  {marker} layer failed. Stopping.\033[0m")
            sys.exit(rc)

    print("\n\033[1;32m✓  All requested layers passed!\033[0m")


if __name__ == "__main__":
    main(sys.argv[1:])