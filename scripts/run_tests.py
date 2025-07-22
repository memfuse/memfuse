#!/usr/bin/env python
"""
Run MemFuse test layers in order and abort on first failure.

Usage:
    python scripts/run_tests.py                                    # run all layers
    python scripts/run_tests.py smoke                              # run up to smoke layer
    python scripts/run_tests.py contract                           # run up to contract layer
    python scripts/run_tests.py tests/integration/api/test_users_api_integration.py  # run specific test file
    python scripts/run_tests.py tests/integration/api/test_users_api_integration.py::TestUsersAPIIntegration::test_create_user_persistence  # run specific test method
    poetry run python scripts/run_tests.py smoke                   # recommended with Poetry
    
    # Client type configuration:
    python scripts/run_tests.py --client-type=server smoke         # Use actual server (default)
    python scripts/run_tests.py --client-type=testclient smoke     # Use in-process TestClient
    
    # Server restart control:
    python scripts/run_tests.py --no-restart-server smoke          # Keep your development server running
"""
from __future__ import annotations
import sys
import subprocess
import time
import os
import argparse
from pathlib import Path

# Try to import requests for health checking, but don't fail if it's not available
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

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

# MemFuse service configuration
MEMFUSE_API_HOST = "localhost"
MEMFUSE_API_PORT = 8000
MEMFUSE_HEALTH_ENDPOINT = f"http://{MEMFUSE_API_HOST}:{MEMFUSE_API_PORT}/api/v1/health"
MEMFUSE_HEALTH_CHECK_TIMEOUT = 5  # seconds
MEMFUSE_HEALTH_CHECK_RETRIES = 3
MEMFUSE_HEALTH_CHECK_INTERVAL = 2  # seconds

# ---------------------------------------------------------------------------


def print_status(message: str, level: str = "INFO") -> None:
    """Print colored status message."""
    colors = {
        "INFO": "\033[1;34m",      # Blue
        "SUCCESS": "\033[1;32m",   # Green
        "WARNING": "\033[1;33m",   # Yellow
        "ERROR": "\033[1;31m",     # Red
    }
    icons = {
        "INFO": "🔍",
        "SUCCESS": "✅",
        "WARNING": "⚠️",
        "ERROR": "❌"
    }
    
    color = colors.get(level, colors["INFO"])
    icon = icons.get(level, "ℹ️")
    print(f"{color}{icon}  {message}\033[0m")


def check_memfuse_health() -> bool:
    """Check if MemFuse server is healthy and accessible."""
    if not REQUESTS_AVAILABLE:
        print_status("Health check skipped (requests module not available)", "WARNING")
        return True  # Assume healthy if we can't check

    for attempt in range(MEMFUSE_HEALTH_CHECK_RETRIES):
        try:
            response = requests.get(
                MEMFUSE_HEALTH_ENDPOINT,
                timeout=MEMFUSE_HEALTH_CHECK_TIMEOUT
            )
            if response.status_code == 200:
                health_data = response.json()
                # Check for both "healthy" and "ok" status
                data_status = health_data.get("data", {}).get("status", "")
                if (health_data.get("status") == "healthy" or
                    health_data.get("status") == "success" and data_status == "ok"):
                    print_status("MemFuse server is healthy and accessible", "SUCCESS")
                    return True
                else:
                    print_status(
                        f"MemFuse server unhealthy: status={health_data.get('status')}, data_status={data_status}",
                        "WARNING"
                    )
            else:
                print_status(
                    f"Health check failed with status {response.status_code}",
                    "WARNING"
                )
        except Exception as e:
            print_status(
                f"Health check attempt {attempt + 1}/{MEMFUSE_HEALTH_CHECK_RETRIES} failed: {e}",
                "INFO" if attempt < MEMFUSE_HEALTH_CHECK_RETRIES - 1 else "WARNING"
            )

        if attempt < MEMFUSE_HEALTH_CHECK_RETRIES - 1:
            time.sleep(MEMFUSE_HEALTH_CHECK_INTERVAL)

    return False


def start_memfuse_services() -> bool:
    """Start MemFuse services using memfuse_launcher.py."""
    print_status("Starting MemFuse services...", "INFO")
    
    # Start services using memfuse_launcher.py in background mode
    launcher_cmd = [
        sys.executable, "scripts/memfuse_launcher.py", 
        "--start-db", "--optimize-db", "--background"
    ]
    
    print(f"Running: {' '.join(launcher_cmd)}")
    result = subprocess.run(launcher_cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
    
    if result.returncode != 0:
        print_status(f"Failed to start services: {result.stderr}", "ERROR")
        return False
    
    print_status("Services started successfully", "SUCCESS")
    
    # Wait longer for services to be fully ready
    print_status("Waiting for services to be fully ready...", "INFO")
    time.sleep(20)
    
    # Verify services are healthy with extended retries
    for retry in range(5):  # Try up to 5 times
        if check_memfuse_health():
            return True
        if retry < 4:
            print_status(f"Health check failed, retrying in 3 seconds... (attempt {retry + 1}/5)", "INFO")
            time.sleep(3)
    
    print_status("Services started but health check failed after extended retries", "WARNING")
    return False


def reset_database(no_restart: bool = False) -> bool:
    """Reset database to clean state before running tests."""
    if no_restart:
        print_status("Resetting database without server restart...", "INFO")
        print_status("⚠️  WARNING: This may cause connection pool conflicts if server is running", "WARNING")
    else:
        print_status("Resetting database with server restart...", "INFO")
    
    if not no_restart:
        # Stop the current server to prevent connection conflicts
        print_status("Stopping server to reset database safely...", "INFO")
        stop_cmd = [
            "pkill", "-f", "memfuse_launcher.py"
        ]
        
        try:
            subprocess.run(stop_cmd, capture_output=True, text=True, timeout=10)
            print_status("Server stopped", "SUCCESS")
        except subprocess.TimeoutExpired:
            print_status("Server stop timed out, continuing...", "WARNING")
        
        # Wait a moment for server to fully stop
        time.sleep(3)
    
    # Reset database using database_manager.py
    reset_cmd = [
        sys.executable, "scripts/database_manager.py", "reset"
    ]
    
    print(f"Running: {' '.join(reset_cmd)}")
    result = subprocess.run(reset_cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
    
    if result.returncode != 0:
        print_status(f"Failed to reset database: {result.stderr}", "ERROR")
        return False
    
    print_status("Database reset successfully", "SUCCESS")
    
    if not no_restart:
        # Start server again with clean database connections
        print_status("Starting server with fresh database connections...", "INFO")
        if not restart_server_after_reset():
            print_status("Failed to start server after database reset", "ERROR")
            return False
    else:
        print_status("Skipping server restart (--no-restart-server flag used)", "INFO")
        print_status("💡 If you encounter connection issues, restart your server manually", "INFO")
    
    return True


def restart_server_after_reset() -> bool:
    """Restart the server after database reset to clear stale connections."""
    # Start server in background mode
    launcher_cmd = [
        sys.executable, "scripts/memfuse_launcher.py", 
        "--start-db", "--optimize-db", "--background"
    ]
    
    print(f"Running: {' '.join(launcher_cmd)}")
    result = subprocess.run(launcher_cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
    
    if result.returncode != 0:
        print_status(f"Failed to restart server: {result.stderr}", "ERROR")
        return False
    
    print_status("Server restarted successfully", "SUCCESS")
    
    # Wait a bit for server to fully restart
    print_status("Waiting for server to be ready...", "INFO")
    time.sleep(5)
    
    # Verify server is healthy after restart
    for retry in range(3):
        if check_memfuse_health():
            return True
        if retry < 2:
            print_status(f"Server not ready yet, waiting... (attempt {retry + 1}/3)", "INFO")
            time.sleep(3)
    
    print_status("Server restart succeeded but health check failed", "WARNING")
    return False


def ensure_services_ready(no_restart: bool = False) -> bool:
    """Ensure MemFuse services are ready for testing."""
    print_status("Ensuring MemFuse services are ready...", "INFO")
    
    # Check if services are already running
    if check_memfuse_health():
        print_status("Services are already running and healthy", "SUCCESS")
    else:
        print_status("Services not running, starting them...", "INFO")
        if not start_memfuse_services():
            print_status("Failed to start services", "ERROR")
            return False
    
    # Reset database to clean state before testing
    if not reset_database(no_restart):
        print_status("Failed to reset database", "ERROR")
        return False
    
    return True


def is_test_file(path_str: str) -> bool:
    """Check if the given string is a test file path."""
    return (path_str.endswith('.py') and 
            ('test_' in path_str or path_str.endswith('_test.py')) and
            ('/' in path_str or '\\' in path_str))


def is_pytest_selection(path_str: str) -> bool:
    """Check if the given string is a pytest-style test selection (contains ::)."""
    return '::' in path_str and ('test_' in path_str or path_str.endswith('_test.py'))


def run_pytest_selection(test_selection: str, extra_args: list = None) -> int:
    """Run pytest for a specific test selection (file::class::method)."""
    print(f"\n\033[1;34m▶▶  Running pytest selection: {test_selection} …\033[0m")
    
    # Validate that the base file exists
    base_file = test_selection.split('::')[0]
    test_path = PROJECT_ROOT / base_file
    
    if not test_path.exists():
        print_status(f"Test file {base_file} does not exist", "ERROR")
        return 1
    
    cmd = PYTEST_CMD + [
        "--tb=short",
        "-v",  # verbose output for specific selections
        test_selection
    ]
    
    # Add extra pytest arguments if provided
    if extra_args:
        cmd.extend(extra_args)
    
    print(f"Running: {' '.join(cmd)}")
    
    # Run the command and capture output for better error handling
    # Pass environment variables to subprocess
    env = os.environ.copy()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT, env=env)
    
    # Print output for transparency
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    return result.returncode


def run_specific_test_file(test_file: str, extra_args: list = None) -> int:
    """Run pytest for a specific test file."""
    test_path = PROJECT_ROOT / test_file
    
    if not test_path.exists():
        print_status(f"Test file {test_file} does not exist", "ERROR")
        return 1
    
    print(f"\n\033[1;34m▶▶  Running specific test file: {test_file} …\033[0m")
    
    cmd = PYTEST_CMD + [
        "--tb=short",
        "-v",  # verbose output for single file
        str(test_path)
    ]
    
    # Add extra pytest arguments if provided
    if extra_args:
        cmd.extend(extra_args)
    
    print(f"Running: {' '.join(cmd)}")
    
    # Run the command and capture output for better error handling
    # Pass environment variables to subprocess
    env = os.environ.copy()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT, env=env)
    
    # Print output for transparency
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    return result.returncode


def run_layer(marker: str, extra_args: list = None) -> int:
    """Run pytest for one marker, return its exit code."""
    print(f"\n\033[1;34m▶▶  Running {marker} layer …\033[0m")
    
    # Get the specific directory for this layer
    test_dir = LAYER_CONFIG.get(marker, "tests")
    test_path = PROJECT_ROOT / test_dir
    
    # Check if the test directory exists
    if not test_path.exists():
        print(f"\033[1;33m⚠  Test directory {test_dir} does not exist, skipping {marker} layer\033[0m")
        return 0
    
    # Base command with common ignore patterns for all layers
    base_cmd = PYTEST_CMD + [
        "--tb=short",
        "--ignore-glob=**/legacy/**",  # Ignore all legacy folders
        "--ignore-glob=**/ignore/**",  # Ignore all ignore folders
    ]
    
    # For layers with specific directories, we can run without markers
    # For layers that might be everywhere, use markers
    if marker == "slow":
        # slow tests might be anywhere, so use marker
        cmd = base_cmd + [
            "-m", marker,
            str(test_path)
        ]
    else:
        # specific directory layers - run all tests in that directory
        cmd = base_cmd + [
            str(test_path)
        ]
    
    # Add extra pytest arguments if provided
    if extra_args:
        cmd.extend(extra_args)
    
    print(f"Running: {' '.join(cmd)}")
    
    # Run the command and capture output for better error handling
    # Pass environment variables to subprocess
    env = os.environ.copy()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT, env=env)
    
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
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run MemFuse test layers in order and abort on first failure.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_tests.py                                    # run all layers
  python scripts/run_tests.py smoke                              # run up to smoke layer
  python scripts/run_tests.py contract                           # run up to contract layer
  python scripts/run_tests.py tests/integration/api/test_users_api_integration.py  # run specific test file
  python scripts/run_tests.py tests/integration/api/test_users_api_integration.py::TestUsersAPIIntegration::test_create_user_persistence  # run specific test method
  
  # Client type configuration:
  python scripts/run_tests.py --client-type=server smoke         # Use actual server (default)
  python scripts/run_tests.py --client-type=testclient smoke     # Use in-process TestClient
  
  # Server restart control:
  python scripts/run_tests.py --no-restart-server smoke          # Keep your development server running
  python scripts/run_tests.py --client-type=server --no-restart-server integration  # Test against running server without restart
        """
    )
    
    parser.add_argument(
        "target",
        nargs="?",
        help="Test target: layer name, test file path, or pytest selection (default: run all layers)"
    )
    
    parser.add_argument(
        "--client-type",
        choices=["server", "testclient"],
        default="server",
        help="Client type for integration tests: 'server' for actual HTTP server, 'testclient' for in-process TestClient (default: server)"
    )

    parser.add_argument(
        "--no-restart-server",
        action="store_true",
        help="Do not restart the MemFuse server after database reset. Keeps your development server running for monitoring, but may cause connection pool conflicts."
    )
    
    # Parse known args only, so pytest flags like -v can be passed through
    args, unknown_args = parser.parse_known_args(argv)
    
    # Set environment variable for client type
    os.environ["MEMFUSE_TEST_CLIENT_TYPE"] = args.client_type
    
    # Print client type configuration
    if args.client_type == "server":
        print_status("Using actual HTTP server for integration tests", "INFO")
    else:
        print_status("Using in-process TestClient for integration tests", "INFO")
    
    # Parse target argument
    target = args.target
    
    if not target:
        # No target - run all layers
        max_layer_idx = len(LAYER_ORDER) - 1
        run_mode = "all_layers"
    elif is_pytest_selection(target):
        # Pytest-style test selection (file::class::method)
        run_mode = "pytest_selection"
        test_selection = target
    elif is_test_file(target):
        # Single test file
        run_mode = "single_file"
        test_file = target
    elif target in LAYER_ORDER:
        # Run up to specific layer
        max_layer_idx = LAYER_ORDER.index(target)
        run_mode = "up_to_layer"
    else:
        print(f"\033[1;31m✖  Unknown layer or invalid test selection: {target}\033[0m")
        print(f"Available layers: {', '.join(LAYER_ORDER)}")
        print(f"Or specify:")
        print(f"  - A test file: tests/integration/api/test_users_api_integration.py")
        print(f"  - A pytest selection: tests/integration/api/test_users_api_integration.py::TestUsersAPIIntegration::test_create_user_persistence")
        sys.exit(1)

    # Ensure MemFuse services are ready before running any tests
    print_status("Preparing MemFuse services for testing...", "INFO")
    if not ensure_services_ready(args.no_restart_server):
        print_status("Failed to prepare services for testing", "ERROR")
        sys.exit(1)

    # Run tests based on mode
    if run_mode == "pytest_selection":
        # Run pytest selection
        rc = run_pytest_selection(test_selection, unknown_args)
        if rc != 0:
            print(f"\n\033[1;31m✖  Pytest selection {test_selection} failed.\033[0m")
            sys.exit(rc)
        else:
            print(f"\n\033[1;32m✓  Pytest selection {test_selection} passed!\033[0m")
    elif run_mode == "single_file":
        # Run single test file
        rc = run_specific_test_file(test_file, unknown_args)
        if rc != 0:
            print(f"\n\033[1;31m✖  Test file {test_file} failed.\033[0m")
            sys.exit(rc)
        else:
            print(f"\n\033[1;32m✓  Test file {test_file} passed!\033[0m")
    else:
        # Run layers up to specified layer
        for marker in LAYER_ORDER[: max_layer_idx + 1]:
            rc = run_layer(marker, unknown_args)
            if rc != 0:
                print(f"\n\033[1;31m✖  {marker} layer failed. Stopping.\033[0m")
                sys.exit(rc)

        print("\n\033[1;32m✓  All requested layers passed!\033[0m")


if __name__ == "__main__":
    main(sys.argv[1:])