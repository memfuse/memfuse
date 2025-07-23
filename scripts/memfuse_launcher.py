#!/usr/bin/env python3
"""
MemFuse Development Launcher

Unified launcher for MemFuse development and deployment.
Handles database startup, optimization, and MemFuse core service management.

By default, starts database container with optimizations and shows logs.
Use --no-start-db or --background to modify default behavior.
"""

import argparse
import subprocess
import sys
import os
import time
import signal
import psutil
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass
from enum import Enum

# Try to import requests, but don't fail if it's not available
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


# Constants
DOCKER_COMPOSE_FILE = "docker/compose/docker-compose.pgai.yml"
POSTGRES_CONTAINER_NAME = "memfuse-pgai-postgres"
MEMFUSE_CORE_COMMAND = ["poetry", "run", "memfuse-core"]
MEMFUSE_API_PORT = 8000
MEMFUSE_API_HOST = "localhost"
MEMFUSE_HEALTH_ENDPOINT = f"http://{MEMFUSE_API_HOST}:{MEMFUSE_API_PORT}/api/v1/health"

# Timeouts and retry settings
DB_STARTUP_TIMEOUT = 120  # seconds
DB_CONNECTIVITY_TIMEOUT = 30  # seconds
DB_CONNECTIVITY_RETRY_INTERVAL = 1  # seconds
MEMFUSE_STARTUP_TIMEOUT = 5  # seconds
MEMFUSE_HEALTH_CHECK_TIMEOUT = 10  # seconds
MEMFUSE_HEALTH_CHECK_RETRIES = 3
MEMFUSE_HEALTH_CHECK_INTERVAL = 2  # seconds
PROCESS_TERMINATION_TIMEOUT = 10  # seconds

# Port management settings
PORT_CHECK_TIMEOUT = 5  # seconds
PORT_CLEANUP_TIMEOUT = 10  # seconds

# Database optimization settings
DB_OPTIMIZATIONS = [
    "ALTER SYSTEM SET lock_timeout = '30s';",
    "ALTER SYSTEM SET max_connections = 50;",
    "ALTER SYSTEM SET shared_buffers = '256MB';",
    "ALTER SYSTEM SET deadlock_timeout = '1s';",
    "ALTER SYSTEM SET max_locks_per_transaction = 256;",
    "SELECT pg_reload_conf();"
]


def check_port_usage(port: int) -> List[int]:
    """
    Check if a port is in use and return list of process IDs using it.

    Args:
        port: Port number to check

    Returns:
        List of process IDs using the port
    """
    pids = []
    try:
        # Use lsof command to find processes using the port
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True,
            text=True,
            timeout=PORT_CHECK_TIMEOUT
        )
        if result.returncode == 0 and result.stdout.strip():
            pids = [int(pid.strip()) for pid in result.stdout.strip().split('\n') if pid.strip().isdigit()]
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError):
        # Fallback to psutil if lsof fails
        try:
            for conn in psutil.net_connections():
                if conn.laddr.port == port and conn.status == psutil.CONN_LISTEN:
                    if conn.pid:
                        pids.append(conn.pid)
        except (psutil.Error, AttributeError):
            pass

    return pids


def kill_processes_on_port(port: int, force: bool = False) -> bool:
    """
    Kill processes using a specific port.

    Args:
        port: Port number to clear
        force: If True, use SIGKILL; if False, use SIGTERM first

    Returns:
        True if all processes were successfully terminated
    """
    pids = check_port_usage(port)
    if not pids:
        return True

    print(f"🔍 Found {len(pids)} process(es) using port {port}: {pids}")

    success = True
    for pid in pids:
        try:
            process = psutil.Process(pid)
            process_name = process.name()
            print(f"🔄 Terminating process {pid} ({process_name})...")

            if force:
                process.kill()  # SIGKILL
            else:
                process.terminate()  # SIGTERM

            # Wait for process to terminate
            try:
                process.wait(timeout=PORT_CLEANUP_TIMEOUT)
                print(f"✅ Process {pid} terminated successfully")
            except psutil.TimeoutExpired:
                print(f"⚠️  Process {pid} didn't terminate gracefully, forcing...")
                process.kill()
                process.wait(timeout=5)
                print(f"✅ Process {pid} force-killed")

        except psutil.NoSuchProcess:
            print(f"✅ Process {pid} already terminated")
        except psutil.AccessDenied:
            print(f"❌ Access denied when trying to terminate process {pid}")
            success = False
        except Exception as e:
            print(f"❌ Error terminating process {pid}: {e}")
            success = False

    # Verify port is now free
    remaining_pids = check_port_usage(port)
    if remaining_pids:
        print(f"⚠️  Port {port} still in use by processes: {remaining_pids}")
        return False

    print(f"✅ Port {port} is now free")
    return success


def ensure_port_available(port: int, force_cleanup: bool = True) -> bool:
    """
    Ensure a port is available for use, cleaning up if necessary.

    Args:
        port: Port number to ensure is available
        force_cleanup: If True, attempt to kill processes using the port

    Returns:
        True if port is available after cleanup
    """
    pids = check_port_usage(port)
    if not pids:
        print(f"✅ Port {port} is available")
        return True

    if not force_cleanup:
        print(f"❌ Port {port} is in use by processes: {pids}")
        return False

    print(f"🧹 Port {port} is in use, attempting cleanup...")

    # First try graceful termination
    if kill_processes_on_port(port, force=False):
        return True

    # If graceful termination failed, try force kill
    print(f"🔨 Graceful termination failed, force-killing processes on port {port}...")
    return kill_processes_on_port(port, force=True)


class StatusLevel(Enum):
    """Status message levels."""
    INFO = "INFO"
    SUCCESS = "SUCCESS"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass
class LauncherConfig:
    """Configuration for MemFuse launcher."""
    start_db: bool = True
    recreate_db: bool = False
    optimize_db: bool = True
    show_logs: bool = True
    background: bool = False
    timeout: Optional[int] = None
    port_cleanup: bool = True

    @classmethod
    def from_env(cls) -> 'LauncherConfig':
        """Create configuration from environment variables."""
        return cls(
            start_db=os.getenv("MEMFUSE_START_DB", "true").lower() != "false",
            recreate_db=os.getenv("MEMFUSE_RECREATE_DB", "false").lower() == "true",
            optimize_db=os.getenv("MEMFUSE_OPTIMIZE_DB", "true").lower() != "false",
            show_logs=os.getenv("MEMFUSE_SHOW_LOGS", "true").lower() != "false",
            background=os.getenv("MEMFUSE_BACKGROUND", "false").lower() == "true",
            timeout=int(os.getenv("MEMFUSE_TIMEOUT", "0")) if os.getenv("MEMFUSE_TIMEOUT", "").isdigit() else None,
            port_cleanup=os.getenv("MEMFUSE_PORT_CLEANUP", "true").lower() != "false"
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'LauncherConfig':
        """Create configuration from command line arguments."""
        config = cls(
            start_db=args.start_db,
            recreate_db=args.recreate_db,
            optimize_db=args.optimize_db,
            show_logs=args.show_logs,
            background=args.background,
            timeout=args.timeout,
            port_cleanup=True  # Default to True
        )

        # Handle negative flags
        if args.no_start_db:
            config.start_db = False
        if args.no_optimize_db:
            config.optimize_db = False
        if args.no_port_cleanup:
            config.port_cleanup = False

        # Adjust show_logs based on background flag
        if config.background:
            config.show_logs = False

        return config


class MemFuseLauncher:
    """MemFuse development launcher and service manager."""

    def __init__(self, config: LauncherConfig):
        self.config = config
        self.memfuse_process: Optional[subprocess.Popen] = None

    def print_status(self, message: str, level: StatusLevel = StatusLevel.INFO):
        """Print colored status messages."""
        colors = {
            StatusLevel.INFO: "\033[0;34m",
            StatusLevel.SUCCESS: "\033[0;32m",
            StatusLevel.WARNING: "\033[1;33m",
            StatusLevel.ERROR: "\033[0;31m",
        }
        reset = "\033[0m"

        icons = {
            StatusLevel.INFO: "ℹ️ ",
            StatusLevel.SUCCESS: "✅ ",
            StatusLevel.WARNING: "⚠️ ",
            StatusLevel.ERROR: "❌ ",
        }

        color = colors.get(level, "")
        icon = icons.get(level, "")
        print(f"{color}{icon}{message}{reset}")

    def _run_command(self, cmd: List[str], timeout: int = 30, capture_output: bool = True) -> subprocess.CompletedProcess:
        """Run a command with proper error handling."""
        try:
            result = subprocess.run(
                cmd,
                capture_output=capture_output,
                text=True,
                timeout=timeout
            )
            return result
        except subprocess.TimeoutExpired as e:
            self.print_status(f"Command timed out after {timeout}s: {' '.join(cmd)}", StatusLevel.ERROR)
            raise e
        except Exception as e:
            self.print_status(f"Command failed: {' '.join(cmd)} - {e}", StatusLevel.ERROR)
            raise e

    def optimize_database(self) -> bool:
        """Apply database optimizations to prevent pgai hanging."""
        self.print_status("Applying database optimizations...", StatusLevel.INFO)

        success_count = 0
        for sql in DB_OPTIMIZATIONS:
            try:
                cmd = [
                    'docker', 'exec', POSTGRES_CONTAINER_NAME,
                    'psql', '-U', 'postgres', '-d', 'memfuse', '-c', sql
                ]
                result = self._run_command(cmd, timeout=10)

                if result.returncode == 0:
                    success_count += 1
                else:
                    self.print_status(
                        f"Failed to apply optimization: {sql[:50]}... (exit code: {result.returncode})",
                        StatusLevel.WARNING
                    )
                    if result.stderr:
                        self.print_status(f"Error details: {result.stderr.strip()}", StatusLevel.WARNING)

            except Exception as e:
                self.print_status(f"Error applying optimization: {e}", StatusLevel.WARNING)

        if success_count == len(DB_OPTIMIZATIONS):
            self.print_status("All database optimizations applied successfully", StatusLevel.SUCCESS)
        else:
            self.print_status(
                f"Applied {success_count}/{len(DB_OPTIMIZATIONS)} optimizations",
                StatusLevel.WARNING
            )

        return success_count > 0

    def check_database_connectivity(self, timeout: int = DB_CONNECTIVITY_TIMEOUT) -> bool:
        """Check if database is accessible."""
        self.print_status("Checking database connectivity...", StatusLevel.INFO)

        start_time = time.time()
        end_time = start_time + timeout
        attempt = 0

        while time.time() < end_time:
            attempt += 1
            try:
                cmd = [
                    'docker', 'exec', POSTGRES_CONTAINER_NAME,
                    'pg_isready', '-U', 'postgres', '-d', 'memfuse'
                ]
                result = self._run_command(cmd, timeout=5, capture_output=True)

                if result.returncode == 0:
                    elapsed = time.time() - start_time
                    self.print_status(
                        f"Database is ready (after {attempt} attempts, {elapsed:.1f}s)",
                        StatusLevel.SUCCESS
                    )
                    return True
                else:
                    self.print_status(
                        f"Database not ready yet (attempt {attempt}/{timeout})",
                        StatusLevel.INFO
                    )

            except subprocess.TimeoutExpired:
                self.print_status(
                    f"Database check timed out (attempt {attempt}/{timeout})",
                    StatusLevel.WARNING
                )
            except Exception as e:
                self.print_status(
                    f"Error checking database: {e}",
                    StatusLevel.WARNING
                )

            time.sleep(DB_CONNECTIVITY_RETRY_INTERVAL)

        self.print_status(
            f"Database is not accessible after {timeout}s ({attempt} attempts)",
            StatusLevel.ERROR
        )
        return False

    def start_database(self, force_recreate: bool = False) -> bool:
        """Start database container."""
        action = "Recreating" if force_recreate else "Starting"
        self.print_status(f"{action} database container...", StatusLevel.INFO)

        cmd = ['docker-compose', '-f', DOCKER_COMPOSE_FILE]

        if force_recreate:
            cmd.extend(['up', '-d', '--force-recreate', 'postgres-pgai'])
        else:
            cmd.extend(['up', '-d', 'postgres-pgai'])

        try:
            result = self._run_command(cmd, timeout=DB_STARTUP_TIMEOUT)

            if result.returncode == 0:
                self.print_status("Database container started successfully", StatusLevel.SUCCESS)
                return True
            else:
                self.print_status(
                    f"Failed to start database (exit code: {result.returncode})",
                    StatusLevel.ERROR
                )
                if result.stderr:
                    self.print_status(f"Error details: {result.stderr.strip()}", StatusLevel.ERROR)
                return False

        except subprocess.TimeoutExpired:
            self.print_status(
                f"Database startup timed out after {DB_STARTUP_TIMEOUT}s",
                StatusLevel.ERROR
            )
            return False
        except Exception as e:
            self.print_status(f"Error starting database: {e}", StatusLevel.ERROR)
            return False

    def check_memfuse_health(self) -> bool:
        """Check if MemFuse server is healthy."""
        if not REQUESTS_AVAILABLE:
            self.print_status(
                "Health check skipped (requests module not available)",
                StatusLevel.WARNING
            )
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
                        self.print_status("MemFuse server health check passed", StatusLevel.SUCCESS)
                        return True
                    else:
                        self.print_status(
                            f"MemFuse server unhealthy: status={health_data.get('status')}, data_status={data_status}",
                            StatusLevel.WARNING
                        )
                else:
                    self.print_status(
                        f"Health check failed with status {response.status_code}",
                        StatusLevel.WARNING
                    )
            except Exception as e:  # Catch all exceptions since requests might not be available
                self.print_status(
                    f"Health check attempt {attempt + 1}/{MEMFUSE_HEALTH_CHECK_RETRIES} failed: {e}",
                    StatusLevel.INFO
                )

            if attempt < MEMFUSE_HEALTH_CHECK_RETRIES - 1:
                time.sleep(MEMFUSE_HEALTH_CHECK_INTERVAL)

        return False

    def start_memfuse(self, show_logs: bool = True, timeout: Optional[int] = None) -> bool:
        """Start MemFuse server."""
        self.print_status("Starting MemFuse server...", StatusLevel.INFO)

        # Check and clean up port before starting (if enabled)
        if self.config.port_cleanup:
            self.print_status(f"Checking port {MEMFUSE_API_PORT} availability...", StatusLevel.INFO)
            if not ensure_port_available(MEMFUSE_API_PORT, force_cleanup=True):
                self.print_status(f"Failed to free port {MEMFUSE_API_PORT}", StatusLevel.ERROR)
                return False
        else:
            self.print_status("Port cleanup disabled, skipping port check", StatusLevel.INFO)

        try:
            if show_logs:
                # Start with visible output
                self.memfuse_process = subprocess.Popen(MEMFUSE_CORE_COMMAND)
            else:
                # Start in background
                self.memfuse_process = subprocess.Popen(
                    MEMFUSE_CORE_COMMAND,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )

            # Wait a bit to see if it starts
            time.sleep(MEMFUSE_STARTUP_TIMEOUT)

            if self.memfuse_process.poll() is None:
                self.print_status("MemFuse process started", StatusLevel.SUCCESS)

                # Perform health check if not showing logs (background mode)
                if not show_logs:
                    if self.check_memfuse_health():
                        self.print_status("MemFuse server is healthy and ready", StatusLevel.SUCCESS)
                    else:
                        self.print_status("MemFuse server started but health check failed", StatusLevel.WARNING)

                return True
            else:
                self.print_status("MemFuse server failed to start", StatusLevel.ERROR)
                return False

        except Exception as e:
            self.print_status(f"Failed to start MemFuse: {e}", StatusLevel.ERROR)
            return False

    def stop_services(self):
        """Stop all services."""
        if self.memfuse_process and self.memfuse_process.poll() is None:
            self.print_status("Stopping MemFuse server...", StatusLevel.INFO)
            self.memfuse_process.terminate()
            try:
                self.memfuse_process.wait(timeout=PROCESS_TERMINATION_TIMEOUT)
                self.print_status("MemFuse server stopped gracefully", StatusLevel.SUCCESS)
            except subprocess.TimeoutExpired:
                self.print_status("Termination timed out, forcing shutdown...", StatusLevel.WARNING)
                self.memfuse_process.kill()
                self.print_status("MemFuse server forcefully terminated", StatusLevel.WARNING)

    def run(self):
        """Run the server with specified options."""
        self.print_status("🚀 MemFuse Server Manager", StatusLevel.INFO)
        self.print_status("=" * 50, StatusLevel.INFO)

        try:
            # Start database if enabled (default) or recreate requested
            if self.config.start_db or self.config.recreate_db:
                if not self.start_database(force_recreate=self.config.recreate_db):
                    self.print_status("Failed to start database. Exiting.", StatusLevel.ERROR)
                    return False

                # Wait for database to be ready
                if not self.check_database_connectivity():
                    self.print_status("Database connectivity check failed. Exiting.", StatusLevel.ERROR)
                    return False

                # Apply database optimizations if enabled (default)
                if self.config.optimize_db:
                    if not self.optimize_database():
                        self.print_status("Database optimization failed, but continuing...", StatusLevel.WARNING)
            else:
                self.print_status("Database startup skipped (--no-start-db)", StatusLevel.INFO)

            # Start MemFuse
            if not self.start_memfuse(
                show_logs=self.config.show_logs,
                timeout=self.config.timeout
            ):
                self.print_status("Failed to start MemFuse server. Exiting.", StatusLevel.ERROR)
                return False

            # Wait for interrupt if showing logs
            if self.config.show_logs:
                self.print_status("Server is running. Press Ctrl+C to stop", StatusLevel.INFO)
                try:
                    while True:
                        if self.memfuse_process and self.memfuse_process.poll() is not None:
                            exit_code = self.memfuse_process.returncode
                            if exit_code == 0:
                                self.print_status(
                                    "MemFuse process exited normally",
                                    StatusLevel.SUCCESS
                                )
                            else:
                                self.print_status(
                                    f"MemFuse process stopped with exit code {exit_code}",
                                    StatusLevel.WARNING
                                )
                            break
                        time.sleep(1)
                except KeyboardInterrupt:
                    self.print_status("Received interrupt signal", StatusLevel.INFO)
                finally:
                    # Only stop services when showing logs (interactive mode)
                    self.stop_services()
            else:
                self.print_status("Server started in background mode", StatusLevel.SUCCESS)

            return True

        except Exception as e:
            self.print_status(f"Server startup failed: {e}", StatusLevel.ERROR)
            # Only stop services on error
            self.stop_services()
            return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="MemFuse Development Launcher",
        epilog="""
Environment Variables:
  MEMFUSE_START_DB      Start database container (default: true)
  MEMFUSE_RECREATE_DB   Force recreate database container (default: false)
  MEMFUSE_OPTIMIZE_DB   Apply database optimizations (default: true)
  MEMFUSE_SHOW_LOGS     Show server logs (default: true)
  MEMFUSE_BACKGROUND    Run in background mode (default: false)
  MEMFUSE_TIMEOUT       Startup timeout in seconds

Examples:
  %(prog)s                          # Start with defaults
  %(prog)s --background             # Run in background
  %(prog)s --no-start-db            # Skip database startup
  %(prog)s --recreate-db            # Force recreate database
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--start-db', action='store_true', default=True,
                        help='Start database container (default: True)')
    parser.add_argument('--no-start-db', action='store_true',
                        help='Skip starting database container')
    parser.add_argument('--recreate-db', action='store_true',
                        help='Force recreate database container')
    parser.add_argument('--optimize-db', action='store_true', default=True,
                        help='Apply database optimizations (default: True)')
    parser.add_argument('--no-optimize-db', action='store_true',
                        help='Skip database optimizations')
    parser.add_argument('--show-logs', action='store_true', default=True,
                        help='Show server logs (default: True)')
    parser.add_argument('--background', action='store_true',
                        help='Run server in background (disables logs)')
    parser.add_argument('--timeout', type=int, default=None,
                        help='Startup timeout in seconds')
    parser.add_argument('--no-port-cleanup', action='store_true',
                        help='Skip automatic port cleanup before starting services')
    parser.add_argument('--version', action='version', version='MemFuse Launcher 2.0')

    args = parser.parse_args()

    # Create configuration from command line arguments (includes environment variable defaults)
    config = LauncherConfig.from_args(args)

    # Change to memfuse directory (use relative path from script location)
    try:
        script_dir = Path(__file__).parent
        memfuse_dir = script_dir.parent
        os.chdir(memfuse_dir)
    except NameError:
        # Handle case when __file__ is not defined (e.g., when running via exec)
        current_dir = Path.cwd()
        if current_dir.name == "scripts":
            os.chdir(current_dir.parent)
        elif (current_dir / "scripts").exists():
            pass  # Already in the right directory
        else:
            print("Warning: Could not determine memfuse directory, using current directory")

    # Create launcher with configuration
    manager = MemFuseLauncher(config)

    # Setup signal handler
    def signal_handler(signum: int, frame: Any) -> None:
        print(f"\nReceived signal {signum}")
        manager.stop_services()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    success = manager.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
