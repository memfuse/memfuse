#!/usr/bin/env python3
"""
MemFuse Development Launcher

Unified launcher for MemFuse development and deployment.
Handles database startup, optimization, and MemFuse core service management.
"""

import argparse
import subprocess
import sys
import os
import time
import signal
from pathlib import Path
from typing import Optional

class MemFuseLauncher:
    """MemFuse development launcher and service manager."""
    
    def __init__(self):
        self.memfuse_process: Optional[subprocess.Popen] = None
        
    def print_status(self, message: str, status: str = "INFO"):
        """Print colored status messages."""
        colors = {
            "INFO": "\033[0;34m",
            "SUCCESS": "\033[0;32m", 
            "WARNING": "\033[1;33m",
            "ERROR": "\033[0;31m",
        }
        reset = "\033[0m"
        
        icons = {
            "INFO": "ℹ️ ",
            "SUCCESS": "✅ ",
            "WARNING": "⚠️ ",
            "ERROR": "❌ ",
        }
        
        color = colors.get(status, "")
        icon = icons.get(status, "")
        print(f"{color}{icon}{message}{reset}")
    
    def optimize_database(self) -> bool:
        """Apply database optimizations to prevent pgai hanging."""
        self.print_status("Applying database optimizations...", "INFO")

        optimizations = [
            "ALTER SYSTEM SET lock_timeout = '30s';",
            "ALTER SYSTEM SET max_connections = 50;",
            "ALTER SYSTEM SET shared_buffers = '256MB';",
            "ALTER SYSTEM SET deadlock_timeout = '1s';",
            "ALTER SYSTEM SET max_locks_per_transaction = 256;",
            "SELECT pg_reload_conf();"
        ]

        container_name = "memfuse-pgai-postgres"

        for sql in optimizations:
            try:
                result = subprocess.run([
                    'docker', 'exec', container_name,
                    'psql', '-U', 'postgres', '-d', 'memfuse', '-c', sql
                ], capture_output=True, text=True, timeout=10)

                if result.returncode != 0:
                    self.print_status(f"Failed to apply: {sql[:50]}...", "WARNING")

            except Exception as e:
                self.print_status(f"Error applying optimization: {e}", "WARNING")

        self.print_status("Database optimizations completed", "SUCCESS")
        return True

    def check_database_connectivity(self, timeout: int = 30) -> bool:
        """Check if database is accessible."""
        self.print_status("Checking database connectivity...", "INFO")

        for attempt in range(timeout):
            try:
                result = subprocess.run([
                    'docker', 'exec', 'memfuse-pgai-postgres',
                    'pg_isready', '-U', 'postgres', '-d', 'memfuse'
                ], capture_output=True, text=True, timeout=5)

                if result.returncode == 0:
                    self.print_status("Database is ready", "SUCCESS")
                    return True

            except:
                pass

            time.sleep(1)

        self.print_status("Database is not accessible", "ERROR")
        return False
    
    def start_database(self, force_recreate: bool = False) -> bool:
        """Start database container."""
        self.print_status("Starting database container...", "INFO")
        
        cmd = ['docker-compose', '-f', 'docker/compose/docker-compose.pgai.yml']
        
        if force_recreate:
            cmd.extend(['up', '-d', '--force-recreate', 'postgres-pgai'])
        else:
            cmd.extend(['up', '-d', 'postgres-pgai'])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                self.print_status("Database container started", "SUCCESS")
                return True
            else:
                self.print_status(f"Failed to start database: {result.stderr}", "ERROR")
                return False
                
        except Exception as e:
            self.print_status(f"Error starting database: {e}", "ERROR")
            return False
    
    def start_memfuse(self, show_logs: bool = True, timeout: Optional[int] = None) -> bool:
        """Start MemFuse server."""
        self.print_status("Starting MemFuse server...", "INFO")
        
        try:
            if show_logs:
                # Start with visible output
                self.memfuse_process = subprocess.Popen([
                    'poetry', 'run', 'memfuse-core'
                ])
            else:
                # Start in background
                self.memfuse_process = subprocess.Popen([
                    'poetry', 'run', 'memfuse-core'
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Wait a bit to see if it starts
            time.sleep(5)
            
            if self.memfuse_process.poll() is None:
                self.print_status("MemFuse server started successfully", "SUCCESS")
                return True
            else:
                self.print_status("MemFuse server failed to start", "ERROR")
                return False
                
        except Exception as e:
            self.print_status(f"Failed to start MemFuse: {e}", "ERROR")
            return False
    
    def stop_services(self):
        """Stop all services."""
        if self.memfuse_process and self.memfuse_process.poll() is None:
            self.print_status("Stopping MemFuse server...", "INFO")
            self.memfuse_process.terminate()
            try:
                self.memfuse_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.memfuse_process.kill()
    
    def run(self, args):
        """Run the server with specified options."""
        self.print_status("🚀 MemFuse Server Manager", "INFO")
        self.print_status("=" * 50, "INFO")
        
        try:
            # Start database if requested
            if args.start_db or args.recreate_db:
                if not self.start_database(force_recreate=args.recreate_db):
                    return False

                # Wait for database to be ready
                if not self.check_database_connectivity():
                    return False

                # Apply database optimizations
                if args.optimize_db:
                    self.optimize_database()
            
            # Start MemFuse
            if not self.start_memfuse(show_logs=args.show_logs, timeout=args.timeout):
                return False
            
            # Wait for interrupt if showing logs
            if args.show_logs:
                self.print_status("Server is running. Press Ctrl+C to stop", "INFO")
                try:
                    while True:
                        if self.memfuse_process.poll() is not None:
                            self.print_status("MemFuse process stopped", "WARNING")
                            break
                        time.sleep(1)
                except KeyboardInterrupt:
                    self.print_status("Received interrupt signal", "INFO")
            
            return True
            
        except Exception as e:
            self.print_status(f"Server startup failed: {e}", "ERROR")
            return False
        finally:
            self.stop_services()

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="MemFuse Development Launcher")
    
    parser.add_argument('--start-db', action='store_true',
                       help='Start database container')
    parser.add_argument('--recreate-db', action='store_true',
                       help='Force recreate database container')
    parser.add_argument('--optimize-db', action='store_true', default=True,
                       help='Apply database optimizations (default: True)')
    parser.add_argument('--show-logs', action='store_true', default=True,
                       help='Show server logs (default: True)')
    parser.add_argument('--background', action='store_true',
                       help='Run server in background')
    parser.add_argument('--timeout', type=int, default=None,
                       help='Startup timeout in seconds')
    
    args = parser.parse_args()
    
    # Adjust show_logs based on background flag
    if args.background:
        args.show_logs = False
    
    # Change to memfuse directory
    os.chdir("/Users/mxue/GitRepos/MemFuse/memfuse")
    
    manager = MemFuseLauncher()
    
    # Setup signal handler
    def signal_handler(signum, frame):
        print(f"\nReceived signal {signum}")
        manager.stop_services()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    success = manager.run(args)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
