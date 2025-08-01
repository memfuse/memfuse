#!/usr/bin/env python3
"""
Real-time PostgreSQL connection monitor for debugging connection leaks.

Usage:
    python tests/integration/database_connection_monitor.py
    
This script continuously monitors active PostgreSQL connections to help
identify when connections are not being properly released.
"""

import psycopg
import time
import sys
from datetime import datetime


class PostgreSQLConnectionMonitor:
    """Real-time PostgreSQL connection monitor."""
    
    def __init__(self, target_database="memfuse"):
        self.target_database = target_database
        self.conn_params = {
            "host": "localhost",
            "port": 5432,
            "database": "postgres",  # Connect to postgres db to monitor
            "user": "postgres",
            "password": "postgres"
        }
        self.previous_count = 0
        
    def get_connection_info(self):
        """Get detailed connection information."""
        try:
            conn = psycopg.connect(**self.conn_params)
            cursor = conn.cursor()
            
            # Get detailed connection info
            cursor.execute("""
                SELECT 
                    pid,
                    usename,
                    application_name,
                    client_addr,
                    state,
                    state_change,
                    query_start,
                    backend_start,
                    LEFT(query, 50) as query_preview
                FROM pg_stat_activity 
                WHERE datname = %s
                ORDER BY backend_start
            """, (self.target_database,))
            
            connections = cursor.fetchall()
            cursor.close()
            conn.close()
            
            return connections
            
        except Exception as e:
            print(f"Error getting connection info: {e}")
            return []
    
    def get_connection_count(self):
        """Get current connection count."""
        connections = self.get_connection_info()
        return len(connections)
    
    def print_connection_summary(self):
        """Print a summary of current connections."""
        connections = self.get_connection_info()
        count = len(connections)
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Show count change
        if count != self.previous_count:
            change = count - self.previous_count
            change_str = f" ({change:+d})" if change != 0 else ""
            print(f"\n[{timestamp}] Connections: {count}{change_str}")
            
            if change > 0:
                print("  🔴 NEW CONNECTIONS:")
                # Show newest connections (assuming they're at the end)
                for conn in connections[-change:]:
                    self.print_connection_details(conn, "    ")
            elif change < 0:
                print("  🟢 CONNECTIONS CLOSED")
        
        self.previous_count = count
        return count, connections
    
    def print_connection_details(self, conn, prefix=""):
        """Print details for a single connection."""
        pid, user, app_name, client_addr, state, state_change, query_start, backend_start, query = conn
        
        print(f"{prefix}PID: {pid}")
        print(f"{prefix}User: {user}")
        print(f"{prefix}App: {app_name or 'N/A'}")
        print(f"{prefix}Client: {client_addr or 'local'}")
        print(f"{prefix}State: {state}")
        print(f"{prefix}Started: {backend_start}")
        if query and query.strip():
            print(f"{prefix}Query: {query}...")
        print()
    
    def print_full_report(self):
        """Print a full connection report."""
        connections = self.get_connection_info()
        count = len(connections)
        
        print(f"\n{'='*50}")
        print(f"PostgreSQL Connection Report - {datetime.now()}")
        print(f"Database: {self.target_database}")
        print(f"Active Connections: {count}")
        print(f"{'='*50}")
        
        if connections:
            for i, conn in enumerate(connections, 1):
                print(f"\nConnection #{i}:")
                self.print_connection_details(conn, "  ")
        else:
            print("No active connections found.")
    
    def monitor_continuously(self, interval=2):
        """Monitor connections continuously."""
        print(f"🔍 Monitoring PostgreSQL connections to '{self.target_database}' database")
        print(f"📊 Checking every {interval} seconds")
        print("Press Ctrl+C to stop\n")
        
        try:
            # Initial report
            self.print_full_report()
            
            while True:
                time.sleep(interval)
                self.print_connection_summary()
                
        except KeyboardInterrupt:
            print("\n\n👋 Monitoring stopped")
            
            # Final report
            self.print_full_report()


def main():
    """Main function to run the monitor."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor PostgreSQL connections")
    parser.add_argument("--database", "-d", default="memfuse", 
                       help="Target database to monitor (default: memfuse)")
    parser.add_argument("--interval", "-i", type=int, default=2,
                       help="Check interval in seconds (default: 2)")
    parser.add_argument("--once", action="store_true",
                       help="Run once and exit (don't monitor continuously)")
    
    args = parser.parse_args()
    
    monitor = PostgreSQLConnectionMonitor(args.database)
    
    if args.once:
        monitor.print_full_report()
    else:
        monitor.monitor_continuously(args.interval)


if __name__ == "__main__":
    main() 