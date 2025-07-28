#!/usr/bin/env python3
"""
Monitor database connections during test runner execution.

This script helps identify whether connection leaks are coming from:
1. The run_tests.py script itself
2. Individual test files
3. Specific test methods

Usage:
    # Terminal 1: Start monitoring
    poetry run python tests/integration/test_runner_connection_monitor.py
    
    # Terminal 2: Run tests
    poetry run python scripts/run_tests.py --no-restart integration -v -s
"""

import psycopg2
import time
import sys
from datetime import datetime
from collections import defaultdict


class TestRunnerConnectionMonitor:
    """Monitor connections during test runner execution."""
    
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
        self.max_connections = 50  # PostgreSQL limit
        self.connection_history = []
        self.warning_threshold = 40  # Warn when approaching limit
        self.critical_threshold = 45  # Critical when very close to limit
        
    def get_connection_info(self):
        """Get detailed connection information."""
        try:
            conn = psycopg2.connect(**self.conn_params)
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
                    LEFT(query, 100) as query_preview
                FROM pg_stat_activity 
                WHERE datname = %s
                ORDER BY backend_start
            """, (self.target_database,))
            
            connections = cursor.fetchall()
            cursor.close()
            conn.close()
            
            return connections
            
        except Exception as e:
            print(f"❌ Error getting connection info: {e}")
            return []
    
    def get_total_connections(self):
        """Get total connection count across all databases."""
        try:
            conn = psycopg2.connect(**self.conn_params)
            cursor = conn.cursor()
            cursor.execute("SELECT count(*) FROM pg_stat_activity")
            total = cursor.fetchone()[0]
            cursor.close()
            conn.close()
            return total
        except Exception as e:
            print(f"❌ Error getting total connections: {e}")
            return 0
    
    def analyze_connection_patterns(self, connections):
        """Analyze connection patterns to identify issues."""
        if not connections:
            return {}
        
        patterns = {
            "total": len(connections),
            "by_state": defaultdict(int),
            "by_app": defaultdict(int),
            "idle_in_transaction": 0,
            "long_running": 0,
            "recent": 0
        }
        
        now = datetime.now()
        
        for conn in connections:
            pid, user, app, client, state, state_change, query_start, backend_start, query = conn
            
            # Analyze by state
            patterns["by_state"][state] += 1
            
            # Analyze by application
            app_name = app or "Unknown"
            patterns["by_app"][app_name] += 1
            
            # Check for problematic states
            if state == "idle in transaction":
                patterns["idle_in_transaction"] += 1
            
            # Check for long-running connections (> 5 minutes)
            if backend_start:
                age = (now - backend_start.replace(tzinfo=None)).total_seconds()
                if age > 300:  # 5 minutes
                    patterns["long_running"] += 1
                if age < 30:  # Recent connections
                    patterns["recent"] += 1
        
        return patterns
    
    def print_connection_summary(self):
        """Print a summary of current connections with analysis."""
        connections = self.get_connection_info()
        count = len(connections)
        total_count = self.get_total_connections()
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Determine alert level
        alert_level = "INFO"
        alert_icon = "📊"
        if count >= self.critical_threshold:
            alert_level = "CRITICAL"
            alert_icon = "🚨"
        elif count >= self.warning_threshold:
            alert_level = "WARNING"
            alert_icon = "⚠️"
        elif count > self.previous_count:
            alert_level = "INCREASE"
            alert_icon = "📈"
        elif count < self.previous_count:
            alert_level = "DECREASE"
            alert_icon = "📉"
        
        # Show count change
        change = count - self.previous_count
        change_str = f" ({change:+d})" if change != 0 else ""
        
        print(f"\n[{timestamp}] {alert_icon} Memfuse: {count}/{self.max_connections}{change_str} | Total: {total_count}/{self.max_connections}")
        
        if count != self.previous_count or count >= self.warning_threshold:
            patterns = self.analyze_connection_patterns(connections)
            
            if patterns:
                print(f"  📋 Analysis:")
                print(f"    • By State: {dict(patterns['by_state'])}")
                print(f"    • By App: {dict(patterns['by_app'])}")
                
                if patterns["idle_in_transaction"] > 0:
                    print(f"    • ⚠️  {patterns['idle_in_transaction']} idle in transaction (potential leaks)")
                
                if patterns["long_running"] > 0:
                    print(f"    • ⏰ {patterns['long_running']} long-running connections (>5min)")
                
                if patterns["recent"] > 0:
                    print(f"    • 🆕 {patterns['recent']} recent connections (<30s)")
        
        # Record history
        self.connection_history.append({
            "timestamp": timestamp,
            "count": count,
            "total": total_count,
            "change": change
        })
        
        # Alert if approaching limits
        if count >= self.critical_threshold:
            print(f"    🚨 CRITICAL: Very close to connection limit!")
        elif count >= self.warning_threshold:
            print(f"    ⚠️  WARNING: Approaching connection limit")
        
        self.previous_count = count
        return count, connections
    
    def print_connection_details(self, connections, limit=5):
        """Print details for recent/problematic connections."""
        if not connections:
            return
        
        print(f"\n📋 Connection Details (showing last {limit}):")
        
        # Sort by backend_start (newest first)
        sorted_connections = sorted(connections, 
                                  key=lambda x: x[7] if x[7] else datetime.min, 
                                  reverse=True)
        
        for i, conn in enumerate(sorted_connections[:limit], 1):
            pid, user, app, client, state, state_change, query_start, backend_start, query = conn
            
            print(f"  {i}. PID: {pid}")
            print(f"     State: {state}")
            print(f"     App: {app or 'N/A'}")
            print(f"     Started: {backend_start}")
            if query and query.strip():
                print(f"     Query: {query[:50]}...")
            print()
    
    def monitor_continuously(self, interval=3):
        """Monitor connections continuously during test execution."""
        print(f"🔍 Monitoring PostgreSQL connections to '{self.target_database}' database")
        print(f"📊 Max connections: {self.max_connections}")
        print(f"⚠️  Warning threshold: {self.warning_threshold}")
        print(f"🚨 Critical threshold: {self.critical_threshold}")
        print(f"📡 Checking every {interval} seconds")
        print("=" * 80)
        
        try:
            while True:
                count, connections = self.print_connection_summary()
                
                # Show details if connection count is high
                if count >= self.warning_threshold:
                    self.print_connection_details(connections, 3)
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n\n👋 Monitoring stopped")
            self.print_final_report()
    
    def print_final_report(self):
        """Print final analysis report."""
        if not self.connection_history:
            return
        
        print("\n" + "="*80)
        print("📊 CONNECTION ANALYSIS REPORT")
        print("="*80)
        
        max_connections = max(h["count"] for h in self.connection_history)
        max_total = max(h["total"] for h in self.connection_history)
        avg_connections = sum(h["count"] for h in self.connection_history) / len(self.connection_history)
        
        print(f"📈 Peak memfuse connections: {max_connections}/{self.max_connections}")
        print(f"📈 Peak total connections: {max_total}/{self.max_connections}")
        print(f"📊 Average memfuse connections: {avg_connections:.1f}")
        
        # Find connection spikes
        spikes = [h for h in self.connection_history if h["change"] >= 5]
        if spikes:
            print(f"\n⚡ Connection spikes (≥5 connections):")
            for spike in spikes:
                print(f"  {spike['timestamp']}: +{spike['change']} → {spike['count']} connections")
        
        # Find connection drops
        drops = [h for h in self.connection_history if h["change"] <= -5]
        if drops:
            print(f"\n📉 Connection drops (≥5 connections):")
            for drop in drops:
                print(f"  {drop['timestamp']}: {drop['change']} → {drop['count']} connections")
        
        print("\n💡 RECOMMENDATIONS:")
        if max_connections >= self.critical_threshold:
            print("  🚨 Connection limit was reached - this causes test failures")
            print("  🔧 Implement better connection cleanup in test fixtures")
            print("  🔧 Consider increasing PostgreSQL max_connections")
        elif max_connections >= self.warning_threshold:
            print("  ⚠️  Close to connection limit - monitor for leaks")
            print("  🔧 Review test cleanup patterns")
        else:
            print("  ✅ Connection usage within reasonable limits")
        
        print("="*80)


def main():
    """Main function to run the monitor."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor PostgreSQL connections during test execution")
    parser.add_argument("--database", "-d", default="memfuse", 
                       help="Target database to monitor (default: memfuse)")
    parser.add_argument("--interval", "-i", type=int, default=3,
                       help="Check interval in seconds (default: 3)")
    
    args = parser.parse_args()
    
    monitor = TestRunnerConnectionMonitor(args.database)
    monitor.monitor_continuously(args.interval)


if __name__ == "__main__":
    main() 