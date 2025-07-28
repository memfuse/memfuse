"""
Diagnostic test to isolate database connection leakage issues.

This test creates a minimal connection test to determine if the issue
is on the server side or in the test code.
"""

import pytest
import psycopg2
import time
import sys
import os
from pathlib import Path

# Add src to path 
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from memfuse_core.database.postgres import PostgresDB
from memfuse_core.services.database_service import DatabaseService


class DatabaseConnectionMonitor:
    """Helper class to monitor PostgreSQL connections."""
    
    def __init__(self):
        self.conn_params = {
            "host": "localhost",
            "port": 5432,
            "database": "postgres",  # Connect to postgres db to monitor
            "user": "postgres",
            "password": "postgres"
        }
    
    def get_connection_count(self, target_database="memfuse"):
        """Get the number of active connections to target database."""
        conn = psycopg2.connect(**self.conn_params)
        cursor = conn.cursor()
        
        # Query to get connection count for specific database
        cursor.execute("""
            SELECT count(*) 
            FROM pg_stat_activity 
            WHERE datname = %s
        """, (target_database,))
        
        count = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        return count
    
    def get_detailed_connections(self, target_database="memfuse"):
        """Get detailed information about active connections."""
        conn = psycopg2.connect(**self.conn_params)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT pid, usename, application_name, client_addr, state, query_start, query
            FROM pg_stat_activity 
            WHERE datname = %s
        """, (target_database,))
        
        connections = cursor.fetchall()
        cursor.close()
        conn.close()
        return connections


@pytest.fixture
def connection_monitor():
    """Provide database connection monitor."""
    return DatabaseConnectionMonitor()


def test_raw_database_connection_cleanup(connection_monitor):
    """Test 1: Raw database connection - should show proper cleanup."""
    print("\n🔍 Test 1: Raw database connection cleanup")
    
    # Get initial connection count
    initial_count = connection_monitor.get_connection_count()
    print(f"Initial connections: {initial_count}")
    
    # Create and close a raw connection
    db = PostgresDB(
        host="localhost",
        port=5432,
        database="memfuse",
        user="postgres", 
        password="postgres"
    )
    
    # Check connection count after creation
    after_create_count = connection_monitor.get_connection_count()
    print(f"After creation: {after_create_count}")
    
    # Explicitly close the connection
    db.close()
    
    # Wait a moment for connection to be released
    time.sleep(0.5)
    
    # Check final connection count
    final_count = connection_monitor.get_connection_count()
    print(f"After close: {final_count}")
    
    # Verify connection was properly released
    assert final_count == initial_count, f"Connection not released! Initial: {initial_count}, Final: {final_count}"
    print("✅ Raw connection properly cleaned up")


def test_database_service_singleton_cleanup(connection_monitor):
    """Test 2: DatabaseService singleton - likely shows leakage."""
    print("\n🔍 Test 2: DatabaseService singleton cleanup")
    
    # Get initial connection count
    initial_count = connection_monitor.get_connection_count()
    print(f"Initial connections: {initial_count}")
    
    # Get database instance through singleton
    db = DatabaseService.get_instance()
    
    # Check connection count after creation
    after_create_count = connection_monitor.get_connection_count()
    print(f"After getting singleton: {after_create_count}")
    
    # Try to close via singleton (this may not work properly)
    try:
        db.close()
        print("Called db.close()")
    except Exception as e:
        print(f"Error calling db.close(): {e}")
    
    # Reset singleton instance
    DatabaseService.reset_instance()
    print("Called DatabaseService.reset_instance()")
    
    # Wait a moment for connection to be released
    time.sleep(0.5)
    
    # Check final connection count
    final_count = connection_monitor.get_connection_count()
    print(f"After reset: {final_count}")
    
    # This test will likely fail, showing the leakage
    connection_leaked = final_count > initial_count
    if connection_leaked:
        print(f"❌ Connection leaked! Initial: {initial_count}, Final: {final_count}")
        print("This indicates the problem is in the DatabaseService singleton management")
    else:
        print("✅ DatabaseService properly cleaned up")
        
    # Show detailed connection info if there's leakage
    if connection_leaked:
        connections = connection_monitor.get_detailed_connections()
        print("\nActive connections:")
        for conn in connections:
            print(f"  PID: {conn[0]}, User: {conn[1]}, App: {conn[2]}, State: {conn[4]}")


def test_multiple_singleton_calls(connection_monitor):
    """Test 3: Multiple calls to singleton - shows accumulation."""
    print("\n🔍 Test 3: Multiple singleton calls")
    
    # Get initial connection count
    initial_count = connection_monitor.get_connection_count()
    print(f"Initial connections: {initial_count}")
    
    # Call singleton multiple times (this should reuse the same instance)
    db1 = DatabaseService.get_instance()
    db2 = DatabaseService.get_instance()
    db3 = DatabaseService.get_instance()
    
    # These should all be the same instance
    assert db1 is db2 is db3, "Singleton not working correctly"
    
    # Check connection count (should only be +1)
    after_create_count = connection_monitor.get_connection_count()
    print(f"After 3 singleton calls: {after_create_count}")
    
    expected_count = initial_count + 1
    if after_create_count == expected_count:
        print("✅ Singleton working correctly - only one connection created")
    else:
        print(f"❌ Singleton issue - expected {expected_count}, got {after_create_count}")
        
    # Cleanup
    DatabaseService.reset_instance()
    time.sleep(0.5)
    final_count = connection_monitor.get_connection_count()
    print(f"After cleanup: {final_count}")


if __name__ == "__main__":
    # Can run this directly for manual testing
    monitor = DatabaseConnectionMonitor()
    
    print("=== Database Connection Diagnostic Test ===")
    
    try:
        print("\n" + "="*50)
        test_raw_database_connection_cleanup(monitor)
        
        print("\n" + "="*50)  
        test_database_service_singleton_cleanup(monitor)
        
        print("\n" + "="*50)
        test_multiple_singleton_calls(monitor)
        
    except Exception as e:
        print(f"Test failed with error: {e}")
    
    print("\n" + "="*50)
    print("Diagnostic test completed!") 