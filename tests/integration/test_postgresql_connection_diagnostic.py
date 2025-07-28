"""
PostgreSQL-specific diagnostic test to isolate database connection leakage issues.

This test specifically tests PostgreSQL connections to determine if the issue
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
from memfuse_core.utils.config import config_manager


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
            SELECT pid, usename, application_name, client_addr, state, query_start, LEFT(query, 100) as query
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


@pytest.fixture
def force_postgresql_config():
    """Force PostgreSQL configuration for testing."""
    # Set environment variables to force PostgreSQL
    original_env = {
        "POSTGRES_HOST": os.getenv("POSTGRES_HOST"),
        "POSTGRES_PORT": os.getenv("POSTGRES_PORT"),
        "POSTGRES_DB": os.getenv("POSTGRES_DB"),
        "POSTGRES_USER": os.getenv("POSTGRES_USER"),
        "POSTGRES_PASSWORD": os.getenv("POSTGRES_PASSWORD"),
    }
    
    os.environ["POSTGRES_HOST"] = "localhost"
    os.environ["POSTGRES_PORT"] = "5432"
    os.environ["POSTGRES_DB"] = "memfuse"
    os.environ["POSTGRES_USER"] = "postgres"
    os.environ["POSTGRES_PASSWORD"] = "postgres"
    
    # Set configuration to PostgreSQL
    config = {
        "database": {
            "type": "postgres",
            "postgres": {
                "host": "localhost",
                "port": 5432,
                "database": "memfuse",
                "user": "postgres",
                "password": "postgres"
            }
        }
    }
    config_manager.set_config(config)
    
    # Reset DatabaseService singleton to pick up new config
    DatabaseService.reset_instance()
    
    yield
    
    # Restore original environment
    for key, value in original_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    
    # Reset singleton again
    DatabaseService.reset_instance()


def test_postgresql_raw_connection_cleanup(connection_monitor):
    """Test 1: Raw PostgreSQL connection - should show proper cleanup."""
    print("\n🔍 Test 1: Raw PostgreSQL connection cleanup")
    
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
    print("✅ Raw PostgreSQL connection properly cleaned up")


def test_postgresql_singleton_cleanup(connection_monitor, force_postgresql_config):
    """Test 2: DatabaseService singleton with PostgreSQL - likely shows leakage."""
    print("\n🔍 Test 2: DatabaseService singleton with PostgreSQL cleanup")
    
    # Get initial connection count
    initial_count = connection_monitor.get_connection_count()
    print(f"Initial connections: {initial_count}")
    
    # Get database instance through singleton (should use PostgreSQL now)
    db = DatabaseService.get_instance()
    
    # Verify it's using PostgreSQL
    if hasattr(db.backend, 'conn_params'):
        print(f"✅ Using PostgreSQL backend: {db.backend.conn_params}")
    else:
        print("❌ Not using PostgreSQL backend!")
        return
    
    # Check connection count after creation
    after_create_count = connection_monitor.get_connection_count()
    print(f"After getting singleton: {after_create_count}")
    expected_count = initial_count + 1
    
    if after_create_count == expected_count:
        print("✅ Singleton created exactly one connection")
    else:
        print(f"❌ Unexpected connection count - expected {expected_count}, got {after_create_count}")
    
    # Try to close via singleton
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
    
    # Check if connection was leaked
    connection_leaked = final_count > initial_count
    if connection_leaked:
        print(f"❌ Connection leaked! Initial: {initial_count}, Final: {final_count}")
        print("This indicates the problem is in the DatabaseService singleton management")
        
        # Show detailed connection info
        connections = connection_monitor.get_detailed_connections()
        print("\nLeaked connections:")
        for conn in connections:
            print(f"  PID: {conn[0]}, User: {conn[1]}, App: {conn[2]}, State: {conn[4]}")
            if conn[6]:  # query
                print(f"    Query: {conn[6]}")
    else:
        print("✅ DatabaseService properly cleaned up")


def test_integration_test_simulation(connection_monitor, force_postgresql_config):
    """Test 3: Simulate what happens in integration tests."""
    print("\n🔍 Test 3: Integration test simulation")
    
    # Get initial connection count
    initial_count = connection_monitor.get_connection_count()
    print(f"Initial connections: {initial_count}")
    
    # Simulate what integration tests do
    for i in range(3):
        print(f"\n--- Simulating test {i+1} ---")
        
        # Each "test" gets a database instance
        db = DatabaseService.get_instance()
        
        # Check connection count
        current_count = connection_monitor.get_connection_count()
        print(f"Test {i+1} - connections: {current_count}")
        
        # Simulate some database operations
        cursor = db.backend.execute("SELECT 1")
        result = cursor.fetchone()
        cursor.close()
        db.commit()
        
        # This is what SHOULD happen after each test but might not be
        DatabaseService.reset_instance()
        
        # Check if connection was cleaned up
        after_reset_count = connection_monitor.get_connection_count()
        print(f"After reset - connections: {after_reset_count}")
        
        if after_reset_count > initial_count:
            print(f"❌ Test {i+1} leaked connections!")
            break
    
    # Final check
    final_count = connection_monitor.get_connection_count()
    if final_count == initial_count:
        print("✅ All connections properly cleaned up")
    else:
        leaked = final_count - initial_count
        print(f"❌ {leaked} connections leaked overall")
        
        # Show detailed connection info
        connections = connection_monitor.get_detailed_connections()
        print("\nLeaked connections:")
        for conn in connections:
            print(f"  PID: {conn[0]}, User: {conn[1]}, State: {conn[4]}")


if __name__ == "__main__":
    # Can run this directly for manual testing
    monitor = DatabaseConnectionMonitor()
    
    print("=== PostgreSQL Connection Diagnostic Test ===")
    
    try:
        print("\n" + "="*50)
        test_postgresql_raw_connection_cleanup(monitor)
        
        print("\n" + "="*50)
        # Create mock force_postgresql_config fixture for manual testing
        class MockConfig:
            def __enter__(self): 
                os.environ["POSTGRES_HOST"] = "localhost"
                os.environ["POSTGRES_PORT"] = "5432"
                os.environ["POSTGRES_DB"] = "memfuse"
                os.environ["POSTGRES_USER"] = "postgres"
                os.environ["POSTGRES_PASSWORD"] = "postgres"
                
                config = {
                    "database": {
                        "type": "postgres",
                        "postgres": {
                            "host": "localhost",
                            "port": 5432,
                            "database": "memfuse",
                            "user": "postgres",
                            "password": "postgres"
                        }
                    }
                }
                config_manager.set_config(config)
                DatabaseService.reset_instance()
                return self
            
            def __exit__(self, *args):
                DatabaseService.reset_instance()
        
        with MockConfig():
            test_postgresql_singleton_cleanup(monitor, None)
            
        print("\n" + "="*50)
        with MockConfig():
            test_integration_test_simulation(monitor, None)
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*50)
    print("PostgreSQL diagnostic test completed!") 