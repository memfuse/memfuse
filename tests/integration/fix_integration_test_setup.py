#!/usr/bin/env python3
"""
Fix for integration test setup to prevent database connection leaks.

This script demonstrates the proper way to set up and tear down
database connections in integration tests.
"""

import pytest
import time
import sys
import os
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from memfuse_core.services.database_service import DatabaseService
from memfuse_core.utils.config import config_manager


class DatabaseConnectionFixer:
    """Helper class to demonstrate proper database connection management."""
    
    def __init__(self):
        self.initial_count = None
        
    def setup_postgresql_config(self):
        """Set up PostgreSQL configuration properly."""
        # Set environment variables
        os.environ["POSTGRES_HOST"] = "localhost"
        os.environ["POSTGRES_PORT"] = "5432"
        os.environ["POSTGRES_DB"] = "memfuse"
        os.environ["POSTGRES_USER"] = "postgres"
        os.environ["POSTGRES_PASSWORD"] = "postgres"
        
        # Set configuration
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
            },
            "embedding": {
                "model": "all-MiniLM-L6-v2",
                "dimension": 384
            },
            "store": {
                "backend": "pgai"
            }
        }
        config_manager.set_config(config)
        
        print("✅ PostgreSQL configuration set up")
        
    def get_connection_count(self):
        """Get current PostgreSQL connection count."""
        import psycopg2
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database="postgres",
            user="postgres",
            password="postgres"
        )
        cursor = conn.cursor()
        cursor.execute("SELECT count(*) FROM pg_stat_activity WHERE datname = 'memfuse'")
        count = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        return count
    
    def demonstrate_broken_test_pattern(self):
        """Demonstrate the BROKEN test pattern that causes leaks."""
        print("\n🔴 BROKEN PATTERN - This causes connection leaks:")
        
        initial_count = self.get_connection_count()
        print(f"Initial connections: {initial_count}")
        
        for i in range(3):
            print(f"\n--- Test {i+1} (BROKEN) ---")
            
            # This is what the current tests do - WRONG!
            db = DatabaseService.get_instance()
            print(f"Created database instance")
            
            # Perform some operations
            try:
                cursor = db.backend.execute("SELECT 1")
                cursor.close()
                db.backend.commit()
                print("Performed database operations")
            except Exception as e:
                print(f"Operation error: {e}")
            
            # The test ends here WITHOUT proper cleanup - THIS IS THE PROBLEM!
            
            current_count = self.get_connection_count()
            print(f"Connections after test {i+1}: {current_count}")
            
            if current_count > initial_count:
                leaked = current_count - initial_count
                print(f"❌ {leaked} connections leaked!")
        
        final_count = self.get_connection_count()
        print(f"\nFinal connection count: {final_count}")
        print(f"Total leaked connections: {final_count - initial_count}")
        
        # Reset for next demonstration
        DatabaseService.reset_instance()
        time.sleep(0.5)
        
    def demonstrate_fixed_test_pattern(self):
        """Demonstrate the FIXED test pattern that prevents leaks."""
        print("\n🟢 FIXED PATTERN - This prevents connection leaks:")
        
        initial_count = self.get_connection_count()
        print(f"Initial connections: {initial_count}")
        
        for i in range(3):
            print(f"\n--- Test {i+1} (FIXED) ---")
            
            # Reset singleton before each test - CRITICAL!
            DatabaseService.reset_instance()
            print("✅ Reset DatabaseService singleton")
            
            # Get database instance
            db = DatabaseService.get_instance()
            print("Created database instance")
            
            # Perform some operations
            try:
                cursor = db.backend.execute("SELECT 1")
                cursor.close()
                db.backend.commit()
                print("Performed database operations")
            except Exception as e:
                print(f"Operation error: {e}")
            
            # CRITICAL: Cleanup after each test
            try:
                db.close()
                print("✅ Called db.close()")
            except Exception as e:
                print(f"Close error: {e}")
            
            # Reset singleton after each test - CRITICAL!
            DatabaseService.reset_instance()
            print("✅ Reset DatabaseService singleton")
            
            # Small delay to allow connection cleanup
            time.sleep(0.1)
            
            current_count = self.get_connection_count()
            print(f"Connections after test {i+1}: {current_count}")
            
            if current_count > initial_count:
                leaked = current_count - initial_count
                print(f"❌ Still {leaked} connections leaked!")
            else:
                print("✅ No connection leaks!")
        
        final_count = self.get_connection_count()
        print(f"\nFinal connection count: {final_count}")
        if final_count == initial_count:
            print("✅ All connections properly cleaned up!")
        else:
            print(f"❌ {final_count - initial_count} connections still leaked")
    
    def demonstrate_service_shutdown_pattern(self):
        """Demonstrate proper service shutdown pattern."""
        print("\n🔵 SERVICE SHUTDOWN PATTERN - For TestClient tests:")
        
        initial_count = self.get_connection_count()
        print(f"Initial connections: {initial_count}")
        
        # Simulate service initialization
        from memfuse_core.services.service_initializer import ServiceInitializer
        from omegaconf import OmegaConf
        import asyncio
        
        # Create proper configuration
        cfg = OmegaConf.create({
            "database": {
                "type": "postgres",
                "postgres": {
                    "host": "localhost",
                    "port": 5432,
                    "database": "memfuse",
                    "user": "postgres",
                    "password": "postgres"
                }
            },
            "embedding": {
                "model": "all-MiniLM-L6-v2",
                "dimension": 384
            },
            "store": {
                "backend": "pgai"
            }
        })
        
        # Initialize services
        service_initializer = ServiceInitializer()
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            print("Initializing services...")
            success = loop.run_until_complete(service_initializer.initialize_all_services(cfg))
            
            if success:
                print("✅ Services initialized")
                
                after_init_count = self.get_connection_count()
                print(f"Connections after service init: {after_init_count}")
                
                # Simulate test operations
                print("Performing test operations...")
                time.sleep(0.5)
                
                # CRITICAL: Proper shutdown
                print("Shutting down services...")
                shutdown_success = loop.run_until_complete(service_initializer.shutdown_all_services())
                
                if shutdown_success:
                    print("✅ Services shut down properly")
                else:
                    print("❌ Service shutdown failed")
                
                # Reset database singleton
                DatabaseService.reset_instance()
                print("✅ Database singleton reset")
                
                # Check final state
                time.sleep(0.5)
                final_count = self.get_connection_count()
                print(f"Final connections: {final_count}")
                
                if final_count <= initial_count:
                    print("✅ Service shutdown prevented connection leaks!")
                else:
                    print(f"❌ {final_count - initial_count} connections still leaked")
            else:
                print("❌ Service initialization failed")
                
        except Exception as e:
            print(f"❌ Error during service test: {e}")
            import traceback
            traceback.print_exc()
        finally:
            loop.close()
    
    def print_solution_summary(self):
        """Print a summary of the solution."""
        print("\n" + "="*60)
        print("🎯 SOLUTION SUMMARY")
        print("="*60)
        print("""
The connection leakage problem is NOT in the server side code.
The issue is in the integration test setup. Here's how to fix it:

🔧 FIXES NEEDED:

1. CONFTEST.PY FIXES:
   ✅ Add DatabaseService.reset_instance() in setup_integration_environment
   ✅ Add proper cleanup in the client fixture
   ✅ Force PostgreSQL configuration consistently
   ✅ Add proper service shutdown

2. TEST PATTERN FIXES:
   ✅ Reset DatabaseService singleton before each test
   ✅ Call db.close() after test operations
   ✅ Reset DatabaseService singleton after each test
   ✅ Use proper service shutdown for TestClient tests

3. CONFIGURATION FIXES:
   ✅ Ensure PostgreSQL config is used consistently
   ✅ Set environment variables properly
   ✅ Don't rely on SQLite fallback

🚀 IMPLEMENTATION:
   Use the 'conftest_fixed.py' file I created, which includes all these fixes.
   
📊 MONITORING:
   Use the database_connection_monitor.py script to watch connections in real-time
   during test runs to verify the fixes work.
        """)
        print("="*60)


def main():
    """Main function to demonstrate the fixes."""
    fixer = DatabaseConnectionFixer()
    
    print("🔍 Database Connection Leak Fix Demonstration")
    print("="*60)
    
    try:
        # Set up proper configuration
        fixer.setup_postgresql_config()
        
        # Demonstrate the broken pattern
        fixer.demonstrate_broken_test_pattern()
        
        # Demonstrate the fixed pattern  
        fixer.demonstrate_fixed_test_pattern()
        
        # Demonstrate service shutdown pattern
        fixer.demonstrate_service_shutdown_pattern()
        
        # Print solution summary
        fixer.print_solution_summary()
        
    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 