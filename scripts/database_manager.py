#!/usr/bin/env python3
"""
MemFuse Database Manager

Unified script for database operations including reset, recreate, and validation.
Replaces reset_database.py and recreate_database_schema.py.

Usage:
    python database_manager.py reset      # Clear all data, keep schema
    python database_manager.py recreate   # Drop and recreate complete schema
    python database_manager.py validate   # Validate current schema
    python database_manager.py status     # Show database status
"""

import subprocess
import sys
import argparse
import time
from datetime import datetime
from typing import Optional, Dict, Any

class DatabaseManager:
    """Unified database management for MemFuse."""
    
    def __init__(self):
        self.container_name = 'memfuse-pgai-postgres-1'
        self.db_name = 'memfuse'
        self.db_user = 'postgres'
    
    def run_sql_command(self, sql_command: str, output_format: str = "table") -> Optional[str]:
        """Execute SQL command in PostgreSQL container."""
        try:
            format_flag = "-t" if output_format == "tuples" else ""
            cmd = [
                'docker', 'exec', '-i', self.container_name,
                'psql', '-U', self.db_user, '-d', self.db_name, format_flag, '-c', sql_command
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                print(f"SQL Error: {result.stderr}")
                return None
        except Exception as e:
            print(f"Failed to execute SQL: {e}")
            return None
    
    def check_container_status(self) -> bool:
        """Check if PostgreSQL container is running."""
        try:
            result = subprocess.run([
                'docker', 'ps', '--filter', f'name={self.container_name}', 
                '--format', 'table {{.Names}}\t{{.Status}}'
            ], capture_output=True, text=True)
            
            if self.container_name in result.stdout and 'Up' in result.stdout:
                print(f"✅ PostgreSQL container ({self.container_name}) is running")
                return True
            else:
                print(f"❌ PostgreSQL container ({self.container_name}) is not running")
                print("   Please start with: docker-compose up -d")
                return False
        except Exception as e:
            print(f"❌ Failed to check container status: {e}")
            return False
    
    def get_database_status(self) -> Dict[str, Any]:
        """Get comprehensive database status."""
        print("📊 Database Status")
        print("=" * 50)
        
        status = {
            'container_running': self.check_container_status(),
            'tables': {},
            'triggers': {},
            'functions': {},
            'extensions': {}
        }
        
        if not status['container_running']:
            return status
        
        # Check tables
        tables_result = self.run_sql_command("""
            SELECT tablename, schemaname 
            FROM pg_tables 
            WHERE schemaname = 'public' 
            ORDER BY tablename;
        """, "tuples")
        
        if tables_result:
            tables = [line.strip().split('|')[0] for line in tables_result.split('\n') if line.strip()]
            status['tables'] = {table: True for table in tables}
            print(f"Tables found: {', '.join(tables)}")
        
        # Check m0_episodic specifically
        if 'm0_episodic' in status['tables']:
            count_result = self.run_sql_command("SELECT COUNT(*) FROM m0_episodic;", "tuples")
            if count_result:
                status['m0_episodic_count'] = int(count_result.strip())
                print(f"m0_episodic records: {status['m0_episodic_count']}")
        
        # Check triggers
        trigger_result = self.run_sql_command("""
            SELECT trigger_name 
            FROM information_schema.triggers 
            WHERE trigger_name = 'm0_episodic_embedding_trigger';
        """, "tuples")
        
        status['triggers']['immediate_trigger'] = bool(trigger_result and 'm0_episodic_embedding_trigger' in trigger_result)
        
        # Check functions
        function_result = self.run_sql_command("""
            SELECT proname 
            FROM pg_proc 
            WHERE proname = 'notify_embedding_needed';
        """, "tuples")
        
        status['functions']['notify_function'] = bool(function_result and 'notify_embedding_needed' in function_result)
        
        # Check extensions
        ext_result = self.run_sql_command("SELECT extname FROM pg_extension;", "tuples")
        if ext_result:
            extensions = [line.strip() for line in ext_result.split('\n') if line.strip()]
            status['extensions'] = {ext: True for ext in extensions}
            print(f"Extensions: {', '.join(extensions)}")
        
        return status
    
    def reset_database(self) -> bool:
        """Reset database (clear data, keep schema)."""
        print("🗑️ Resetting Database (Clear Data, Keep Schema)")
        print("=" * 60)
        
        if not self.check_container_status():
            return False
        
        # Get current stats
        current_status = self.get_database_status()
        
        # Clear m0_episodic table if it exists
        if 'm0_episodic' in current_status.get('tables', {}):
            print("Clearing m0_episodic table...")
            result = self.run_sql_command("TRUNCATE TABLE m0_episodic RESTART IDENTITY CASCADE;")
            
            if result is not None:
                print("✅ m0_episodic table cleared")
                
                # Verify empty
                count_result = self.run_sql_command("SELECT COUNT(*) FROM m0_episodic;", "tuples")
                if count_result and int(count_result.strip()) == 0:
                    print("✅ Table is empty")
                else:
                    print("❌ Table is not empty after reset")
                    return False
            else:
                print("❌ Failed to clear m0_episodic table")
                return False
        else:
            print("⚠️  m0_episodic table not found")
        
        print("✅ Database reset completed successfully")
        return True
    
    def recreate_database_schema(self) -> bool:
        """Recreate complete database schema."""
        print("🔄 Recreating Complete Database Schema")
        print("=" * 60)
        print("⚠️  WARNING: This will DROP ALL EXISTING TABLES!")
        
        if not self.check_container_status():
            return False
        
        # Drop all existing tables
        print("\n1. Dropping existing tables...")
        tables_result = self.run_sql_command("""
            SELECT tablename FROM pg_tables WHERE schemaname = 'public';
        """, "tuples")
        
        if tables_result:
            tables = [line.strip() for line in tables_result.split('\n') if line.strip()]
            for table in tables:
                print(f"   Dropping {table}...")
                result = self.run_sql_command(f"DROP TABLE IF EXISTS {table} CASCADE;")
                if result is not None:
                    print(f"   ✅ {table} dropped")
                else:
                    print(f"   ❌ Failed to drop {table}")
                    return False
        
        # Create extensions
        print("\n2. Creating extensions...")
        extensions = ["CREATE EXTENSION IF NOT EXISTS vector;"]
        
        for ext_sql in extensions:
            result = self.run_sql_command(ext_sql)
            if result is not None:
                print(f"   ✅ Extension created")
            else:
                print(f"   ❌ Failed to create extension")
                return False
        
        # Create m0_episodic table
        print("\n3. Creating m0_episodic table...")
        table_sql = """
        CREATE TABLE m0_episodic (
            id              TEXT PRIMARY KEY,
            content         TEXT NOT NULL,
            metadata        JSONB DEFAULT '{}'::jsonb,
            embedding       VECTOR(384),
            needs_embedding BOOLEAN DEFAULT TRUE,
            retry_count     INTEGER DEFAULT 0,
            last_retry_at   TIMESTAMP,
            retry_status    TEXT DEFAULT 'pending',
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        result = self.run_sql_command(table_sql)
        if result is not None:
            print("   ✅ m0_episodic table created")
        else:
            print("   ❌ Failed to create m0_episodic table")
            return False
        
        # Create indexes
        print("\n4. Creating indexes...")
        indexes = [
            "CREATE INDEX m0_episodic_needs_embedding_idx ON m0_episodic (needs_embedding) WHERE needs_embedding = TRUE;",
            "CREATE INDEX m0_episodic_retry_status_idx ON m0_episodic (retry_status);",
            "CREATE INDEX m0_episodic_retry_count_idx ON m0_episodic (retry_count);",
            "CREATE INDEX m0_episodic_created_at_idx ON m0_episodic (created_at);",
            "CREATE INDEX m0_episodic_embedding_idx ON m0_episodic USING hnsw (embedding vector_cosine_ops);"
        ]
        
        for index_sql in indexes:
            result = self.run_sql_command(index_sql)
            if result is not None:
                print("   ✅ Index created")
            else:
                print("   ❌ Failed to create index")
                return False
        
        # Create immediate trigger system
        print("\n5. Creating immediate trigger system...")
        
        # Create notification function
        function_sql = """
        CREATE OR REPLACE FUNCTION notify_embedding_needed()
        RETURNS TRIGGER AS $$
        BEGIN
            IF NEW.needs_embedding = TRUE THEN
                PERFORM pg_notify('embedding_needed', NEW.id::text);
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        """
        
        result = self.run_sql_command(function_sql)
        if result is not None:
            print("   ✅ Notification function created")
        else:
            print("   ❌ Failed to create notification function")
            return False
        
        # Create trigger
        trigger_sql = """
        CREATE TRIGGER m0_episodic_embedding_trigger
            AFTER INSERT OR UPDATE OF needs_embedding ON m0_episodic
            FOR EACH ROW
            EXECUTE FUNCTION notify_embedding_needed();
        """
        
        result = self.run_sql_command(trigger_sql)
        if result is not None:
            print("   ✅ Immediate trigger created")
        else:
            print("   ❌ Failed to create immediate trigger")
            return False
        
        print("\n✅ Database schema recreation completed successfully")
        return True
    
    def validate_schema(self) -> bool:
        """Validate current database schema."""
        print("✅ Validating Database Schema")
        print("=" * 50)
        
        if not self.check_container_status():
            return False
        
        validation_results = []
        
        # Check m0_episodic table
        table_result = self.run_sql_command("\\d m0_episodic")
        if table_result and 'needs_embedding' in table_result:
            validation_results.append(("m0_episodic table", True))
            print("✅ m0_episodic table exists with correct structure")
        else:
            validation_results.append(("m0_episodic table", False))
            print("❌ m0_episodic table missing or incorrect")
        
        # Check trigger
        trigger_result = self.run_sql_command("""
            SELECT COUNT(*) FROM information_schema.triggers 
            WHERE trigger_name = 'm0_episodic_embedding_trigger';
        """, "tuples")
        
        if trigger_result and int(trigger_result.strip()) > 0:
            validation_results.append(("immediate trigger", True))
            print("✅ Immediate trigger configured")
        else:
            validation_results.append(("immediate trigger", False))
            print("❌ Immediate trigger missing")
        
        # Check function
        function_result = self.run_sql_command("""
            SELECT COUNT(*) FROM pg_proc WHERE proname = 'notify_embedding_needed';
        """, "tuples")
        
        if function_result and int(function_result.strip()) > 0:
            validation_results.append(("notification function", True))
            print("✅ Notification function exists")
        else:
            validation_results.append(("notification function", False))
            print("❌ Notification function missing")
        
        # Check vector extension
        vector_result = self.run_sql_command("""
            SELECT COUNT(*) FROM pg_extension WHERE extname = 'vector';
        """, "tuples")
        
        if vector_result and int(vector_result.strip()) > 0:
            validation_results.append(("vector extension", True))
            print("✅ Vector extension installed")
        else:
            validation_results.append(("vector extension", False))
            print("❌ Vector extension missing")
        
        # Summary
        passed = sum(1 for _, result in validation_results if result)
        total = len(validation_results)
        
        print(f"\nValidation Summary: {passed}/{total} checks passed")
        
        if passed == total:
            print("🎉 Schema validation PASSED - System ready!")
            return True
        else:
            print("💥 Schema validation FAILED - System needs attention")
            return False


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="MemFuse Database Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python database_manager.py status      # Show current status
    python database_manager.py reset       # Clear data, keep schema
    python database_manager.py recreate    # Drop and recreate schema
    python database_manager.py validate    # Validate schema
        """
    )
    
    parser.add_argument(
        'action',
        choices=['status', 'reset', 'recreate', 'validate'],
        help='Action to perform'
    )
    
    args = parser.parse_args()
    
    print(f"🔧 MemFuse Database Manager")
    print(f"Action: {args.action}")
    print(f"Started at: {datetime.now()}")
    print("=" * 60)
    
    manager = DatabaseManager()
    
    try:
        if args.action == 'status':
            status = manager.get_database_status()
            success = status['container_running']
            
        elif args.action == 'reset':
            success = manager.reset_database()
            
        elif args.action == 'recreate':
            success = manager.recreate_database_schema()
            
        elif args.action == 'validate':
            success = manager.validate_schema()
            
        else:
            print(f"Unknown action: {args.action}")
            success = False
        
        print(f"\nCompleted at: {datetime.now()}")
        
        if success:
            print(f"✅ {args.action.title()} completed successfully!")
            sys.exit(0)
        else:
            print(f"❌ {args.action.title()} failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
