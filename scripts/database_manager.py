#!/usr/bin/env python3
"""
MemFuse Database Manager

Unified script for database operations including reset, recreate, and validation.
Replaces reset_database.py and recreate_database_schema.py.

Features:
- Environment variable configuration support
- Robust error handling and retry mechanisms
- Detailed status reporting with color-coded output
- Configurable timeouts and connection settings
- Comprehensive validation and health checks

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
import os
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum


# Constants
DEFAULT_CONTAINER_NAME = "memfuse-pgai-postgres"
DEFAULT_DB_NAME = "memfuse"
DEFAULT_DB_USER = "postgres"
DEFAULT_TIMEOUT = 60
DEFAULT_RETRY_COUNT = 3
DEFAULT_RETRY_DELAY = 2

# Required extensions for MemFuse
REQUIRED_EXTENSIONS_SQL = [
    "CREATE EXTENSION IF NOT EXISTS vector;",  # pgvector for embedding storage
]

# Optional extensions (not required for MemFuse operation)
# Note: MemFuse implements its own pgai-like functionality and doesn't require TimescaleDB's pgai extension
OPTIONAL_EXTENSIONS_SQL = [
    "CREATE EXTENSION IF NOT EXISTS timescaledb;",  # TimescaleDB features (if available)
    # "CREATE EXTENSION IF NOT EXISTS pgai;"  # Not needed - we have our own implementation
]

M0_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_m0_session_id ON m0_raw (session_id);",
    "CREATE INDEX IF NOT EXISTS idx_m0_user_id ON m0_raw (user_id);",
    "CREATE INDEX IF NOT EXISTS idx_m0_message_role ON m0_raw (message_role);",
    "CREATE INDEX IF NOT EXISTS idx_m0_round_id ON m0_raw (round_id);",
    "CREATE INDEX IF NOT EXISTS idx_m0_needs_embedding ON m0_raw (needs_embedding) WHERE needs_embedding = TRUE;",
    "CREATE INDEX IF NOT EXISTS idx_m0_retry_status ON m0_raw (retry_status);",
    "CREATE INDEX IF NOT EXISTS idx_m0_retry_count ON m0_raw (retry_count);",
    "CREATE INDEX IF NOT EXISTS idx_m0_created_at ON m0_raw (created_at);",
    "CREATE INDEX IF NOT EXISTS idx_m0_updated_at ON m0_raw (updated_at);",
    "CREATE INDEX IF NOT EXISTS idx_m0_metadata_gin ON m0_raw USING gin (metadata);",
    "CREATE INDEX IF NOT EXISTS idx_m0_embedding_cosine ON m0_raw USING hnsw (embedding vector_cosine_ops);"
]

M1_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_m1_source_id ON m1_episodic (source_id);",
    "CREATE INDEX IF NOT EXISTS idx_m1_source_session ON m1_episodic (source_session_id);",
    "CREATE INDEX IF NOT EXISTS idx_m1_source_user ON m1_episodic (source_user_id);",
    "CREATE INDEX IF NOT EXISTS idx_m1_episode_type ON m1_episodic (episode_type) WHERE episode_type IS NOT NULL;",
    "CREATE INDEX IF NOT EXISTS idx_m1_episode_category_gin ON m1_episodic USING gin (episode_category);",
    "CREATE INDEX IF NOT EXISTS idx_m1_confidence ON m1_episodic (confidence);",
    "CREATE INDEX IF NOT EXISTS idx_m1_needs_embedding ON m1_episodic (needs_embedding) WHERE needs_embedding = TRUE;",
    "CREATE INDEX IF NOT EXISTS idx_m1_retry_status ON m1_episodic (retry_status);",
    "CREATE INDEX IF NOT EXISTS idx_m1_retry_count ON m1_episodic (retry_count);",
    "CREATE INDEX IF NOT EXISTS idx_m1_created_at ON m1_episodic (created_at);",
    "CREATE INDEX IF NOT EXISTS idx_m1_updated_at ON m1_episodic (updated_at);",
    "CREATE INDEX IF NOT EXISTS idx_m1_entities_gin ON m1_episodic USING gin (entities);",
    "CREATE INDEX IF NOT EXISTS idx_m1_temporal_gin ON m1_episodic USING gin (temporal_info);",
    "CREATE INDEX IF NOT EXISTS idx_m1_metadata_gin ON m1_episodic USING gin (metadata);",
    "CREATE INDEX IF NOT EXISTS idx_m1_embedding_cosine ON m1_episodic USING hnsw (embedding vector_cosine_ops);"
]

M0_TABLE_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS m0_raw (
    -- Primary identification
    id              TEXT PRIMARY KEY,

    -- Core content
    content         TEXT NOT NULL,

    -- Context tracking
    session_id      TEXT,  -- Session context
    user_id         TEXT,  -- User context
    message_role    TEXT,  -- Message role (user, assistant, system)
    round_id        TEXT,  -- Round context

    -- General metadata
    metadata        JSONB DEFAULT '{}'::jsonb,

    -- PgAI embedding infrastructure
    embedding       VECTOR(384),  -- 384-dimensional embedding vector
    needs_embedding BOOLEAN DEFAULT TRUE,  -- Flag for automatic embedding generation
    retry_count     INTEGER DEFAULT 0,  -- Number of embedding retry attempts
    last_retry_at   TIMESTAMP,  -- Timestamp of last retry attempt
    retry_status    TEXT DEFAULT 'pending' CHECK (retry_status IN ('pending', 'processing', 'completed', 'failed')),

    -- Audit timestamps
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

M1_TABLE_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS m1_episodic (
    -- Primary identification
    id TEXT PRIMARY KEY,

    -- Source tracking (links back to M0 raw data)
    source_id TEXT,  -- References m0_raw.id
    source_session_id TEXT,  -- Session context for episode
    source_user_id TEXT,  -- User context for episode

    -- Episode content and metadata
    episode_content TEXT NOT NULL,
    episode_type TEXT,  -- Open-ended episode type, no constraints for extensibility
    episode_category JSONB DEFAULT '{}'::jsonb,  -- Flexible categorization system
    confidence FLOAT NOT NULL DEFAULT 1.0 CHECK (confidence >= 0.0 AND confidence <= 1.0),

    -- Structured episode data
    entities JSONB DEFAULT '[]'::jsonb,  -- Extracted entities from episode
    temporal_info JSONB DEFAULT '{}'::jsonb,  -- Temporal information (dates, times, etc.)
    source_context TEXT,  -- Brief context about where episode came from

    -- General metadata
    metadata JSONB DEFAULT '{}'::jsonb,

    -- PgAI embedding infrastructure (identical to M0)
    embedding VECTOR(384),  -- 384-dimensional embedding vector
    needs_embedding BOOLEAN DEFAULT TRUE,  -- Flag for automatic embedding generation
    retry_count INTEGER DEFAULT 0,  -- Number of embedding retry attempts
    last_retry_at TIMESTAMP,  -- Timestamp of last retry attempt
    retry_status TEXT DEFAULT 'pending' CHECK (retry_status IN ('pending', 'processing', 'completed', 'failed')),

    -- Audit timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

M0_NOTIFICATION_FUNCTION_SQL = """
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

M0_TRIGGER_SQL = """
CREATE TRIGGER m0_raw_embedding_trigger
    AFTER INSERT OR UPDATE OF needs_embedding ON m0_raw
    FOR EACH ROW
    EXECUTE FUNCTION notify_embedding_needed();
"""

M1_UPDATE_FUNCTION_SQL = """
CREATE OR REPLACE FUNCTION update_m1_episodic_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
"""

M1_UPDATE_TRIGGER_SQL = """
DROP TRIGGER IF EXISTS trigger_update_m1_episodic_updated_at ON m1_episodic;
CREATE TRIGGER trigger_update_m1_episodic_updated_at
    BEFORE UPDATE ON m1_episodic
    FOR EACH ROW
    EXECUTE FUNCTION update_m1_episodic_updated_at();
"""

M1_NOTIFICATION_FUNCTION_SQL = """
CREATE OR REPLACE FUNCTION notify_m1_embedding_needed()
RETURNS TRIGGER AS $$
BEGIN
    -- Only notify if needs_embedding is TRUE and content exists
    IF NEW.needs_embedding = TRUE AND NEW.episode_content IS NOT NULL THEN
        PERFORM pg_notify('embedding_needed', 'm1_episodic:' || NEW.id::text);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
"""

M1_EMBEDDING_TRIGGER_SQL = """
DROP TRIGGER IF EXISTS trigger_m1_embedding_notification ON m1_episodic;
CREATE TRIGGER trigger_m1_embedding_notification
    AFTER INSERT OR UPDATE OF needs_embedding ON m1_episodic
    FOR EACH ROW
    EXECUTE FUNCTION notify_m1_embedding_needed();
"""

# Note: This implements MemFuse's custom pgai-like functionality
# We don't need TimescaleDB's pgai extension as we have our own event-driven embedding system


class StatusLevel(Enum):
    """Status message levels."""
    INFO = "INFO"
    SUCCESS = "SUCCESS"
    WARNING = "WARNING"
    ERROR = "ERROR"


class OperationResult(Enum):
    """Operation result types."""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    SKIPPED = "skipped"


@dataclass
class DatabaseConfig:
    """Configuration for database operations."""
    container_name: str = DEFAULT_CONTAINER_NAME
    db_name: str = DEFAULT_DB_NAME
    db_user: str = DEFAULT_DB_USER
    timeout: int = DEFAULT_TIMEOUT
    retry_count: int = DEFAULT_RETRY_COUNT
    retry_delay: int = DEFAULT_RETRY_DELAY

    @classmethod
    def from_env(cls) -> 'DatabaseConfig':
        """Create configuration from environment variables."""
        return cls(
            container_name=os.getenv("MEMFUSE_DB_CONTAINER", DEFAULT_CONTAINER_NAME),
            db_name=os.getenv("MEMFUSE_DB_NAME", DEFAULT_DB_NAME),
            db_user=os.getenv("MEMFUSE_DB_USER", DEFAULT_DB_USER),
            timeout=int(os.getenv("MEMFUSE_DB_TIMEOUT", str(DEFAULT_TIMEOUT))),
            retry_count=int(os.getenv("MEMFUSE_DB_RETRY_COUNT", str(DEFAULT_RETRY_COUNT))),
            retry_delay=int(os.getenv("MEMFUSE_DB_RETRY_DELAY", str(DEFAULT_RETRY_DELAY)))
        )

class DatabaseManager:
    """Unified database management for MemFuse."""

    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or DatabaseConfig.from_env()

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

    def _run_command_with_retry(self, cmd: List[str], timeout: int = None, retries: int = None) -> subprocess.CompletedProcess:
        """Run a command with retry mechanism."""
        timeout = timeout or self.config.timeout
        retries = retries or self.config.retry_count

        for attempt in range(retries):
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
                return result
            except subprocess.TimeoutExpired as e:
                if attempt == retries - 1:
                    raise e
                self.print_status(
                    f"Command timed out (attempt {attempt + 1}/{retries}), retrying...",
                    StatusLevel.WARNING
                )
                time.sleep(self.config.retry_delay)
            except Exception as e:
                if attempt == retries - 1:
                    raise e
                self.print_status(
                    f"Command failed (attempt {attempt + 1}/{retries}): {e}",
                    StatusLevel.WARNING
                )
                time.sleep(self.config.retry_delay)

        raise RuntimeError(f"Command failed after {retries} attempts")
    
    def run_sql_command(self, sql_command: str, output_format: str = "table", ignore_errors: bool = False) -> Optional[str]:
        """Execute SQL command in PostgreSQL container with retry mechanism."""
        try:
            format_flag = "-t" if output_format == "tuples" else ""
            cmd = [
                'docker', 'exec', '-i', self.config.container_name,
                'psql', '-U', self.config.db_user, '-d', self.config.db_name
            ]

            # Only add format flag if it's not empty
            if format_flag:
                cmd.append(format_flag)

            cmd.extend(['-c', sql_command])

            result = self._run_command_with_retry(cmd, timeout=self.config.timeout)

            if result.returncode == 0:
                return result.stdout.strip()
            else:
                error_msg = result.stderr.strip()
                if ignore_errors:
                    self.print_status(f"SQL Warning: {error_msg}", StatusLevel.WARNING)
                    return None
                else:
                    self.print_status(f"SQL Error: {error_msg}", StatusLevel.ERROR)
                    return None

        except subprocess.TimeoutExpired:
            self.print_status(f"SQL command timed out after {self.config.timeout}s", StatusLevel.ERROR)
            return None
        except Exception as e:
            self.print_status(f"Failed to execute SQL: {e}", StatusLevel.ERROR)
            return None
    
    def check_container_status(self) -> bool:
        """Check if PostgreSQL container is running."""
        try:
            cmd = [
                'docker', 'ps', '--filter', f'name={self.config.container_name}',
                '--format', 'table {{.Names}}\t{{.Status}}'
            ]
            result = self._run_command_with_retry(cmd, timeout=10, retries=2)

            if self.config.container_name in result.stdout and 'Up' in result.stdout:
                self.print_status(
                    f"PostgreSQL container ({self.config.container_name}) is running",
                    StatusLevel.SUCCESS
                )
                return True
            else:
                self.print_status(
                    f"PostgreSQL container ({self.config.container_name}) is not running",
                    StatusLevel.ERROR
                )
                self.print_status("Please start with: docker-compose up -d", StatusLevel.INFO)
                return False
        except Exception as e:
            self.print_status(f"Failed to check container status: {e}", StatusLevel.ERROR)
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
            tables = []
            for line in tables_result.split('\n'):
                if line.strip():
                    # Handle both pipe-separated and space-separated formats
                    if '|' in line:
                        table_name = line.strip().split('|')[0].strip()
                    else:
                        table_name = line.strip().split()[0].strip()
                    if table_name:
                        tables.append(table_name)
            status['tables'] = {table: True for table in tables}
            print(f"Tables found: {', '.join(f'{table:<18}' for table in tables)}")
        
        # Check m0_raw specifically
        if 'm0_raw' in status['tables']:
            count_result = self.run_sql_command("SELECT COUNT(*) FROM m0_raw;", "tuples")
            if count_result:
                status['m0_raw_count'] = int(count_result.strip())
                print(f"m0_raw records: {status['m0_raw_count']}")
        
        # Check triggers
        trigger_result = self.run_sql_command("""
            SELECT trigger_name 
            FROM information_schema.triggers 
            WHERE trigger_name = 'm0_raw_embedding_trigger';
        """, "tuples")
        
        status['triggers']['immediate_trigger'] = bool(trigger_result and 'm0_raw_embedding_trigger' in trigger_result)
        
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

        # Get all tables to clear
        tables_to_clear = [
            'messages', 'rounds', 'sessions', 'agents', 'users', 'api_keys', 'knowledge', 'm0_raw', 'm1_episodic', 'm2_semantic'
        ]

        # Clear all tables
        for table in tables_to_clear:
            if table in current_status.get('tables', {}):
                print(f"Clearing {table} table...")
                result = self.run_sql_command(f"TRUNCATE TABLE {table} RESTART IDENTITY CASCADE;")

                if result is not None:
                    print(f"✅ {table} table cleared")

                    # Verify empty
                    count_result = self.run_sql_command(f"SELECT COUNT(*) FROM {table};", "tuples")
                    if count_result and int(count_result.strip()) == 0:
                        print(f"✅ {table} table is empty")
                    else:
                        print(f"❌ {table} table is not empty after reset")
                        return False
                else:
                    print(f"❌ Failed to clear {table} table")
                    return False
            else:
                print(f"⚠️  {table} table not found")

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
        
        # Create required extensions
        self.print_status("Creating required extensions...", StatusLevel.INFO)
        success_count = 0
        for ext_sql in REQUIRED_EXTENSIONS_SQL:
            result = self.run_sql_command(ext_sql)
            if result is not None:
                success_count += 1
                self.print_status("Required extension created successfully", StatusLevel.SUCCESS)
            else:
                self.print_status("Failed to create required extension", StatusLevel.ERROR)
                return False

        # Create optional extensions (don't fail if they're not available)
        self.print_status("Creating optional extensions...", StatusLevel.INFO)
        optional_success_count = 0
        for ext_sql in OPTIONAL_EXTENSIONS_SQL:
            result = self.run_sql_command(ext_sql, ignore_errors=True)
            if result is not None:
                optional_success_count += 1
                self.print_status("Optional extension created successfully", StatusLevel.SUCCESS)
            else:
                self.print_status("Optional extension not available (skipping)", StatusLevel.WARNING)

        if optional_success_count > 0:
            self.print_status(
                f"Created {optional_success_count}/{len(OPTIONAL_EXTENSIONS_SQL)} optional extensions",
                StatusLevel.INFO
            )
        else:
            self.print_status("No optional extensions were available", StatusLevel.INFO)

        # Create m0_raw table
        self.print_status("Creating m0_raw table...", StatusLevel.INFO)
        result = self.run_sql_command(M0_TABLE_SCHEMA_SQL)
        if result is not None:
            self.print_status("m0_raw table created successfully", StatusLevel.SUCCESS)
        else:
            self.print_status("Failed to create m0_raw table", StatusLevel.ERROR)
            return False

        # Create m1_episodic table
        self.print_status("Creating m1_episodic table...", StatusLevel.INFO)
        result = self.run_sql_command(M1_TABLE_SCHEMA_SQL)
        if result is not None:
            self.print_status("m1_episodic table created successfully", StatusLevel.SUCCESS)
        else:
            self.print_status("Failed to create m1_episodic table", StatusLevel.ERROR)
            return False

        # Create M0 indexes
        self.print_status("Creating M0 indexes...", StatusLevel.INFO)
        m0_index_success_count = 0
        for index_sql in M0_INDEXES_SQL:
            result = self.run_sql_command(index_sql)
            if result is not None:
                m0_index_success_count += 1
                self.print_status("M0 index created successfully", StatusLevel.SUCCESS)
            else:
                self.print_status("Failed to create M0 index", StatusLevel.WARNING)

        # Create M1 indexes
        self.print_status("Creating M1 indexes...", StatusLevel.INFO)
        m1_index_success_count = 0
        for index_sql in M1_INDEXES_SQL:
            result = self.run_sql_command(index_sql)
            if result is not None:
                m1_index_success_count += 1
                self.print_status("M1 index created successfully", StatusLevel.SUCCESS)
            else:
                self.print_status("Failed to create M1 index", StatusLevel.WARNING)

        total_indexes = len(M0_INDEXES_SQL) + len(M1_INDEXES_SQL)
        total_success = m0_index_success_count + m1_index_success_count

        if total_success == 0:
            self.print_status("Failed to create any indexes", StatusLevel.ERROR)
            return False
        elif total_success < total_indexes:
            self.print_status(
                f"Created {total_success}/{total_indexes} indexes",
                StatusLevel.WARNING
            )

        # Create M0 trigger system
        self.print_status("Creating M0 trigger system...", StatusLevel.INFO)

        # Create M0 notification function
        result = self.run_sql_command(M0_NOTIFICATION_FUNCTION_SQL)
        if result is not None:
            self.print_status("M0 notification function created successfully", StatusLevel.SUCCESS)
        else:
            self.print_status("Failed to create M0 notification function", StatusLevel.ERROR)
            return False

        # Create M0 trigger
        result = self.run_sql_command(M0_TRIGGER_SQL)
        if result is not None:
            self.print_status("M0 trigger created successfully", StatusLevel.SUCCESS)
        else:
            self.print_status("Failed to create M0 trigger", StatusLevel.ERROR)
            return False

        # Create M1 trigger system
        self.print_status("Creating M1 trigger system...", StatusLevel.INFO)

        # Create M1 update function
        result = self.run_sql_command(M1_UPDATE_FUNCTION_SQL)
        if result is not None:
            self.print_status("M1 update function created successfully", StatusLevel.SUCCESS)
        else:
            self.print_status("Failed to create M1 update function", StatusLevel.ERROR)
            return False

        # Create M1 update trigger
        result = self.run_sql_command(M1_UPDATE_TRIGGER_SQL)
        if result is not None:
            self.print_status("M1 update trigger created successfully", StatusLevel.SUCCESS)
        else:
            self.print_status("Failed to create M1 update trigger", StatusLevel.ERROR)
            return False

        # Create M1 notification function
        result = self.run_sql_command(M1_NOTIFICATION_FUNCTION_SQL)
        if result is not None:
            self.print_status("M1 notification function created successfully", StatusLevel.SUCCESS)
        else:
            self.print_status("Failed to create M1 notification function", StatusLevel.ERROR)
            return False

        # Create M1 embedding trigger
        result = self.run_sql_command(M1_EMBEDDING_TRIGGER_SQL)
        if result is not None:
            self.print_status("M1 embedding trigger created successfully", StatusLevel.SUCCESS)
        else:
            self.print_status("Failed to create M1 embedding trigger", StatusLevel.ERROR)
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
        
        # Check m0_raw table
        table_result = self.run_sql_command("\\d m0_raw")
        if table_result and 'needs_embedding' in table_result:
            validation_results.append(("m0_raw table", True))
            print("✅ m0_raw table exists with correct structure")
        else:
            validation_results.append(("m0_raw table", False))
            print("❌ m0_raw table missing or incorrect")
        
        # Check trigger
        trigger_result = self.run_sql_command("""
            SELECT COUNT(*) FROM information_schema.triggers
            WHERE trigger_name = 'm0_raw_embedding_trigger';
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
Environment Variables:
  MEMFUSE_DB_CONTAINER    Database container name (default: memfuse-pgai-postgres)
  MEMFUSE_DB_NAME         Database name (default: memfuse)
  MEMFUSE_DB_USER         Database user (default: postgres)
  MEMFUSE_DB_TIMEOUT      Command timeout in seconds (default: 60)
  MEMFUSE_DB_RETRY_COUNT  Number of retries for failed operations (default: 3)
  MEMFUSE_DB_RETRY_DELAY  Delay between retries in seconds (default: 2)

Examples:
  %(prog)s status      # Show current status
  %(prog)s reset       # Clear data, keep schema
  %(prog)s recreate    # Drop and recreate schema
  %(prog)s validate    # Validate schema

  # Using environment variables
  export MEMFUSE_DB_TIMEOUT=120
  %(prog)s recreate    # Use longer timeout
        """
    )

    parser.add_argument(
        'action',
        choices=['status', 'reset', 'recreate', 'validate'],
        help='Action to perform'
    )

    parser.add_argument(
        '--container',
        help='Database container name (overrides MEMFUSE_DB_CONTAINER)'
    )

    parser.add_argument(
        '--timeout',
        type=int,
        help='Command timeout in seconds (overrides MEMFUSE_DB_TIMEOUT)'
    )

    parser.add_argument(
        '--retry-count',
        type=int,
        help='Number of retries (overrides MEMFUSE_DB_RETRY_COUNT)'
    )

    parser.add_argument('--version', action='version', version='MemFuse Database Manager 2.0')

    args = parser.parse_args()

    # Create configuration from environment variables, then override with command line args
    config = DatabaseConfig.from_env()

    if args.container:
        config.container_name = args.container
    if args.timeout:
        config.timeout = args.timeout
    if args.retry_count:
        config.retry_count = args.retry_count

    print("🔧 MemFuse Database Manager v2.0")
    print(f"Action: {args.action}")
    print(f"Container: {config.container_name}")
    print(f"Database: {config.db_name}")
    print(f"Started at: {datetime.now()}")
    print("=" * 60)

    manager = DatabaseManager(config)

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
            manager.print_status(f"Unknown action: {args.action}", StatusLevel.ERROR)
            success = False

        print(f"\nCompleted at: {datetime.now()}")

        if success:
            manager.print_status(f"{args.action.title()} completed successfully!", StatusLevel.SUCCESS)
            sys.exit(0)
        else:
            manager.print_status(f"{args.action.title()} failed!", StatusLevel.ERROR)
            sys.exit(1)

    except KeyboardInterrupt:
        manager.print_status("Operation cancelled by user", StatusLevel.WARNING)
        sys.exit(1)
    except Exception as e:
        manager.print_status(f"Unexpected error: {e}", StatusLevel.ERROR)
        sys.exit(1)


if __name__ == "__main__":
    main()
