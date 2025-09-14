"""M3 migration manager for database schema management."""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from loguru import logger

from ...services.database_service import DatabaseService


class M3MigrationManager:
    """Manages M3 database migrations."""
    
    def __init__(self, db: Optional[DatabaseService] = None):
        self.db = db
        self.migrations_dir = Path(__file__).parent
        
    async def get_db(self) -> DatabaseService:
        """Get database service instance."""
        if self.db is None:
            self.db = await DatabaseService.get_instance()
        return self.db
    
    async def ensure_migration_table(self) -> None:
        """Ensure the schema_migrations table exists."""
        db = await self.get_db()
        
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version TEXT PRIMARY KEY,
            applied_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            description TEXT
        );
        """
        
        try:
            await db.backend.execute(create_table_sql, tuple())
            logger.info("Schema migrations table ensured")
        except Exception as e:
            logger.error(f"Failed to create schema_migrations table: {e}")
            raise
    
    async def get_applied_migrations(self) -> List[str]:
        """Get list of applied migration versions."""
        db = await self.get_db()
        
        try:
            query = "SELECT version FROM schema_migrations ORDER BY version;"
            rows = await db.backend.execute(query, tuple())
            return [row.get("version", "") for row in rows]
        except Exception as e:
            logger.warning(f"Failed to get applied migrations: {e}")
            return []
    
    async def get_pending_migrations(self) -> List[Dict[str, Any]]:
        """Get list of pending migrations."""
        applied_migrations = await self.get_applied_migrations()
        all_migrations = self.get_available_migrations()
        
        pending = []
        for migration in all_migrations:
            if migration["version"] not in applied_migrations:
                pending.append(migration)
        
        return pending
    
    def get_available_migrations(self) -> List[Dict[str, Any]]:
        """Get list of available migration files."""
        migrations = []
        
        for sql_file in sorted(self.migrations_dir.glob("*.sql")):
            if sql_file.name.startswith("00"):  # Migration files start with version numbers
                version = sql_file.stem.split("_")[0]
                description = " ".join(sql_file.stem.split("_")[1:]).replace("_", " ").title()
                
                migrations.append({
                    "version": version,
                    "file": sql_file,
                    "description": description
                })
        
        return migrations
    
    async def apply_migration(self, migration: Dict[str, Any]) -> bool:
        """Apply a single migration."""
        db = await self.get_db()
        
        try:
            # Read migration SQL
            sql_content = migration["file"].read_text()
            
            logger.info(f"Applying migration {migration['version']}: {migration['description']}")
            
            # Execute migration SQL
            await db.backend.execute(sql_content, tuple())
            
            logger.info(f"Migration {migration['version']} applied successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply migration {migration['version']}: {e}")
            return False
    
    async def apply_all_pending_migrations(self) -> bool:
        """Apply all pending migrations."""
        await self.ensure_migration_table()
        
        pending_migrations = await self.get_pending_migrations()
        
        if not pending_migrations:
            logger.info("No pending M3 migrations found")
            return True
        
        logger.info(f"Found {len(pending_migrations)} pending M3 migrations")
        
        success_count = 0
        for migration in pending_migrations:
            if await self.apply_migration(migration):
                success_count += 1
            else:
                logger.error(f"Migration {migration['version']} failed, stopping migration process")
                break
        
        if success_count == len(pending_migrations):
            logger.info("All M3 migrations applied successfully")
            return True
        else:
            logger.error(f"Only {success_count}/{len(pending_migrations)} migrations applied")
            return False
    
    async def rollback_migration(self, version: str) -> bool:
        """Rollback a specific migration (if rollback script exists)."""
        rollback_file = self.migrations_dir / f"{version}_rollback.sql"
        
        if not rollback_file.exists():
            logger.error(f"No rollback script found for migration {version}")
            return False
        
        db = await self.get_db()
        
        try:
            # Read rollback SQL
            sql_content = rollback_file.read_text()
            
            logger.info(f"Rolling back migration {version}")
            
            # Execute rollback SQL
            await db.backend.execute(sql_content, tuple())
            
            # Remove from migrations table
            await db.backend.execute(
                "DELETE FROM schema_migrations WHERE version = %s",
                (version,)
            )
            
            logger.info(f"Migration {version} rolled back successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to rollback migration {version}: {e}")
            return False
    
    async def get_migration_status(self) -> Dict[str, Any]:
        """Get current migration status."""
        await self.ensure_migration_table()
        
        applied_migrations = await self.get_applied_migrations()
        pending_migrations = await self.get_pending_migrations()
        all_migrations = self.get_available_migrations()
        
        return {
            "total_migrations": len(all_migrations),
            "applied_count": len(applied_migrations),
            "pending_count": len(pending_migrations),
            "applied_migrations": applied_migrations,
            "pending_migrations": [m["version"] for m in pending_migrations],
            "up_to_date": len(pending_migrations) == 0
        }
    
    async def validate_m3_schema(self) -> Dict[str, bool]:
        """Validate that M3 schema is properly set up."""
        db = await self.get_db()
        
        validation_results = {}
        
        # Check required tables
        required_tables = [
            "message_workflows",
            "procedural_memory", 
            "procedural_lessons"
        ]
        
        for table in required_tables:
            try:
                query = """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = %s
                );
                """
                rows = await db.backend.execute(query, (table,))
                exists = rows[0].get("exists", False) if rows else False
                validation_results[f"table_{table}"] = exists
                
                if not exists:
                    logger.warning(f"M3 table {table} does not exist")
                
            except Exception as e:
                logger.error(f"Failed to check table {table}: {e}")
                validation_results[f"table_{table}"] = False
        
        # Check required indexes
        required_indexes = [
            "idx_message_workflows_message_id",
            "idx_procedural_memory_usage_count",
            "idx_procedural_lessons_agent"
        ]
        
        for index in required_indexes:
            try:
                query = """
                SELECT EXISTS (
                    SELECT FROM pg_indexes 
                    WHERE indexname = %s
                );
                """
                rows = await db.backend.execute(query, (index,))
                exists = rows[0].get("exists", False) if rows else False
                validation_results[f"index_{index}"] = exists
                
                if not exists:
                    logger.warning(f"M3 index {index} does not exist")
                
            except Exception as e:
                logger.error(f"Failed to check index {index}: {e}")
                validation_results[f"index_{index}"] = False
        
        # Check vector extension
        try:
            query = """
            SELECT EXISTS (
                SELECT FROM pg_extension 
                WHERE extname = 'vector'
            );
            """
            rows = await db.backend.execute(query, tuple())
            vector_available = rows[0].get("exists", False) if rows else False
            validation_results["vector_extension"] = vector_available
            
            if not vector_available:
                logger.warning("Vector extension not available - M3 will work but without vector similarity")
                
        except Exception as e:
            logger.error(f"Failed to check vector extension: {e}")
            validation_results["vector_extension"] = False
        
        # Overall validation status
        validation_results["schema_valid"] = all(
            validation_results.get(f"table_{table}", False) 
            for table in required_tables
        )
        
        return validation_results


# Convenience functions for common operations
async def initialize_m3_schema() -> bool:
    """Initialize M3 schema by applying all migrations."""
    manager = M3MigrationManager()
    return await manager.apply_all_pending_migrations()


async def get_m3_migration_status() -> Dict[str, Any]:
    """Get current M3 migration status."""
    manager = M3MigrationManager()
    return await manager.get_migration_status()


async def validate_m3_schema() -> Dict[str, bool]:
    """Validate M3 schema setup."""
    manager = M3MigrationManager()
    return await manager.validate_m3_schema()