"""
Schema management for multi-layer PgAI embedding system.

This module handles database schema creation and validation
for M0 and M1 memory layers with automatic embedding support.
"""

from pathlib import Path
from typing import Dict, List, Any
from loguru import logger

try:
    from psycopg_pool import AsyncConnectionPool
    PSYCOPG_AVAILABLE = True
except ImportError:
    PSYCOPG_AVAILABLE = False
    AsyncConnectionPool = None
    logger.warning("psycopg not available for schema management")


class SchemaManager:
    """
    Schema manager for multi-layer PgAI system.

    Handles database schema creation and validation for M0 and M1 layers
    with their associated triggers, indexes, and embedding infrastructure.
    """
    
    def __init__(self, pool):
        """Initialize schema manager with database connection pool.

        Args:
            pool: AsyncConnectionPool for database connections
        """
        if not PSYCOPG_AVAILABLE:
            raise ImportError("psycopg required for schema management")

        self.pool = pool
        self.schema_dir = Path(__file__).parent / "schemas"
        self.supported_layers = ["m0", "m1"]
        
        logger.info("SchemaManager initialized")
    
    async def initialize_all_schemas(self, enabled_layers: List[str]) -> bool:
        """Initialize schemas for all enabled memory layers.
        
        Args:
            enabled_layers: List of layer names to initialize (e.g., ['m0', 'm1'])
            
        Returns:
            True if all schemas initialized successfully
        """
        try:
            logger.info(f"Initializing schemas for layers: {enabled_layers}")
            
            # Initialize each enabled layer
            for layer in enabled_layers:
                if layer in self.supported_layers:
                    success = await self._initialize_layer_schema(layer)
                    if not success:
                        logger.error(f"Failed to initialize schema for layer {layer}")
                        return False
                else:
                    logger.warning(f"Unsupported layer: {layer}")
            
            logger.info("All schemas initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Schema initialization failed: {e}")
            return False
    
    async def _initialize_layer_schema(self, layer: str) -> bool:
        """Initialize schema for a specific memory layer.
        
        Args:
            layer: Layer name ('m0' or 'm1')
            
        Returns:
            True if schema initialized successfully
        """
        try:
            logger.info(f"Initializing schema for layer: {layer}")
            
            # Load and execute schema SQL
            if layer == "m0":
                success = await self._initialize_m0_schema()
            elif layer == "m1":
                success = await self._initialize_m1_schema()
            else:
                logger.error(f"Unknown layer: {layer}")
                return False
            
            if success:
                logger.info(f"Schema for {layer} initialized successfully")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to initialize schema for {layer}: {e}")
            return False
    
    async def _initialize_m0_schema(self) -> bool:
        """Initialize M0 episodic memory schema."""
        try:
            async with self.pool.connection() as conn:
                # Create M0 table with basic structure
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS m0_episodic (
                        id TEXT PRIMARY KEY,
                        content TEXT NOT NULL,
                        embedding VECTOR(384),
                        needs_embedding BOOLEAN DEFAULT TRUE,
                        session_id TEXT,
                        user_id TEXT,
                        chunk_id TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                
                # Create basic indexes
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_m0_session_id ON m0_episodic (session_id);
                    CREATE INDEX IF NOT EXISTS idx_m0_user_id ON m0_episodic (user_id);
                    CREATE INDEX IF NOT EXISTS idx_m0_needs_embedding ON m0_episodic (needs_embedding) WHERE needs_embedding = TRUE;
                    CREATE INDEX IF NOT EXISTS idx_m0_embedding_hnsw ON m0_episodic USING hnsw (embedding vector_cosine_ops);
                """)
                
                # Create embedding trigger
                await conn.execute("""
                    CREATE OR REPLACE FUNCTION notify_m0_embedding_needed()
                    RETURNS TRIGGER AS $$
                    BEGIN
                        IF NEW.needs_embedding = TRUE THEN
                            PERFORM pg_notify('m0_embedding_needed', NEW.id);
                        END IF;
                        RETURN NEW;
                    END;
                    $$ LANGUAGE plpgsql;
                    
                    DROP TRIGGER IF EXISTS trigger_m0_embedding_notification ON m0_episodic;
                    CREATE TRIGGER trigger_m0_embedding_notification
                        AFTER INSERT OR UPDATE OF needs_embedding ON m0_episodic
                        FOR EACH ROW
                        EXECUTE FUNCTION notify_m0_embedding_needed();
                """)
                
                await conn.commit()
                logger.info("M0 schema created successfully")
                return True
                
        except Exception as e:
            logger.error(f"Failed to create M0 schema: {e}")
            return False
    
    async def _initialize_m1_schema(self) -> bool:
        """Initialize M1 semantic memory schema."""
        try:
            schema_file = self.schema_dir / "m1_semantic.sql"
            if not schema_file.exists():
                logger.error(f"M1 schema file not found: {schema_file}")
                return False
            
            async with self.pool.connection() as conn:
                schema_sql = schema_file.read_text()
                await conn.execute(schema_sql)
                await conn.commit()
                
                logger.info("M1 schema created successfully")
                return True
                
        except Exception as e:
            logger.error(f"Failed to create M1 schema: {e}")
            return False
    
    async def validate_schemas(self, enabled_layers: List[str]) -> Dict[str, bool]:
        """Validate schemas for enabled layers.
        
        Args:
            enabled_layers: List of layer names to validate
            
        Returns:
            Dict mapping layer names to validation results
        """
        results = {}
        
        for layer in enabled_layers:
            if layer not in self.supported_layers:
                results[layer] = False
                continue
                
            try:
                if layer == "m0":
                    results[layer] = await self._validate_m0_schema()
                elif layer == "m1":
                    results[layer] = await self._validate_m1_schema()
                else:
                    results[layer] = False
                    
            except Exception as e:
                logger.error(f"Schema validation failed for {layer}: {e}")
                results[layer] = False
        
        return results
    
    async def _validate_m0_schema(self) -> bool:
        """Validate M0 schema structure."""
        try:
            async with self.pool.connection() as conn:
                # Check if table exists
                result = await conn.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'm0_episodic'
                    );
                """)
                table_exists = (await result.fetchone())[0]
                
                if not table_exists:
                    return False
                
                # Check if embedding column exists
                result = await conn.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.columns 
                        WHERE table_name = 'm0_episodic' AND column_name = 'embedding'
                    );
                """)
                embedding_exists = (await result.fetchone())[0]
                
                return embedding_exists
                
        except Exception as e:
            logger.error(f"M0 schema validation failed: {e}")
            return False
    
    async def _validate_m1_schema(self) -> bool:
        """Validate M1 schema structure."""
        try:
            async with self.pool.connection() as conn:
                # Check if table exists
                result = await conn.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'm1_semantic'
                    );
                """)
                table_exists = (await result.fetchone())[0]
                
                if not table_exists:
                    return False
                
                # Check if key columns exist
                result = await conn.execute("""
                    SELECT COUNT(*) FROM information_schema.columns 
                    WHERE table_name = 'm1_semantic' 
                    AND column_name IN ('fact_content', 'embedding', 'fact_type');
                """)
                column_count = (await result.fetchone())[0]
                
                return column_count >= 3
                
        except Exception as e:
            logger.error(f"M1 schema validation failed: {e}")
            return False
