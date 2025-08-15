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
        """Initialize M0 raw data memory schema."""
        try:
            async with self.pool.connection() as conn:
                # Create M0 table with demo-compatible structure
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS m0_raw (
                        message_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                        content TEXT NOT NULL,
                        role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
                        conversation_id UUID NOT NULL,
                        sequence_number INTEGER NOT NULL,
                        token_count INTEGER NOT NULL DEFAULT 0,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        processed_at TIMESTAMP WITH TIME ZONE,
                        processing_status VARCHAR(20) DEFAULT 'pending'
                            CHECK (processing_status IN ('pending', 'processing', 'completed', 'failed')),
                        chunk_assignments UUID[] DEFAULT '{}',
                        CONSTRAINT unique_conversation_sequence
                            UNIQUE (conversation_id, sequence_number)
                    );
                """)
                
                # Create demo-compatible indexes
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_m0_conversation_sequence
                        ON m0_raw (conversation_id, sequence_number);
                    CREATE INDEX IF NOT EXISTS idx_m0_processing_status
                        ON m0_raw (processing_status)
                        WHERE processing_status != 'completed';
                    CREATE INDEX IF NOT EXISTS idx_m0_created_at
                        ON m0_raw (created_at DESC);
                    CREATE INDEX IF NOT EXISTS idx_m0_role
                        ON m0_raw (role);
                    CREATE INDEX IF NOT EXISTS idx_m0_token_count
                        ON m0_raw (token_count);
                    CREATE INDEX IF NOT EXISTS idx_m0_chunk_assignments_gin
                        ON m0_raw USING gin (chunk_assignments);
                """)

                # Create M1 processing trigger (M0 -> M1 pipeline)
                await conn.execute("""
                    CREATE OR REPLACE FUNCTION notify_m1_processing_needed()
                    RETURNS TRIGGER AS $$
                    BEGIN
                        -- Notify M1 layer that new raw data is available for processing
                        IF NEW.content IS NOT NULL THEN
                            PERFORM pg_notify('m1_processing_needed', NEW.message_id::text);
                        END IF;
                        RETURN NEW;
                    END;
                    $$ LANGUAGE plpgsql;

                    DROP TRIGGER IF EXISTS trigger_m0_m1_processing_notification ON m0_raw;
                    CREATE TRIGGER trigger_m0_m1_processing_notification
                        AFTER INSERT ON m0_raw
                        FOR EACH ROW
                        EXECUTE FUNCTION notify_m1_processing_needed();
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
            schema_file = self.schema_dir / "m1_episodic.sql"
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
                        WHERE table_name = 'm0_raw'
                    );
                """)
                table_exists = (await result.fetchone())[0]
                
                if not table_exists:
                    return False
                
                # Check if embedding column exists
                result = await conn.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.columns 
                        WHERE table_name = 'm0_raw' AND column_name = 'embedding'
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
                        WHERE table_name = 'm1_episodic'
                    );
                """)
                table_exists = (await result.fetchone())[0]
                
                if not table_exists:
                    return False
                
                # Check if key columns exist
                result = await conn.execute("""
                    SELECT COUNT(*) FROM information_schema.columns 
                    WHERE table_name = 'm1_episodic'
                    AND column_name IN ('episode_content', 'embedding', 'episode_type');
                """)
                column_count = (await result.fetchone())[0]
                
                return column_count >= 3
                
        except Exception as e:
            logger.error(f"M1 schema validation failed: {e}")
            return False
