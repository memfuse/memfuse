"""pgai-based unified storage for MemFuse M0 layer."""

import asyncio
import json
import numpy as np
from typing import List, Dict, Any, Optional
from loguru import logger

try:
    from pgai.vectorizer import Worker
    import psycopg
    from psycopg_pool import AsyncConnectionPool
    from pgvector.psycopg import register_vector_async
    PGAI_AVAILABLE = True
except ImportError:
    PGAI_AVAILABLE = False
    # Create dummy classes for type hints when imports fail

    class psycopg:
        class AsyncConnection:
            pass

    class AsyncConnectionPool:
        pass

    class Worker:
        pass

    logger.warning("pgai dependencies not available. Install with: pip install pgai psycopg pgvector")

from ...interfaces.chunk_store import ChunkStoreInterface
from ...rag.chunk.base import ChunkData
from ...models.core import Query
from ...utils.config import config_manager


class GlobalEmbeddingModelWrapper:
    """Wrapper to make global embedding model compatible with encoder interface."""

    def __init__(self, embedding_model):
        """Initialize wrapper with global embedding model.

        Args:
            embedding_model: Global SentenceTransformer model instance
        """
        self.model = embedding_model
        self.model_name = getattr(embedding_model, 'model_name', 'all-MiniLM-L6-v2')

    async def encode_text(self, text: str) -> np.ndarray:
        """Encode a single text using the global model.

        Args:
            text: Text to encode

        Returns:
            Embedding vector as numpy array
        """
        try:
            # Use asyncio.to_thread to run the synchronous encode in a thread
            embedding = await asyncio.to_thread(self.model.encode, text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            logger.error(f"GlobalEmbeddingModelWrapper: Failed to encode text: {e}")
            raise

    async def encode_texts(self, texts: List[str]) -> List[np.ndarray]:
        """Encode multiple texts using the global model.

        Args:
            texts: List of texts to encode

        Returns:
            List of embedding vectors as numpy arrays
        """
        try:
            # Use asyncio.to_thread to run the synchronous encode in a thread
            embeddings = await asyncio.to_thread(self.model.encode, texts, convert_to_numpy=True)
            return embeddings
        except Exception as e:
            logger.error(f"GlobalEmbeddingModelWrapper: Failed to encode texts: {e}")
            raise


class PgaiStore(ChunkStoreInterface):
    """pgai-based unified storage for MemFuse M0 layer.
    
    This store uses pgai's vectorizer to automatically manage embeddings
    while maintaining compatibility with existing MemFuse schema.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, table_name: str = "m0_raw"):
        """Initialize pgai store.

        Args:
            config: Database configuration dict (optional, uses config_manager if None)
            table_name: Name of the table to store messages
        """
        if not PGAI_AVAILABLE:
            raise ImportError("pgai dependencies required for PgaiStore")

        self.table_name = table_name
        self.embedding_view = f"{table_name}_embedding"
        self.vectorizer_name = f"{table_name}_vectorizer"

        self.pool: Optional[AsyncConnectionPool] = None
        self.vectorizer_worker: Optional[Worker] = None
        self.initialized = False

        # Get configuration
        if config:
            # Use provided config for testing
            self.db_config = config
            self.pgai_config = config.get("pgai", {})
        else:
            # Use config manager for production
            config = config_manager.get_config()
            self.db_config = config.get("database", {})
            # pgai config is under database, not at root level
            self.pgai_config = config.get("database", {}).get("pgai", {})

        # Build database URL
        self.db_url = self._build_database_url()
    
    def _build_database_url(self) -> str:
        """Build PostgreSQL database URL from configuration and environment variables."""
        import os

        # Priority order: 1. Environment variables, 2. Config, 3. Defaults
        if "postgres" in self.db_config:
            # Config manager format
            postgres_config = self.db_config.get("postgres", {})
            host = os.getenv("POSTGRES_HOST", postgres_config.get("host", "localhost"))
            port = int(os.getenv("POSTGRES_PORT", postgres_config.get("port", 5432)))
            database = os.getenv("POSTGRES_DB", postgres_config.get("database", "memfuse"))
            user = os.getenv("POSTGRES_USER", postgres_config.get("user", "postgres"))
            password = os.getenv("POSTGRES_PASSWORD", postgres_config.get("password", "postgres"))
        else:
            # Direct config format (for testing) - still check environment variables first
            host = os.getenv("POSTGRES_HOST", self.db_config.get("host", "localhost"))
            port = int(os.getenv("POSTGRES_PORT", self.db_config.get("port", 5432)))
            database = os.getenv("POSTGRES_DB", self.db_config.get("database", "memfuse"))
            user = os.getenv("POSTGRES_USER", self.db_config.get("user", "postgres"))
            password = os.getenv("POSTGRES_PASSWORD", self.db_config.get("password", "postgres"))

        return f"postgresql://{user}:{password}@{host}:{port}/{database}"

    async def _check_required_extensions(self) -> bool:
        """Check if required PostgreSQL extensions are available."""
        try:
            # Create a temporary connection to check extensions
            import psycopg

            logger.debug("Checking required PostgreSQL extensions...")

            async with await psycopg.AsyncConnection.connect(self.db_url) as conn:
                async with conn.cursor() as cur:
                    # Check if pgvector extension exists
                    await cur.execute(
                        "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')"
                    )
                    vector_exists = (await cur.fetchone())[0]

                    if not vector_exists:
                        logger.error("pgvector extension is not installed")
                        logger.error("Please ensure pgvector extension is installed in your PostgreSQL database")
                        logger.error("Run: CREATE EXTENSION IF NOT EXISTS vector;")
                        return False

                    # Test vector functionality
                    try:
                        await cur.execute("SELECT '[1,2,3]'::vector")
                        logger.debug("pgvector extension verified successfully")
                    except Exception as e:
                        logger.error(f"pgvector functionality test failed: {e}")
                        return False

            return True

        except Exception as e:
            logger.error(f"Extension check failed: {e}")
            logger.error("Please ensure PostgreSQL database is accessible and pgvector extension is installed")
            return False

    async def initialize(self) -> bool:
        """Initialize pgai store and vectorizer."""
        if self.initialized:
            return True
            
        try:
            logger.info(f"Initializing PgaiStore with table: {self.table_name}")
            
            # Check required extensions first
            if not await self._check_required_extensions():
                logger.error("Required extensions are not available")
                return False

            # Create connection pool with conservative settings
            logger.debug("Creating connection pool...")
            postgres_config = self.db_config.get("postgres", {})

            # Use minimal pool size to avoid lock conflicts
            safe_min_size = 1  # Always start with 1 connection
            safe_max_size = 2   # Maximum 2 connections to avoid deadlocks

            # First try: Create pool without pgvector configuration
            self.pool = AsyncConnectionPool(
                self.db_url,
                min_size=safe_min_size,
                max_size=safe_max_size,
                open=False,
                configure=None  # No configuration initially
            )
            logger.debug(f"Opening basic connection pool (min_size={safe_min_size}, max_size={safe_max_size})")

            # Open basic pool first
            try:
                await asyncio.wait_for(self.pool.open(), timeout=15.0)
                logger.debug("Basic connection pool opened successfully")

                # Now configure pgvector on existing connections
                await self._configure_pgvector_on_pool()

            except asyncio.TimeoutError:
                logger.error("Basic connection pool opening timed out, trying single connection fallback")
                await self._fallback_single_connection()
            except Exception as e:
                logger.error(f"Connection pool creation failed: {e}")
                await self._fallback_single_connection()
            
            # Create tables and setup vectorizer
            logger.debug("Setting up database schema...")
            await self._setup_schema()
            logger.debug("Database schema setup completed")

            # Create vectorizer if enabled and not exists
            auto_embedding = self.pgai_config.get("auto_embedding", False)
            logger.debug(f"Auto embedding enabled: {auto_embedding}")

            if auto_embedding:
                logger.debug("Creating vectorizer...")
                await self._create_vectorizer()
                logger.debug("Vectorizer creation completed")

                # Choose processing mode based on configuration
                immediate_trigger = self.pgai_config.get("immediate_trigger", False)
                logger.debug(f"Immediate trigger enabled: {immediate_trigger}")

                if immediate_trigger:
                    # Event-driven mode will be handled by EventDrivenPgaiStore
                    logger.info("pgai auto-embedding configured for immediate trigger mode")
                else:
                    # Start traditional background embedding processor
                    logger.debug("Starting background embedding processor...")
                    asyncio.create_task(self._process_pending_embeddings())
                    logger.info("pgai auto-embedding enabled with background processor")
            else:
                logger.debug("pgai auto_embedding disabled, using manual embedding approach")
            
            self.initialized = True
            logger.info("PgaiStore initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize PgaiStore: {e}")
            return False
    
    async def _configure_pgvector_on_pool(self):
        """Configure pgvector on existing pool connections."""
        logger.debug("Configuring pgvector on existing connections...")

        try:
            # Test a connection and configure pgvector
            async with self.pool.connection() as conn:
                await self._setup_pgvector_connection_safe(conn)
            logger.debug("pgvector configured successfully on pool")
        except Exception as e:
            logger.warning(f"Failed to configure pgvector on pool: {e}")
            # Continue anyway - basic operations should still work

    async def _fallback_single_connection(self):
        """Fallback to single connection when pool fails."""
        logger.info("Using single connection fallback")

        # Close any existing pool
        if self.pool:
            try:
                await self.pool.close()
            except:
                pass

        # Create minimal pool with single connection and no initial configuration
        self.pool = AsyncConnectionPool(
            self.db_url,
            min_size=1,
            max_size=1,
            open=False,
            configure=None  # Configure manually after opening
        )

        try:
            # Open with minimal timeout
            await asyncio.wait_for(self.pool.open(), timeout=10.0)
            logger.info("Single connection pool opened successfully")

            # Configure pgvector manually
            await self._configure_pgvector_on_pool()

        except Exception as e:
            logger.error(f"Single connection fallback failed: {e}")
            raise
    
    async def _setup_pgvector_connection_safe(self, conn: psycopg.AsyncConnection):
        """Safe pgvector setup with error handling."""
        try:
            await register_vector_async(conn)
            logger.debug("pgvector registered successfully for connection")
        except Exception as e:
            logger.warning(f"pgvector registration failed, continuing anyway: {e}")
            # Don't fail the entire initialization for pgvector registration issues
    
    async def _setup_pgvector_connection(self, conn: psycopg.AsyncConnection):
        """Setup pgvector for connection with timeout and retry."""
        max_retries = 2  # Reduce retries to fail faster
        base_timeout = 5.0  # Shorter timeout

        for attempt in range(max_retries):
            try:
                # Progressive timeout: shorter on retries
                timeout = base_timeout / (attempt + 1)
                await asyncio.wait_for(register_vector_async(conn), timeout=timeout)
                logger.debug(f"pgvector registered successfully (attempt {attempt + 1})")
                return
            except asyncio.TimeoutError:
                logger.warning(f"pgvector registration timed out in {timeout}s (attempt {attempt + 1})")
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.5)  # Shorter delay
                else:
                    logger.error("pgvector registration failed after all retries - continuing without vector support")
                    # Don't raise exception - allow connection to work without vector support
            except Exception as e:
                logger.warning(f"pgvector registration error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.5)
                else:
                    logger.error(f"pgvector registration failed permanently: {e}")
                    # Don't raise exception - allow basic PostgreSQL operations
    
    async def _run_vectorizer_worker(self):
        """Run vectorizer worker in background."""
        try:
            if self.vectorizer_worker:
                await self.vectorizer_worker.run()
        except Exception as e:
            logger.error(f"Vectorizer worker error: {e}")
            # Restart worker after delay
            await asyncio.sleep(10)
            asyncio.create_task(self._run_vectorizer_worker())

    async def _setup_schema(self):
        """Create the messages table schema compatible with existing MemFuse schema."""
        logger.debug(f"Setting up schema for table: {self.table_name}")

        # First check if schema is already properly set up
        if await self._is_schema_ready():
            logger.debug(f"Schema for {self.table_name} is already properly configured")
            return

        logger.debug(f"Schema needs setup for {self.table_name}, proceeding with creation...")
        async with self.pool.connection() as conn:
            try:
                # Create table compatible with existing m0_raw schema
                logger.debug(f"Creating table {self.table_name} if not exists...")
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        id TEXT PRIMARY KEY,
                        content TEXT NOT NULL,
                        metadata JSONB DEFAULT '{{}}',
                        embedding VECTOR(384),
                        needs_embedding BOOLEAN DEFAULT TRUE,
                        retry_count INTEGER DEFAULT 0,
                        last_retry_at TIMESTAMP,
                        retry_status TEXT DEFAULT 'pending',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                logger.debug(f"Table {self.table_name} creation completed")

                # Add columns if they don't exist (for existing tables)
                logger.debug("Checking and adding missing columns...")
                await conn.execute(f"""
                    DO $$
                    BEGIN
                        IF NOT EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_name = '{self.table_name}' AND column_name = 'embedding'
                        ) THEN
                            ALTER TABLE {self.table_name} ADD COLUMN embedding VECTOR(384);
                        END IF;

                        IF NOT EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_name = '{self.table_name}' AND column_name = 'needs_embedding'
                        ) THEN
                            ALTER TABLE {self.table_name} ADD COLUMN needs_embedding BOOLEAN DEFAULT TRUE;
                        END IF;

                        IF NOT EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_name = '{self.table_name}' AND column_name = 'retry_count'
                        ) THEN
                            ALTER TABLE {self.table_name} ADD COLUMN retry_count INTEGER DEFAULT 0;
                        END IF;

                        IF NOT EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_name = '{self.table_name}' AND column_name = 'last_retry_at'
                        ) THEN
                            ALTER TABLE {self.table_name} ADD COLUMN last_retry_at TIMESTAMP;
                        END IF;

                        IF NOT EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_name = '{self.table_name}' AND column_name = 'retry_status'
                        ) THEN
                            ALTER TABLE {self.table_name} ADD COLUMN retry_status TEXT DEFAULT 'pending';
                        END IF;
                    END $$;
                """)
                logger.debug("Column checks completed")

                # Create indexes for performance
                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.table_name}_created
                    ON {self.table_name}(created_at)
                """)

                # Create index for retry mechanism
                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.table_name}_needs_embedding
                    ON {self.table_name}(needs_embedding, retry_status)
                    WHERE needs_embedding = TRUE
                """)

                # Create trigger to update updated_at timestamp
                await conn.execute(f"""
                    CREATE OR REPLACE FUNCTION update_{self.table_name}_updated_at()
                    RETURNS TRIGGER AS $$
                    BEGIN
                        NEW.updated_at = CURRENT_TIMESTAMP;
                        RETURN NEW;
                    END;
                    $$ LANGUAGE plpgsql;
                """)

                await conn.execute(f"""
                    DROP TRIGGER IF EXISTS trigger_update_{self.table_name}_updated_at
                    ON {self.table_name};
                """)

                await conn.execute(f"""
                    CREATE TRIGGER trigger_update_{self.table_name}_updated_at
                        BEFORE UPDATE ON {self.table_name}
                        FOR EACH ROW
                        EXECUTE FUNCTION update_{self.table_name}_updated_at();
                """)

                # Setup immediate trigger mechanism if enabled
                if self.pgai_config.get("immediate_trigger", False):
                    await self._setup_immediate_trigger_in_transaction(conn)

                # Commit all schema changes
                await conn.commit()
                logger.debug(f"Schema setup completed for table: {self.table_name}")

            except Exception as e:
                await conn.rollback()
                logger.error(f"Schema setup failed for {self.table_name}: {e}")
                raise

    async def _is_schema_ready(self) -> bool:
        """Check if the schema is already properly set up to avoid unnecessary operations."""
        try:
            async with self.pool.connection() as conn:
                async with conn.cursor() as cur:
                    # Check if table exists with required columns
                    await cur.execute(f"""
                        SELECT column_name
                        FROM information_schema.columns
                        WHERE table_name = '{self.table_name}'
                        AND table_schema = 'public'
                        ORDER BY column_name
                    """)
                    columns = await cur.fetchall()

                    if not columns:
                        logger.debug(f"Table {self.table_name} does not exist")
                        return False

                    # Check for required columns
                    column_names = {col[0] for col in columns}
                    required_columns = {
                        'id', 'content', 'metadata', 'embedding', 'needs_embedding',
                        'retry_count', 'last_retry_at', 'retry_status', 'created_at', 'updated_at'
                    }

                    missing_columns = required_columns - column_names
                    if missing_columns:
                        logger.debug(f"Table {self.table_name} missing columns: {missing_columns}")
                        return False

                    # Check if required indexes exist
                    await cur.execute(f"""
                        SELECT indexname
                        FROM pg_indexes
                        WHERE tablename = '{self.table_name}'
                        AND schemaname = 'public'
                    """)
                    indexes = await cur.fetchall()
                    index_names = {idx[0] for idx in indexes}

                    required_indexes = {
                        f'idx_{self.table_name}_created',
                        f'idx_{self.table_name}_needs_embedding'
                    }

                    missing_indexes = required_indexes - index_names
                    if missing_indexes:
                        logger.debug(f"Table {self.table_name} missing indexes: {missing_indexes}")
                        return False

                    # Check if required functions exist
                    await cur.execute(f"""
                        SELECT proname
                        FROM pg_proc
                        WHERE proname IN ('update_{self.table_name}_updated_at')
                    """)
                    functions = await cur.fetchall()

                    if not functions:
                        logger.debug(f"Table {self.table_name} missing required functions")
                        return False

                    logger.debug(f"Schema for {self.table_name} is complete and ready")
                    return True

        except Exception as e:
            logger.warning(f"Error checking schema readiness for {self.table_name}: {e}")
            return False

    async def _create_vectorizer(self):
        """Setup embedding automation using our custom background processor.

        Note: We use our own background processing approach instead of pgai's vectorizer
        because pgai doesn't support local embedding models like MiniLM directly.
        Our approach provides better control and supports local models.
        """
        try:
            # We don't actually need pgai's vectorizer table since we use our own
            # background processing approach. This method is kept for compatibility
            # but the actual work is done by _process_pending_embeddings()

            logger.debug("Using custom background processor instead of pgai vectorizer")
            logger.info(f"Embedding automation setup completed for {self.table_name}")

        except Exception as e:
            logger.error(f"Failed to setup embedding automation: {e}")
            logger.info("Falling back to manual embedding generation")

    async def _setup_immediate_trigger_in_transaction(self, conn):
        """Setup immediate trigger mechanism within existing transaction."""
        logger.debug(f"Starting immediate trigger setup for {self.table_name}")

        # Create notification function for immediate embedding trigger
        logger.debug(f"Creating notification function for {self.table_name}")
        await conn.execute(f"""
            CREATE OR REPLACE FUNCTION notify_embedding_needed_{self.table_name}()
            RETURNS TRIGGER AS $$
            BEGIN
                -- Send notification with record ID for immediate processing
                PERFORM pg_notify('embedding_needed_{self.table_name}', NEW.id);
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
        """)
        logger.debug(f"Notification function created for {self.table_name}")

        # Create trigger for immediate notification on INSERT
        logger.debug(f"Creating immediate trigger for {self.table_name}")
        await conn.execute(f"""
            DROP TRIGGER IF EXISTS trigger_immediate_embedding_{self.table_name} ON {self.table_name};
            CREATE TRIGGER trigger_immediate_embedding_{self.table_name}
                AFTER INSERT ON {self.table_name}
                FOR EACH ROW
                WHEN (NEW.needs_embedding = TRUE AND NEW.content IS NOT NULL)
                EXECUTE FUNCTION notify_embedding_needed_{self.table_name}();
        """)
        logger.debug(f"Immediate trigger created for {self.table_name}")

    async def _setup_immediate_trigger(self):
        """Setup immediate trigger mechanism using PostgreSQL NOTIFY/LISTEN."""
        logger.debug(f"Starting immediate trigger setup for {self.table_name}")

        async with self.pool.connection() as conn:
            try:
                await self._setup_immediate_trigger_in_transaction(conn)
                await conn.commit()
                logger.debug(f"Immediate trigger mechanism setup completed for {self.table_name}")
            except Exception as e:
                await conn.rollback()
                logger.error(f"Failed to setup immediate trigger for {self.table_name}: {e}")
                raise

    async def _create_embedding_trigger(self):
        """Create database trigger for automatic embedding generation."""
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                # Create a function that will be called by the trigger
                await cur.execute(f"""
                    CREATE OR REPLACE FUNCTION {self.table_name}_auto_embed()
                    RETURNS TRIGGER AS $$
                    BEGIN
                        -- Mark record for embedding processing
                        -- The embedding will be generated by the application
                        NEW.embedding = NULL;
                        NEW.needs_embedding = TRUE;
                        NEW.retry_count = 0;
                        NEW.retry_status = 'pending';
                        RETURN NEW;
                    END;
                    $$ LANGUAGE plpgsql;
                """)

                # Create trigger for INSERT and UPDATE
                await cur.execute(f"""
                    DROP TRIGGER IF EXISTS {self.table_name}_embedding_trigger ON {self.table_name};
                    CREATE TRIGGER {self.table_name}_embedding_trigger
                        BEFORE INSERT OR UPDATE OF content ON {self.table_name}
                        FOR EACH ROW
                        WHEN (NEW.content IS NOT NULL AND NEW.content != '')
                        EXECUTE FUNCTION {self.table_name}_auto_embed();
                """)

            await conn.commit()
            logger.debug(f"Embedding trigger created for {self.table_name}")

    async def _process_pending_embeddings(self):
        """Background task to process records that need embeddings with retry support."""
        max_retries = self.pgai_config.get("max_retries", 3)
        retry_interval = self.pgai_config.get("retry_interval", 5.0)

        while True:
            try:
                await asyncio.sleep(retry_interval)  # Use configurable interval

                if not self.initialized:
                    continue

                # Get records that need embeddings, excluding permanently failed ones
                async with self.pool.connection() as conn:
                    async with conn.cursor() as cur:
                        await cur.execute(f"""
                            SELECT id, content, retry_count, last_retry_at
                            FROM {self.table_name}
                            WHERE needs_embedding = TRUE
                            AND content IS NOT NULL
                            AND (retry_status != 'failed' OR retry_status IS NULL)
                            AND (retry_count < %s OR retry_count IS NULL)
                            AND (
                                last_retry_at IS NULL
                                OR last_retry_at < NOW() - INTERVAL '%s seconds'
                            )
                            ORDER BY
                                CASE WHEN retry_count IS NULL THEN 0 ELSE retry_count END,
                                created_at ASC
                            LIMIT 10
                        """, (max_retries, retry_interval))

                        pending_records = await cur.fetchall()

                        if not pending_records:
                            continue

                        logger.info(f"Processing {len(pending_records)} pending embeddings")

                        # Process each record
                        for record_id, content, retry_count, last_retry_at in pending_records:
                            current_retry_count = retry_count or 0
                            is_retry = current_retry_count > 0

                            try:
                                # Mark as processing and increment retry count
                                await cur.execute(f"""
                                    UPDATE {self.table_name}
                                    SET retry_count = %s,
                                        last_retry_at = CURRENT_TIMESTAMP,
                                        retry_status = 'processing'
                                    WHERE id = %s
                                """, (current_retry_count + 1, record_id))

                                # Generate embedding
                                embedding = await self._generate_embedding(content)

                                # Update record with embedding and mark as completed
                                await cur.execute(f"""
                                    UPDATE {self.table_name}
                                    SET embedding = %s,
                                        needs_embedding = FALSE,
                                        retry_count = 0,
                                        retry_status = 'completed',
                                        last_retry_at = NULL
                                    WHERE id = %s
                                """, (embedding, record_id))

                                logger.debug(f"Generated embedding for record {record_id} (retry: {is_retry})")

                            except Exception as e:
                                logger.error(f"Failed to generate embedding for record {record_id}: {e}")

                                # Check if we should retry or mark as failed
                                if current_retry_count + 1 >= max_retries:
                                    # Mark as permanently failed
                                    await cur.execute(f"""
                                        UPDATE {self.table_name}
                                        SET retry_status = 'failed'
                                        WHERE id = %s
                                    """, (record_id,))
                                    logger.warning(f"Record {record_id} marked as failed after {max_retries} retries")
                                else:
                                    # Mark as pending for retry
                                    await cur.execute(f"""
                                        UPDATE {self.table_name}
                                        SET retry_status = 'pending'
                                        WHERE id = %s
                                    """, (record_id,))
                                    logger.info(f"Record {record_id} will be retried (attempt {current_retry_count + 1}/{max_retries})")

                    await conn.commit()

            except Exception as e:
                logger.error(f"Error in background embedding processor: {e}")
                await asyncio.sleep(30)  # Wait longer on error

    async def add(self, chunks: List[ChunkData]) -> List[str]:
        """Add chunks to pgai store with automatic or manual embedding generation.

        Args:
            chunks: List of ChunkData objects to store

        Returns:
            List of chunk IDs that were added
        """
        if not self.initialized:
            await self.initialize()

        if not chunks:
            return []

        # Check if auto-embedding is enabled
        auto_embedding = self.pgai_config.get("auto_embedding", False)

        chunk_ids = []
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                if auto_embedding:
                    # Auto-embedding mode: insert without embeddings, let background task handle it
                    for chunk in chunks:
                        metadata_json = self._prepare_metadata(chunk.metadata)

                        await cur.execute(f"""
                            INSERT INTO {self.table_name}
                            (id, content, metadata, needs_embedding)
                            VALUES (%s, %s, %s, TRUE)
                            ON CONFLICT (id) DO UPDATE SET
                                content = EXCLUDED.content,
                                metadata = EXCLUDED.metadata,
                                needs_embedding = TRUE,
                                updated_at = CURRENT_TIMESTAMP
                        """, (
                            chunk.chunk_id,
                            chunk.content,
                            metadata_json
                        ))
                        chunk_ids.append(chunk.chunk_id)

                    logger.debug(f"Added {len(chunk_ids)} chunks to {self.table_name} for auto-embedding")
                else:
                    # Manual embedding mode: generate embeddings immediately
                    contents = [chunk.content for chunk in chunks]
                    embeddings = await self._generate_embeddings_batch(contents)

                    for chunk, embedding in zip(chunks, embeddings):
                        metadata_json = self._prepare_metadata(chunk.metadata)

                        await cur.execute(f"""
                            INSERT INTO {self.table_name}
                            (id, content, metadata, embedding, needs_embedding)
                            VALUES (%s, %s, %s, %s, FALSE)
                            ON CONFLICT (id) DO UPDATE SET
                                content = EXCLUDED.content,
                                metadata = EXCLUDED.metadata,
                                embedding = EXCLUDED.embedding,
                                needs_embedding = FALSE,
                                updated_at = CURRENT_TIMESTAMP
                        """, (
                            chunk.chunk_id,
                            chunk.content,
                            metadata_json,
                            embedding
                        ))
                        chunk_ids.append(chunk.chunk_id)

                    logger.debug(f"Added {len(chunk_ids)} chunks to {self.table_name} with immediate embeddings")

            await conn.commit()

        return chunk_ids

    def _get_encoder(self):
        """Get encoder with priority for global singleton model."""
        # Priority 1: Use the encoder from PgaiVectorWrapper if available
        if hasattr(self, 'encoder') and self.encoder is not None:
            logger.debug("Using pre-configured encoder")
            return self.encoder

        # Priority 2: Get encoder from ModelRegistry (global model provider)
        try:
            from ...interfaces.model_provider import ModelRegistry
            model_provider = ModelRegistry.get_provider()
            if model_provider and hasattr(model_provider, 'get_encoder'):
                encoder = model_provider.get_encoder()
                if encoder is not None:
                    logger.info("Using encoder from ModelRegistry (global provider)")
                    # Cache the encoder to avoid repeated lookups
                    self.encoder = encoder
                    return encoder
        except Exception as e:
            logger.debug(f"ModelRegistry access failed: {e}")

        # Priority 3: Get global embedding model from ServiceFactory
        try:
            from ..services.service_factory import ServiceFactory
            global_embedding_model = ServiceFactory.get_global_embedding_model()

            if global_embedding_model is not None:
                logger.info("Using global singleton embedding model from ServiceFactory")
                # Create a wrapper to make the model compatible with encoder interface
                wrapper = GlobalEmbeddingModelWrapper(global_embedding_model)
                # Cache the wrapper
                self.encoder = wrapper
                return wrapper
        except Exception as e:
            logger.debug(f"Failed to get global embedding model: {e}")

        # Priority 4: Create direct encoder as last resort
        try:
            logger.info("Creating direct MiniLM encoder as fallback")
            from ..rag.encode.MiniLM import MiniLMEncoder
            # Use the configured model name
            try:
                from ...utils.config import config_manager
                config = config_manager.get_config()
                model_name = config.get("embedding", {}).get("model", "all-MiniLM-L6-v2")
            except:
                model_name = "all-MiniLM-L6-v2"

            encoder = MiniLMEncoder(model_name=model_name)
            # Cache the encoder
            self.encoder = encoder
            return encoder
        except Exception as e:
            logger.error(f"Failed to create fallback encoder: {e}")
            raise RuntimeError(f"All encoder initialization methods failed. Cannot proceed without embedding capability. Last error: {e}")

    def _convert_embedding_to_list(self, embedding) -> List[float]:
        """Convert embedding to list format."""
        return embedding.tolist() if hasattr(embedding, 'tolist') else embedding

    def _prepare_metadata(self, metadata: Optional[Dict[str, Any]]) -> str:
        """Convert metadata to JSON string."""
        return json.dumps(metadata) if metadata else '{}'

    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using the configured encoder."""
        encoder = self._get_encoder()
        if encoder is None:
            raise RuntimeError("No encoder available for embedding generation. Ensure ModelService is initialized.")

        try:
            embedding = await encoder.encode_text(text)
            embedding_list = self._convert_embedding_to_list(embedding)

            # Validate embedding is not zero vector
            if all(abs(x) < 1e-10 for x in embedding_list):
                raise RuntimeError("Generated zero vector - encoder may not be working correctly")

            return embedding_list

        except Exception as e:
            logger.error(f"Failed to generate embedding for text: {text[:50]}... Error: {e}")
            raise

    async def _generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts in batch for better performance.

        Args:
            texts: List of texts to generate embeddings for

        Returns:
            List of embedding vectors as lists of floats
        """
        if not texts:
            return []

        encoder = self._get_encoder()
        if encoder is None:
            raise RuntimeError("No encoder available for batch embedding generation. Ensure ModelService is initialized.")

        try:
            embeddings = await encoder.encode_texts(texts)
            embedding_lists = [self._convert_embedding_to_list(embedding) for embedding in embeddings]

            # Validate no zero vectors
            for i, embedding_list in enumerate(embedding_lists):
                if all(abs(x) < 1e-10 for x in embedding_list):
                    raise RuntimeError(f"Generated zero vector for text {i}: {texts[i][:50]}...")

            return embedding_lists

        except Exception as e:
            logger.error(f"Failed to generate batch embeddings for {len(texts)} texts. Error: {e}")
            raise

    async def read(self, chunk_ids: List[str], filters: Optional[Dict[str, Any]] = None) -> List[Optional[ChunkData]]:
        """Read chunks by their IDs with optional metadata filters."""
        if not self.initialized:
            await self.initialize()

        if not chunk_ids:
            return []

        results = []
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                for chunk_id in chunk_ids:
                    await cur.execute(f"""
                        SELECT id, content, metadata, created_at, updated_at
                        FROM {self.table_name}
                        WHERE id = %s
                    """, (chunk_id,))

                    row = await cur.fetchone()
                    if row:
                        # JSONB column returns dict directly, no need to parse
                        metadata = row[2] if row[2] else {}

                        # Apply filters if provided
                        if filters:
                            match = True
                            for key, value in filters.items():
                                if metadata.get(key) != value:
                                    match = False
                                    break
                            if not match:
                                results.append(None)
                                continue

                        # Add timestamps to metadata
                        metadata['created_at'] = row[3].isoformat() if row[3] else None
                        metadata['updated_at'] = row[4].isoformat() if row[4] else None

                        chunk_data = ChunkData(
                            content=row[1],
                            chunk_id=row[0],
                            metadata=metadata
                        )
                        results.append(chunk_data)
                    else:
                        results.append(None)

        return results

    async def update(self, chunk_id: str, chunk: ChunkData) -> bool:
        """Update an existing chunk with new embedding."""
        if not self.initialized:
            await self.initialize()

        # Generate new embedding for updated content
        embedding = await self._generate_embedding(chunk.content)

        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                metadata_json = self._prepare_metadata(chunk.metadata)

                await cur.execute(f"""
                    UPDATE {self.table_name}
                    SET content = %s, metadata = %s, embedding = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                """, (chunk.content, metadata_json, embedding, chunk_id))

                await conn.commit()
                return cur.rowcount > 0

    async def update_batch(self, chunk_ids: List[str], chunks: List[ChunkData]) -> List[bool]:
        """Update multiple chunks with batch embedding generation.

        Args:
            chunk_ids: List of chunk IDs to update
            chunks: List of new ChunkData objects

        Returns:
            List of success flags for each update
        """
        if not self.initialized:
            await self.initialize()

        if not chunk_ids or not chunks or len(chunk_ids) != len(chunks):
            return [False] * len(chunk_ids)

        # Generate embeddings in batch for better performance
        contents = [chunk.content for chunk in chunks]
        embeddings = await self._generate_embeddings_batch(contents)

        results = []
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                for chunk_id, chunk, embedding in zip(chunk_ids, chunks, embeddings):
                    metadata_json = self._prepare_metadata(chunk.metadata)

                    await cur.execute(f"""
                        UPDATE {self.table_name}
                        SET content = %s, metadata = %s, embedding = %s, updated_at = CURRENT_TIMESTAMP
                        WHERE id = %s
                    """, (chunk.content, metadata_json, embedding, chunk_id))

                    results.append(cur.rowcount > 0)

            await conn.commit()

        logger.debug(f"Updated {sum(results)} out of {len(chunk_ids)} chunks with batch embeddings")
        return results

    async def delete(self, chunk_ids: List[str]) -> List[bool]:
        """Delete chunks by their IDs."""
        if not self.initialized:
            await self.initialize()

        if not chunk_ids:
            return []

        results = []
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                for chunk_id in chunk_ids:
                    await cur.execute(f"""
                        DELETE FROM {self.table_name}
                        WHERE id = %s
                    """, (chunk_id,))

                    results.append(cur.rowcount > 0)

                await conn.commit()

        return results

    async def query(self, query: Query, top_k: int = 5) -> List[ChunkData]:
        """Query: Semantic search for relevant chunks based on query text."""
        if not self.initialized:
            await self.initialize()

        # Use text-based search for now (could be enhanced with embeddings)
        query_text = query.text if hasattr(query, 'text') else str(query)

        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(f"""
                    SELECT id, content, metadata, created_at, updated_at
                    FROM {self.table_name}
                    WHERE content ILIKE %s
                    ORDER BY created_at DESC
                    LIMIT %s
                """, (f"%{query_text}%", top_k))

                results = []
                async for row in cur:
                    # JSONB column returns dict directly, no need to parse
                    metadata = row[2] if row[2] else {}
                    metadata['created_at'] = row[3].isoformat() if row[3] else None
                    metadata['updated_at'] = row[4].isoformat() if row[4] else None

                    chunk_data = ChunkData(
                        content=row[1],
                        chunk_id=row[0],
                        metadata=metadata
                    )
                    results.append(chunk_data)

                return results

    async def count(self) -> int:
        """Get total number of chunks in the store."""
        if not self.initialized:
            await self.initialize()

        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(f"SELECT COUNT(*) FROM {self.table_name}")
                row = await cur.fetchone()
                return row[0] if row else 0

    async def clear(self) -> bool:
        """Clear all chunks from the store."""
        if not self.initialized:
            await self.initialize()

        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(f"DELETE FROM {self.table_name}")
                await conn.commit()
                return True

    async def search_similar(self, query_embedding: List[float],
                            top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """Search for similar chunks using pgvector.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            **kwargs: Additional search parameters

        Returns:
            List of similar chunks with metadata and distances
        """
        if not self.initialized:
            await self.initialize()

        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(f"""
                    SELECT
                        id, content, metadata, created_at, updated_at,
                        embedding <=> %s as distance
                    FROM {self.table_name}
                    WHERE embedding IS NOT NULL
                    ORDER BY distance
                    LIMIT %s
                """, (query_embedding, top_k))

                results = []
                async for row in cur:
                    # Parse metadata JSON
                    metadata = json.loads(row[2]) if row[2] else {}

                    results.append({
                        'id': row[0],
                        'content': row[1],
                        'metadata': metadata,
                        'created_at': row[3].isoformat() if row[3] else None,
                        'updated_at': row[4].isoformat() if row[4] else None,
                        'distance': float(row[5])
                    })

                return results

    async def search_by_text(self, query_text: str, top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """Search for similar chunks using text query.

        This method generates embeddings for the query text and then searches.

        Args:
            query_text: Text query to search for
            top_k: Number of results to return
            **kwargs: Additional search parameters

        Returns:
            List of similar chunks
        """
        # Generate embedding for the query text
        query_embedding = await self._generate_embedding(query_text)

        # Use vector similarity search
        return await self.search_similar(query_embedding, top_k, **kwargs)

    async def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific chunk by ID.

        Args:
            chunk_id: ID of the chunk to retrieve

        Returns:
            Chunk data or None if not found
        """
        if not self.initialized:
            await self.initialize()

        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(f"""
                    SELECT id, content, metadata, created_at, updated_at
                    FROM {self.table_name}
                    WHERE id = %s
                """, (chunk_id,))

                row = await cur.fetchone()
                if row:
                    metadata = json.loads(row[2]) if row[2] else {}
                    return {
                        'id': row[0],
                        'content': row[1],
                        'metadata': metadata,
                        'created_at': row[3].isoformat() if row[3] else None,
                        'updated_at': row[4].isoformat() if row[4] else None
                    }

                return None

    async def delete_chunk(self, chunk_id: str) -> bool:
        """Delete a chunk by ID.

        Args:
            chunk_id: ID of the chunk to delete

        Returns:
            True if deleted, False if not found
        """
        if not self.initialized:
            await self.initialize()

        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(f"""
                    DELETE FROM {self.table_name}
                    WHERE id = %s
                """, (chunk_id,))

                await conn.commit()
                return cur.rowcount > 0

    async def count_chunks(self) -> int:
        """Get total number of chunks in the store.

        Returns:
            Number of chunks
        """
        if not self.initialized:
            await self.initialize()

        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(f"SELECT COUNT(*) FROM {self.table_name}")
                row = await cur.fetchone()
                return row[0] if row else 0

    async def close(self):
        """Close the store and cleanup resources."""
        try:
            if self.pool:
                logger.debug("Closing connection pool...")
                # Cancel all pending tasks in the pool
                if hasattr(self.pool, '_workers'):
                    for worker in self.pool._workers:
                        if not worker.done():
                            worker.cancel()

                # Close the pool
                await self.pool.close()
                logger.debug("Connection pool closed successfully")

            if self.vectorizer_worker:
                # Stop the worker (implementation depends on pgai version)
                logger.debug("Stopping vectorizer worker...")
                pass

            self.initialized = False
            logger.debug("PgaiStore closed successfully")

        except Exception as e:
            logger.error(f"Error during PgaiStore cleanup: {e}")
            self.initialized = False

    async def get_chunks_by_session(self, session_id: str) -> List[ChunkData]:
        """Get all chunks for a specific session."""
        if not self.initialized:
            await self.initialize()

        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(f"""
                    SELECT id, content, metadata, created_at, updated_at
                    FROM {self.table_name}
                    WHERE metadata->>'session_id' = %s
                    ORDER BY created_at
                """, (session_id,))

                results = []
                async for row in cur:
                    metadata = json.loads(row[2]) if row[2] else {}
                    metadata['created_at'] = row[3].isoformat() if row[3] else None
                    metadata['updated_at'] = row[4].isoformat() if row[4] else None

                    chunk_data = ChunkData(
                        content=row[1],
                        chunk_id=row[0],
                        metadata=metadata
                    )
                    results.append(chunk_data)

                return results

    async def get_chunks_by_round(self, round_id: str) -> List[ChunkData]:
        """Get all chunks for a specific round."""
        if not self.initialized:
            await self.initialize()

        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(f"""
                    SELECT id, content, metadata, created_at, updated_at
                    FROM {self.table_name}
                    WHERE metadata->>'round_id' = %s
                    ORDER BY created_at
                """, (round_id,))

                results = []
                async for row in cur:
                    metadata = json.loads(row[2]) if row[2] else {}
                    metadata['created_at'] = row[3].isoformat() if row[3] else None
                    metadata['updated_at'] = row[4].isoformat() if row[4] else None

                    chunk_data = ChunkData(
                        content=row[1],
                        chunk_id=row[0],
                        metadata=metadata
                    )
                    results.append(chunk_data)

                return results

    async def get_chunks_by_user(self, user_id: str) -> List[ChunkData]:
        """Get all chunks for a specific user."""
        if not self.initialized:
            await self.initialize()

        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(f"""
                    SELECT id, content, metadata, created_at, updated_at
                    FROM {self.table_name}
                    WHERE metadata->>'user_id' = %s
                    ORDER BY created_at
                """, (user_id,))

                results = []
                async for row in cur:
                    metadata = json.loads(row[2]) if row[2] else {}
                    metadata['created_at'] = row[3].isoformat() if row[3] else None
                    metadata['updated_at'] = row[4].isoformat() if row[4] else None

                    chunk_data = ChunkData(
                        content=row[1],
                        chunk_id=row[0],
                        metadata=metadata
                    )
                    results.append(chunk_data)

                return results

    async def get_chunks_by_strategy(self, strategy_type: str) -> List[ChunkData]:
        """Get all chunks created by a specific strategy."""
        if not self.initialized:
            await self.initialize()

        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(f"""
                    SELECT id, content, metadata, created_at, updated_at
                    FROM {self.table_name}
                    WHERE metadata->>'strategy_type' = %s
                    ORDER BY created_at
                """, (strategy_type,))

                results = []
                async for row in cur:
                    metadata = json.loads(row[2]) if row[2] else {}
                    metadata['created_at'] = row[3].isoformat() if row[3] else None
                    metadata['updated_at'] = row[4].isoformat() if row[4] else None

                    chunk_data = ChunkData(
                        content=row[1],
                        chunk_id=row[0],
                        metadata=metadata
                    )
                    results.append(chunk_data)

                return results

    async def get_chunks_stats(self, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get statistics about chunks in the store."""
        if not self.initialized:
            await self.initialize()

        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                # Get total count
                await cur.execute(f"SELECT COUNT(*) FROM {self.table_name}")
                total_chunks = (await cur.fetchone())[0]

                # Get count by session
                await cur.execute(f"""
                    SELECT metadata->>'session_id' as session_id, COUNT(*)
                    FROM {self.table_name}
                    WHERE metadata->>'session_id' IS NOT NULL
                    GROUP BY metadata->>'session_id'
                """)
                by_session = {row[0]: row[1] async for row in cur}

                # Get count by strategy
                await cur.execute(f"""
                    SELECT metadata->>'strategy_type' as strategy_type, COUNT(*)
                    FROM {self.table_name}
                    WHERE metadata->>'strategy_type' IS NOT NULL
                    GROUP BY metadata->>'strategy_type'
                """)
                by_strategy = {row[0]: row[1] async for row in cur}

                # Get count by user
                await cur.execute(f"""
                    SELECT metadata->>'user_id' as user_id, COUNT(*)
                    FROM {self.table_name}
                    WHERE metadata->>'user_id' IS NOT NULL
                    GROUP BY metadata->>'user_id'
                """)
                by_user = {row[0]: row[1] async for row in cur}

                return {
                    'total_chunks': total_chunks,
                    'by_session': by_session,
                    'by_strategy': by_strategy,
                    'by_user': by_user,
                    'storage_size': 'N/A'  # Could be calculated if needed
                }

    def __del__(self):
        """Cleanup when object is destroyed."""
        if hasattr(self, 'initialized') and self.initialized:
            # Try to cleanup synchronously if possible
            if hasattr(self, 'pool') and self.pool:
                try:
                    # Cancel any pending tasks
                    if hasattr(self.pool, '_workers'):
                        for worker in self.pool._workers:
                            if not worker.done():
                                worker.cancel()

                    # Force close without waiting
                    if hasattr(self.pool, '_closed') and not self.pool._closed:
                        logger.debug("Force closing connection pool in destructor")
                        # This is a synchronous close, may not be perfect but better than nothing
                        try:
                            import asyncio
                            loop = asyncio.get_event_loop()
                            if loop.is_running():
                                # Schedule cleanup for later
                                loop.create_task(self.close())
                            else:
                                # Try to run cleanup
                                asyncio.run(self.close())
                        except Exception:
                            # Last resort: just mark as closed
                            self.initialized = False
                            logger.debug("PgaiStore destroyed with forced cleanup")
                except Exception as e:
                    logger.debug(f"Error in PgaiStore destructor: {e}")

            self.initialized = False
