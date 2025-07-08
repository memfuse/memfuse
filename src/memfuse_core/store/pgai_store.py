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

from ..interfaces.chunk_store import ChunkStoreInterface
from ..rag.chunk.base import ChunkData
from ..models import Query
from ..utils.config import config_manager


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
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, table_name: str = "m0_messages"):
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
            self.pgai_config = config.get("database", {}).get("pgai", {})

        # Build database URL
        self.db_url = self._build_database_url()
    
    def _build_database_url(self) -> str:
        """Build PostgreSQL database URL from configuration."""
        # Handle both config manager format and direct config format
        if "postgres" in self.db_config:
            # Config manager format
            postgres_config = self.db_config.get("postgres", {})
            host = postgres_config.get("host", "localhost")
            port = postgres_config.get("port", 5432)
            database = postgres_config.get("database", "memfuse")
            user = postgres_config.get("user", "postgres")
            password = postgres_config.get("password", "password")
        else:
            # Direct config format (for testing)
            host = self.db_config.get("host", "localhost")
            port = self.db_config.get("port", 5432)
            database = self.db_config.get("database", "memfuse")
            user = self.db_config.get("user", "postgres")
            password = self.db_config.get("password", "password")

        return f"postgresql://{user}:{password}@{host}:{port}/{database}"
    
    async def initialize(self) -> bool:
        """Initialize pgai store and vectorizer."""
        if self.initialized:
            return True
            
        try:
            logger.info(f"Initializing PgaiStore with table: {self.table_name}")
            
            # Skip pgai.install() as extensions are already installed in Docker
            logger.debug("Skipping pgai.install() - extensions already available")
            
            # Create connection pool
            logger.debug("Creating connection pool...")
            self.pool = AsyncConnectionPool(
                self.db_url,
                min_size=self.db_config.get("postgres", {}).get("pool_size", 5),
                max_size=self.db_config.get("postgres", {}).get("pool_size", 10),
                open=False,
                configure=self._setup_pgvector_connection
            )
            await self.pool.open()
            
            # Create tables only (skip vectorizer for now)
            await self._setup_schema()
            # TODO: Implement vectorizer creation when pgai API is clarified
            logger.debug("Skipping vectorizer creation - using manual embedding approach")
            
            # Skip vectorizer worker for now
            logger.debug("Skipping vectorizer worker - not implemented yet")
            
            self.initialized = True
            logger.info("PgaiStore initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize PgaiStore: {e}")
            return False
    
    async def _setup_pgvector_connection(self, conn: psycopg.AsyncConnection):
        """Setup pgvector for connection."""
        await register_vector_async(conn)
    
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
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                # Create table compatible with existing m0_messages schema
                await cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        id TEXT PRIMARY KEY,
                        content TEXT NOT NULL,
                        metadata JSONB DEFAULT '{{}}',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Create indexes for performance
                await cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.table_name}_created
                    ON {self.table_name}(created_at)
                """)

                # Create trigger to update updated_at timestamp
                await cur.execute(f"""
                    CREATE OR REPLACE FUNCTION update_{self.table_name}_updated_at()
                    RETURNS TRIGGER AS $$
                    BEGIN
                        NEW.updated_at = CURRENT_TIMESTAMP;
                        RETURN NEW;
                    END;
                    $$ LANGUAGE plpgsql;
                """)

                await cur.execute(f"""
                    DROP TRIGGER IF EXISTS trigger_update_{self.table_name}_updated_at
                    ON {self.table_name};
                """)

                await cur.execute(f"""
                    CREATE TRIGGER trigger_update_{self.table_name}_updated_at
                        BEFORE UPDATE ON {self.table_name}
                        FOR EACH ROW
                        EXECUTE FUNCTION update_{self.table_name}_updated_at();
                """)

            await conn.commit()
            logger.debug(f"Schema setup completed for table: {self.table_name}")

    async def _create_vectorizer(self):
        """Create pgai vectorizer for automatic embedding generation."""
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                # Get embedding configuration
                embedding_model = self.pgai_config.get("embedding_model", "text-embedding-3-small")
                embedding_dimensions = self.pgai_config.get("embedding_dimensions", 1536)
                chunk_size = self.pgai_config.get("chunk_size", 1000)
                chunk_overlap = self.pgai_config.get("chunk_overlap", 200)

                await cur.execute(f"""
                    SELECT ai.create_vectorizer(
                        '{self.table_name}'::regclass,
                        name => '{self.vectorizer_name}',
                        if_not_exists => true,
                        embedding => ai.embedding_openai(
                            model=>'{embedding_model}',
                            dimensions=>{embedding_dimensions}
                        ),
                        chunking => ai.chunking_recursive_character_text_splitter(
                            chunk_size => {chunk_size},
                            chunk_overlap => {chunk_overlap}
                        ),
                        formatting => ai.formatting_python_template(
                            'ID: $id\\nContent: $chunk\\nMetadata: $metadata'
                        ),
                        destination => ai.destination_table(
                            view_name => '{self.embedding_view}'
                        ),
                        scheduling => ai.scheduling_default()
                    )
                """)

            await conn.commit()
            logger.debug(f"Vectorizer created: {self.vectorizer_name}")

    async def add(self, chunks: List[ChunkData]) -> List[str]:
        """Add chunks to pgai store with batch embedding generation.

        Args:
            chunks: List of ChunkData objects to store

        Returns:
            List of chunk IDs that were added
        """
        if not self.initialized:
            await self.initialize()

        if not chunks:
            return []

        # Extract contents for batch embedding generation
        contents = [chunk.content for chunk in chunks]

        # Generate embeddings in batch for better performance
        embeddings = await self._generate_embeddings_batch(contents)

        chunk_ids = []
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                for chunk, embedding in zip(chunks, embeddings):
                    metadata_json = self._prepare_metadata(chunk.metadata)

                    await cur.execute(f"""
                        INSERT INTO {self.table_name}
                        (id, content, metadata, embedding)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (id) DO UPDATE SET
                            content = EXCLUDED.content,
                            metadata = EXCLUDED.metadata,
                            embedding = EXCLUDED.embedding,
                            updated_at = CURRENT_TIMESTAMP
                    """, (
                        chunk.chunk_id,
                        chunk.content,
                        metadata_json,
                        embedding
                    ))
                    chunk_ids.append(chunk.chunk_id)

            await conn.commit()

        logger.debug(f"Added {len(chunk_ids)} chunks to {self.table_name} with batch embeddings")
        return chunk_ids

    def _get_encoder(self):
        """Get encoder with priority for global singleton model."""
        # Priority 1: Use the encoder from PgaiVectorWrapper if available
        if hasattr(self, 'encoder') and self.encoder is not None:
            return self.encoder

        # Priority 2: Get encoder from ModelRegistry (global model provider)
        from ..interfaces.model_provider import ModelRegistry
        model_provider = ModelRegistry.get_provider()
        if model_provider and hasattr(model_provider, 'get_encoder'):
            encoder = model_provider.get_encoder()
            if encoder is not None:
                logger.debug("Using encoder from ModelRegistry")
                return encoder

        # Priority 3: Get global embedding model from ServiceFactory
        try:
            from ..services.service_factory import ServiceFactory
            global_embedding_model = ServiceFactory.get_global_embedding_model()

            if global_embedding_model is not None:
                logger.info("Using global singleton embedding model from ServiceFactory")
                # Create a wrapper to make the model compatible with encoder interface
                return GlobalEmbeddingModelWrapper(global_embedding_model)
        except Exception as e:
            logger.warning(f"Failed to get global embedding model: {e}")

        # Priority 4: Create direct encoder as last resort
        try:
            logger.info("Creating direct MiniLM encoder as fallback")
            from ..rag.encode.MiniLM import MiniLMEncoder
            # Use the configured model name
            from ..utils.config import config_manager
            config = config_manager.get_config()
            model_name = config.get("embedding", {}).get("model", "all-MiniLM-L6-v2")
            return MiniLMEncoder(model_name=model_name)
        except Exception as e:
            logger.error(f"Failed to create fallback encoder: {e}")
            return None

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
        if self.pool:
            await self.pool.close()

        if self.vectorizer_worker:
            # Stop the worker (implementation depends on pgai version)
            pass

        self.initialized = False
        logger.debug("PgaiStore closed")

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
        if self.initialized:
            # Note: Can't use async in __del__, so we just log
            logger.debug("PgaiStore being destroyed, resources may not be properly cleaned up")
