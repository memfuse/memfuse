"""
PgVectorScale vector store implementation for MemFuse.

This implementation provides high-performance vector similarity search using
pgvectorscale with StreamingDiskANN, supporting the M0/M1 memory layer architecture.
"""

import json
import uuid
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
from loguru import logger

from .base import VectorStore
from ...rag.chunk.base import ChunkData
from ...rag.chunk.integrated import IntegratedChunkingProcessor
from ...rag.encode.MiniLM import MiniLMEncoder
from ...memory import M0Processor, M1Processor
from ...models.schema import SchemaManager
from ...rag.encode.embedding_service import EmbeddingService


class PgVectorScaleStore(VectorStore):
    """
    PgVectorScale vector store implementation with M0/M1 memory layer support.
    
    This store implements the simplified MemFuse architecture:
    - M0 Layer: Raw streaming messages with metadata
    - M1 Layer: Intelligent chunks with embeddings and StreamingDiskANN indexing
    - Normalized similarity scores (0-1 range) for cross-system compatibility
    """
    
    def __init__(
        self,
        data_dir: str = "",
        encoder: Optional[Any] = None,
        model_name: str = "all-MiniLM-L6-v2",
        cache_size: int = 100,
        buffer_size: int = 10,
        db_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Initialize the PgVectorScale store.
        
        Args:
            data_dir: Not used for PostgreSQL, kept for compatibility
            encoder: Encoder to use (if None, a MiniLMEncoder will be created)
            model_name: Name of the embedding model
            cache_size: Size of the query cache
            buffer_size: Size of the write buffer
            db_config: Database connection configuration
            **kwargs: Additional arguments
        """
        super().__init__(data_dir, encoder, model_name, cache_size, buffer_size, **kwargs)
        
        # Database configuration
        self.db_config = db_config or {
            'host': 'localhost',
            'port': 5432,
            'database': 'memfuse',
            'user': 'postgres',
            'password': 'postgres'
        }
        
        self.conn = None
        self.embedding_dim = 384  # sentence-transformers/all-MiniLM-L6-v2

        # M0/M1 layer configuration
        self.chunk_token_limit = 200  # Token limit per M1 chunk

        # Initialize new components
        self.schema_manager = SchemaManager()
        self.m0_processor = M0Processor()
        self.m1_processor = M1Processor(embedding_model=model_name)
        self.embedding_service = EmbeddingService(
            model_name=model_name,
            cache_size=cache_size
        )

        # Keep legacy chunking processor for compatibility
        self.chunking_processor = IntegratedChunkingProcessor(
            strategy_name='contextual',
            max_words_per_chunk=self.chunk_token_limit
        )

        logger.info(f"PgVectorScaleStore: Initialized with model {model_name}")
    
    async def initialize(self) -> None:
        """Initialize the database connection, verify schema, and set up global encoder."""
        try:
            self.conn = psycopg2.connect(**self.db_config)
            self.conn.autocommit = True

            # Verify pgvectorscale extension
            with self.conn.cursor() as cur:
                cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vectorscale';")
                result = cur.fetchone()
                if result:
                    logger.info(f"PgVectorScaleStore: Connected with pgvectorscale version {result[0]}")
                else:
                    logger.warning("PgVectorScaleStore: pgvectorscale extension not found, using standard pgvector")

            # Verify tables exist using schema manager
            await self._verify_schema_with_manager()

            # Also run legacy schema verification for compatibility
            await self._verify_schema()

            # Initialize new components
            await self.m1_processor.initialize()
            await self.embedding_service.initialize()

            # Initialize encoder with global model management (for compatibility)
            await self._initialize_encoder()

            # Initialize chunking processor
            await self.chunking_processor.initialize()

            logger.info("PgVectorScaleStore: Initialization complete")

        except Exception as e:
            logger.error(f"PgVectorScaleStore: Initialization failed: {e}")
            raise

    async def _initialize_encoder(self) -> None:
        """Initialize encoder using global model management."""
        try:
            # Try to get global embedding model first
            global_model = None
            try:
                from ...services.global_model_manager import get_global_model_manager
                global_model_manager = get_global_model_manager()
                global_model = global_model_manager.get_embedding_model()

                if global_model is not None:
                    logger.info("PgVectorScaleStore: Using global embedding model")

                    # If encoder is not already set, create one with the global model
                    if self.encoder is None:
                        # Check if global_model is already a MiniLMEncoder
                        if hasattr(global_model, 'encode_text'):
                            self.encoder = global_model
                            logger.info("PgVectorScaleStore: Using global MiniLMEncoder directly")
                        else:
                            # Create MiniLMEncoder with existing model
                            self.encoder = MiniLMEncoder(
                                model_name=self.model_name,
                                existing_model=global_model,
                                cache_size=self.cache_size
                            )
                            logger.info("PgVectorScaleStore: Created MiniLMEncoder with global model")

                    return

            except Exception as e:
                logger.warning(f"PgVectorScaleStore: Could not get global model: {e}")

            # Fallback: create encoder if not already set
            if self.encoder is None:
                logger.info("PgVectorScaleStore: Creating new MiniLMEncoder")
                self.encoder = MiniLMEncoder(
                    model_name=self.model_name,
                    cache_size=self.cache_size
                )

            logger.info("PgVectorScaleStore: Encoder initialization complete")

        except Exception as e:
            logger.error(f"PgVectorScaleStore: Encoder initialization failed: {e}")
            raise

    async def _verify_schema_with_manager(self) -> None:
        """Verify schema using the schema manager."""
        try:
            # Check if tables exist
            table_names = self.schema_manager.get_table_names()

            with self.conn.cursor() as cur:
                for table_name in table_names:
                    cur.execute("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables
                            WHERE table_name = %s
                        );
                    """, (table_name,))

                    exists = cur.fetchone()[0]
                    if not exists:
                        logger.warning(f"PgVectorScaleStore: Table {table_name} does not exist")
                        # Could auto-create here if needed
                    else:
                        logger.debug(f"PgVectorScaleStore: Table {table_name} exists")

            logger.info("PgVectorScaleStore: Schema verification with manager completed")

        except Exception as e:
            logger.error(f"PgVectorScaleStore: Schema verification with manager failed: {e}")
            raise

    async def _verify_schema(self) -> None:
        """Verify that required M0/M1 tables exist."""
        try:
            with self.conn.cursor() as cur:
                # Check M0 table
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_name = 'm0_raw'
                    );
                """)
                m0_exists = cur.fetchone()[0]

                # Check M1 table
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_name = 'm1_episodic'
                    );
                """)
                m1_exists = cur.fetchone()[0]

                if not (m0_exists and m1_exists):
                    raise Exception(f"Required tables missing: M0={m0_exists}, M1={m1_exists}")

                # Ensure StreamingDiskANN index exists on m1_episodic
                await self._ensure_diskann_index()

                logger.info("PgVectorScaleStore: Schema verification passed")

        except Exception as e:
            logger.error(f"PgVectorScaleStore: Schema verification failed: {e}")
            raise

    async def _ensure_diskann_index(self) -> None:
        """Ensure StreamingDiskANN index exists on m1_episodic table."""
        try:
            with self.conn.cursor() as cur:
                # Check if StreamingDiskANN index exists
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM pg_indexes
                        WHERE tablename = 'm1_episodic'
                        AND indexdef LIKE '%diskann%'
                    );
                """)

                index_exists = cur.fetchone()[0]

                if not index_exists:
                    logger.info("PgVectorScaleStore: Creating StreamingDiskANN index on m1_episodic")

                    # Create StreamingDiskANN index for optimal vector similarity search
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS idx_m1_episodic_embedding_diskann
                        ON m1_episodic
                        USING diskann (embedding vector_cosine_ops)
                        WITH (
                            storage_layout = 'memory_optimized',
                            num_neighbors = 50,
                            search_list_size = 100,
                            max_alpha = 1.2,
                            num_dimensions = 384,
                            num_bits_per_dimension = 2
                        );
                    """)

                    logger.info("PgVectorScaleStore: StreamingDiskANN index created successfully")
                else:
                    logger.debug("PgVectorScaleStore: StreamingDiskANN index already exists")

        except Exception as e:
            logger.warning(f"PgVectorScaleStore: Failed to create StreamingDiskANN index: {e}")
            # Don't raise - fallback to regular HNSW index if available

    async def add(self, chunks: List[ChunkData]) -> List[str]:
        """Add chunks to the vector store using M0 -> M1 processing pipeline.
        
        Args:
            chunks: List of chunks to add
            
        Returns:
            List of added chunk IDs
        """
        if not chunks:
            return []
        
        if not self.conn:
            await self.initialize()
        
        start_time = time.time()
        added_ids = []
        
        try:
            # Step 1: Process messages for M0 layer using M0 processor
            session_id = chunks[0].metadata.get('session_id', str(uuid.uuid4())) if chunks else str(uuid.uuid4())

            # Convert chunks to messages for M0 processing
            messages = []
            for chunk in chunks:
                message = {
                    'content': chunk.content,
                    'role': chunk.metadata.get('role', 'user')
                }
                messages.append(message)

            # Process with M0 processor
            m0_records = await self.m0_processor.process_messages(messages, session_id)

            # Insert M0 records into database
            m0_raw_ids = await self._insert_m0_records(m0_records)

            # Step 2: Process chunks for M1 layer using M1 processor
            m1_records = await self.m1_processor.process_chunks(chunks, m0_raw_ids, session_id)

            # Insert M1 records into database
            m1_chunk_ids = await self._insert_m1_records(m1_records)

            added_ids.extend(m1_chunk_ids)

            # Update metrics
            self.metrics["add_time"] += time.time() - start_time
            self.metrics["add_count"] += len(chunks)

            # Invalidate query cache
            self.query_cache = {}

            logger.info(f"PgVectorScaleStore: Added {len(chunks)} chunks -> {len(m0_raw_ids)} M0 messages -> {len(m1_chunk_ids)} M1 chunks")

            return added_ids

        except Exception as e:
            logger.error(f"PgVectorScaleStore: Add operation failed: {e}")
            raise

    async def _insert_m0_records(self, m0_records: List[Dict[str, Any]]) -> List[str]:
        """Insert M0 records into database."""
        m0_ids = []

        try:
            with self.conn.cursor() as cur:
                insert_query = """
                    INSERT INTO m0_raw
                    (message_id, content, role, conversation_id, sequence_number, token_count,
                     processing_status, chunk_assignments, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING message_id
                """

                for record in m0_records:
                    cur.execute(insert_query, (
                        record['message_id'],
                        record['content'],
                        record['role'],
                        record['conversation_id'],
                        record['sequence_number'],
                        record['token_count'],
                        record['processing_status'],
                        record['chunk_assignments'],
                        record['created_at'],
                        record['updated_at']
                    ))

                    result = cur.fetchone()
                    if result:
                        m0_ids.append(result[0])

                logger.info(f"PgVectorScaleStore: Inserted {len(m0_ids)} M0 records")
                return m0_ids

        except Exception as e:
            logger.error(f"PgVectorScaleStore: M0 record insertion failed: {e}")
            raise

    async def _insert_m1_records(self, m1_records: List[Dict[str, Any]]) -> List[str]:
        """Insert M1 records into database."""
        m1_ids = []

        try:
            with self.conn.cursor() as cur:
                insert_query = """
                    INSERT INTO m1_episodic
                    (chunk_id, content, chunking_strategy, token_count, embedding, needs_embedding,
                     m0_raw_ids, conversation_id, created_at, updated_at,
                     embedding_generated_at, embedding_model, chunk_quality_score)
                    VALUES (%s, %s, %s, %s, %s, %s, %s::uuid[], %s, %s, %s, %s, %s, %s)
                    RETURNING chunk_id
                """

                for record in m1_records:
                    # Handle embedding conversion - can be None for async processing
                    embedding_str = None
                    needs_embedding = True

                    if 'embedding' in record and record['embedding'] is not None:
                        embedding_vector = record['embedding']
                        if isinstance(embedding_vector, list):
                            # Convert list to string format for vector type: '[1,2,3]'
                            embedding_str = '[' + ','.join(map(str, embedding_vector)) + ']'
                            needs_embedding = False
                        elif hasattr(embedding_vector, 'tolist'):
                            # It's a numpy array
                            embedding_str = '[' + ','.join(map(str, embedding_vector.tolist())) + ']'
                            needs_embedding = False
                        elif hasattr(embedding_vector, '__iter__'):
                            # It's some other iterable
                            embedding_str = '[' + ','.join(map(str, embedding_vector)) + ']'
                            needs_embedding = False
                        else:
                            logger.warning(f"Unexpected embedding format: {type(embedding_vector)}, treating as needs_embedding=True")

                    # Override needs_embedding if explicitly set in record
                    if 'needs_embedding' in record:
                        needs_embedding = record['needs_embedding']

                    cur.execute(insert_query, (
                        record['chunk_id'],
                        record['content'],
                        record['chunking_strategy'],
                        record['token_count'],
                        embedding_str,  # Can be None for async processing
                        needs_embedding,
                        record['m0_raw_ids'],
                        record['conversation_id'],
                        record['created_at'],
                        record['updated_at'],
                        record.get('embedding_generated_at'),  # Can be None
                        record['embedding_model'],
                        record['chunk_quality_score']
                    ))

                    result = cur.fetchone()
                    if result:
                        m1_ids.append(result[0])

                logger.info(f"PgVectorScaleStore: Inserted {len(m1_ids)} M1 records")
                return m1_ids

        except Exception as e:
            logger.error(f"PgVectorScaleStore: M1 record insertion failed: {e}")
            raise

    async def _insert_m0_messages(self, chunks: List[ChunkData]) -> List[str]:
        """Insert raw messages into M0 raw layer."""
        m0_ids = []

        try:
            with self.conn.cursor() as cur:
                insert_query = """
                    INSERT INTO m0_raw
                    (message_id, content, role, conversation_id, sequence_number, token_count,
                     processing_status, chunk_assignments, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING message_id
                """

                for i, chunk in enumerate(chunks):
                    message_id = str(uuid.uuid4())

                    # Extract metadata
                    role = chunk.metadata.get('role', 'user')
                    conversation_id = chunk.metadata.get('conversation_id', str(uuid.uuid4()))
                    sequence_number = chunk.metadata.get('sequence_number', i + 1)
                    token_count = max(1, len(chunk.content) // 4)

                    cur.execute(insert_query, (
                        message_id,
                        chunk.content,
                        role,
                        conversation_id,
                        sequence_number,
                        token_count,
                        'pending',  # processing_status
                        [],  # chunk_assignments (empty initially)
                        datetime.now()
                    ))

                    result = cur.fetchone()
                    if result:
                        m0_ids.append(result[0])

                logger.info(f"PgVectorScaleStore: Inserted {len(m0_ids)} M0 raw messages")
                return m0_ids

        except Exception as e:
            logger.error(f"PgVectorScaleStore: M0 insertion failed: {e}")
            raise
    
    async def _create_m1_chunks(self, chunks: List[ChunkData], m0_raw_ids: List[str]) -> List[str]:
        """Create M1 chunks with embeddings using intelligent chunking strategy."""
        m1_ids = []

        try:
            # Convert ChunkData to messages for re-chunking with integrated processor
            messages = []
            for i, chunk in enumerate(chunks):
                message = {
                    'content': chunk.content,
                    'role': chunk.metadata.get('role', 'user'),
                    'conversation_id': chunk.metadata.get('conversation_id', str(uuid.uuid4())),
                    'sequence_number': i + 1,
                    'chunk_id': chunk.chunk_id,
                    'metadata': chunk.metadata
                }
                messages.append(message)

            # Use integrated chunking processor to create optimized chunks
            processed_chunks = await self.chunking_processor.process_messages_to_chunks(messages)

            # Validate chunks
            validated_chunks = await self.chunking_processor.validate_chunks(processed_chunks)

            if not validated_chunks:
                logger.warning("PgVectorScaleStore: No valid chunks after processing")
                return []

            # Generate embeddings for processed chunks
            contents = [chunk.content for chunk in validated_chunks]
            embeddings = await self._generate_embeddings(contents)

            with self.conn.cursor() as cur:
                insert_query = """
                    INSERT INTO m1_episodic
                    (chunk_id, content, chunking_strategy, token_count, embedding,
                     m0_raw_ids, conversation_id, embedding_generated_at,
                     embedding_model, chunk_quality_score, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s::uuid[], %s, %s, %s, %s, %s)
                    RETURNING chunk_id
                """

                for i, (chunk, embedding) in enumerate(zip(validated_chunks, embeddings)):
                    chunk_id = str(uuid.uuid4())

                    # Extract metadata
                    conversation_id = chunk.metadata.get('conversation_id', str(uuid.uuid4()))
                    token_count = chunk.metadata.get('estimated_tokens', max(1, len(chunk.content) // 4))
                    chunking_strategy = chunk.metadata.get('chunking_strategy', 'token_based')

                    # Convert UUIDs to strings for the array (psycopg2 compatibility)
                    m0_uuid_array = [str(uid) for uid in m0_raw_ids]

                    cur.execute(insert_query, (
                        chunk_id,
                        chunk.content,
                        chunking_strategy,
                        token_count,
                        embedding.tolist(),  # embedding
                        m0_uuid_array,  # m0_raw_ids array (as UUID objects)
                        conversation_id,
                        datetime.now(),  # embedding_generated_at
                        self.model_name,  # embedding_model
                        1.0,  # chunk_quality_score
                        datetime.now()  # created_at
                    ))

                    result = cur.fetchone()
                    if result:
                        m1_ids.append(result[0])

                logger.info(f"PgVectorScaleStore: Created {len(m1_ids)} M1 episodic memories with embeddings using integrated chunking")
                return m1_ids

        except Exception as e:
            logger.error(f"PgVectorScaleStore: M1 chunk creation failed: {e}")
            raise

    async def query(self, query_text: str, top_k: int = 10, **kwargs) -> List[ChunkData]:
        """Query similar chunks using high-performance vector similarity search.

        Args:
            query_text: Query text to search for
            top_k: Number of top results to return
            **kwargs: Additional query parameters

        Returns:
            List of similar ChunkData objects
        """
        if not self.conn:
            await self.initialize()

        start_time = time.time()

        try:
            # Generate query embedding
            query_embedding = await self.encoder.encode_text(query_text)

            # Perform similarity search using the custom function
            similarity_threshold = kwargs.get('similarity_threshold', 0.1)

            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Direct query on m1_episodic with normalized similarity scores
                cur.execute("""
                    SELECT
                        chunk_id,
                        content,
                        (1.0 - (embedding <=> %s::vector) / 2.0) as similarity_score,
                        (embedding <=> %s::vector) as distance,
                        chunking_strategy,
                        token_count,
                        m0_raw_ids,
                        session_id,
                        embedding_model,
                        chunk_quality_score,
                        created_at
                    FROM m1_episodic
                    WHERE (1.0 - (embedding <=> %s::vector) / 2.0) >= %s
                    ORDER BY embedding <=> %s::vector ASC
                    LIMIT %s
                """, (
                    query_embedding.tolist(), query_embedding.tolist(),
                    query_embedding.tolist(), similarity_threshold,
                    query_embedding.tolist(), top_k
                ))

                rows = cur.fetchall()

                # Convert results to ChunkData objects
                results = []
                for row in rows:
                    metadata = {
                        'chunk_id': row['chunk_id'],
                        'similarity_score': float(row['similarity_score']),
                        'distance': float(row['distance']),
                        'chunking_strategy': row['chunking_strategy'],
                        'token_count': row['token_count'],
                        'm0_raw_ids': row['m0_raw_ids'],
                        'conversation_id': row['conversation_id'],
                        'embedding_model': row['embedding_model'],
                        'chunk_quality_score': row['chunk_quality_score'],
                        'created_at': row['created_at'],
                        'source': 'pgvectorscale_m1_episodic'
                    }

                    chunk = ChunkData(
                        content=row['content'],
                        chunk_id=row['chunk_id'],
                        metadata=metadata
                    )
                    results.append(chunk)

                # Update metrics
                self.metrics["query_time"] += time.time() - start_time
                self.metrics["query_count"] += 1

                logger.info(f"PgVectorScaleStore: Query returned {len(results)} results in {time.time() - start_time:.3f}s")

                return results

        except Exception as e:
            logger.error(f"PgVectorScaleStore: Query failed: {e}")
            return []

    async def delete(self, chunk_ids: List[str]) -> bool:
        """Delete chunks from both M0 and M1 layers.

        Args:
            chunk_ids: List of chunk IDs to delete

        Returns:
            True if deletion was successful
        """
        if not chunk_ids or not self.conn:
            return False

        try:
            with self.conn.cursor() as cur:
                # Delete from M1 episodic memories
                cur.execute("""
                    DELETE FROM m1_episodic WHERE chunk_id = ANY(%s)
                """, (chunk_ids,))

                deleted_m1 = cur.rowcount

                # Note: We don't delete from M0 to preserve data lineage
                # M0 raw messages are immutable raw data

                logger.info(f"PgVectorScaleStore: Deleted {deleted_m1} M1 episodic memories")

                # Invalidate query cache
                self.query_cache = {}

                return deleted_m1 > 0

        except Exception as e:
            logger.error(f"PgVectorScaleStore: Delete failed: {e}")
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """Get store statistics including M0/M1 layer information."""
        if not self.conn:
            return {}

        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Get M0 raw statistics
                cur.execute("SELECT COUNT(*) as count FROM m0_raw")
                m0_count = cur.fetchone()['count']

                # Get M1 episodic statistics
                cur.execute("SELECT COUNT(*) as count FROM m1_episodic")
                m1_count = cur.fetchone()['count']

                # Get M1 embeddings statistics
                cur.execute("SELECT COUNT(*) as count FROM m1_episodic WHERE embedding IS NOT NULL")
                m1_embeddings_count = cur.fetchone()['count']

                # Get vector index information
                cur.execute("""
                    SELECT
                        schemaname,
                        tablename,
                        indexname,
                        CASE
                            WHEN indexdef LIKE '%diskann%' THEN 'StreamingDiskANN (pgvectorscale)'
                            WHEN indexdef LIKE '%hnsw%' THEN 'HNSW (pgvector)'
                            ELSE 'Other'
                        END as index_type
                    FROM pg_indexes
                    WHERE tablename IN ('m1_episodic')
                    AND indexdef LIKE '%vector%'
                """)
                index_stats = cur.fetchall()

                stats = {
                    'layer_stats': [
                        {'layer': 'M0 Raw', 'record_count': m0_count},
                        {'layer': 'M1 Episodic', 'record_count': m1_count},
                        {'layer': 'M1 Embeddings', 'record_count': m1_embeddings_count}
                    ],
                    'lineage_summary': {
                        'total_m0_messages': m0_count,
                        'total_m1_episodes': m1_count,
                        'embedding_coverage': m1_embeddings_count / max(1, m1_count)
                    },
                    'index_stats': [dict(row) for row in index_stats],
                    'store_metrics': self.metrics.copy()
                }

                return stats

        except Exception as e:
            logger.error(f"PgVectorScaleStore: Stats retrieval failed: {e}")
            return {}

    async def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.info("PgVectorScaleStore: Connection closed")

    async def add_with_embeddings(self, chunks: List[ChunkData], embeddings: List[np.ndarray]) -> List[str]:
        """Add chunks with pre-computed embeddings."""
        if not chunks or not embeddings:
            return []

        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")

        # For now, delegate to the regular add method which will re-compute embeddings
        # In a full implementation, this would use the provided embeddings directly
        return await self.add(chunks)

    async def get_chunks_by_ids(self, chunk_ids: List[str]) -> List[Optional[ChunkData]]:
        """Get chunks by their IDs."""
        if not chunk_ids or not self.conn:
            return [None] * len(chunk_ids)

        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT chunk_id, content, chunking_strategy, token_count,
                           m0_raw_ids, conversation_id, created_at
                    FROM m1_episodic
                    WHERE chunk_id = ANY(%s)
                """, (chunk_ids,))

                rows = cur.fetchall()

                # Create a mapping of chunk_id to chunk data
                chunk_map = {}
                for row in rows:
                    metadata = {
                        'chunk_id': row['chunk_id'],
                        'chunking_strategy': row['chunking_strategy'],
                        'token_count': row['token_count'],
                        'm0_raw_ids': row['m0_raw_ids'],
                        'conversation_id': row['conversation_id'],
                        'created_at': row['created_at'],
                        'source': 'pgvectorscale_m1'
                    }

                    chunk = ChunkData(
                        content=row['content'],
                        chunk_id=row['chunk_id'],
                        metadata=metadata
                    )
                    chunk_map[row['chunk_id']] = chunk

                # Return chunks in the same order as requested IDs
                return [chunk_map.get(chunk_id) for chunk_id in chunk_ids]

        except Exception as e:
            logger.error(f"PgVectorScaleStore: Get chunks by IDs failed: {e}")
            return [None] * len(chunk_ids)

    async def get_embedding(self, chunk_id: str) -> Optional[np.ndarray]:
        """Get the embedding for a chunk."""
        if not chunk_id or not self.conn:
            return None

        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT embedding FROM m1_episodic WHERE chunk_id = %s
                """, (chunk_id,))

                result = cur.fetchone()
                if result and result[0]:
                    return np.array(result[0])

                return None

        except Exception as e:
            logger.error(f"PgVectorScaleStore: Get embedding failed: {e}")
            return None

    async def query_by_embedding_chunks(self, embedding: np.ndarray, top_k: int = 5, query: Optional[Any] = None) -> List[ChunkData]:
        """Query the store by embedding and return chunks."""
        if not self.conn:
            return []

        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM search_similar_chunks(%s::vector, %s, %s)
                """, (embedding.tolist(), 0.1, top_k))

                rows = cur.fetchall()

                results = []
                for row in rows:
                    metadata = {
                        'chunk_id': row['chunk_id'],
                        'similarity_score': row['similarity_score'],
                        'distance': row['distance'],
                        'm0_message_count': row['m0_message_count'],
                        'chunking_strategy': row['chunking_strategy'],
                        'created_at': row['created_at'],
                        'source': 'pgvectorscale_m1'
                    }

                    chunk = ChunkData(
                        content=row['content'],
                        chunk_id=row['chunk_id'],
                        metadata=metadata
                    )
                    results.append(chunk)

                return results

        except Exception as e:
            logger.error(f"PgVectorScaleStore: Query by embedding failed: {e}")
            return []

    async def update_chunk_with_embedding(self, chunk_id: str, chunk: ChunkData, embedding: np.ndarray) -> bool:
        """Update a chunk with pre-computed embedding."""
        if not chunk_id or not self.conn:
            return False

        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    UPDATE m1_episodic
                    SET content = %s, embedding = %s, embedding_generated_at = %s
                    WHERE chunk_id = %s
                """, (chunk.content, embedding.tolist(), datetime.now(), chunk_id))

                return cur.rowcount > 0

        except Exception as e:
            logger.error(f"PgVectorScaleStore: Update chunk failed: {e}")
            return False

    async def get_chunk_count(self) -> int:
        """Get the total number of chunks in the store."""
        if not self.conn:
            return 0

        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM m1_episodic")
                result = cur.fetchone()
                return result[0] if result else 0

        except Exception as e:
            logger.error(f"PgVectorScaleStore: Get chunk count failed: {e}")
            return 0

    async def clear_all_chunks(self) -> bool:
        """Clear all chunks from the store."""
        if not self.conn:
            return False

        try:
            with self.conn.cursor() as cur:
                # Clear M1 chunks
                cur.execute("DELETE FROM m1_episodic")
                m1_deleted = cur.rowcount

                # Clear M0 messages
                cur.execute("DELETE FROM m0_raw")
                m0_deleted = cur.rowcount

                logger.info(f"PgVectorScaleStore: Cleared {m1_deleted} M1 chunks and {m0_deleted} M0 messages")
                return True

        except Exception as e:
            logger.error(f"PgVectorScaleStore: Clear all chunks failed: {e}")
            return False

    async def delete_chunks_by_ids(self, chunk_ids: List[str]) -> List[bool]:
        """Delete chunks by their IDs."""
        if not chunk_ids or not self.conn:
            return [False] * len(chunk_ids)

        try:
            with self.conn.cursor() as cur:
                # Delete from M1 chunks
                cur.execute("""
                    DELETE FROM m1_episodic WHERE chunk_id = ANY(%s)
                    RETURNING chunk_id
                """, (chunk_ids,))

                deleted_ids = {row[0] for row in cur.fetchall()}

                # Return success status for each requested ID
                return [chunk_id in deleted_ids for chunk_id in chunk_ids]

        except Exception as e:
            logger.error(f"PgVectorScaleStore: Delete chunks by IDs failed: {e}")
            return [False] * len(chunk_ids)

    # Additional methods for ChunkStoreInterface compatibility
    async def get_chunks_by_user(self, user_id: str) -> List[ChunkData]:
        """Get all chunks for a specific user."""
        # For now, return empty list as we don't have user-specific filtering
        return []

    async def get_chunks_by_session(self, session_id: str) -> List[ChunkData]:
        """Get all chunks for a specific session."""
        if not session_id or not self.conn:
            return []

        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT chunk_id, content, chunking_strategy, token_count,
                           m0_raw_ids, session_id, created_at
                    FROM m1_episodic
                    WHERE session_id = %s
                    ORDER BY created_at
                """, (session_id,))

                rows = cur.fetchall()

                results = []
                for row in rows:
                    metadata = {
                        'chunk_id': row['chunk_id'],
                        'chunking_strategy': row['chunking_strategy'],
                        'token_count': row['token_count'],
                        'm0_raw_ids': row['m0_raw_ids'],
                        'session_id': row['session_id'],
                        'created_at': row['created_at'],
                        'source': 'pgvectorscale_m1'
                    }

                    chunk = ChunkData(
                        content=row['content'],
                        chunk_id=row['chunk_id'],
                        metadata=metadata
                    )
                    results.append(chunk)

                return results

        except Exception as e:
            logger.error(f"PgVectorScaleStore: Get chunks by session failed: {e}")
            return []

    async def get_chunks_by_strategy(self, strategy: str) -> List[ChunkData]:
        """Get all chunks created with a specific chunking strategy."""
        if not strategy or not self.conn:
            return []

        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT chunk_id, content, chunking_strategy, token_count,
                           m0_raw_ids, conversation_id, created_at
                    FROM m1_episodic
                    WHERE chunking_strategy = %s
                    ORDER BY created_at
                """, (strategy,))

                rows = cur.fetchall()

                results = []
                for row in rows:
                    metadata = {
                        'chunk_id': row['chunk_id'],
                        'chunking_strategy': row['chunking_strategy'],
                        'token_count': row['token_count'],
                        'm0_raw_ids': row['m0_raw_ids'],
                        'conversation_id': row['conversation_id'],
                        'created_at': row['created_at'],
                        'source': 'pgvectorscale_m1'
                    }

                    chunk = ChunkData(
                        content=row['content'],
                        chunk_id=row['chunk_id'],
                        metadata=metadata
                    )
                    results.append(chunk)

                return results

        except Exception as e:
            logger.error(f"PgVectorScaleStore: Get chunks by strategy failed: {e}")
            return []

    async def get_chunks_by_round(self, round_id: str) -> List[ChunkData]:
        """Get all chunks for a specific round."""
        # For now, return empty list as we don't have round-specific filtering
        return []

    async def get_chunks_stats(self) -> Dict[str, Any]:
        """Get statistics about stored chunks."""
        return await self.get_stats()

    def __del__(self):
        """Cleanup on object destruction."""
        if hasattr(self, 'conn') and self.conn:
            try:
                self.conn.close()
            except Exception:
                pass
