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
from ...rag.chunk.integrated import IntegratedChunkingProcessor
from ...rag.chunk.base import ChunkData
from ...rag.encode.base import EncoderBase
from ...rag.encode.MiniLM import MiniLMEncoder


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
        encoder: Optional[EncoderBase] = None,
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

        # Initialize chunking processor
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

            # Verify tables exist
            await self._verify_schema()

            # Initialize encoder with global model management
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
            # Step 1: Insert raw messages into M0 layer
            m0_message_ids = await self._insert_m0_messages(chunks)
            
            # Step 2: Create M1 chunks with intelligent chunking
            m1_chunk_ids = await self._create_m1_chunks(chunks, m0_message_ids)
            
            added_ids.extend(m1_chunk_ids)
            
            # Update metrics
            self.metrics["add_time"] += time.time() - start_time
            self.metrics["add_count"] += len(chunks)
            
            # Invalidate query cache
            self.query_cache = {}
            
            logger.info(f"PgVectorScaleStore: Added {len(chunks)} chunks -> {len(m0_message_ids)} M0 messages -> {len(m1_chunk_ids)} M1 chunks")
            
            return added_ids
            
        except Exception as e:
            logger.error(f"PgVectorScaleStore: Add operation failed: {e}")
            raise
    
    async def _insert_m0_messages(self, chunks: List[ChunkData]) -> List[str]:
        """Insert raw messages into M0 raw layer."""
        m0_ids = []

        try:
            with self.conn.cursor() as cur:
                insert_query = """
                    INSERT INTO m0_raw
                    (id, content, metadata, created_at)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id
                """

                for i, chunk in enumerate(chunks):
                    message_id = str(uuid.uuid4())

                    # Extract metadata
                    role = chunk.metadata.get('role', 'user')
                    conversation_id = chunk.metadata.get('conversation_id', str(uuid.uuid4()))
                    sequence_number = chunk.metadata.get('sequence_number', i + 1)
                    batch_index = chunk.metadata.get('batch_index', 0)
                    message_index = chunk.metadata.get('message_index', i)

                    # Prepare metadata for M0 raw storage
                    m0_metadata = {
                        'conversation_id': conversation_id,
                        'sequence_number': sequence_number,
                        'batch_index': batch_index,
                        'message_index': message_index,
                        'token_count': max(1, len(chunk.content) // 4),
                        'processing_status': 'pending',
                        **chunk.metadata
                    }

                    cur.execute(insert_query, (
                        message_id,
                        chunk.content,
                        json.dumps(m0_metadata),
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
    
    async def _create_m1_chunks(self, chunks: List[ChunkData], m0_message_ids: List[str]) -> List[str]:
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
                    (id, source_id, source_session_id, source_user_id, episode_content,
                     episode_type, confidence, entities, temporal_info, source_context,
                     metadata, embedding, needs_embedding, retry_status, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """

                for i, (chunk, embedding) in enumerate(zip(validated_chunks, embeddings)):
                    chunk_id = str(uuid.uuid4())

                    # Map to corresponding M0 message IDs (simplified mapping)
                    source_id = m0_message_ids[min(i, len(m0_message_ids) - 1)]

                    # Extract metadata
                    conversation_id = chunk.metadata.get('conversation_id', str(uuid.uuid4()))
                    token_count = chunk.metadata.get('estimated_tokens', max(1, len(chunk.content) // 4))
                    chunking_strategy = chunk.metadata.get('chunking_strategy', 'token_based')
                    user_id = chunk.metadata.get('user_id', 'demo_user')

                    # Prepare episodic memory metadata
                    episode_metadata = {
                        'chunking_strategy': chunking_strategy,
                        'token_count': token_count,
                        'm0_message_ids': m0_message_ids,
                        'conversation_id': conversation_id,
                        'embedding_model': self.model_name,
                        **chunk.metadata
                    }

                    cur.execute(insert_query, (
                        chunk_id,
                        source_id,  # source_id (first M0 message)
                        conversation_id,  # source_session_id
                        user_id,  # source_user_id
                        chunk.content,  # episode_content
                        'conversation_chunk',  # episode_type
                        1.0,  # confidence
                        json.dumps([]),  # entities (empty for now)
                        json.dumps({}),  # temporal_info (empty for now)
                        f"Chunked from {len(m0_message_ids)} M0 messages",  # source_context
                        json.dumps(episode_metadata),  # metadata
                        embedding.tolist(),  # embedding
                        False,  # needs_embedding (already generated)
                        'completed',  # retry_status
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
                        id,
                        episode_content,
                        (1.0 - (embedding <=> %s::vector) / 2.0) as similarity_score,
                        (embedding <=> %s::vector) as distance,
                        source_id,
                        source_session_id,
                        source_user_id,
                        episode_type,
                        confidence,
                        metadata,
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
                    # Parse metadata JSON (handle both string and dict cases)
                    if row['metadata']:
                        if isinstance(row['metadata'], str):
                            episode_metadata = json.loads(row['metadata'])
                        else:
                            episode_metadata = row['metadata']
                    else:
                        episode_metadata = {}

                    metadata = {
                        'chunk_id': row['id'],
                        'similarity_score': float(row['similarity_score']),
                        'distance': float(row['distance']),
                        'source_id': row['source_id'],
                        'source_session_id': row['source_session_id'],
                        'source_user_id': row['source_user_id'],
                        'episode_type': row['episode_type'],
                        'confidence': row['confidence'],
                        'created_at': row['created_at'],
                        'source': 'pgvectorscale_m1_episodic',
                        **episode_metadata
                    }

                    chunk = ChunkData(
                        content=row['episode_content'],
                        chunk_id=row['id'],
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
                    DELETE FROM m1_episodic WHERE id = ANY(%s)
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
                           m0_message_ids, conversation_id, created_at
                    FROM m1_chunks
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
                        'm0_message_ids': row['m0_message_ids'],
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
                    SELECT embedding FROM m1_chunks WHERE chunk_id = %s
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
                    UPDATE m1_chunks
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
                cur.execute("SELECT COUNT(*) FROM m1_chunks")
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
                cur.execute("DELETE FROM m1_chunks")
                m1_deleted = cur.rowcount

                # Clear M0 messages
                cur.execute("DELETE FROM m0_messages")
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
                    DELETE FROM m1_chunks WHERE chunk_id = ANY(%s)
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
                           m0_message_ids, conversation_id, created_at
                    FROM m1_chunks
                    WHERE conversation_id = %s
                    ORDER BY created_at
                """, (session_id,))

                rows = cur.fetchall()

                results = []
                for row in rows:
                    metadata = {
                        'chunk_id': row['chunk_id'],
                        'chunking_strategy': row['chunking_strategy'],
                        'token_count': row['token_count'],
                        'm0_message_ids': row['m0_message_ids'],
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
                           m0_message_ids, conversation_id, created_at
                    FROM m1_chunks
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
                        'm0_message_ids': row['m0_message_ids'],
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
