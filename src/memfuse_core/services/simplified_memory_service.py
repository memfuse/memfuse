"""
Simplified Memory Service for MemFuse

This is a simplified implementation of the Memory Service based on the MVP
pgvectorscale demo. It maintains interface compatibility with the existing
MemoryService but uses a much simpler internal implementation.

Key features:
- Direct M0/M1 processing without complex parallel layers
- pgvectorscale-based vector similarity search
- Normalized similarity scores (0-1 range)
- Compatible with existing QueryBuffer integration
- Simplified database connection management
"""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
from loguru import logger

import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
from sentence_transformers import SentenceTransformer

from ..interfaces import MessageInterface
from ..interfaces.message_interface import MessageBatchList
from .sync_connection_pool import sync_connection_pool


class SimplifiedDatabaseManager:
    """Simplified database connection and management."""

    def __init__(self, db_config: Dict[str, Any]):
        self.db_config = db_config
        self.conn = None
        self._connection_pool = None

    async def connect(self) -> None:
        """Establish database connection with retry logic."""
        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                self.conn = psycopg2.connect(**self.db_config)
                self.conn.autocommit = True
                logger.info("✅ Connected to pgvectorscale database")

                # Verify extensions
                await self._verify_extensions()
                return

            except Exception as e:
                logger.warning(f"Database connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"❌ Database connection failed after {max_retries} attempts")
                    raise

    async def _verify_extensions(self) -> None:
        """Verify required database extensions."""
        try:
            with self.conn.cursor() as cur:
                # Check vector extension (required)
                cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector';")
                result = cur.fetchone()
                if result:
                    logger.info(f"✅ pgvector extension version: {result[0]}")
                else:
                    raise Exception("pgvector extension not found")

                # Check vectorscale extension (optional but preferred)
                cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vectorscale';")
                result = cur.fetchone()
                if result:
                    logger.info(f"✅ pgvectorscale extension version: {result[0]}")
                else:
                    logger.warning("⚠️ pgvectorscale extension not found, using standard pgvector")

        except Exception as e:
            logger.error(f"❌ Extension verification failed: {e}")
            raise

    async def initialize_schema(self) -> None:
        """Initialize database schema if needed."""
        try:
            # Check if required tables exist
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT table_name FROM information_schema.tables
                    WHERE table_schema = 'public' AND table_name IN ('m0_raw', 'm1_episodic')
                """)
                existing_tables = [row[0] for row in cur.fetchall()]

                if 'm0_raw' not in existing_tables:
                    logger.error("❌ m0_raw table not found. Please run database initialization first.")
                    raise Exception("Required tables not found")

                if 'm1_episodic' not in existing_tables:
                    logger.error("❌ m1_episodic table not found. Please run database initialization first.")
                    raise Exception("Required tables not found")

                # Verify required functions exist
                await self._verify_functions()

                logger.info("✅ Database schema verification complete")

        except Exception as e:
            logger.error(f"❌ Schema initialization failed: {e}")
            raise

    async def _verify_functions(self) -> None:
        """Verify required database functions exist."""
        required_functions = [
            'search_similar_chunks',
            'normalize_cosine_similarity',
            'get_data_lineage_stats'
        ]

        with self.conn.cursor() as cur:
            for func_name in required_functions:
                cur.execute("""
                    SELECT COUNT(*) FROM pg_proc
                    WHERE proname = %s AND pronamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'public')
                """, (func_name,))

                count = cur.fetchone()[0]
                if count == 0:
                    logger.error(f"❌ Required function '{func_name}' not found")
                    raise Exception(f"Required function '{func_name}' not found")
                else:
                    logger.debug(f"✅ Function '{func_name}' found")

    async def health_check(self) -> bool:
        """Perform database health check."""
        try:
            if not self.conn:
                return False

            with self.conn.cursor() as cur:
                cur.execute("SELECT 1")
                result = cur.fetchone()
                return result is not None

        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False

    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            try:
                self.conn.close()
                logger.info("Database connection closed")
            except Exception as e:
                logger.warning(f"Error closing database connection: {e}")
            finally:
                self.conn = None


class SimplifiedChunkProcessor:
    """Simplified chunking processor based on MVP implementation."""
    
    def __init__(self, chunk_token_limit: int = 200):
        self.chunk_token_limit = chunk_token_limit

    def create_chunks(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create M1 chunks using token-based strategy."""
        chunks = []
        current_chunk_messages = []
        current_token_count = 0

        for message in messages:
            # Estimate token count (rough approximation: 1 token ≈ 4 characters)
            content = message.get('content', '')
            token_count = max(1, len(content) // 4)

            # Check if adding this message would exceed token limit
            if current_token_count + token_count > self.chunk_token_limit and current_chunk_messages:
                # Create chunk from accumulated messages
                chunk = self._create_chunk_from_messages(current_chunk_messages)
                chunks.append(chunk)

                # Start new chunk
                current_chunk_messages = [message]
                current_token_count = token_count
            else:
                # Add message to current chunk
                current_chunk_messages.append(message)
                current_token_count += token_count

        # Handle remaining messages
        if current_chunk_messages:
            chunk = self._create_chunk_from_messages(current_chunk_messages)
            chunks.append(chunk)

        logger.info(f"✅ Created {len(chunks)} M1 chunks from {len(messages)} messages")
        return chunks
    
    def _create_chunk_from_messages(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a single M1 chunk from a list of M0 messages."""
        # Combine message contents
        combined_content = " ".join([msg.get('content', '') for msg in messages])
        
        # Calculate total token count
        total_tokens = sum([max(1, len(msg.get('content', '')) // 4) for msg in messages])
        
        # Extract conversation_id and message_ids
        conversation_id = messages[0].get('metadata', {}).get('conversation_id')
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
            
        # Generate valid UUIDs for message_ids
        message_ids = []
        for msg in messages:
            msg_id = msg.get('id', str(uuid.uuid4()))
            try:
                # Validate UUID format
                uuid.UUID(msg_id)
                message_ids.append(msg_id)
            except ValueError:
                # If not a valid UUID, generate a new one
                message_ids.append(str(uuid.uuid4()))
        
        chunk = {
            'chunk_id': str(uuid.uuid4()),
            'content': combined_content,
            'chunking_strategy': 'token_based',
            'token_count': total_tokens,
            'conversation_id': conversation_id,
            'm0_message_ids': message_ids,
            'created_at': datetime.now()
        }
        
        return chunk


class SimplifiedEmbeddingGenerator:
    """Simplified embedding generator using global model manager."""

    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = None
        self.embedding_dim = 384  # for all-MiniLM-L6-v2

    async def initialize(self) -> None:
        """Initialize the embedding model using global model manager."""
        try:
            # Try to get global embedding model first
            from ..services.global_model_manager import get_global_model_manager

            global_manager = get_global_model_manager()
            global_model = global_manager.get_embedding_model()

            if global_model:
                # Check if the global model is a MiniLMEncoder or SentenceTransformer
                if hasattr(global_model, 'model') and hasattr(global_model.model, 'encode'):
                    # It's a MiniLMEncoder, use the underlying SentenceTransformer
                    self.model = global_model.model
                    logger.info("✅ Using global embedding model instance (MiniLMEncoder)")
                elif hasattr(global_model, 'encode'):
                    # It's a SentenceTransformer directly
                    self.model = global_model
                    logger.info("✅ Using global embedding model instance (SentenceTransformer)")
                else:
                    logger.warning(f"Global model type not recognized: {type(global_model)}")
                    raise RuntimeError("Global model is not compatible")
                return

            # Fallback: load model directly if global manager doesn't have it
            logger.info(f"🧠 Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("✅ Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            raise

    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        if not self.model:
            raise RuntimeError("Embedding model not initialized")

        return self.model.encode(text)


class SimplifiedMemoryService(MessageInterface):
    """Simplified Memory Service implementation based on MVP."""
    
    def __init__(
        self,
        cfg=None,
        user: str = "user_default",
        agent: Optional[str] = None,
        session: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        """Initialize the Simplified Memory Service."""
        # Store parameters
        self.user = user
        self.agent = agent or "agent_default"
        self.session = session
        self.session_id = session_id

        # Configuration
        self.config = cfg or {}

        # Database configuration with environment variable support
        self.db_config = self._get_database_config()

        # Components
        self.db_manager = SimplifiedDatabaseManager(self.db_config)
        self.chunk_processor = SimplifiedChunkProcessor(
            chunk_token_limit=self.config.get('chunk_token_limit', 200)
        )
        self.embedding_generator = SimplifiedEmbeddingGenerator(
            model_name=self.config.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
        )

        # Add compatibility attribute for BufferService (set early for compatibility)
        self.multi_path_retrieval = self  # Point to self for compatibility

        # State
        self._initialized = False

        logger.info(f"SimplifiedMemoryService: Initialized for user: {user}")

    def _get_database_config(self) -> Dict[str, Any]:
        """Get database configuration from config and environment variables."""
        import os

        # Default configuration
        db_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'memfuse',
            'user': 'postgres',
            'password': 'postgres'
        }

        # Override with config values (filter out non-connection parameters)
        if 'database' in self.config:
            config_db = self.config['database']
            # Only use connection-related parameters
            connection_params = ['host', 'port', 'database', 'user', 'password']
            for param in connection_params:
                if param in config_db:
                    db_config[param] = config_db[param]

        # Override with environment variables (highest priority)
        env_mapping = {
            'POSTGRES_HOST': 'host',
            'POSTGRES_PORT': 'port',
            'POSTGRES_DB': 'database',
            'POSTGRES_USER': 'user',
            'POSTGRES_PASSWORD': 'password'
        }

        for env_var, config_key in env_mapping.items():
            env_value = os.getenv(env_var)
            if env_value:
                if config_key == 'port':
                    db_config[config_key] = int(env_value)
                else:
                    db_config[config_key] = env_value

        logger.info(f"Database config: {db_config['user']}@{db_config['host']}:{db_config['port']}/{db_config['database']}")
        return db_config

    async def initialize(self):
        """Initialize the service components."""
        if self._initialized:
            return self

        logger.info("SimplifiedMemoryService: Starting initialization...")

        # Initialize synchronous connection pool
        sync_connection_pool.initialize(self.db_config)

        # Initialize database connection (fallback for schema operations)
        await self.db_manager.connect()
        await self.db_manager.initialize_schema()

        # Initialize embedding generator
        await self.embedding_generator.initialize()

        # Add compatibility attribute for BufferService
        self.multi_path_retrieval = self  # Point to self for compatibility

        self._initialized = True
        logger.info("SimplifiedMemoryService: Initialization complete")
        return self
    
    async def add_batch(self, message_batch_list: MessageBatchList, **kwargs) -> Dict[str, Any]:
        """Add a batch of message lists with simplified M0/M1 processing."""
        try:
            if not message_batch_list:
                return self._success_response([], "No message lists to process")

            logger.info(f"SimplifiedMemoryService: Processing {len(message_batch_list)} message lists")

            # Flatten message batch list
            all_messages = []
            for message_list in message_batch_list:
                all_messages.extend(message_list)

            if not all_messages:
                return self._success_response([], "No messages to process")

            # Step 1: Create session and round (like traditional MemoryService)
            session_id, round_id = await self._prepare_session_and_round(message_batch_list)
            logger.info(f"SimplifiedMemoryService: Prepared session_id={session_id}, round_id={round_id}")

            # Step 2: Store to messages and rounds tables (for compatibility)
            await self._store_to_messages_rounds_tables(all_messages, session_id, round_id)
            logger.info(f"✅ Stored {len(all_messages)} messages to messages/rounds tables")

            # Step 3: Store M0 messages
            message_ids = await self._store_m0_messages(all_messages)
            logger.info(f"✅ Stored {len(message_ids)} M0 messages")

            # Step 4: Create and store M1 chunks
            chunks = self.chunk_processor.create_chunks(all_messages)
            chunk_ids = await self._store_m1_chunks(chunks)
            logger.info(f"✅ Stored {len(chunk_ids)} M1 chunks")

            response = self._success_response(
                {"message_ids": message_ids, "chunk_count": len(chunks)},
                f"Processed {len(all_messages)} messages into {len(chunks)} chunks"
            )
            logger.debug(f"SimplifiedMemoryService: Returning response: {response}")
            return response

        except Exception as e:
            logger.error(f"SimplifiedMemoryService: Error in add_batch: {e}")
            return self._error_response(f"Error processing message batch: {str(e)}")

    async def _store_m0_messages(self, messages: List[Dict[str, Any]]) -> List[str]:
        """Store M0 messages to database."""
        message_ids = []

        try:
            with self.db_manager.conn.cursor() as cur:
                insert_query = """
                    INSERT INTO m0_raw
                    (message_id, content, role, conversation_id, sequence_number, token_count, created_at, processing_status)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING message_id
                """

                for i, message in enumerate(messages):
                    # Generate UUID if message_id is not a valid UUID
                    message_id = message.get('id', str(uuid.uuid4()))
                    try:
                        # Validate UUID format
                        uuid.UUID(message_id)
                    except ValueError:
                        # If not a valid UUID, generate a new one
                        message_id = str(uuid.uuid4())

                    content = message.get('content', '')
                    role = message.get('role', 'user')

                    # Extract or generate conversation_id
                    conversation_id = message.get('metadata', {}).get('conversation_id')
                    if not conversation_id:
                        conversation_id = str(uuid.uuid4())
                    else:
                        # Validate conversation_id UUID format
                        try:
                            uuid.UUID(conversation_id)
                        except ValueError:
                            conversation_id = str(uuid.uuid4())

                    # Estimate token count
                    token_count = max(1, len(content) // 4)

                    # Use existing created_at or current time
                    created_at = message.get('created_at', datetime.now())
                    if isinstance(created_at, (int, float)):
                        created_at = datetime.fromtimestamp(created_at)

                    cur.execute(insert_query, (
                        message_id,
                        content,
                        role,
                        conversation_id,
                        i + 1,  # sequence_number
                        token_count,
                        created_at,
                        'pending'
                    ))

                    result = cur.fetchone()
                    if result:
                        message_ids.append(result[0])
                    else:
                        message_ids.append(message_id)

        except Exception as e:
            logger.error(f"Error storing M0 messages: {e}")
            raise

        return message_ids

    async def _prepare_session_and_round(self, message_batch_list: MessageBatchList) -> tuple[str, str]:
        """Prepare session and round IDs from message batch list."""
        import uuid

        # Extract session_id from first message if available
        session_id = None
        for message_list in message_batch_list:
            for message in message_list:
                metadata = message.get('metadata', {})
                if 'session_id' in metadata:
                    session_id = metadata['session_id']
                    break
                # Also check conversation_id as fallback
                if 'conversation_id' in metadata:
                    session_id = metadata['conversation_id']
                    break
            if session_id:
                break

        # Generate session_id if not found
        if not session_id:
            session_id = str(uuid.uuid4())

        # Generate round_id
        round_id = str(uuid.uuid4())

        return session_id, round_id

    async def _store_to_messages_rounds_tables(self, messages: List[Dict[str, Any]], session_id: str, round_id: str) -> None:
        """Store messages to messages and rounds tables for compatibility."""
        try:
            # Create the round first
            with self.db_manager.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO rounds (id, session_id, created_at, updated_at)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (id) DO NOTHING
                """, (round_id, session_id, datetime.now(), datetime.now()))

            # Store messages to messages table
            with self.db_manager.conn.cursor() as cur:
                insert_query = """
                    INSERT INTO messages (id, round_id, role, content, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO NOTHING
                """

                for message in messages:
                    message_id = message.get('id') or message.get('message_id') or str(uuid.uuid4())
                    role = message.get('role', 'user')
                    content = message.get('content', '')
                    created_at = message.get('created_at', datetime.now())
                    if isinstance(created_at, (int, float)):
                        created_at = datetime.fromtimestamp(created_at)

                    cur.execute(insert_query, (
                        message_id,
                        round_id,
                        role,
                        content,
                        created_at,
                        datetime.now()
                    ))

            self.db_manager.conn.commit()

        except Exception as e:
            logger.error(f"Error storing to messages/rounds tables: {e}")
            self.db_manager.conn.rollback()
            raise

    async def _store_m1_chunks(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """Store M1 chunks with embeddings to database using connection pool."""
        chunk_ids = []

        try:
            # Use connection pool for better concurrency and connection management
            with sync_connection_pool.get_connection() as conn:
                # Process chunks in small batches to reduce lock contention
                batch_size = 3  # Very small batches to minimize lock time

                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i:i + batch_size]

                    with conn.cursor() as cur:
                        insert_query = """
                            INSERT INTO m1_episodic
                            (chunk_id, content, chunking_strategy, token_count, embedding,
                             m0_message_ids, conversation_id, created_at, embedding_generated_at)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                            RETURNING chunk_id
                        """

                        for chunk in batch:
                            # Generate embedding
                            embedding = self.embedding_generator.generate_embedding(chunk['content'])

                            # Format m0_message_ids as PostgreSQL UUID array
                            m0_ids_array = '{' + ','.join(chunk['m0_message_ids']) + '}'

                            cur.execute(insert_query, (
                                chunk['chunk_id'],
                                chunk['content'],
                                chunk['chunking_strategy'],
                                chunk['token_count'],
                                embedding.tolist(),  # Convert numpy array to list
                                m0_ids_array,
                                chunk['conversation_id'],
                                chunk['created_at'],
                                datetime.now()  # embedding_generated_at
                            ))

                            result = cur.fetchone()
                            if result:
                                chunk_ids.append(result[0])
                            else:
                                chunk_ids.append(chunk['chunk_id'])

                    # Commit each small batch immediately
                    conn.commit()

        except Exception as e:
            logger.error(f"Error storing M1 chunks: {e}")
            raise

        return chunk_ids

    async def query_similar_chunks(
        self,
        query_text: str,
        top_k: int = 10,
        similarity_threshold: float = 0.1
    ) -> List[Dict[str, Any]]:
        """Query similar chunks using vector similarity search."""
        try:
            # Generate query embedding
            query_embedding = self.embedding_generator.generate_embedding(query_text)

            # Execute similarity search
            with self.db_manager.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM search_similar_chunks(%s::vector, %s, %s)
                """, (query_embedding.tolist(), similarity_threshold, top_k))

                results = []
                for row in cur.fetchall():
                    # Convert to QueryBuffer-compatible format
                    result = {
                        'id': str(row['chunk_id']),
                        'content': row['content'],
                        'score': row['similarity_score'],  # Already normalized 0-1
                        'distance': row['distance'],
                        'created_at': row['created_at'].isoformat() if row['created_at'] else None,
                        'metadata': {
                            'source': 'memory_database',
                            'chunking_strategy': row['chunking_strategy'],
                            'm0_message_count': row['m0_message_count'],
                            'type': 'chunk'
                        }
                    }
                    results.append(result)

                logger.info(f"✅ Vector search returned {len(results)} results for query: '{query_text[:50]}...'")
                return results

        except Exception as e:
            logger.error(f"Error in vector similarity search: {e}")
            return []

    def _success_response(self, data: Any, message: str) -> Dict[str, Any]:
        """Create a success response compatible with BufferService expectations."""
        return {
            "status": "success",
            "code": 200,
            "data": data,
            "message": message,
            "errors": None
        }

    def _error_response(self, message: str, code: int = 500) -> Dict[str, Any]:
        """Create an error response."""
        return {
            "status": "error",
            "code": code,
            "data": None,
            "message": message,
            "errors": [{"field": "general", "message": message}]
        }

    # Additional interface methods for compatibility

    async def read(self, message_ids: List[str]) -> Dict[str, Any]:
        """Read messages from memory."""
        try:
            messages = []
            not_found_ids = []

            with self.db_manager.conn.cursor(cursor_factory=RealDictCursor) as cur:
                for message_id in message_ids:
                    cur.execute("SELECT * FROM m0_raw WHERE message_id = %s", (message_id,))
                    row = cur.fetchone()

                    if row:
                        messages.append({
                            "id": str(row["message_id"]),
                            "role": row["role"],
                            "content": row["content"],
                            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                        })
                    else:
                        not_found_ids.append(message_id)

            if not_found_ids:
                return self._error_response(
                    f"Some message IDs were not found: {', '.join(not_found_ids)}",
                    404
                )

            return self._success_response({"messages": messages}, f"Read {len(messages)} messages")

        except Exception as e:
            logger.error(f"Error reading messages: {e}")
            return self._error_response(f"Error reading messages: {str(e)}")

    async def get_messages_by_session(
        self,
        session_id: str,
        limit: Optional[int] = None,
        sort_by: str = 'timestamp',
        order: str = 'desc',
        buffer_only: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """Get messages for a session."""
        try:
            with self.db_manager.conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Build query
                query = "SELECT * FROM m0_raw WHERE conversation_id = %s"
                params = [session_id]

                # Add ordering
                if sort_by == 'timestamp':
                    query += f" ORDER BY created_at {order.upper()}"
                else:
                    query += f" ORDER BY sequence_number {order.upper()}"

                # Add limit
                if limit:
                    query += " LIMIT %s"
                    params.append(limit)

                cur.execute(query, params)
                rows = cur.fetchall()

                messages = []
                for row in rows:
                    messages.append({
                        "id": str(row["message_id"]),
                        "role": row["role"],
                        "content": row["content"],
                        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                        "metadata": {
                            "conversation_id": str(row["conversation_id"]),
                            "sequence_number": row["sequence_number"],
                            "token_count": row["token_count"]
                        }
                    })

                logger.info(f"Retrieved {len(messages)} messages for session {session_id}")
                return messages

        except Exception as e:
            logger.error(f"Error getting messages by session: {e}")
            return []

    async def query(
        self,
        query_text: str,
        top_k: int = 10,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Query interface for compatibility with existing code."""
        return await self.query_similar_chunks(query_text, top_k)

    async def close(self):
        """Close database connections."""
        if self.db_manager:
            self.db_manager.close()

        # Close sync connection pool if it exists
        try:
            from .sync_connection_pool import sync_connection_pool
            if sync_connection_pool._initialized:
                sync_connection_pool.close()
        except Exception as e:
            logger.debug(f"Error closing sync connection pool: {e}")

        return None  # Ensure we return something for await
