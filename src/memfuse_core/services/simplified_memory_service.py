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
import json
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
                    WHERE table_schema = 'public' AND table_name IN ('users', 'sessions', 'rounds', 'messages', 'm0_raw', 'm1_episodic')
                """)
                existing_tables = [row[0] for row in cur.fetchall()]

                # Create missing basic tables
                missing_tables = []
                required_tables = ['users', 'sessions', 'rounds', 'messages', 'm0_raw', 'm1_episodic']
                for table in required_tables:
                    if table not in existing_tables:
                        missing_tables.append(table)

                if missing_tables:
                    logger.warning(f"⚠️ Missing tables: {missing_tables}. Creating them now...")
                    await self._create_missing_tables(missing_tables)

                # Verify required functions exist
                await self._verify_functions()

                logger.info("✅ Database schema verification complete")

        except Exception as e:
            logger.error(f"❌ Schema initialization failed: {e}")
            raise

    async def _create_missing_tables(self, missing_tables: List[str]) -> None:
        """Create missing database tables."""
        try:
            # Serialize DDL with an advisory transaction lock to avoid races
            original_autocommit = self.conn.autocommit
            self.conn.autocommit = False
            with self.conn.cursor() as cur:
                # Acquire transaction-scoped advisory lock
                cur.execute("SELECT pg_advisory_xact_lock(448820728)")

                # Create users table
                if 'users' in missing_tables:
                    cur.execute('''
                    CREATE TABLE IF NOT EXISTS users (
                        id TEXT PRIMARY KEY,
                        name TEXT UNIQUE NOT NULL,
                        description TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP
                    )
                    ''')
                    logger.info("✅ Created users table")

                # Create sessions table
                if 'sessions' in missing_tables:
                    cur.execute('''
                    CREATE TABLE IF NOT EXISTS sessions (
                        id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        agent_id TEXT NOT NULL DEFAULT 'default-agent',
                        name TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                    )
                    ''')
                    logger.info("✅ Created sessions table")

                # Create rounds table
                if 'rounds' in missing_tables:
                    cur.execute('''
                    CREATE TABLE IF NOT EXISTS rounds (
                        id TEXT PRIMARY KEY,
                        session_id TEXT,
                        created_at TIMESTAMP,
                        updated_at TIMESTAMP,
                        FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE
                    )
                    ''')
                    logger.info("✅ Created rounds table")

                # Create messages table
                if 'messages' in missing_tables:
                    cur.execute('''
                    CREATE TABLE IF NOT EXISTS messages (
                        id TEXT PRIMARY KEY,
                        round_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP,
                        FOREIGN KEY (round_id) REFERENCES rounds (id) ON DELETE CASCADE
                    )
                    ''')
                    logger.info("✅ Created messages table")

                # Create M0 and M1 tables using SchemaManager
                if 'm0_raw' in missing_tables or 'm1_episodic' in missing_tables:
                    from memfuse_core.models.schema.manager import SchemaManager
                    schema_manager = SchemaManager()

                    if 'm0_raw' in missing_tables:
                        m0_schema = schema_manager.get_schema('m0_raw')
                        cur.execute(m0_schema.generate_create_table_sql())
                        logger.info("✅ Created m0_raw table")

                    if 'm1_episodic' in missing_tables:
                        m1_schema = schema_manager.get_schema('m1_episodic')
                        cur.execute(m1_schema.generate_create_table_sql())
                        logger.info("✅ Created m1_episodic table")
            # Commit DDL batch
            self.conn.commit()
            # Restore autocommit
            self.conn.autocommit = original_autocommit

        except Exception as e:
            logger.error(f"❌ Failed to create missing tables: {e}")
            try:
                self.conn.rollback()
            except Exception:
                pass
            # Restore autocommit on failure as well
            try:
                self.conn.autocommit = original_autocommit
            except Exception:
                pass
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


class TokenBasedChunker:
    """
    Intelligent token-based chunker that optimizes for retrieval quality.

    Key features:
    - Target chunk size: 500-800 tokens
    - User boundary awareness
    - Metadata preservation
    - Conversation context preservation
    """

    def __init__(self,
                 target_tokens: int = 700,
                 min_tokens: int = 500,
                 max_tokens: int = 800,
                 strict_user_boundaries: bool = True):
        """
        Initialize the token-based chunker with strict user boundary enforcement.

        Args:
            target_tokens: Target token count per chunk
            min_tokens: Minimum tokens before creating a chunk
            max_tokens: Maximum tokens allowed in a chunk
            strict_user_boundaries: If True, NEVER mix different users in same chunk (security)
        """
        self.target_tokens = target_tokens
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.strict_user_boundaries = strict_user_boundaries
        from ..utils.token_counter import get_token_counter
        self.token_counter = get_token_counter()

        logger.info(f"TokenBasedChunker initialized: target={target_tokens}, "
                   f"range=[{min_tokens}, {max_tokens}], strict_user_boundaries={strict_user_boundaries}")

    def create_chunks(self, messages: List[Dict[str, Any]], session_id: Optional[str] = None, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Create optimally-sized chunks from messages with strict user boundary enforcement.

        Strategy:
        1. Group messages by user_id first (strict security boundary)
        2. Within each user group, optimize for token count (500-800 tokens)
        3. Never mix different users in the same chunk

        Args:
            messages: List of message dictionaries
            session_id: Session ID for the messages
            user_id: Default user ID if not in message metadata

        Returns:
            List of chunk dictionaries with embedded metadata
        """
        if not messages:
            return []

        # Convert messages to metadata format expected by original TokenBasedChunker
        metadata_list = []
        for message in messages:
            meta = {
                'message_id': message.get('id', str(uuid.uuid4())),
                'user_id': user_id,
                'session_id': session_id,
                'conversation_id': session_id  # For compatibility
            }
            # Extract user_id from message metadata if available
            if 'metadata' in message and isinstance(message['metadata'], dict):
                meta['user_id'] = message['metadata'].get('user_id', user_id)
            metadata_list.append(meta)

        # Step 1: Group messages by user_id for strict security
        user_groups = self._group_messages_by_user(messages, metadata_list)

        all_chunks = []

        # Step 2: Process each user group separately
        for group_user_id, (user_messages, user_metadata) in user_groups.items():
            user_chunks = self._create_chunks_for_user(user_messages, user_metadata, group_user_id)
            all_chunks.extend(user_chunks)

        logger.info(f"TokenBasedChunker: Created {len(all_chunks)} chunks from {len(messages)} messages "
                   f"across {len(user_groups)} users")
        return all_chunks

    def _group_messages_by_user(self, messages: List[Dict[str, Any]],
                               metadata_list: List[Dict[str, Any]]) -> Dict[str, tuple]:
        """
        Group messages by user_id for strict security boundaries.

        Args:
            messages: List of message dictionaries
            metadata_list: List of metadata for each message

        Returns:
            Dictionary mapping user_id to (messages, metadata) tuple
        """
        user_groups = {}

        for message, meta in zip(messages, metadata_list):
            user_id = meta.get('user_id', 'unknown_user')

            if user_id not in user_groups:
                user_groups[user_id] = ([], [])

            user_groups[user_id][0].append(message)
            user_groups[user_id][1].append(meta)

        return user_groups

    def _create_chunks_for_user(self, messages: List[Dict[str, Any]],
                               metadata_list: List[Dict[str, Any]],
                               user_id: str) -> List[Dict[str, Any]]:
        """
        Create optimally-sized chunks for a single user's messages.

        Args:
            messages: List of message dictionaries for this user
            metadata_list: List of metadata for each message
            user_id: User ID for these messages

        Returns:
            List of chunk dictionaries for this user
        """
        chunks = []
        current_chunk_messages = []
        current_chunk_metadata = []
        current_tokens = 0

        for message, meta in zip(messages, metadata_list):
            content = message.get('content', '')
            message_tokens = self.token_counter.count_tokens(content)

            # Check if adding this message would exceed max tokens
            if current_tokens + message_tokens > self.max_tokens and current_chunk_messages:
                # Create chunk from current messages
                chunk = self._create_chunk_from_messages(
                    current_chunk_messages, current_chunk_metadata, current_tokens, user_id
                )
                chunks.append(chunk)

                # Start new chunk
                current_chunk_messages = [message]
                current_chunk_metadata = [meta]
                current_tokens = message_tokens
            else:
                # Add to current chunk
                current_chunk_messages.append(message)
                current_chunk_metadata.append(meta)
                current_tokens += message_tokens

        # Handle remaining messages
        if current_chunk_messages:
            chunk = self._create_chunk_from_messages(
                current_chunk_messages, current_chunk_metadata, current_tokens, user_id
            )
            chunks.append(chunk)

        logger.debug(f"TokenBasedChunker: Created {len(chunks)} chunks for user {user_id}")
        return chunks

    def _create_chunk_from_messages(self, messages: List[Dict[str, Any]],
                                   metadata_list: List[Dict[str, Any]],
                                   token_count: int, user_id: str = None) -> Dict[str, Any]:
        """
        Create a chunk dictionary from messages and metadata.

        Args:
            messages: List of messages in this chunk
            metadata_list: List of metadata for each message
            token_count: Total token count for this chunk
            user_id: User ID for this chunk

        Returns:
            Chunk dictionary ready for database insertion
        """
        # Combine message contents
        combined_content = self._format_chunk_content(messages)

        # Extract user_ids from Buffer data first, fallback to conversion only if needed
        import uuid

        user_ids = []

        # Priority 1: Extract user_ids from Buffer metadata (these are already correct UUIDs)
        for meta in metadata_list:
            buffer_user_id = meta.get('user_id')
            if buffer_user_id:
                try:
                    # Buffer should provide proper UUIDs from users table
                    uuid.UUID(buffer_user_id)
                    user_ids.append(buffer_user_id)
                    logger.debug(f"Using user_id from Buffer metadata: {buffer_user_id}")
                except (ValueError, TypeError):
                    logger.warning(f"Invalid user_id format in Buffer metadata: {buffer_user_id}")

        # Priority 2: Use provided user_id parameter if no Buffer data found
        if not user_ids and user_id:
            try:
                # Check if it's already a valid UUID
                uuid.UUID(user_id)
                user_ids = [user_id]
                logger.debug(f"Using provided user_id (UUID): {user_id}")
            except (ValueError, TypeError):
                # Only convert short names when Buffer is disabled
                if isinstance(user_id, str) and len(user_id) < 36:
                    # This should only happen when Buffer is disabled
                    user_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, user_id))
                    user_ids = [user_uuid]
                    logger.debug(f"Converted short user_id to UUID (Buffer disabled): {user_id} -> {user_uuid}")
                else:
                    logger.warning(f"Invalid user_id format: {user_id}")
                    user_ids = [str(uuid.uuid4())]

        # Remove duplicates while preserving order
        user_ids = list(dict.fromkeys(user_ids))

        # Collect all message roles
        roles = [msg.get('role', 'unknown') for msg in messages]

        # Get session info (unified from conversation_id)
        session_ids = list(set(
            meta.get('session_id', meta.get('conversation_id')) for meta in metadata_list
            if meta.get('session_id') or meta.get('conversation_id')
        ))

        # Create comprehensive metadata
        chunk_metadata = {
            'user_ids': user_ids,
            'session_ids': session_ids,
            'source_message_roles': roles,
            'source_message_count': len(messages),
            'chunking_method': 'token_based_user_safe',
            'token_target': self.target_tokens,
            'actual_tokens': token_count,
            'multi_user': False,  # Always False with strict boundaries
            'strict_user_boundary': True  # Security flag
        }

        return {
            'chunk_id': str(uuid.uuid4()),
            'content': combined_content,
            'chunking_strategy': 'token_based',
            'token_count': token_count,
            'user_id': user_ids[0] if user_ids else str(uuid.uuid4()),  # Use first user_id
            'session_id': session_ids[0] if session_ids else None,  # Unified to session_id
            'metadata': chunk_metadata,  # Use correct field name
            'm0_raw_ids': [
                meta.get('message_id') for meta in metadata_list
                if meta.get('message_id')
            ]
        }

    def _format_chunk_content(self, messages: List[Dict[str, Any]]) -> str:
        """
        Format messages into a coherent chunk content.

        Args:
            messages: List of messages to format

        Returns:
            Formatted chunk content string
        """
        formatted_parts = []

        for msg in messages:
            role = msg.get('role', 'unknown').upper()
            content = msg.get('content', '').strip()

            if content:
                formatted_parts.append(f"[{role}]: {content}")

        return "\n\n".join(formatted_parts)


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
        self.chunk_processor = TokenBasedChunker(
            target_tokens=self.config.get('chunk_token_limit', 700),
            min_tokens=self.config.get('min_chunk_tokens', 500),
            max_tokens=self.config.get('max_chunk_tokens', 800),
            strict_user_boundaries=True
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

        # 🔧 CRITICAL FIX: Initialize _user_id from users table
        # This ensures we have the correct UUID for the user
        await self._initialize_user_id()

        # Initialize embedding generator
        await self.embedding_generator.initialize()

        # Add compatibility attribute for BufferService
        self.multi_path_retrieval = self  # Point to self for compatibility

        self._initialized = True
        logger.info("SimplifiedMemoryService: Initialization complete")
        return self

    async def _initialize_user_id(self):
        """Initialize _user_id from users table to ensure ID consistency."""
        try:
            # Use the context manager to get database connection
            with sync_connection_pool.get_connection() as conn:
                cur = conn.cursor()

                # First, try to find existing user
                cur.execute("SELECT id FROM users WHERE name = %s", (self.user,))
                result = cur.fetchone()

                if result:
                    self._user_id = result[0]
                    logger.info(f"✅ Found existing user '{self.user}' with ID: {self._user_id}")
                else:
                    # Create new user if not exists
                    import uuid
                    new_user_id = str(uuid.uuid4())
                    cur.execute(
                        "INSERT INTO users (id, name, created_at) VALUES (%s, %s, NOW()) ON CONFLICT (name) DO NOTHING RETURNING id",
                        (new_user_id, self.user)
                    )
                    conn.commit()

                    # Get the actual ID (in case of race condition)
                    cur.execute("SELECT id FROM users WHERE name = %s", (self.user,))
                    result = cur.fetchone()
                    self._user_id = result[0] if result else new_user_id
                    logger.info(f"✅ Created new user '{self.user}' with ID: {self._user_id}")

        except Exception as e:
            logger.error(f"❌ Failed to initialize user_id for '{self.user}': {e}")
            # Fallback to generating a UUID (should not happen in normal operation)
            import uuid
            self._user_id = str(uuid.uuid4())
            logger.warning(f"⚠️ Using fallback user_id: {self._user_id}")
    
    async def add_batch(self, message_batch_list: MessageBatchList, **kwargs) -> Dict[str, Any]:
        """Add a batch of message lists with simplified M0/M1 processing."""
        try:
            if not message_batch_list:
                return self._success_response([], "No message lists to process")

            logger.info(f"SimplifiedMemoryService: Processing {len(message_batch_list)} message lists")

            # Extract session_id from kwargs (passed from API)
            provided_session_id = kwargs.get('session_id')

            # Flatten message batch list
            all_messages = []
            for message_list in message_batch_list:
                all_messages.extend(message_list)

            if not all_messages:
                return self._success_response([], "No messages to process")

            # Step 1: Create session and round (like traditional MemoryService)
            session_id, round_id = await self._prepare_session_and_round(message_batch_list, provided_session_id)
            logger.info(f"SimplifiedMemoryService: Prepared session_id={session_id}, round_id={round_id}")

            # Step 2: Store to messages and rounds tables (for compatibility)
            await self._store_to_messages_rounds_tables(all_messages, session_id, round_id)
            logger.info(f"✅ Stored {len(all_messages)} messages to messages/rounds tables")

            # Step 3: Store M0 messages with session_id, user_id, and round_id
            # Priority 1: Extract user_id from Buffer data (message metadata)
            user_id = None

            for message in all_messages:
                metadata = message.get('metadata', {})
                if 'user_id' in metadata:
                    buffer_user_id = metadata['user_id']
                    try:
                        # Validate it's a proper UUID from Buffer
                        uuid.UUID(buffer_user_id)
                        user_id = buffer_user_id
                        logger.debug(f"Using valid UUID user_id from Buffer: {user_id}")
                        break
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid user_id in Buffer metadata: {buffer_user_id}")

            # Priority 2: Use provided user_id from kwargs
            if not user_id:
                kwargs_user_id = kwargs.get('user_id')
                if kwargs_user_id:
                    try:
                        uuid.UUID(kwargs_user_id)
                        user_id = kwargs_user_id
                        logger.debug(f"Using user_id from kwargs: {user_id}")
                    except (ValueError, TypeError):
                        # Convert short user name to UUID (Buffer disabled case)
                        if isinstance(kwargs_user_id, str) and len(kwargs_user_id) < 36:
                            user_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, kwargs_user_id))
                            logger.debug(f"Converted kwargs user_id to UUID: {kwargs_user_id} -> {user_id}")

            # Priority 3: Use service user_id (from users table lookup)
            if not user_id and hasattr(self, '_user_id') and self._user_id:
                user_id = str(self._user_id)
                logger.debug(f"Using service _user_id: {user_id}")

            # Priority 4: Generate user_id only as last resort
            if not user_id:
                user_id = str(uuid.uuid4())
                logger.warning(f"Generated new user_id (last resort - Buffer disabled and no user data): {user_id}")

            # 🔧 CRITICAL FIX: Always use the correct user_id from _user_id if available
            # This ensures ID consistency between Buffer and Memory layers
            if hasattr(self, '_user_id') and self._user_id:
                correct_user_id = str(self._user_id)
                if user_id != correct_user_id:
                    logger.warning(f"FIXING ID MISMATCH: Changing user_id from {user_id} to {correct_user_id}")
                    user_id = correct_user_id

            message_ids = await self._store_m0_messages(all_messages, session_id, user_id, round_id)
            logger.info(f"✅ Stored {len(message_ids)} M0 messages")

            # Step 4: Create and store M1 chunks with session_id and user_id
            chunks = self.chunk_processor.create_chunks(all_messages, session_id, user_id)
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

    async def _store_m0_messages(self, messages: List[Dict[str, Any]], session_id: str, user_id: str = None, round_id: str = None) -> List[str]:
        """Store M0 messages to database with session_id."""
        message_ids = []

        # user_id should already be provided from Buffer data or parent method
        # Only generate as absolute fallback (should rarely happen)
        if user_id is None:
            user_id = str(uuid.uuid4())
            logger.warning("Generated fallback user_id - this should rarely happen if Buffer is working correctly")

        try:
            with self.db_manager.conn.cursor() as cur:
                insert_query = """
                    INSERT INTO m0_raw
                    (message_id, content, role, user_id, session_id, round_id, sequence_number, token_count, created_at, processing_status)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (session_id, sequence_number) DO UPDATE SET
                        message_id = EXCLUDED.message_id,
                        content = EXCLUDED.content,
                        role = EXCLUDED.role,
                        user_id = EXCLUDED.user_id,
                        round_id = EXCLUDED.round_id,
                        token_count = EXCLUDED.token_count,
                        created_at = EXCLUDED.created_at,
                        processing_status = EXCLUDED.processing_status
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
                        user_id,
                        session_id,
                        round_id,
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

    async def _prepare_session_and_round(self, message_batch_list: MessageBatchList, provided_session_id: Optional[str] = None) -> tuple[str, str]:
        """Prepare session and round IDs from message batch list, prioritizing Buffer data."""
        import uuid

        # Priority 1: Use provided session_id (from API parameter)
        session_id = provided_session_id
        round_id = None

        # Priority 2: Extract session_id and round_id from Buffer data (message metadata)
        if not session_id:
            for message_list in message_batch_list:
                for message in message_list:
                    metadata = message.get('metadata', {})

                    # Try to get session_id from Buffer metadata
                    if not session_id and 'session_id' in metadata:
                        session_id = metadata['session_id']

                    # Try to get round_id from Buffer metadata
                    if not round_id and 'round_id' in metadata:
                        round_id = metadata['round_id']

                    # Also check for legacy conversation_id as session_id fallback
                    if not session_id and 'conversation_id' in metadata:
                        session_id = metadata['conversation_id']

                    # Break if we found both
                    if session_id and round_id:
                        break

                if session_id and round_id:
                    break

        # Priority 3: Generate IDs only if Buffer is disabled or no data found
        if not session_id:
            session_id = str(uuid.uuid4())
            logger.debug("Generated new session_id (Buffer disabled or no Buffer data)")
        else:
            logger.debug(f"Using session_id from Buffer: {session_id}")

        if not round_id:
            round_id = str(uuid.uuid4())
            logger.debug("Generated new round_id (Buffer disabled or no Buffer data)")
        else:
            logger.debug(f"Using round_id from Buffer: {round_id}")

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
                             m0_raw_ids, user_id, session_id, created_at, embedding_generated_at, metadata)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (chunk_id) DO UPDATE SET
                                content = EXCLUDED.content,
                                chunking_strategy = EXCLUDED.chunking_strategy,
                                token_count = EXCLUDED.token_count,
                                embedding = EXCLUDED.embedding,
                                m0_raw_ids = EXCLUDED.m0_raw_ids,
                                user_id = EXCLUDED.user_id,
                                session_id = EXCLUDED.session_id,
                                created_at = EXCLUDED.created_at,
                                embedding_generated_at = EXCLUDED.embedding_generated_at,
                                metadata = EXCLUDED.metadata
                            RETURNING chunk_id
                        """

                        for chunk in batch:
                            # Generate embedding
                            embedding = self.embedding_generator.generate_embedding(chunk['content'])

                            # Format m0_raw_ids as PostgreSQL UUID array - fix field name
                            m0_ids = chunk.get('m0_message_ids', chunk.get('m0_raw_ids', []))
                            m0_ids_array = '{' + ','.join(str(id) for id in m0_ids if id) + '}' if m0_ids else '{}'

                            # Convert metadata dict to JSON string
                            metadata = chunk.get('metadata', {})
                            metadata_json = json.dumps(metadata) if metadata else '{}'

                            cur.execute(insert_query, (
                                chunk['chunk_id'],
                                chunk['content'],
                                chunk['chunking_strategy'],
                                chunk['token_count'],
                                embedding.tolist(),  # Convert numpy array to list
                                m0_ids_array,
                                chunk.get('user_id', str(uuid.uuid4())),  # user_id
                                chunk['session_id'],
                                chunk.get('created_at', datetime.now()),  # created_at
                                datetime.now(),  # embedding_generated_at
                                metadata_json  # metadata as JSON string
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
        similarity_threshold: float = 0.0,  # Lower threshold to get more results
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Query similar chunks using vector similarity search."""
        try:
            # Generate query embedding
            query_embedding = self.embedding_generator.generate_embedding(query_text)

            # Apply user filtering if user_id is provided
            if user_id:
                # Use user-filtered search - NO similarity threshold, use top_k only
                logger.debug(f"SimplifiedMemoryService: Querying with user_id filter: {user_id}")
                with self.db_manager.conn.cursor(cursor_factory=RealDictCursor) as cur:
                    # Check if user_id is a UUID (direct user_id) or a name (needs lookup)
                    try:
                        import uuid
                        uuid.UUID(user_id)
                        # It's a UUID, query directly by user_id
                        cur.execute("""
                            SELECT
                                c.chunk_id,
                                c.content,
                                (1.0 - (c.embedding <=> %s::vector) / 2.0) as similarity_score,
                                (c.embedding <=> %s::vector) as distance,
                                array_length(c.m0_raw_ids, 1) as m0_message_count,
                                c.chunking_strategy,
                                c.user_id,
                                c.created_at
                            FROM m1_episodic c
                            WHERE c.user_id = %s
                            ORDER BY c.embedding <=> %s::vector ASC
                            LIMIT %s
                        """, (query_embedding.tolist(), query_embedding.tolist(), user_id,
                              query_embedding.tolist(), top_k))
                    except ValueError:
                        # It's a name, query by joining with users table
                        cur.execute("""
                            SELECT
                                c.chunk_id,
                                c.content,
                                (1.0 - (c.embedding <=> %s::vector) / 2.0) as similarity_score,
                                (c.embedding <=> %s::vector) as distance,
                                array_length(c.m0_raw_ids, 1) as m0_message_count,
                                c.chunking_strategy,
                                c.user_id,
                                c.created_at
                            FROM m1_episodic c
                            JOIN sessions s ON c.session_id::text = s.id
                            JOIN users u ON s.user_id = u.id
                            WHERE u.name = %s
                            ORDER BY c.embedding <=> %s::vector ASC
                            LIMIT %s
                        """, (query_embedding.tolist(), query_embedding.tolist(), user_id,
                              query_embedding.tolist(), top_k))
                    rows = cur.fetchall()
            else:
                # No user filtering (fallback) - NO similarity threshold
                logger.warning("SimplifiedMemoryService: No user_id provided, querying all data (potential security issue)")
                with self.db_manager.conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT
                            chunk_id,
                            content,
                            (1.0 - (embedding <=> %s::vector) / 2.0) as similarity_score,
                            (embedding <=> %s::vector) as distance,
                            array_length(m0_raw_ids, 1) as m0_message_count,
                            chunking_strategy,
                            user_id,
                            created_at
                        FROM m1_episodic
                        ORDER BY embedding <=> %s::vector ASC
                        LIMIT %s
                    """, (query_embedding.tolist(), query_embedding.tolist(),
                          query_embedding.tolist(), top_k))
                    rows = cur.fetchall()

            results = []
            for row in rows:
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
                query = "SELECT * FROM m0_raw WHERE session_id = %s"
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
                            "session_id": str(row["session_id"]),
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
        query: Optional[str] = None,
        query_text: Optional[str] = None,
        top_k: int = 10,
        store_type: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        include_messages: bool = True,
        include_knowledge: bool = True,
        include_chunks: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Query interface for compatibility with existing code.

        Args:
            query: Query string (BufferService compatibility)
            query_text: Query string (alternative parameter name)
            top_k: Maximum number of results to return
            store_type: Type of store to query (ignored in current implementation)
            session_id: Session ID to filter results (optional)
            include_messages: Whether to include messages in results
            include_knowledge: Whether to include knowledge in results
            include_chunks: Whether to include chunks in results
            **kwargs: Additional parameters

        Returns:
            Dictionary with status, code, and query results (BufferService compatible format)
        """
        # Handle parameter compatibility - accept both 'query' and 'query_text'
        actual_query = query or query_text
        if not actual_query:
            return self._error_response("Query text is required", 400)

        try:
            # Increase search scope to get more diverse results
            # This helps when the correct answer might not be in the top few results
            search_top_k = max(top_k * 3, 15)  # Search more broadly, then filter

            # Get raw results from similarity search with user filtering
            raw_results = await self.query_similar_chunks(
                actual_query,
                search_top_k,
                user_id=user_id,
                session_id=session_id
            )

            # Take the requested top_k from the broader search
            results = raw_results[:top_k]

            # Format response to match BufferService expectations
            response = {
                "status": "success",
                "code": 200,
                "data": {
                    "results": results,
                    "total": len(results)
                },
                "message": f"Retrieved {len(results)} results from memory database (searched {len(raw_results)} candidates)",
                "errors": None
            }

            logger.info(f"SimplifiedMemoryService.query: Returning {len(results)} results for query: '{actual_query[:50]}...' (searched {search_top_k} candidates)")
            return response

        except Exception as e:
            logger.error(f"SimplifiedMemoryService.query: Error: {e}")
            return self._error_response(f"Query failed: {str(e)}")

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
