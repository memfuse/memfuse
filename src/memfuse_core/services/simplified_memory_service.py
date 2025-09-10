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
from typing import Dict, List, Any, Optional, Union
from loguru import logger

import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
from sentence_transformers import SentenceTransformer

from ..interfaces import MessageInterface
from ..interfaces.message_interface import MessageBatchList
from ..models.core import M2Status, Chunk, Fact, M2FactStatus
from .sync_connection_pool import sync_connection_pool
from ..llm.base import LLMProviderError
from ..models.m2_extraction import FactExtractionResponse, ExtractedFact

# Import TYPE_CHECKING to avoid circular imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..llm.base import LLMRequest

# Expose PromptManager at module level for test patching and safe usage
try:
    from ..prompts.prompt_manager import PromptManager  # type: ignore
except Exception:  # pragma: no cover - fallback for environments without prompts
    PromptManager = None  # type: ignore


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
                    WHERE table_schema = 'public' AND table_name IN ('users', 'sessions', 'rounds', 'messages', 'm0_raw', 'm1_episodic', 'm2_semantic')
                """)
                existing_tables = [row[0] for row in cur.fetchall()]

                # Create missing basic tables
                missing_tables = []
                required_tables = ['users', 'sessions', 'rounds', 'messages', 'm0_raw', 'm1_episodic', 'm2_semantic']
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

                # Create M0, M1, and M2 tables using SchemaManager and SQL files
                if 'm0_raw' in missing_tables or 'm1_episodic' in missing_tables or 'm2_semantic' in missing_tables:
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

                    if 'm2_semantic' in missing_tables:
                        # Create M2 table using SQL file (SchemaManager doesn't support M2)
                        await self._create_m2_semantic_table(cur)
                        logger.info("✅ Created m2_semantic table")
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

    async def _create_m2_semantic_table(self, cur) -> None:
        """Create M2 semantic table using SQL schema file."""
        import os
        from pathlib import Path
        
        try:
            # Path to M2 semantic SQL schema file
            schema_file = Path(__file__).parent.parent / "store" / "pgai_store" / "schemas" / "m2_semantic.sql"
            
            if not schema_file.exists():
                logger.error(f"❌ M2 schema file not found: {schema_file}")
                raise FileNotFoundError(f"M2 schema file not found: {schema_file}")
            
            # Read and execute M2 schema SQL
            with open(schema_file, 'r') as f:
                m2_sql = f.read()
            
            logger.debug(f"📝 Loading M2 schema from: {schema_file}")
            cur.execute(m2_sql)
            logger.debug("✅ M2 semantic table schema executed successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to create M2 semantic table: {e}")
            raise

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
        # M2 autonomous background processing state
        self.m2_running: bool = False
        self.m2_processor_task: Optional[asyncio.Task] = None

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

        # Start autonomous M2 background processing if enabled via config
        try:
            m2_enabled = self._resolve_m2_enabled_from_config()
            if m2_enabled:
                await self._start_m2_background_processing()
            else:
                logger.opt(colors=True).info(
                    "<magenta>[M2]</magenta> Background worker disabled by configuration"
                )
        except Exception as e:
            logger.error(f"Failed to evaluate/start M2 background processing: {e}")

        self._initialized = True
        logger.info("SimplifiedMemoryService: Initialization complete")
        return self

    def _resolve_m2_enabled_from_config(self) -> bool:
        """Resolve the m2_enabled flag from multiple config locations.

        Supports both top-level and nested Hydra layouts:
        - cfg["m2_enabled"]
        - cfg["memory_service"]["m2_enabled"]
        - cfg["memory"]["m2_enabled"]
        - cfg["memory"]["memory_service"]["m2_enabled"]
        """
        try:
            cfg = self.config or {}

            # Direct top-level
            v1 = cfg.get("m2_enabled") if hasattr(cfg, 'get') else None
            if isinstance(v1, bool):
                logger.opt(colors=True).debug(
                    "<magenta>[M2]</magenta> Config resolved: top-level m2_enabled=%s",
                    v1,
                )
                return v1

            # Top-level memory_service
            ms = cfg.get("memory_service", {}) if hasattr(cfg, 'get') else {}
            v2 = ms.get("m2_enabled") if hasattr(ms, 'get') else None
            if isinstance(v2, bool):
                logger.opt(colors=True).debug(
                    "<magenta>[M2]</magenta> Config resolved: memory_service.m2_enabled=%s",
                    v2,
                )
                return v2

            # Nested under memory
            memory = cfg.get("memory", {}) if hasattr(cfg, 'get') else {}
            v3 = memory.get("m2_enabled") if hasattr(memory, 'get') else None
            if isinstance(v3, bool):
                logger.opt(colors=True).debug(
                    "<magenta>[M2]</magenta> Config resolved: memory.m2_enabled=%s",
                    v3,
                )
                return v3

            # Nested under memory.memory_service
            memory_ms = memory.get("memory_service", {}) if hasattr(memory, 'get') else {}
            v4 = memory_ms.get("m2_enabled") if hasattr(memory_ms, 'get') else None
            if isinstance(v4, bool):
                logger.opt(colors=True).debug(
                    "<magenta>[M2]</magenta> Config resolved: memory.memory_service.m2_enabled=%s",
                    v4,
                )
                return v4

            # Default false if not specified
            logger.opt(colors=True).debug(
                "<magenta>[M2]</magenta> Config not found; defaulting m2_enabled=False"
            )
            return False
        except Exception as e:
            logger.warning(f"M2 config resolution error: {e}")
            return False

    async def _start_m2_background_processing(self):
        """Start autonomous M2 fact extraction background tasks."""
        if self.m2_running and self.m2_processor_task and not self.m2_processor_task.done():
            logger.debug("M2 autonomous background processing already running")
            return

        self.m2_running = True
        # M2 processor - completely independent of other operations
        self.m2_processor_task = asyncio.create_task(self._m2_autonomous_processor())
        logger.opt(colors=True).info(
            "<magenta>[M2]</magenta> Autonomous background processing started"
        )

    async def _m2_autonomous_processor(self):
        """Main loop that autonomously processes pending M2 chunks in batches."""
        # Configuration knobs with safe defaults
        batch_size: int = int(self.config.get("m2_batch_size", 5))
        interval_secs: float = float(self.config.get("m2_interval_secs", 5.0))

        # For global processing, we do not scope by user when scanning/locking.
        # We'll look up the chunk's user_id per item to save facts with correct ownership.
        logger.opt(colors=True).info(
            "<magenta>[M2]</magenta> Processor configured | batch_size=%s | interval=%ss",
            batch_size,
            interval_secs,
        )

        # One-time visibility probe flag
        probe_logged = False

        try:
            while self.m2_running:
                try:
                    import time
                    batch_started = time.monotonic()

                    # One-time probe: report total pending across DB for visibility
                    if not probe_logged:
                        try:
                            with sync_connection_pool.get_connection() as conn:
                                with conn.cursor() as cur:
                                    cur.execute(
                                        "SELECT COUNT(*) FROM m1_episodic WHERE m2_status = %s",
                                        (M2Status.PENDING.value,),
                                    )
                                    total_pending = cur.fetchone()[0]
                                    logger.opt(colors=True).info(
                                        "<magenta>[M2]</magenta> Probe: total pending chunks in DB = %s",
                                        total_pending,
                                    )
                        except Exception as probe_err:
                            logger.opt(colors=True).warning(
                                "<magenta>[M2]</magenta> Probe failed: %s",
                                probe_err,
                            )
                        finally:
                            probe_logged = True

                    # Fetch pending chunks
                    pending_ids = await self._get_pending_m2_chunks(
                        batch_size=batch_size, user_id=None
                    )

                    if not pending_ids:
                        logger.opt(colors=True).info(
                            "<magenta>[M2]</magenta> No pending chunks. Sleeping %ss",
                            interval_secs,
                        )
                        await asyncio.sleep(interval_secs)
                        continue

                    logger.opt(colors=True).info(
                        "<magenta>[M2]</magenta> Found %s pending chunk(s)",
                        len(pending_ids),
                    )

                    processed = 0
                    succeeded = 0
                    failed = 0

                    for chunk_id in pending_ids:
                        if not self.m2_running:
                            break

                        # Lock chunk for processing
                        locked = await self._lock_chunk_for_m2_processing(
                            chunk_id=chunk_id, user_id=None
                        )
                        if not locked:
                            # Could be racing with another worker; skip
                            short_id = str(chunk_id)[:8]
                            logger.opt(colors=True).info(
                                "<magenta>[M2]</magenta> Skip chunk %s: could not "
                                "acquire lock",
                                short_id,
                            )
                            continue

                        processed += 1
                        short_id = str(chunk_id)[:8]
                        logger.opt(colors=True).info(
                            "<magenta>[M2]</magenta> Locked chunk %s for processing",
                            short_id,
                        )

                        try:
                            # Retrieve chunk to determine its user_id for saving facts
                            chunk_data = await self._get_m1_chunk(chunk_id, user_id=None)
                            chunk_user_id = None
                            if isinstance(chunk_data, dict):
                                chunk_user_id = chunk_data.get("user_id")
                            # Extract structured facts (content + optional confidence)
                            ext_started = time.monotonic()
                            extracted_facts = await self._extract_list_of_structured_facts_from_chunk(
                                chunk_id=chunk_id, user_id=chunk_user_id
                            )
                            ext_duration = time.monotonic() - ext_started
                            logger.opt(colors=True).info(
                                "<magenta>[M2]</magenta> Extracted %s fact strings "
                                "for chunk %s in %.2fs",
                                len(extracted_facts or []),
                                short_id,
                                ext_duration,
                            )

                            # Convert to fact dicts for saving/marking completed
                            fact_dicts: List[Dict[str, Any]] = []
                            for item in extracted_facts or []:
                                # Support both dicts (with confidence) and raw strings
                                if isinstance(item, dict):
                                    raw_text = item.get("text") or item.get("content")
                                    conf = item.get("confidence")
                                else:
                                    raw_text = str(item)
                                    conf = None

                                if not isinstance(raw_text, str):
                                    continue
                                cleaned = raw_text.strip()
                                if not cleaned:
                                    continue

                                fact_entry: Dict[str, Any] = {
                                    "text": cleaned,
                                    "chunk_ids": [chunk_id],
                                    "user_id": chunk_user_id,
                                    "metadata": {
                                        "source": "m2-autonomous-processor",
                                        "strategy": "m2-background",
                                    },
                                }
                                # Only include confidence if LLM provided one; otherwise
                                # allow downstream defaulting/clamping to apply
                                if isinstance(conf, (int, float)):
                                    fact_entry["confidence"] = float(conf)

                                fact_dicts.append(fact_entry)

                            # If nothing extracted, mark failed to avoid infinite retries
                            if not fact_dicts:
                                failed += 1
                                logger.opt(colors=True).warning(
                                    "<magenta>[M2]</magenta> No facts extracted for "
                                    "chunk %s; marking failed",
                                    short_id,
                                )
                                await self._mark_chunk_m2_failed(
                                    chunk_id, "No facts extracted", user_id=user_id
                                )
                                continue

                            # Mark completed (will also save facts via _save_m2_facts)
                            completed = await self._mark_chunk_m2_completed(
                                chunk_id, fact_dicts, user_id=chunk_user_id
                            )
                            if not completed:
                                failed += 1
                                logger.opt(colors=True).error(
                                    "<magenta>[M2]</magenta> Failed to save/complete "
                                    "for chunk %s; marking failed",
                                    short_id,
                                )
                                await self._mark_chunk_m2_failed(
                                    chunk_id, "Failed to complete/save facts", user_id=chunk_user_id
                                )
                            else:
                                succeeded += 1
                                logger.opt(colors=True).info(
                                    "<magenta>[M2]</magenta> Completed chunk %s | "
                                    "facts_saved=%s",
                                    short_id,
                                    len(fact_dicts),
                                )

                        except asyncio.CancelledError:
                            raise
                        except Exception as e:
                            logger.error(
                                f"Error processing chunk {chunk_id} in M2 autonomous processor: {e}"
                            )
                            try:
                                await self._mark_chunk_m2_failed(
                                    chunk_id, f"Processing error: {e}", user_id=user_id
                                )
                            except Exception:
                                # Ensure loop continues even if marking failed
                                pass

                    batch_duration = time.monotonic() - batch_started
                    logger.opt(colors=True).info(
                        "<magenta>[M2]</magenta> Batch summary | scanned=%s | "
                        "processed=%s | succeeded=%s | failed=%s | duration=%.2fs | "
                        "sleep=%ss",
                        len(pending_ids),
                        processed,
                        succeeded,
                        failed,
                        batch_duration,
                        interval_secs,
                    )

                    # Yield control between batches
                    await asyncio.sleep(0)

                except asyncio.CancelledError:
                    break
                except Exception as loop_error:
                    logger.error(f"M2 autonomous processor loop error: {loop_error}")
                    # Backoff before next iteration to avoid tight error loops
                    await asyncio.sleep(interval_secs)
        finally:
            logger.opt(colors=True).info(
                "<magenta>[M2]</magenta> Autonomous background processing stopped"
            )

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
                             m0_raw_ids, user_id, session_id, created_at, embedding_generated_at, m2_status, metadata)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                                m2_status = EXCLUDED.m2_status,
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
                                M2Status.PENDING.value,  # m2_status
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

    async def _get_pending_m2_chunks(self, batch_size: int = 10, user_id: Optional[str] = None) -> List[str]:
        """
        Fetch chunk IDs for M1 chunks with m2_status = 'pending' for M2 processing.
        
        Args:
            batch_size: Maximum number of chunk IDs to return (default 10)
            user_id: DEPRECATED - ignored. Processing now scans all users.
            
        Returns:
            List of chunk ID strings ready for M2 fact extraction
        """
        try:
            results = []
            
            with sync_connection_pool.get_connection() as conn:
                with conn.cursor() as cur:
                    # Global scan (no user filter): process pending chunks across all users
                    cur.execute(
                        """
                        SELECT chunk_id
                        FROM m1_episodic
                        WHERE m2_status = %s
                        ORDER BY created_at ASC
                        LIMIT %s
                        """,
                        (M2Status.PENDING.value, batch_size),
                    )
                    
                    rows = cur.fetchall()
                    
                    for row in rows:
                        results.append(str(row[0]))
            
            logger.opt(colors=True).info(
                "<magenta>[M2]</magenta> Found %s pending chunk ID(s) (batch_size=%s)",
                len(results),
                batch_size,
            )
            if results:
                sample = ", ".join([rid[:8] for rid in results[: min(5, len(results))]])
                logger.opt(colors=True).info(
                    "<magenta>[M2]</magenta> Pending sample (first %s): %s",
                    min(5, len(results)),
                    sample,
                )
            return results
            
        except Exception as e:
            logger.error(f"❌ Error fetching pending M2 chunk IDs: {e}")
            return []

    async def _get_m1_chunk(self, chunk_id: str, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Retrieve a single M1 chunk by its ID from the m1_episodic table.
        
        Args:
            chunk_id: The UUID of the chunk to retrieve
            user_id: Optional user_id filter for security scoping
            
        Returns:
            Chunk dictionary with all fields, or None if not found
        """
        try:
            # Validate chunk_id is a valid UUID format
            import uuid
            try:
                uuid.UUID(chunk_id)
            except (ValueError, TypeError):
                logger.error(f"❌ Invalid chunk_id format: {chunk_id}")
                return None
            
            with sync_connection_pool.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    if user_id:
                        # User-scoped query for security
                        cur.execute("""
                            SELECT 
                                chunk_id,
                                content,
                                user_id,
                                session_id,
                                token_count,
                                created_at,
                                m0_raw_ids,
                                metadata,
                                chunking_strategy,
                                m2_status,
                                m2_processing_started_at,
                                m2_processing_ended_at,
                                embedding_generated_at
                            FROM m1_episodic 
                            WHERE chunk_id = %s AND user_id = %s
                        """, (chunk_id, user_id))
                    else:
                        # No user filtering (fallback)
                        logger.warning("_get_m1_chunk: No user_id provided, querying without user filter")
                        cur.execute("""
                            SELECT 
                                chunk_id,
                                content,
                                user_id,
                                session_id,
                                token_count,
                                created_at,
                                m0_raw_ids,
                                metadata,
                                chunking_strategy,
                                m2_status,
                                m2_processing_started_at,
                                m2_processing_ended_at,
                                embedding_generated_at
                            FROM m1_episodic 
                            WHERE chunk_id = %s
                        """, (chunk_id,))
                    
                    row = cur.fetchone()
                    
                    if not row:
                        logger.debug(f"Chunk not found: {chunk_id}")
                        return None
                    
                    # Convert to dictionary format for M2 processing
                    chunk_data = {
                        'chunk_id': str(row['chunk_id']),
                        'content': row['content'],
                        'user_id': str(row['user_id']),
                        'session_id': str(row['session_id']) if row['session_id'] else None,
                        'token_count': row['token_count'],
                        'created_at': row['created_at'],
                        'm0_raw_ids': list(row['m0_raw_ids']) if row['m0_raw_ids'] else [],
                        'chunking_strategy': row['chunking_strategy'],
                        'metadata': json.loads(row['metadata']) if isinstance(row['metadata'], str) else (row['metadata'] or {}),
                        'm2_status': row['m2_status'],
                        'm2_processing_started_at': row['m2_processing_started_at'],
                        'm2_processing_ended_at': row['m2_processing_ended_at'],
                        'embedding_generated_at': row['embedding_generated_at']
                    }
                    
                    logger.debug(f"✅ Retrieved M1 chunk: {chunk_id}")
                    return chunk_data
            
        except Exception as e:
            logger.error(f"❌ Error retrieving M1 chunk {chunk_id}: {e}")
            return None

    async def _lock_chunk_for_m2_processing(self, chunk_id: str, user_id: Optional[str] = None) -> bool:
        """
        Lock a chunk for M2 processing by updating its status to 'processing'.
        
        This method transitions a chunk from 'pending' to 'processing' status and sets
        the m2_processing_started_at timestamp. It includes safety checks to prevent
        race conditions and double-locking.
        
        Args:
            chunk_id: The UUID of the chunk to lock for processing
            user_id: Optional user_id filter for security scoping
            
        Returns:
            True if chunk was successfully locked, False otherwise
        """
        try:
            # Validate chunk_id is a valid UUID format
            import uuid
            try:
                uuid.UUID(chunk_id)
            except (ValueError, TypeError):
                logger.error(f"❌ Invalid chunk_id format: {chunk_id}")
                return False
            
            with sync_connection_pool.get_connection() as conn:
                with conn.cursor() as cur:
                    # Construct UPDATE query with safety checks
                    if user_id:
                        # User-scoped update for security
                        cur.execute("""
                            UPDATE m1_episodic 
                            SET m2_status = %s, m2_processing_started_at = NOW()
                            WHERE chunk_id = %s AND user_id = %s AND m2_status = %s
                        """, (M2Status.PROCESSING.value, chunk_id, user_id, M2Status.PENDING.value))
                    else:
                        # No user filtering (fallback - should be avoided in production)
                        logger.warning("_lock_chunk_for_m2_processing: No user_id provided, updating without user filter")
                        cur.execute("""
                            UPDATE m1_episodic 
                            SET m2_status = %s, m2_processing_started_at = NOW()
                            WHERE chunk_id = %s AND m2_status = %s
                        """, (M2Status.PROCESSING.value, chunk_id, M2Status.PENDING.value))
                    
                    # Check if any row was updated
                    affected_rows = cur.rowcount
                    conn.commit()
                    
                    if affected_rows > 0:
                        logger.info(f"✅ Successfully locked chunk {chunk_id} for M2 processing (user_id={user_id})")
                        return True
                    else:
                        logger.warning(f"⚠️ Chunk {chunk_id} could not be locked - may already be processing or not found (user_id={user_id})")
                        return False
            
        except Exception as e:
            logger.error(f"❌ Error locking chunk {chunk_id} for M2 processing: {e}")
            return False

    async def _mark_chunk_m2_completed(self, chunk_id: str, facts: List[Dict[str, Any]], user_id: Optional[str] = None) -> bool:
        """
        Mark a chunk as M2 processing completed and store extracted facts.
        
        This method transitions a chunk from 'processing' to 'completed' status,
        sets the m2_processing_ended_at timestamp, and stores the extracted facts
        in the m2_semantic table with proper lineage tracking.
        
        Args:
            chunk_id: The UUID of the chunk to mark as completed
            facts: List of fact dictionaries to store in m2_semantic table
            user_id: Optional user_id filter for security scoping
            
        Returns:
            True if chunk was successfully marked as completed and facts stored, False otherwise
        """
        try:
            # Validate chunk_id is a valid UUID format
            import uuid
            try:
                uuid.UUID(chunk_id)
            except (ValueError, TypeError):
                logger.error(f"❌ Invalid chunk_id format: {chunk_id}")
                return False
            
            # Validate facts structure
            if not isinstance(facts, list):
                logger.error(f"❌ Facts must be a list, got {type(facts)}")
                return False
            
            with sync_connection_pool.get_connection() as conn:
                with conn.cursor() as cur:
                    # Step 1: Update M1 chunk status to completed
                    if user_id:
                        # User-scoped update for security
                        cur.execute("""
                            UPDATE m1_episodic 
                            SET m2_status = %s, m2_processing_ended_at = NOW()
                            WHERE chunk_id = %s AND user_id = %s AND m2_status = %s
                        """, (M2Status.COMPLETED.value, chunk_id, user_id, M2Status.PROCESSING.value))
                    else:
                        # No user filtering (fallback - should be avoided in production)
                        logger.warning("_mark_chunk_m2_completed: No user_id provided, updating without user filter")
                        cur.execute("""
                            UPDATE m1_episodic 
                            SET m2_status = %s, m2_processing_ended_at = NOW()
                            WHERE chunk_id = %s AND m2_status = %s
                        """, (M2Status.COMPLETED.value, chunk_id, M2Status.PROCESSING.value))
                    
                    # Check if chunk was updated
                    affected_rows = cur.rowcount
                    if affected_rows == 0:
                        logger.warning(f"⚠️ Chunk {chunk_id} could not be marked completed - may not be processing or not found (user_id={user_id})")
                        return False
                    
                    # Step 2: Store facts in m2_semantic table using new _save_m2_facts method
                    facts_count = 0  # Initialize facts count
                    if facts:
                        # Ensure each fact has the chunk_id for lineage tracking
                        facts_with_lineage = []
                        for fact in facts:
                            if isinstance(fact, dict):
                                fact_copy = fact.copy()
                                # Add chunk_id to chunk_ids list if not already present
                                chunk_ids = fact_copy.get('chunk_ids', [])
                                if chunk_id not in chunk_ids:
                                    chunk_ids.append(chunk_id)
                                fact_copy['chunk_ids'] = chunk_ids
                                facts_with_lineage.append(fact_copy)
                            else:
                                facts_with_lineage.append(fact)
                        
                        # Use the new _save_m2_facts method
                        save_result = await self._save_m2_facts(facts_with_lineage, user_id)
                        
                        if save_result['status'] != 'success':
                            logger.error(f"❌ Failed to save M2 facts: {save_result['message']}")
                            return False
                        
                        facts_count = save_result['data']['facts_saved']
                        logger.info(f"✅ Saved {facts_count} M2 facts for chunk {chunk_id}")
                
                
                # Explicit commit to ensure changes are persisted
                conn.commit()
                
                logger.info(f"✅ Successfully marked chunk {chunk_id} as M2 completed and stored {facts_count} facts (user_id={user_id})")
                return True
            
        except Exception as e:
            logger.error(f"❌ Error marking chunk {chunk_id} as M2 completed: {e}")
            return False

    async def _mark_chunk_m2_failed(self, chunk_id: str, error: str, user_id: Optional[str] = None) -> bool:
        """
        Mark a chunk as M2 processing failed with error information.
        
        This method transitions a chunk from 'processing' to 'failed' status,
        sets the m2_processing_ended_at timestamp, and logs the error information
        for debugging purposes.
        
        Args:
            chunk_id: The UUID of the chunk to mark as failed
            error: Error message describing why processing failed
            user_id: Optional user_id filter for security scoping
            
        Returns:
            True if chunk was successfully marked as failed, False otherwise
        """
        try:
            # Validate chunk_id is a valid UUID format
            import uuid
            try:
                uuid.UUID(chunk_id)
            except (ValueError, TypeError):
                logger.error(f"❌ Invalid chunk_id format: {chunk_id}")
                return False
            
            # Validate error message
            if not error or not error.strip():
                logger.warning("⚠️ Empty error message provided, using default")
                error = "Unknown M2 processing error"
            
            with sync_connection_pool.get_connection() as conn:
                with conn.cursor() as cur:
                    # Update M1 chunk status to failed
                    if user_id:
                        # User-scoped update for security
                        cur.execute("""
                            UPDATE m1_episodic 
                            SET m2_status = %s, m2_processing_ended_at = NOW()
                            WHERE chunk_id = %s AND user_id = %s AND m2_status = %s
                        """, (M2Status.FAILED.value, chunk_id, user_id, M2Status.PROCESSING.value))
                    else:
                        # No user filtering (fallback - should be avoided in production)
                        logger.warning("_mark_chunk_m2_failed: No user_id provided, updating without user filter")
                        cur.execute("""
                            UPDATE m1_episodic 
                            SET m2_status = %s, m2_processing_ended_at = NOW()
                            WHERE chunk_id = %s AND m2_status = %s
                        """, (M2Status.FAILED.value, chunk_id, M2Status.PROCESSING.value))
                    
                    # Check if any row was updated
                    affected_rows = cur.rowcount
                    conn.commit()
                    
                    if affected_rows > 0:
                        logger.error(f"❌ Marked chunk {chunk_id} as M2 failed (user_id={user_id}). Error: {error}")
                        return True
                    else:
                        logger.warning(f"⚠️ Chunk {chunk_id} could not be marked failed - may not be processing or not found (user_id={user_id})")
                        return False
            
        except Exception as e:
            logger.error(f"❌ Error marking chunk {chunk_id} as M2 failed: {e}")
            return False

    async def _save_m2_facts(
        self, 
        facts: List[Union[Fact, Dict[str, Any]]], 
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Save M2 semantic facts to the m2_semantic table.
        
        This method handles both Pydantic Fact objects and dictionary facts,
        validates their structure, generates missing fields (embeddings, hashes),
        and stores them in the database with proper idempotency handling.
        
        Args:
            facts: List of Fact objects or fact dictionaries to save
            user_id: Optional user_id for security scoping (required if not in facts)
            
        Returns:
            Success/error response dictionary with operation details
        """
        try:
            # Validate input
            if not isinstance(facts, list):
                return self._error_response("Facts must be a list", 400)
            
            if not facts:
                return self._success_response(
                    {"facts_saved": 0, "facts_skipped": 0}, 
                    "No facts to save"
                )
            
            logger.info(f"🔧 Saving {len(facts)} M2 facts to database (user_id={user_id})")
            logger.debug(f"DEBUG: _save_m2_facts called with user_id={user_id}")
            
            # Process and validate facts
            processed_facts = []
            for i, fact in enumerate(facts):
                logger.debug(f"DEBUG: Processing fact {i}: {type(fact)} - {str(fact)[:100]}")
                try:
                    processed_fact = self._process_fact(fact, user_id, i)
                    if processed_fact:
                        processed_facts.append(processed_fact)
                except Exception as e:
                    logger.warning(f"⚠️ Skipping invalid fact at index {i}: {e}")
                    continue
            
            if not processed_facts:
                return self._error_response("No valid facts to save after processing", 400)
            
            # Save facts to database
            return await self._insert_facts_to_database(processed_facts, user_id)
            
        except Exception as e:
            logger.error(f"❌ Error saving M2 facts: {e}")
            return self._error_response(f"Error saving M2 facts: {str(e)}")
    
    def _process_fact(
        self, 
        fact: Union[Fact, Dict[str, Any]], 
        default_user_id: Optional[str], 
        index: int
    ) -> Optional[Dict[str, Any]]:
        """
        Process and validate a single fact, converting Pydantic objects to dictionaries
        and generating missing fields.
        
        Args:
            fact: Fact object or dictionary to process
            default_user_id: Default user_id to use if not in fact
            index: Index of fact in list (for error reporting)
            
        Returns:
            Processed fact dictionary, or None if invalid
        """
        try:
            # Convert Pydantic Fact to dictionary
            if isinstance(fact, Fact):
                fact_dict = {
                    'text': fact.text,
                    'hash': fact.hash,
                    'embedding': fact.embedding,
                    'confidence': fact.confidence,
                    'status': fact.status.value if fact.status else M2FactStatus.ACTIVE.value,
                    'chunk_ids': fact.chunk_ids,
                    'user_id': fact.user_id,
                    'policy_version': fact.policy_version,
                    'metadata': fact.metadata
                }
            elif isinstance(fact, dict):
                fact_dict = fact.copy()
            else:
                logger.error(f"❌ Invalid fact type at index {index}: {type(fact)}")
                return None
            
            # Validate required fields
            fact_text = fact_dict.get('text', '').strip()
            if not fact_text:
                logger.error(f"❌ Empty or missing 'text' field in fact at index {index}")
                return None
            
            # Determine user_id for this fact
            fact_user_id = fact_dict.get('user_id') or default_user_id
            if not fact_user_id:
                logger.error(f"❌ No user_id available for fact at index {index}")
                return None
            
            # Debug logging
            logger.debug(f"DEBUG: _process_fact index {index}: fact_dict.get('user_id')={fact_dict.get('user_id')}, default_user_id={default_user_id}, final_user_id={fact_user_id}")
            
            # Validate user_id format (should be UUID)
            try:
                uuid.UUID(fact_user_id)
            except (ValueError, TypeError):
                logger.error(f"❌ Invalid user_id format for fact at index {index}: {fact_user_id}")
                return None
            
            # Generate hash if not provided
            if not fact_dict.get('hash'):
                import hashlib
                metadata_str = json.dumps(fact_dict.get('metadata', {}), sort_keys=True)
                hash_content = fact_text + metadata_str
                fact_dict['hash'] = hashlib.sha256(hash_content.encode('utf-8')).hexdigest()
            
            # Generate embedding if not provided
            if fact_dict.get('embedding') is None:
                fact_dict['embedding'] = self.embedding_generator.generate_embedding(fact_text)
            
            # Validate and clamp confidence score
            confidence = float(fact_dict.get('confidence', 0.8))
            fact_dict['confidence'] = max(0.0, min(1.0, confidence))
            
            # Set defaults for optional fields
            fact_dict['status'] = fact_dict.get('status', M2FactStatus.ACTIVE.value)
            fact_dict['policy_version'] = fact_dict.get('policy_version', 'v1.0')
            fact_dict['chunk_ids'] = fact_dict.get('chunk_ids', [])
            fact_dict['metadata'] = fact_dict.get('metadata', {})
            fact_dict['user_id'] = fact_user_id
            fact_dict['text'] = fact_text
            
            logger.debug(f"DEBUG: Final processed fact {index}: user_id={fact_dict['user_id']}, text='{fact_dict['text'][:50]}...'")
            logger.debug(f"✅ Processed fact at index {index}: hash={fact_dict['hash'][:8]}...")
            return fact_dict
            
        except Exception as e:
            logger.error(f"❌ Error processing fact at index {index}: {e}")
            return None
    
    async def _insert_facts_to_database(
        self, 
        processed_facts: List[Dict[str, Any]], 
        user_id: Optional[str]
    ) -> Dict[str, Any]:
        """
        Insert processed facts into the m2_semantic database table.
        
        Args:
            processed_facts: List of validated and processed fact dictionaries
            user_id: User ID for security scoping
            
        Returns:
            Success/error response dictionary with operation details
        """
        facts_saved = 0
        facts_skipped = 0
        
        try:
            with sync_connection_pool.get_connection() as conn:
                with conn.cursor() as cur:
                    for fact in processed_facts:
                        try:
                            # Convert embedding to list format for PostgreSQL
                            embedding_list = None
                            if fact['embedding'] is not None:
                                if hasattr(fact['embedding'], 'tolist'):
                                    embedding_list = fact['embedding'].tolist()
                                elif isinstance(fact['embedding'], list):
                                    embedding_list = fact['embedding']
                                else:
                                    logger.warning(f"⚠️ Unknown embedding format: {type(fact['embedding'])}")
                                    continue
                            
                            # Prepare chunk_ids as PostgreSQL UUID array
                            chunk_ids = fact.get('chunk_ids', [])
                            if chunk_ids:
                                # Validate all chunk_ids are UUIDs
                                valid_chunk_ids = []
                                for chunk_id in chunk_ids:
                                    try:
                                        uuid.UUID(str(chunk_id))
                                        valid_chunk_ids.append(str(chunk_id))
                                    except (ValueError, TypeError):
                                        logger.warning(f"⚠️ Invalid chunk_id format: {chunk_id}")
                                chunk_ids = valid_chunk_ids
                            
                            # Insert fact with conflict resolution
                            logger.debug(f"DEBUG: Inserting fact into DB: user_id={fact['user_id']}, text='{fact['text'][:50]}...'")
                            cur.execute("""
                                INSERT INTO m2_semantic 
                                (text, hash, embedding, confidence, status, chunk_ids, user_id, 
                                 policy_version, embedding_generated_at, metadata, embedding_model)
                                VALUES (%s, %s, %s, %s, %s, %s::uuid[], %s, %s, NOW(), %s, %s)
                                ON CONFLICT (hash) DO UPDATE SET
                                    chunk_ids = array_append(m2_semantic.chunk_ids, %s::uuid),
                                    updated_at = NOW(),
                                    confidence = GREATEST(m2_semantic.confidence, %s),
                                    status = CASE 
                                        WHEN m2_semantic.status = 'deprecated' THEN EXCLUDED.status
                                        ELSE m2_semantic.status 
                                    END
                            """, (
                                fact['text'],
                                fact['hash'],
                                embedding_list,
                                fact['confidence'],
                                fact['status'],
                                chunk_ids,  # PostgreSQL UUID array
                                fact['user_id'],
                                fact['policy_version'],
                                json.dumps(fact['metadata']),
                                'sentence-transformers/all-MiniLM-L6-v2',  # Default embedding model
                                chunk_ids[0] if chunk_ids else None,  # For conflict resolution
                                fact['confidence']  # For conflict resolution
                            ))
                            
                            facts_saved += 1
                            logger.debug(f"✅ Saved fact: {fact['hash'][:8]}...")
                            
                        except Exception as e:
                            logger.error(f"❌ Error saving individual fact: {e}")
                            facts_skipped += 1
                            continue
                
                # Commit all changes
                conn.commit()
                
            success_message = f"Saved {facts_saved} M2 facts to database"
            if facts_skipped > 0:
                success_message += f" ({facts_skipped} skipped due to errors)"
            
            logger.info(f"✅ {success_message} (user_id={user_id})")
            return self._success_response(
                {"facts_saved": facts_saved, "facts_skipped": facts_skipped},
                success_message
            )
            
        except Exception as e:
            logger.error(f"❌ Error saving M2 facts: {e}")
            return self._error_response(f"Error saving M2 facts: {str(e)}")
    
    async def _get_session_context_for_chunk(
        self, 
        chunk_id: str, 
        context_chunks_before: int = 5,
        user_id: Optional[str] = None
    ) -> Optional[List[Chunk]]:
        """
        Retrieve session context chunks created before the target chunk for M2 processing.
        
        Args:
            chunk_id: The target chunk ID to get context for
            context_chunks_before: Number of chunks before the target chunk to retrieve (default 5)
            user_id: Optional user ID for security scoping
            
        Returns:
            List of Chunk models ordered chronologically (oldest first), or None if target chunk not found
        """
        try:
            # Validate chunk_id is a valid UUID format
            import uuid
            try:
                uuid.UUID(chunk_id)
            except (ValueError, TypeError):
                logger.error(f"❌ Invalid chunk_id format: {chunk_id}")
                return None
            
            with sync_connection_pool.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    # First, get the target chunk to extract session_id and created_at
                    if user_id:
                        cur.execute("""
                            SELECT session_id, created_at 
                            FROM m1_episodic 
                            WHERE chunk_id = %s AND user_id = %s
                        """, (chunk_id, user_id))
                    else:
                        logger.warning("_get_session_context_for_chunk: No user_id provided, querying without user filter")
                        cur.execute("""
                            SELECT session_id, created_at 
                            FROM m1_episodic 
                            WHERE chunk_id = %s
                        """, (chunk_id,))
                    
                    target_row = cur.fetchone()
                    if not target_row:
                        logger.debug(f"Target chunk not found: {chunk_id}")
                        return None
                    
                    target_session_id = target_row['session_id']
                    target_created_at = target_row['created_at']
                    
                    if not target_session_id:
                        logger.warning(f"Target chunk {chunk_id} has no session_id, cannot retrieve context")
                        return []
                    
                    # Query for context chunks in the same session created before target chunk
                    if user_id:
                        cur.execute("""
                            SELECT 
                                chunk_id, content, token_count, user_id, session_id, 
                                created_at, updated_at, m2_status, chunking_strategy, 
                                m0_raw_ids, metadata
                            FROM m1_episodic 
                            WHERE session_id = %s AND user_id = %s AND created_at < %s
                            ORDER BY created_at DESC
                            LIMIT %s
                        """, (target_session_id, user_id, target_created_at, context_chunks_before))
                    else:
                        cur.execute("""
                            SELECT 
                                chunk_id, content, token_count, user_id, session_id, 
                                created_at, updated_at, m2_status, chunking_strategy, 
                                m0_raw_ids, metadata
                            FROM m1_episodic 
                            WHERE session_id = %s AND created_at < %s
                            ORDER BY created_at DESC
                            LIMIT %s
                        """, (target_session_id, target_created_at, context_chunks_before))
                    
                    rows = cur.fetchall()
                    
                    # Convert to Chunk models and reverse to get chronological order (oldest first)
                    chunks = []
                    for row in reversed(rows):
                        chunk = Chunk(
                            chunk_id=str(row['chunk_id']),
                            content=row['content'],
                            token_count=row['token_count'],
                            user_id=str(row['user_id']),
                            session_id=str(row['session_id']) if row['session_id'] else None,
                            created_at=row['created_at'],
                            updated_at=row['updated_at'],
                            m2_status=M2Status(row['m2_status']),
                            chunking_strategy=row['chunking_strategy'],
                            m0_raw_ids=list(row['m0_raw_ids']) if row['m0_raw_ids'] else [],
                            metadata=json.loads(row['metadata']) if isinstance(row['metadata'], str) else (row['metadata'] or {})
                        )
                        chunks.append(chunk)
                    
                    logger.info(f"✅ Retrieved {len(chunks)} context chunks for chunk {chunk_id} (user_id={user_id})")
                    return chunks
            
        except Exception as e:
            logger.error(f"❌ Error retrieving session context for chunk {chunk_id}: {e}")
            return None

    def _apply_token_budget_limit(
        self, 
        context: List[Chunk], 
        limit: int
    ) -> List[Chunk]:
        """
        Apply token budget limit to context chunks, keeping most recent chunks that fit within budget.
        
        Args:
            context: List of Chunk models ordered chronologically (oldest first)
            limit: Maximum token budget for the context
            
        Returns:
            Truncated list of Chunk models that fit within token budget (chronological order preserved)
        """
        if not context or limit <= 0:
            return []
        
        # Calculate total tokens and truncate if needed
        total_tokens = sum(chunk.token_count for chunk in context)
        
        if total_tokens <= limit:
            logger.debug(f"✅ Context fits within token budget: {total_tokens}/{limit} tokens")
            return context
        
        # Keep oldest chunks that fit within budget (starting from beginning of list)
        truncated_context = []
        current_tokens = 0
        
        # Work forwards through the list (oldest first)
        for chunk in context:
            if current_tokens + chunk.token_count <= limit:
                truncated_context.append(chunk)  # Append to maintain chronological order
                current_tokens += chunk.token_count
            else:
                # This chunk would exceed budget, stop here
                break
        
        logger.info(f"✅ Applied token budget limit: {len(truncated_context)}/{len(context)} chunks, "
                    f"{current_tokens}/{limit} tokens")
        return truncated_context
    
    def _build_fact_extraction_prompt(
        self,
        target_chunk: Optional[Dict[str, Any]],
        context_chunks: Optional[List[Chunk]]
    ) -> List[Dict[str, str]]:
        """Build messages for LLM fact extraction using existing prompt templates.
        
        Args:
            target_chunk: The main chunk to extract facts from
            context_chunks: List of context chunks for additional information
            
        Returns:
            List of message dictionaries for LLM consumption
        """
        try:
            # Normalize inputs
            context_list: List[Chunk] = context_chunks or []
            # Ensure chronological ordering (oldest first)
            try:
                context_list = sorted(
                    context_list,
                    key=lambda c: getattr(c, 'created_at', None) or datetime.now()
                )
            except Exception:
                # If sorting fails, keep original order
                pass

            # Format context chunks as chronological text
            if context_list:
                context_parts = []
                for i, chunk in enumerate(context_list):
                    created = getattr(chunk, 'created_at', None)
                    context_parts.append(
                        f"**Context Chunk {i+1}:**\n{getattr(chunk, 'content', '')}\n"
                        f"timestamp: {created}"
                    )
                chunk_context = "\n\n".join(context_parts)
            else:
                chunk_context = "No additional context chunks available."

            # Get the key chunk content safely
            key_chunk = ''
            if isinstance(target_chunk, dict):
                key_chunk = target_chunk.get('content', '')

            # Use module-level PromptManager if available
            if PromptManager is None:
                raise ImportError("PromptManager not available")

            system_prompt = PromptManager.get_prompt(
                "m2_extractor_system",
                chunk_context=chunk_context
            )
            user_prompt = PromptManager.get_prompt(
                "m2_extractor_user",
                key_chunk=key_chunk
            )

            # Validate prompt types; fallback if unexpected
            if not isinstance(system_prompt, str) or not isinstance(user_prompt, str):
                raise ValueError("PromptManager returned non-string content")

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            logger.debug(f"Built fact extraction prompt with {len(context_list)} context chunks")
            return messages

        except Exception as e:
            logger.error(f"Failed to build fact extraction prompt: {e}")
            # Fallback to basic prompt
            content = ''
            if isinstance(target_chunk, dict):
                content = target_chunk.get('content', '')
            fallback_prompt = f"""Extract semantic facts from this memory chunk:

{content}

Please extract clear, factual statements that would be useful for future memory retrieval. Return your response as JSON with a 'facts' array containing objects with 'content' and 'source_chunk_ids' fields."""
            
            return [
                {"role": "system", "content": "You are an expert at extracting semantic facts from conversational content."},
                {"role": "user", "content": fallback_prompt}
            ]

    async def _extract_list_of_fact_content_from_chunk(
        self, 
        chunk_id: str, 
        context: Optional[List[Chunk]] = None,
        user_id: Optional[str] = None
    ) -> List[str]:
        """Extract fact content strings from chunk using LLM with structured outputs.
        
        Args:
            chunk_id: The UUID of the chunk to extract facts from
            context: Optional context chunks for additional information  
            user_id: Optional user ID for security scoping
            
        Returns:
            List of fact content strings extracted from the chunk
        """
        try:
            # Step 1: Get the target chunk
            target_chunk = await self._get_m1_chunk(chunk_id, user_id)
            if not target_chunk:
                logger.error(f"Target chunk not found: {chunk_id}")
                return []
            
            # Step 2: Get session context if not provided
            if context is None:
                context = await self._get_session_context_for_chunk(
                    chunk_id, 
                    context_chunks_before=5, 
                    user_id=user_id
                )
                if context is None:
                    context = []
            
            # Step 3: Apply token budget limit
            context = self._apply_token_budget_limit(context, limit=2000)
            
            # Step 4: Build extraction prompt
            messages = self._build_fact_extraction_prompt(target_chunk, context)
            
            # Step 5: Initialize LLM provider
            llm_provider = await self._get_llm_provider()
            if not llm_provider:
                logger.error("No LLM provider available for fact extraction")
                return []
            
            # Step 6: Create LLM request with structured output
            from ..llm.base import LLMRequest
            from ..models.m2_extraction import FactExtractionResponse
            
            # Choose model - prefer structured output capable model
            model = self._get_preferred_extraction_model()
            
            request = LLMRequest(
                messages=messages,
                model=model,
                temperature=0.3,  # Lower temperature for more consistent extraction
                max_tokens=1500
            )
            
            # Step 7: Generate with retry logic
            facts = await self._extract_facts_with_retry(llm_provider, request)
            
            logger.info(f"✅ Extracted {len(facts)} facts from chunk {chunk_id}")
            return facts
            
        except Exception as e:
            logger.error(f"❌ Error extracting facts from chunk {chunk_id}: {e}")
            return []

    async def _extract_list_of_structured_facts_from_chunk(
        self,
        chunk_id: str,
        context: Optional[List[Chunk]] = None,
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Extract structured facts (text + optional confidence) from a chunk.

        Mirrors `_extract_list_of_fact_content_from_chunk` but preserves confidence
        from the LLM when available.
        """
        try:
            # Step 1: Get the target chunk
            target_chunk = await self._get_m1_chunk(chunk_id, user_id)
            if not target_chunk:
                logger.error(f"Target chunk not found: {chunk_id}")
                return []

            # Step 2: Get session context if not provided
            if context is None:
                context = await self._get_session_context_for_chunk(
                    chunk_id,
                    context_chunks_before=5,
                    user_id=user_id,
                )
                if context is None:
                    context = []

            # Step 3: Apply token budget limit
            context = self._apply_token_budget_limit(context, limit=2000)

            # Step 4: Build extraction prompt
            messages = self._build_fact_extraction_prompt(target_chunk, context)

            # Step 5: Initialize LLM provider
            llm_provider = await self._get_llm_provider()
            if not llm_provider:
                logger.error("No LLM provider available for fact extraction")
                return []

            # Step 6: Create LLM request
            from ..llm.base import LLMRequest
            model = self._get_preferred_extraction_model()
            request = LLMRequest(
                messages=messages,
                model=model,
                temperature=0.3,
                max_tokens=1500,
            )

            # Step 7: Generate with retry (structured)
            facts = await self._extract_facts_with_retry_structured(llm_provider, request)

            logger.info(
                f"✅ Extracted {len(facts)} structured facts from chunk {chunk_id}"
            )
            return facts

        except Exception as e:
            logger.error(f"❌ Error extracting structured facts from chunk {chunk_id}: {e}")
            return []
    
    async def _get_llm_provider(self):
        """Get LLM provider for fact extraction.

        Behavior:
        - If using official OpenAI endpoint (or no base_url), use OpenAIProvider
          to leverage native structured outputs when available.
        - If using a custom OpenAI-compatible base URL, use LiteLLMProvider,
          forced to OpenAI provider semantics and with structured API disabled.
        """
        import os

        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL") or ""

        # Official OpenAI endpoint? Use OpenAIProvider to enable native structured
        is_official_openai = (not base_url) or ("api.openai.com" in base_url)

        if is_official_openai:
            try:
                from ..llm.providers.openai import OpenAIProvider

                config = {
                    "api_key": api_key,
                    "base_url": base_url if base_url else None,
                    "timeout": 30.0,
                }
                provider = OpenAIProvider(config)
                return provider
            except Exception as e:
                raise LLMProviderError(f"Failed to initialize OpenAI provider: {e}")

        # Custom OpenAI-compatible proxy: use LiteLLMProvider with structured API enabled
        try:
            from ..llm.providers.litellm import LiteLLMProvider

            config = {
                "api_key": api_key,
                "base_url": base_url,
                "timeout": 30.0,
                # Allow provider to handle structured outputs when supported
                # (match scripts/smoke_litellm usage)
            }
            provider = LiteLLMProvider(config)
            return provider
        except Exception as e:
            raise LLMProviderError(f"Failed to initialize LiteLLM provider: {e}")
    
    async def _extract_facts_with_retry(
        self,
        llm_provider,
        request: "LLMRequest",
        max_retries: int = 3
    ) -> List[str]:
        """Extract facts with retry logic and fallback parsing."""
        from ..models.m2_extraction import FactExtractionResponse
        import asyncio
        
        last_error = None
        fallback_facts: Optional[List[str]] = None
        skip_calls = False  # If we already have fallback facts, skip further network calls

        for attempt in range(max_retries):
            try:
                if not skip_calls:
                    # Try structured output first if provider supports it
                    if hasattr(llm_provider, 'generate_structured'):
                        try:
                            response = await llm_provider.generate_structured(
                                request, FactExtractionResponse
                            )

                            # Primary path: parsed structured data
                            if response.success and getattr(response, 'parsed_data', None):
                                facts: List[str] = []
                                for fact in response.parsed_data.facts:  # type: ignore[attr-defined]
                                    if hasattr(fact, 'content'):
                                        facts.append(fact.content)
                                    elif isinstance(fact, str):
                                        facts.append(fact)

                                logger.debug(f"Structured extraction: {len(facts)} facts")
                                return facts

                            # Secondary path: structured call succeeded but only raw JSON content provided
                            if response.success and getattr(response, 'content', None):
                                parsed = self._parse_fact_extraction_response(response.content)
                                if parsed:
                                    logger.debug(f"Structured (content-parse) extraction: {len(parsed)} facts")
                                    return parsed

                        except Exception as structured_error:
                            logger.warning(f"Structured extraction failed on attempt {attempt + 1}: {structured_error}")

                    # Fall back to regular generation with JSON parsing
                    try:
                        response = await llm_provider.generate(request)

                        if response.success and response.content:
                            facts = self._parse_fact_extraction_response(response.content)
                            if facts:
                                logger.debug(f"JSON fallback extraction: {len(facts)} facts")
                                fallback_facts = facts
                                skip_calls = True  # Do not make more provider calls; still honor backoff timing

                        last_error = getattr(response, 'error', None) if not response.success else "No facts extracted"
                    except Exception as gen_err:
                        last_error = str(gen_err)
                        logger.warning(f"Regular generation failed on attempt {attempt + 1}: {gen_err}")

            except Exception as e:
                last_error = str(e)
                logger.warning(f"Fact extraction attempt {attempt + 1} failed: {e}")
            
            # Wait before retry
            if attempt < max_retries - 1:
                await asyncio.sleep(1.0 * (2 ** attempt))  # Exponential backoff
        
        # If we captured fallback facts, return them after completing retries/backoff
        if fallback_facts:
            return fallback_facts

        logger.error(f"All fact extraction attempts failed. Last error: {last_error}")
        return []

    async def _extract_facts_with_retry_structured(
        self,
        llm_provider,
        request: "LLMRequest",
        max_retries: int = 3
    ) -> List[Dict[str, Any]]:
        """Extract structured facts (text + optional confidence) with retry.

        Returns a list of dicts: {"text": str, "confidence": Optional[float]}.
        """
        from ..models.m2_extraction import FactExtractionResponse
        import asyncio

        last_error = None
        fallback_facts: Optional[List[Dict[str, Any]]] = None
        skip_calls = False

        for attempt in range(max_retries):
            try:
                if not skip_calls:
                    if hasattr(llm_provider, 'generate_structured'):
                        try:
                            response = await llm_provider.generate_structured(
                                request, FactExtractionResponse
                            )

                            if response.success and getattr(response, 'parsed_data', None):
                                results: List[Dict[str, Any]] = []
                                for fact in response.parsed_data.facts:  # type: ignore[attr-defined]
                                    try:
                                        content = getattr(fact, 'content', None)
                                        confidence = getattr(fact, 'confidence', None)
                                        if isinstance(content, str) and content.strip():
                                            entry: Dict[str, Any] = {"text": content.strip()}
                                            if isinstance(confidence, (int, float)):
                                                entry["confidence"] = float(confidence)
                                            results.append(entry)
                                    except Exception:
                                        continue

                                if results:
                                    logger.debug(
                                        f"Structured extraction (parsed_data): {len(results)} facts"
                                    )
                                    return results

                            if response.success and getattr(response, 'content', None):
                                parsed = self._parse_fact_extraction_response_structured(
                                    response.content
                                )
                                if parsed:
                                    logger.debug(
                                        f"Structured (content-parse) extraction: {len(parsed)} facts"
                                    )
                                    return parsed

                        except Exception as structured_error:
                            logger.warning(
                                f"Structured extraction failed on attempt {attempt + 1}: {structured_error}"
                            )

                    try:
                        response = await llm_provider.generate(request)
                        if response.success and response.content:
                            parsed = self._parse_fact_extraction_response_structured(
                                response.content
                            )
                            if parsed:
                                logger.debug(
                                    f"JSON fallback extraction (structured): {len(parsed)} facts"
                                )
                                fallback_facts = parsed
                                skip_calls = True

                        last_error = getattr(response, 'error', None) if not response.success else "No facts extracted"
                    except Exception as gen_err:
                        last_error = str(gen_err)
                        logger.warning(
                            f"Regular generation failed on attempt {attempt + 1}: {gen_err}"
                        )

            except Exception as e:
                last_error = str(e)
                logger.warning(f"Fact extraction attempt {attempt + 1} failed: {e}")

            if attempt < max_retries - 1:
                await asyncio.sleep(1.0 * (2 ** attempt))

        if fallback_facts:
            return fallback_facts

        logger.error(f"All structured fact extraction attempts failed. Last error: {last_error}")
        return []
    
    def _get_preferred_extraction_model(self) -> str:
        """Get the preferred model for fact extraction from environment variable.
        
        Returns:
            Model name string for LLM requests
        """
        import os
        
        # Try to get model from OPENAI_COMPATIBLE_MODEL environment variable
        env_model = os.getenv("OPENAI_COMPATIBLE_MODEL")
        if env_model:
            logger.debug(f"Using model from OPENAI_COMPATIBLE_MODEL: {env_model}")
            return env_model
        
        # Fallback hierarchy based on available API keys
        openai_key = os.getenv("OPENAI_API_KEY")
        xai_key = os.getenv("XAI_API_KEY")
        
        if openai_key:
            # Prefer newer OpenAI models with structured output support
            logger.debug("Using OpenAI model for fact extraction")
            return "gpt-4o-2024-08-06"
        elif xai_key:
            # Fallback to XAI/Grok model
            logger.debug("Using XAI model for fact extraction")
            return "grok-3-mini"
        else:
            # Default fallback
            logger.warning("No API keys found, using default model")
            return "gpt-4o"
    
    def _parse_fact_extraction_response(self, response_content: str) -> List[str]:
        """Parse LLM response content to extract fact strings.
        
        Args:
            response_content: Raw response content from LLM
            
        Returns:
            List of fact content strings
        """
        try:
            import json
            import re

            if not response_content or not response_content.strip():
                return []

            original_text = response_content.strip()

            # Try JSON parsing first (without losing original text)
            try:
                json_candidate = None
                # Handle JSON wrapped in markdown code blocks
                if '```json' in original_text:
                    json_match = re.search(r'```json\s*(\{.*?\})\s*```', original_text, re.DOTALL)
                    if json_match:
                        json_candidate = json_match.group(1)
                elif original_text.startswith('{') and original_text.endswith('}'):
                    # Already looks like JSON
                    json_candidate = original_text
                else:
                    # Try to find JSON object within the response
                    json_match = re.search(r'\{.*\}', original_text, re.DOTALL)
                    if json_match:
                        json_candidate = json_match.group(0)

                data = json.loads(json_candidate) if json_candidate else None
                
                # Extract facts from JSON structure
                facts = []
                if isinstance(data, dict):
                    items = data.get('facts')
                    if isinstance(items, list):
                        for fact_item in items:
                            if isinstance(fact_item, dict):
                                # Handle structured fact objects
                                content = fact_item.get('content') or fact_item.get('text')
                                if content and isinstance(content, str):
                                    content = content.strip()
                                    if content:
                                        facts.append(content)
                            elif isinstance(fact_item, str):
                                # Handle simple string facts
                                if fact_item.strip():
                                    facts.append(fact_item.strip())
                    # If JSON is present but no facts extracted, do NOT fall back to
                    # line-based parsing to avoid saving keys like "facts" or "processing_notes".
                    if facts:
                        logger.debug(f"Parsed {len(facts)} facts from JSON response")
                        return facts
                    else:
                        logger.debug("JSON detected but no facts found; returning empty list")
                        return []

            except (json.JSONDecodeError, KeyError) as e:
                logger.debug(f"JSON parsing failed, falling back to text parsing: {e}")
            
            # Fallback: Text parsing for unstructured responses
            facts: List[str] = []
            # Prefer bullet-style lines if present
            bullet_matches = re.findall(r"^[\s]*[\-\*\•]\s*(.+)$", original_text, flags=re.MULTILINE)
            if bullet_matches:
                for item in bullet_matches:
                    item = item.strip()
                    if len(item) >= 3:
                        facts.append(item)
            else:
                # General line-based extraction
                lines = original_text.split('\n')
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    # Remove list markers and prefixes
                    line = re.sub(r'^[\-\*\•]\s*', '', line)
                    line = re.sub(r'^\d+\.\s*', '', line)
                    line = re.sub(r'^Facts?\s*:\s*', '', line, flags=re.IGNORECASE)
                    # Filter out narrations
                    lower = line.lower()
                    if lower.startswith(('here', 'the following', 'extracted', 'based on', 'this should')):
                        continue
                    if len(line) >= 10:
                        facts.append(line)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_facts = []
            for fact in facts:
                fact_lower = fact.lower()
                if fact_lower not in seen:
                    seen.add(fact_lower)
                    unique_facts.append(fact)
            
            logger.debug(f"Parsed {len(unique_facts)} facts from text fallback")
            return unique_facts[:20]  # Limit to 20 facts max
        
        except Exception as e:
            logger.error(f"Error parsing fact extraction response: {e}")
            return []

    def _parse_fact_extraction_response_structured(self, response_content: str) -> List[Dict[str, Any]]:
        """Parse LLM response into structured facts with optional confidence.

        Returns a list of dictionaries: {"text": str, "confidence": Optional[float]}.
        """
        try:
            import json
            import re

            if not response_content or not response_content.strip():
                return []

            original_text = response_content.strip()

            # Attempt to isolate JSON object
            json_candidate = None
            if '```json' in original_text:
                m = re.search(r'```json\s*(\{.*?\})\s*```', original_text, re.DOTALL)
                if m:
                    json_candidate = m.group(1)
            elif original_text.startswith('{') and original_text.endswith('}'):
                json_candidate = original_text
            else:
                m = re.search(r'\{.*\}', original_text, re.DOTALL)
                if m:
                    json_candidate = m.group(0)

            if json_candidate:
                try:
                    data = json.loads(json_candidate)
                    items = data.get('facts') if isinstance(data, dict) else None
                    results: List[Dict[str, Any]] = []
                    if isinstance(items, list):
                        for it in items:
                            if isinstance(it, dict):
                                text = it.get('content') or it.get('text')
                                if isinstance(text, str) and text.strip():
                                    entry: Dict[str, Any] = {"text": text.strip()}
                                    conf = it.get('confidence')
                                    if isinstance(conf, (int, float)):
                                        entry["confidence"] = float(conf)
                                    results.append(entry)
                            elif isinstance(it, str):
                                s = it.strip()
                                if s:
                                    results.append({"text": s})
                    if results:
                        logger.debug(
                            f"Parsed {len(results)} structured facts from JSON response"
                        )
                        return results[:20]
                    else:
                        logger.debug(
                            "JSON detected but no structured facts found; returning empty list"
                        )
                        return []
                except Exception as json_err:
                    logger.debug(f"JSON parsing failed in structured parser: {json_err}")

            # Fallback: extract lines/bullets without confidence
            facts: List[str] = []
            bullet_matches = re.findall(r"^[\s]*[\-\*\•]\s*(.+)$", original_text, flags=re.MULTILINE)
            if bullet_matches:
                for item in bullet_matches:
                    item = item.strip()
                    if len(item) >= 3:
                        facts.append(item)
            else:
                lines = original_text.split('\n')
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    line = re.sub(r'^[\-\*\•]\s*', '', line)
                    line = re.sub(r'^\d+\.\s*', '', line)
                    line = re.sub(r'^Facts?\s*:\s*', '', line, flags=re.IGNORECASE)
                    lower = line.lower()
                    if lower.startswith(('here', 'the following', 'extracted', 'based on', 'this should')):
                        continue
                    if len(line) >= 10:
                        facts.append(line)

            seen = set()
            unique: List[Dict[str, Any]] = []
            for f in facts:
                fl = f.lower()
                if fl not in seen:
                    seen.add(fl)
                    unique.append({"text": f})

            logger.debug(
                f"Parsed {len(unique)} structured facts from text fallback"
            )
            return unique[:20]
        except Exception as e:
            logger.error(f"Error parsing structured fact extraction response: {e}")
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

            all_results = []



            # Search messages/chunks if requested
            if include_messages or include_chunks:
                chunk_results = await self.query_similar_chunks(
                    actual_query,
                    search_top_k,
                    user_id=user_id,
                    session_id=session_id
                )
                all_results.extend(chunk_results)

            # Sort by relevance score and take top_k
            all_results.sort(key=lambda x: x.get('relevance_score', x.get('similarity_score', 0)), reverse=True)
            results = all_results[:top_k]

            # Format response to match BufferService expectations
            response = {
                "status": "success",
                "code": 200,
                "data": {
                    "results": results,
                    "total": len(results)
                },
                "message": f"Retrieved {len(results)} results from memory database (searched {len(all_results)} candidates)",
                "errors": None
            }

            logger.info(f"SimplifiedMemoryService.query: Returning {len(results)} results for query: '{actual_query[:50]}...' (searched {search_top_k} candidates, messages={include_messages})")
            return response

        except Exception as e:
            logger.error(f"SimplifiedMemoryService.query: Error: {e}")
            return self._error_response(f"Query failed: {str(e)}")

    async def close(self):
        """Close database connections."""
        # Stop M2 background task if running
        if self.m2_processor_task is not None:
            self.m2_running = False
            try:
                self.m2_processor_task.cancel()
                try:
                    await self.m2_processor_task
                except asyncio.CancelledError:
                    pass
            finally:
                self.m2_processor_task = None

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
