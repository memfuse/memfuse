"""PostgreSQL backend for MemFuse database."""

import asyncio
import json
from typing import Dict, List, Any, Optional

from loguru import logger

from .base import DBBase

try:
    import psycopg
    from psycopg.rows import dict_row
    PSYCOPG_AVAILABLE = True
except ImportError:
    PSYCOPG_AVAILABLE = False
    logger.warning("psycopg not available, PostgreSQL backend will not work. To use PostgreSQL, install psycopg with: pip install psycopg or poetry add psycopg")


class PostgresDB(DBBase):
    """PostgreSQL backend for MemFuse database.

    This class provides an async PostgreSQL implementation of the database backend.
    """

    def __init__(self, host: str, port: int, database: str, user: str, password: str):
        """Initialize the PostgreSQL backend.

        Args:
            host: Database host
            port: Database port
            database: Database name
            user: Database user
            password: Database password
        """
        if not PSYCOPG_AVAILABLE:
            raise ImportError("psycopg is required for PostgreSQL backend")

        # Store connection parameters for pool creation
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password

        # Create database URL for connection pool
        self.db_url = f"postgresql://{user}:{password}@{host}:{port}/{database}"

        # Connection manager will be initialized lazily in async context
        self.connection_manager = None
        self._initialized = False

        logger.info(f"PostgreSQL backend initialized with async connection pool at {host}:{port}/{database}")

    async def _ensure_initialized(self):
        """Ensure the connection manager is initialized."""
        if not self._initialized:
            from ..services.global_connection_manager import get_global_connection_manager
            self.connection_manager = get_global_connection_manager()
            self._initialized = True

    async def execute(self, query: str, params: tuple = ()) -> Any:
        """Execute a SQL query using simplified connection management.

        Args:
            query: SQL query
            params: Query parameters

        Returns:
            For SELECT: List of dictionaries
            For INSERT/UPDATE/DELETE: Number of affected rows
        """
        # Ensure connection manager is initialized
        await self._ensure_initialized()

        # Use timeout optimized for streaming scenarios
        connection_timeout = 45.0  # Generous timeout for streaming operations

        try:
            # Direct execution with simplified connection management
            return await asyncio.wait_for(
                self._execute_with_simplified_connection(query, params),
                timeout=connection_timeout
            )
        except asyncio.TimeoutError:
            logger.error(f"PostgresDB: Query execution timed out after {connection_timeout}s: {query[:100]}...")
            raise
        except Exception as e:
            msg = str(e).lower()
            # Downgrade log level for expected constraint conflicts
            if (
                'duplicate key' in msg or 'already exists' in msg or 'unique constraint' in msg
            ):
                logger.warning(f"PostgresDB: Query resulted in constraint conflict: {e}")
            else:
                logger.error(f"PostgresDB: Query execution failed: {e}")
            raise

    async def _execute_with_simplified_connection(self, query: str, params: tuple) -> Any:
        """Execute query with simplified connection management optimized for streaming."""
        pool = None
        conn = None

        try:
            # Get connection pool using global connection manager
            pool = await self.connection_manager.get_connection_pool(self.db_url)
            # Get connection from pool
            conn = await pool.getconn()

            async with conn.cursor(row_factory=dict_row) as cursor:
                await cursor.execute(query, params)

                # Fetch results if it's a SELECT query
                if query.strip().upper().startswith('SELECT'):
                    results = await cursor.fetchall()
                    return results
                else:
                    # For INSERT/UPDATE/DELETE, commit and return rowcount
                    rowcount = cursor.rowcount
                    await conn.commit()
                    return rowcount

        except psycopg.Error as e:
            # Handle transaction errors by rolling back
            if "current transaction is aborted" in str(e) and conn:
                logger.warning("Transaction aborted, rolling back...")
                try:
                    await conn.rollback()
                except Exception:
                    pass
                # Retry the query after rollback
                async with conn.cursor(row_factory=dict_row) as cursor:
                    await cursor.execute(query, params)

                    if query.strip().upper().startswith('SELECT'):
                        results = await cursor.fetchall()
                        return results
                    else:
                        rowcount = cursor.rowcount
                        await conn.commit()
                        return rowcount
            else:
                # Re-raise other types of errors
                raise
        finally:
            # CRITICAL: Always return connection to pool immediately
            if conn is not None:
                try:
                    await pool.putconn(conn)
                    logger.debug("PostgresDB: Connection returned to connection pool")
                except Exception as e:
                    logger.error(f"PostgresDB: Failed to return connection to pool: {e}")

    async def commit(self):
        """Commit changes to the database using connection pool."""
        # Note: With async connection pool, commits are handled per-connection
        # This method is kept for compatibility but actual commits happen
        # when connections are returned to the pool
        logger.debug("PostgresDB.commit: Using async connection pool, commits handled automatically")

    async def close(self):
        """Close the database connection pool."""
        # Ensure connection manager is initialized
        await self._ensure_initialized()

        # Close all pools when the database backend is closed
        await self.connection_manager.close_all_pools()
        logger.info("PostgresDB: Simplified connection pools closed")

    async def create_tables(self):
        """Create database tables if they don't exist."""
        await self._initialize_tables()

    async def _initialize_tables(self):
        """Initialize database tables with proper schema.

        Uses a single connection, single transaction, and an advisory transaction lock
        to serialize DDL and avoid concurrent "type ... already exists" errors.
        """
        # Ensure connection manager is ready
        await self._ensure_initialized()

        pool = await self.connection_manager.get_connection_pool(self.db_url)
        conn = await pool.getconn()
        try:
            # Start transaction and acquire an advisory lock for schema init
            await conn.execute("BEGIN")
            # Transaction-scoped advisory lock; releases automatically on COMMIT/ROLLBACK
            await conn.execute("SELECT pg_advisory_xact_lock(448820728)")

            # Base tables in dependency order
            await conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP
            )
            ''')

            await conn.execute('''
            CREATE TABLE IF NOT EXISTS agents (
                id TEXT PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP
            )
            ''')

            await conn.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
                FOREIGN KEY (agent_id) REFERENCES agents (id) ON DELETE CASCADE
            )
            ''')

            await conn.execute('''
            CREATE TABLE IF NOT EXISTS rounds (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE
            )
            ''')

            await conn.execute('''
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

            await conn.execute('''
            CREATE TABLE IF NOT EXISTS knowledge (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
            ''')

            await conn.execute('''
            CREATE TABLE IF NOT EXISTS api_keys (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                key TEXT UNIQUE NOT NULL,
                name TEXT,
                permissions TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
            ''')

            # Initialize MemFuse memory layer tables using SchemaManager within same tx
            await self._initialize_memory_layer_tables(conn=conn)

            await conn.commit()
        except Exception as e:
            try:
                await conn.rollback()
            except Exception:
                pass
            raise
        finally:
            try:
                await pool.putconn(conn)
            except Exception:
                pass

    async def _initialize_memory_layer_tables(self, conn=None):
        """Initialize memory layer tables using SchemaManager.

        If a connection is provided, use it; otherwise, fall back to execute().
        """
        from memfuse_core.models.schema.manager import SchemaManager

        schema_manager = SchemaManager()

        # Create M0 raw table
        m0_schema = schema_manager.get_schema('m0_raw')
        if conn is not None:
            await conn.execute(m0_schema.generate_create_table_sql())
        else:
            await self.execute(m0_schema.generate_create_table_sql())

        # Create M1 episodic table
        m1_schema = schema_manager.get_schema('m1_episodic')
        if conn is not None:
            await conn.execute(m1_schema.generate_create_table_sql())
        else:
            await self.execute(m1_schema.generate_create_table_sql())

        logger.info("PostgresDB: Memory layer tables initialized using SchemaManager")

    async def add(self, table: str, data: Dict[str, Any]) -> str:
        """Add data to a table.

        Args:
            table: Table name
            data: Data to add

        Returns:
            ID of the added row
        """
        # Convert any dictionary values to JSONB
        processed_data = {}
        for key, value in data.items():
            if isinstance(value, dict):
                processed_data[key] = json.dumps(value)
            else:
                processed_data[key] = value

        # Build the query
        columns = ', '.join(processed_data.keys())
        placeholders = ', '.join(['%s'] * len(processed_data))
        query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"

        # Execute the query
        await self.execute(query, tuple(processed_data.values()))

        # Return the appropriate ID field based on table schema
        if table == 'm0_raw':
            return data.get('message_id')
        else:
            return data.get('id')

    async def select(self, table: str, conditions: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Select data from a table.

        Args:
            table: Table name
            conditions: Selection conditions (optional)

        Returns:
            List of selected rows
        """
        query = f"SELECT * FROM {table}"
        params = ()

        if conditions:
            where_clauses = []
            params_list = []

            for key, value in conditions.items():
                where_clauses.append(f"{key} = %s")
                params_list.append(value)

            query += " WHERE " + " AND ".join(where_clauses)
            params = tuple(params_list)

        results = await self.execute(query, params)

        return [dict(row) for row in results]

    async def select_one(self, table: str, conditions: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Select a single row from a table.

        Args:
            table: Table name
            conditions: Selection conditions

        Returns:
            Selected row or None if not found
        """
        rows = await self.select(table, conditions)

        if not rows:
            return None

        return rows[0]

    async def update(self, table: str, data: Dict[str, Any], conditions: Dict[str, Any]) -> int:
        """Update data in a table.

        Args:
            table: Table name
            data: Data to update
            conditions: Update conditions

        Returns:
            Number of rows updated
        """
        # Convert any dictionary values to JSONB
        processed_data = {}
        for key, value in data.items():
            if isinstance(value, dict):
                processed_data[key] = json.dumps(value)
            else:
                processed_data[key] = value

        # Build the query
        set_clauses = []
        params_list = []

        for key, value in processed_data.items():
            set_clauses.append(f"{key} = %s")
            params_list.append(value)

        where_clauses = []

        for key, value in conditions.items():
            where_clauses.append(f"{key} = %s")
            params_list.append(value)

        query = f"UPDATE {table} SET " + ", ".join(set_clauses) + " WHERE " + " AND ".join(where_clauses)

        # Execute the query
        rowcount = await self.execute(query, tuple(params_list))

        return rowcount

    async def delete(self, table: str, conditions: Dict[str, Any]) -> int:
        """Delete data from a table.

        Args:
            table: Table name
            conditions: Delete conditions

        Returns:
            Number of rows deleted
        """
        # Build the query
        where_clauses = []
        params_list = []

        for key, value in conditions.items():
            where_clauses.append(f"{key} = %s")
            params_list.append(value)

        query = f"DELETE FROM {table} WHERE " + " AND ".join(where_clauses)

        # Execute the query
        rowcount = await self.execute(query, tuple(params_list))

        return rowcount

    async def batch_add(self, table: str, data_list: List[Dict[str, Any]]) -> List[str]:
        """Batch add data to a table for improved performance with proper connection management.

        This method provides optimized batch addition for PostgreSQL,
        which is particularly important for high-throughput scenarios.

        Args:
            table: Table name
            data_list: List of data dictionaries to add

        Returns:
            List of IDs of the added rows
        """
        if not data_list:
            return []

        # Process all data items
        processed_data_list = []
        for data in data_list:
            processed_data = {}
            for key, value in data.items():
                if isinstance(value, dict):
                    processed_data[key] = json.dumps(value)
                else:
                    processed_data[key] = value
            processed_data_list.append(processed_data)

        # Build the batch query
        if processed_data_list:
            columns = list(processed_data_list[0].keys())
            columns_str = ', '.join(columns)
            placeholders = ', '.join(['%s'] * len(columns))

            # Use VALUES clause for batch add
            values_clause = ', '.join([f"({placeholders})" for _ in processed_data_list])
            query = f"INSERT INTO {table} ({columns_str}) VALUES {values_clause}"

            # Flatten the parameters
            params = []
            for data in processed_data_list:
                params.extend([data[col] for col in columns])

            # Execute the batch query with timeout
            try:
                await asyncio.wait_for(
                    self.execute(query, tuple(params)),
                    timeout=120.0  # 2 minutes for batch operations
                )
                logger.debug(f"PostgresDB: Batch added {len(processed_data_list)} records to {table}")
            except asyncio.TimeoutError:
                logger.error(f"PostgresDB: Batch add timed out for {len(processed_data_list)} records to {table}")
                raise

            # Return the IDs based on table schema
            if table == 'm0_raw':
                return [data.get('message_id', '') for data in processed_data_list]
            else:
                return [data.get('id', '') for data in processed_data_list]

        return []

