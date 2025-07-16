"""PostgreSQL backend for MemFuse database."""

import json
from typing import Dict, List, Any, Optional

from loguru import logger

from .base import DBBase

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    logger.warning("psycopg2 not available, PostgreSQL backend will not work. To use PostgreSQL, install psycopg2 with: pip install psycopg2-binary or poetry add psycopg2-binary")


class PostgresDB(DBBase):
    """PostgreSQL backend for MemFuse database.
    
    This class provides a PostgreSQL implementation of the database backend.
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
        if not PSYCOPG2_AVAILABLE:
            raise ImportError("psycopg2 is required for PostgreSQL backend")
        
        self.conn_params = {
            "host": host,
            "port": port,
            "database": database,
            "user": user,
            "password": password
        }
        
        # Connect to database
        self.conn = psycopg2.connect(**self.conn_params)
        
        # Initialize database tables
        self._initialize_tables()
        
        logger.info(f"PostgreSQL backend initialized at {host}:{port}/{database}")
    
    def execute(self, query: str, params: tuple = ()) -> Any:
        """Execute a SQL query.
        
        Args:
            query: SQL query
            params: Query parameters
            
        Returns:
            PostgreSQL cursor
        """
        cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(query, params)
        return cursor
    
    def commit(self):
        """Commit changes to the database."""
        self.conn.commit()
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
    
    def create_tables(self):
        """Create database tables if they don't exist."""
        self._initialize_tables()

    def _initialize_tables(self):
        """Initialize database tables with proper schema."""
        # Create users table
        self.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP
        )
        ''')
        
        # Create agents table
        self.execute('''
        CREATE TABLE IF NOT EXISTS agents (
            id TEXT PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP
        )
        ''')
        
        # Create sessions table
        self.execute('''
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
        
        # Create rounds table
        self.execute('''
        CREATE TABLE IF NOT EXISTS rounds (
            id TEXT PRIMARY KEY,
            session_id TEXT,
            created_at TIMESTAMP,
            updated_at TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE
        )
        ''')
        
        # Create messages table
        self.execute('''
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
        
        # Create knowledge table
        self.execute('''
        CREATE TABLE IF NOT EXISTS knowledge (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        )
        ''')
        
        # Create API keys table
        self.execute('''
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
        
        self.commit()
    
    def add(self, table: str, data: Dict[str, Any]) -> str:
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
        self.execute(query, tuple(processed_data.values()))
        self.commit()

        return data.get('id')
    
    def select(self, table: str, conditions: Dict[str, Any] = None) -> List[Dict[str, Any]]:
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
        
        cursor = self.execute(query, params)
        rows = cursor.fetchall()
        
        return [dict(row) for row in rows]
    
    def select_one(self, table: str, conditions: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Select a single row from a table.
        
        Args:
            table: Table name
            conditions: Selection conditions
            
        Returns:
            Selected row or None if not found
        """
        rows = self.select(table, conditions)
        
        if not rows:
            return None
        
        return rows[0]
    
    def update(self, table: str, data: Dict[str, Any], conditions: Dict[str, Any]) -> int:
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
        cursor = self.execute(query, tuple(params_list))
        self.commit()
        
        return cursor.rowcount
    
    def delete(self, table: str, conditions: Dict[str, Any]) -> int:
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
        cursor = self.execute(query, tuple(params_list))
        self.commit()

        return cursor.rowcount

    def batch_add(self, table: str, data_list: List[Dict[str, Any]]) -> List[str]:
        """Batch add data to a table for improved performance.

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

            # Execute the batch query
            self.execute(query, tuple(params))
            self.commit()

            # Return the IDs
            return [data.get('id', '') for data in processed_data_list]

        return []


