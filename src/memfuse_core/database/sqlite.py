"""SQLite backend for MemFuse database."""

import sqlite3
import os
import json
from typing import Dict, List, Any, Optional
from datetime import datetime

from loguru import logger

from ..utils.path_manager import PathManager
from .base import DBBase


class SQLiteDB(DBBase):
    """SQLite backend for MemFuse database.

    This class provides a SQLite implementation of the database backend.
    """

    def __init__(self, db_path: str):
        """Initialize the SQLite backend.

        Args:
            db_path: Path to the SQLite database file
        """
        # Create directory if it doesn't exist
        PathManager.ensure_directory(os.path.dirname(db_path))

        # Connect to database
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row

        logger.info(f"SQLite backend initialized at {db_path}")

    def execute(self, query: str, params: tuple = ()) -> sqlite3.Cursor:
        """Execute a SQL query.

        Args:
            query: SQL query
            params: Query parameters

        Returns:
            SQLite cursor
        """
        cursor = self.conn.cursor()
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
        # ARCHITECTURAL CHANGE (2025-05-24): Added composite unique constraint on (name, user_id)
        # This ensures session names are unique within each user's scope, preventing data isolation bugs
        self.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id),
            FOREIGN KEY (agent_id) REFERENCES agents (id),
            UNIQUE(name, user_id)
        )
        ''')

        # Create rounds table
        self.execute('''
        CREATE TABLE IF NOT EXISTS rounds (
            id TEXT PRIMARY KEY,
            session_id TEXT,
            created_at TIMESTAMP,
            updated_at TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions (id)
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
            FOREIGN KEY (round_id) REFERENCES rounds (id)
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
            FOREIGN KEY (user_id) REFERENCES users (id)
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
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        ''')

        self.commit()

    def insert(self, table: str, data: Dict[str, Any]) -> str:
        """Insert data into a table.

        Args:
            table: Table name
            data: Data to insert

        Returns:
            ID of the inserted row
        """
        # Convert any dictionary values to JSON
        for key, value in data.items():
            if isinstance(value, dict):
                data[key] = json.dumps(value)

        # Build the query
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['?'] * len(data))
        query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"

        # Execute the query
        self.execute(query, tuple(data.values()))
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
                where_clauses.append(f"{key} = ?")
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
        # Convert any dictionary values to JSON
        for key, value in data.items():
            if isinstance(value, dict):
                data[key] = json.dumps(value)

        # Build the query
        set_clauses = []
        params_list = []

        for key, value in data.items():
            set_clauses.append(f"{key} = ?")
            params_list.append(value)

        where_clauses = []

        for key, value in conditions.items():
            where_clauses.append(f"{key} = ?")
            params_list.append(value)

        query = f"UPDATE {table} SET " + \
            ", ".join(set_clauses) + " WHERE " + " AND ".join(where_clauses)

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
            where_clauses.append(f"{key} = ?")
            params_list.append(value)

        query = f"DELETE FROM {table} WHERE " + " AND ".join(where_clauses)

        # Execute the query
        cursor = self.execute(query, tuple(params_list))
        self.commit()

        return cursor.rowcount
