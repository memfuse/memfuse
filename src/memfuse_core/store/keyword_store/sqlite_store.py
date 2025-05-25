"""SQLite keyword store implementation for MemFuse server.

This module provides a keyword store implementation using SQLite FTS5 for full-text search.
It creates a separate index.db file to store the search indices, keeping the metadata.db
clean and focused on raw data storage.
"""

import os
import sqlite3
import asyncio
from typing import List, Optional
import time
from loguru import logger
import json

from ...models.core import Item, QueryResult
from ...utils.path_manager import PathManager
from .base import KeywordStore


class SQLiteKeywordStore(KeywordStore):
    """SQLite keyword store implementation using FTS5 for full-text search."""

    def __init__(
        self,
        data_dir: str,
        cache_size: int = 100,
        buffer_size: int = 10,
        k1: float = 1.2,  # Optimized BM25 parameter for short texts
        b: float = 0.75,  # Standard BM25 parameter for document length normalization
        **kwargs
    ):
        """Initialize the SQLite keyword store.

        Args:
            data_dir: Directory to store data
            cache_size: Size of the query cache
            buffer_size: Size of the write buffer
            k1: BM25 parameter for term frequency scaling (default: 1.2, optimized for short texts)
            b: BM25 parameter for document length normalization (default: 0.75)
            **kwargs: Additional arguments
        """
        super().__init__(data_dir, cache_size=cache_size, **kwargs)
        self.buffer_size = buffer_size
        self.k1 = k1
        self.b = b

        # index.db should be at the same level as metadata.db, not in user subdirectory
        # Extract the parent directory if data_dir is a user subdirectory
        if os.path.basename(os.path.dirname(data_dir)) == "data":
            # We're already at the user directory level, so go up one level
            root_data_dir = os.path.dirname(data_dir)
        else:
            # We might be at the root data directory already
            root_data_dir = data_dir

        self.index_db_path = os.path.join(root_data_dir, "index.db")
        self.conn = None
        self.write_buffer = []
        self.write_lock = asyncio.Lock()
        self.initialized = False

    async def initialize(self) -> bool:
        """Initialize the keyword store.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            PathManager.ensure_directory(os.path.dirname(self.index_db_path))

            # Connect to the database
            self.conn = sqlite3.connect(self.index_db_path)
            self.conn.row_factory = sqlite3.Row

            # Create tables if they don't exist
            self._create_tables()

            # Log initialization
            logger.debug(
                f"Initialized SQLite keyword store at {self.index_db_path}")
            logger.info(
                f"Database exists: {os.path.exists(self.index_db_path)}")

            self.initialized = True
            return True
        except Exception as e:
            logger.error(f"Error initializing SQLite keyword store: {e}")
            return False

    def _create_tables(self):
        """Create the necessary tables for the keyword store."""
        cursor = self.conn.cursor()

        # Create items table to store document metadata
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS items (
            id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            metadata TEXT
        )
        """)

        # Create FTS5 virtual table for full-text search with better tokenizer
        try:
            # First try with advanced tokenizer options
            cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS items_fts USING fts5(
                id UNINDEXED,
                content,
                type UNINDEXED,
                metadata UNINDEXED,
                tokenize='porter unicode61 remove_diacritics 1'
            )
            """)
            logger.debug("Created FTS5 table with porter tokenizer")
        except sqlite3.OperationalError:
            # Fall back to simpler configuration if advanced options not available
            try:
                cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS items_fts USING fts5(
                    id UNINDEXED,
                    content,
                    type UNINDEXED,
                    metadata UNINDEXED,
                    tokenize='unicode61'
                )
                """)
                logger.debug("Created FTS5 table with unicode61 tokenizer")
            except sqlite3.OperationalError:
                # Fall back to default tokenizer as last resort
                cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS items_fts USING fts5(
                    id UNINDEXED,
                    content,
                    type UNINDEXED,
                    metadata UNINDEXED
                )
                """)
                logger.debug("Created FTS5 table with default tokenizer")

        self.conn.commit()

    async def add_document(self, item: Item) -> str:
        """Add a document to the store.

        Args:
            item: Item to add

        Returns:
            ID of the added document
        """
        async with self.write_lock:
            # Add to write buffer
            self.write_buffer.append(("add", item))

            # Flush buffer if it's full
            if len(self.write_buffer) >= self.buffer_size:
                await self._flush_buffer()

            return item.id

    async def add_documents(self, items: List[Item]) -> List[str]:
        """Add multiple documents to the store.

        Args:
            items: Items to add

        Returns:
            List of IDs of the added documents
        """
        if not items:
            return []

        async with self.write_lock:
            # Add all items to write buffer
            for item in items:
                self.write_buffer.append(("add", item))

            # Flush buffer if it's full
            if len(self.write_buffer) >= self.buffer_size:
                await self._flush_buffer()

            return [item.id for item in items]

    async def get_document(self, item_id: str) -> Optional[Item]:
        """Get a document by ID.

        Args:
            item_id: ID of the document to get

        Returns:
            Document if found, None otherwise
        """
        # Flush buffer to ensure we have the latest data
        await self._flush_buffer()

        cursor = self.conn.cursor()
        cursor.execute("""
        SELECT i.id, f.content, i.type, i.created_at, i.updated_at, i.metadata
        FROM items i
        JOIN items_fts f ON i.id = f.id
        WHERE i.id = ?
        """, (item_id,))

        row = cursor.fetchone()
        if not row:
            return None

        return Item(
            id=row["id"],
            content=row["content"],
            metadata=eval(row["metadata"]) if row["metadata"] else {}
        )

    async def get_documents(self, item_ids: List[str]) -> List[Optional[Item]]:
        """Get multiple documents by ID.

        Args:
            item_ids: IDs of the documents to get

        Returns:
            List of documents (None for documents not found)
        """
        if not item_ids:
            return []

        # Flush buffer to ensure we have the latest data
        await self._flush_buffer()

        # Prepare placeholders for the query
        placeholders = ", ".join(["?"] * len(item_ids))

        cursor = self.conn.cursor()
        cursor.execute(f"""
        SELECT i.id, f.content, i.type, i.created_at, i.updated_at, i.metadata
        FROM items i
        JOIN items_fts f ON i.id = f.id
        WHERE i.id IN ({placeholders})
        """, item_ids)

        rows = cursor.fetchall()
        result = {row["id"]: Item(
            id=row["id"],
            content=row["content"],
            metadata=eval(row["metadata"]) if row["metadata"] else {}
        ) for row in rows}

        # Return items in the same order as item_ids
        return [result.get(item_id) for item_id in item_ids]

    async def update_document(self, item_id: str, item: Item) -> bool:
        """Update a document.

        Args:
            item_id: ID of the document to update
            item: New document data

        Returns:
            True if successful, False otherwise
        """
        async with self.write_lock:
            # Add to write buffer
            self.write_buffer.append(("update", item_id, item))

            # Flush buffer if it's full
            if len(self.write_buffer) >= self.buffer_size:
                await self._flush_buffer()

            return True

    async def update_documents(self, item_ids: List[str], items: List[Item]) -> List[bool]:
        """Update multiple documents.

        Args:
            item_ids: IDs of the documents to update
            items: New document data

        Returns:
            List of success flags
        """
        if not item_ids or not items or len(item_ids) != len(items):
            return [False] * len(item_ids)

        async with self.write_lock:
            # Add all items to write buffer
            for item_id, item in zip(item_ids, items):
                self.write_buffer.append(("update", item_id, item))

            # Flush buffer if it's full
            if len(self.write_buffer) >= self.buffer_size:
                await self._flush_buffer()

            return [True] * len(item_ids)

    async def delete_document(self, item_id: str) -> bool:
        """Delete a document.

        Args:
            item_id: ID of the document to delete

        Returns:
            True if successful, False otherwise
        """
        async with self.write_lock:
            # Add to write buffer
            self.write_buffer.append(("delete", item_id))

            # Flush buffer if it's full
            if len(self.write_buffer) >= self.buffer_size:
                await self._flush_buffer()

            return True

    async def delete_documents(self, item_ids: List[str]) -> List[bool]:
        """Delete multiple documents.

        Args:
            item_ids: IDs of the documents to delete

        Returns:
            List of success flags
        """
        if not item_ids:
            return []

        async with self.write_lock:
            # Add all items to write buffer
            for item_id in item_ids:
                self.write_buffer.append(("delete", item_id))

            # Flush buffer if it's full
            if len(self.write_buffer) >= self.buffer_size:
                await self._flush_buffer()

            return [True] * len(item_ids)

    async def search(self, query_text: str, top_k: int = 5, user_id: Optional[str] = None) -> List[QueryResult]:
        """Search the store.

        Args:
            query_text: Query text
            top_k: Number of results to return
            user_id: User ID to filter by (optional)

        Returns:
            List of query results
        """
        # Log search query
        logger.info(f"Searching keyword store for: {query_text}")
        if user_id:
            logger.info(f"Filtering by user_id: {user_id}")

        # Flush buffer to ensure we have the latest data
        await self._flush_buffer()

        # Count total documents in the store
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM items_fts")
        total_docs = cursor.fetchone()[0]
        logger.info(f"Total documents in keyword store: {total_docs}")

        # If no documents, return empty list
        if total_docs == 0:
            logger.warning("No documents in keyword store")
            return []

        # Prepare FTS5 query
        fts_query = self._prepare_fts_query(query_text)

        # Base SQL for all queries
        base_select = """
        SELECT i.id, f.content, i.type, i.metadata,
               {score_expr} AS score
        FROM items_fts f
        JOIN items i ON f.id = i.id
        """

        # Add user_id filter if provided
        user_filter = ""
        user_params = []
        if user_id:
            user_filter = " AND json_extract(i.metadata, '$.user_id') = ? "
            user_params = [user_id]

        # If query is empty or wildcard, return all documents
        if not fts_query or fts_query == "*":
            logger.info("Empty query or wildcard, returning all documents")
            query = base_select.format(
                score_expr="0.7") + user_filter + " LIMIT ?"
            cursor.execute(query, user_params + [top_k])
        else:
            # Try different query approaches
            try:
                # # First try: exact FTS5 query with optimized BM25 scoring
                # logger.info(
                #     f"Executing FTS query with BM25 (k1={self.k1}, b={self.b}): {fts_query}")

                # Use custom BM25 parameters if SQLite version supports it
                try:
                    # Try with custom BM25 parameters
                    query = base_select.format(score_expr="(1.0 / (1.0 + ABS(bm25(items_fts, ?, ?))))") + \
                        " WHERE items_fts MATCH ? " + user_filter + \
                        " ORDER BY score DESC LIMIT ?"
                    cursor.execute(
                        query, [self.k1, self.b, fts_query] + user_params + [top_k])
                except sqlite3.OperationalError:
                    # Fall back to default BM25 parameters
                    logger.warning(
                        "SQLite version doesn't support custom BM25 parameters, using defaults")
                    query = base_select.format(score_expr="(1.0 / (1.0 + ABS(bm25(items_fts))))") + \
                        " WHERE items_fts MATCH ? " + user_filter + \
                        " ORDER BY score DESC LIMIT ?"
                    cursor.execute(query, [fts_query] + user_params + [top_k])

                # Check if we got any results
                results = cursor.fetchall()
                if not results:
                    # Second try: fallback to simple LIKE query
                    logger.info("No FTS results, trying LIKE query")
                    like_terms = [f"%{term}%" for term in query_text.split()]
                    like_clause = " OR ".join(
                        ["f.content LIKE ?" for _ in like_terms])

                    query = base_select.format(score_expr="0.6") + \
                        f" WHERE ({like_clause}) " + user_filter + \
                        " LIMIT ?"

                    cursor.execute(query, like_terms + user_params + [top_k])
                    results = cursor.fetchall()

                    # If still no results, return all documents (with user filter)
                    if not results:
                        logger.info(
                            "No LIKE results either, returning all documents")
                        query = base_select.format(
                            score_expr="0.5") + user_filter + " LIMIT ?"
                        cursor.execute(query, user_params + [top_k])
                else:
                    # We got results from the first query, reset cursor position
                    query = base_select.format(score_expr="(1.0 / (1.0 + ABS(bm25(items_fts))))") + \
                        " WHERE items_fts MATCH ? " + user_filter + \
                        " ORDER BY score DESC LIMIT ?"
                    cursor.execute(query, [fts_query] + user_params + [top_k])

            except Exception as e:
                # If all queries fail, log error and return all documents
                logger.error(
                    f"All queries failed: {e}, returning all documents")
                query = base_select.format(
                    score_expr="0.5") + user_filter + " LIMIT ?"
                cursor.execute(query, user_params + [top_k])

        rows = cursor.fetchall()
        results = []

        for row in rows:
            try:
                metadata = json.loads(
                    row["metadata"]) if row["metadata"] else {}
            except (json.JSONDecodeError, TypeError):
                metadata = {}

            # Add retrieval method to metadata
            if "retrieval" not in metadata:
                metadata["retrieval"] = {}
            metadata["retrieval"]["method"] = "keyword"

            results.append(QueryResult(
                id=row["id"],
                content=row["content"],
                score=float(row["score"]),
                metadata=metadata,
                store_type=self.store_type
            ))

        return results

    def _prepare_fts_query(self, query_text: str) -> str:
        """Prepare a query for FTS5 with improved tokenization and query strategies.

        Args:
            query_text: Original query text

        Returns:
            FTS5 compatible query
        """
        # Log original query
        logger.info(f"Preparing FTS query for: {query_text}")

        # Handle empty query
        if not query_text or query_text.strip() == "":
            logger.warning("Empty query text")
            return "*"  # Match all documents

        # Handle wildcard query
        if query_text.strip() == "*":
            return "*"

        import re
        from collections import Counter

        # Convert to lowercase for case-insensitive matching
        query_text = query_text.lower()

        # Define stopwords (common words that add little value to search)
        stopwords = {
            'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were',
            'be', 'been', 'being', 'in', 'on', 'at', 'to', 'for', 'with',
            'by', 'about', 'of', 'from', 'as', 'into', 'like', 'through',
            'after', 'before', 'between', 'under', 'above', 'below', 'up',
            'down', 'out', 'off', 'over', 'under', 'again', 'further', 'then',
            'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
            'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
            'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
            'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should',
            'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren',
            'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma',
            'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren',
            'won', 'wouldn'
        }

        # Extract terms (words) from the query
        all_terms = re.findall(r'\b\w+\b', query_text)

        # Filter out stopwords for certain query types, but keep them for exact phrase matching
        terms = [
            term for term in all_terms if term not in stopwords or len(term) > 5]

        # If all terms were stopwords, use the original terms
        if not terms and all_terms:
            terms = all_terms

        if not terms:
            logger.warning(
                "No terms found after tokenization and stopword removal")
            return "*"  # Match all documents

        # Count term frequency to identify important terms
        term_counts = Counter(terms)
        important_terms = [
            term for term, count in term_counts.items() if count > 1 or len(term) > 3]

        # If no important terms, use all terms
        if not important_terms:
            important_terms = terms

        # Create different query strategies
        exact_phrase = []      # For exact phrase matching
        bigram_phrases = []    # For bigram phrase matching
        exact_terms = []       # For exact term matching
        near_terms = []        # For NEAR operator matching
        prefix_terms = []      # For prefix matching
        wildcard_terms = []    # For wildcard matching

        # Add exact phrase if original query is not too long
        if len(query_text) < 100:
            exact_phrase.append(f'"{query_text}"')

        # Create bigrams (pairs of adjacent words)
        if len(all_terms) >= 2:
            for i in range(len(all_terms) - 1):
                bigram = f'"{all_terms[i]} {all_terms[i+1]}"'
                bigram_phrases.append(bigram)

        # Process each term
        for term in terms:
            # Skip very short terms (less than 3 chars) for exact matching
            if len(term) >= 3:
                exact_terms.append(f'"{term}"')  # Exact match

                # Add wildcard for terms longer than 4 chars (handles plurals and variations)
                if len(term) > 4:
                    wildcard_terms.append(f'"{term}"*')

            # Add prefix matching for all terms
            prefix_terms.append(f"{term}*")

        # Add NEAR operator for term pairs (not just adjacent)
        if len(terms) >= 2:
            for i in range(len(terms)):
                # Consider terms within a window of 3
                for j in range(i + 1, min(i + 4, len(terms))):
                    if len(terms[i]) >= 3 and len(terms[j]) >= 3:
                        near_terms.append(f'"{terms[i]}" NEAR "{terms[j]}"')

        # Combine strategies with OR
        query_parts = []

        # Add exact phrase matching (highest priority)
        if exact_phrase:
            query_parts.append(" OR ".join(exact_phrase))

        # Add bigram phrase matching
        if bigram_phrases:
            query_parts.append(" OR ".join(bigram_phrases))

        # Add NEAR operator matching
        if near_terms:
            query_parts.append(" OR ".join(near_terms))

        # Add exact term matching
        if exact_terms:
            query_parts.append(" OR ".join(exact_terms))

        # Add individual terms (lowest priority)
        if terms:
            query_parts.append(" OR ".join(terms))

        # Add prefix matching
        if prefix_terms:
            query_parts.append(" OR ".join(prefix_terms))

        # Add wildcard matching
        if wildcard_terms:
            query_parts.append(" OR ".join(wildcard_terms))

        # Combine all strategies
        query = " OR ".join(query_parts)

        # logger.info(f"Prepared FTS query: {query}")
        return query

    async def _flush_buffer(self):
        """Flush the write buffer to the database."""
        if not self.write_buffer:
            return

        # Create a connection if it doesn't exist
        if not self.conn:
            await self.initialize()

        cursor = self.conn.cursor()

        try:
            # Begin transaction
            cursor.execute("BEGIN TRANSACTION")

            # Log buffer size
            logger.info(
                f"Flushing buffer with {len(self.write_buffer)} operations")

            for operation in self.write_buffer:
                if operation[0] == "add":
                    item = operation[1]
                    now = time.time()

                    # Log item being added
                    logger.info(f"Adding item to keyword store: {item.id}")
                    logger.info(f"Item content: {item.content[:50]}...")

                    # Insert into items table
                    cursor.execute("""
                    INSERT OR REPLACE INTO items (id, type, created_at, updated_at, metadata)
                    VALUES (?, ?, ?, ?, ?)
                    """, (
                        item.id,
                        item.metadata.get("type", "unknown"),
                        now,
                        now,
                        json.dumps(item.metadata)
                    ))

                    # Insert into FTS table
                    cursor.execute("""
                    INSERT OR REPLACE INTO items_fts (id, content, type, metadata)
                    VALUES (?, ?, ?, ?)
                    """, (
                        item.id,
                        item.content,
                        item.metadata.get("type", "unknown"),
                        json.dumps(item.metadata)
                    ))

                elif operation[0] == "update":
                    item_id, item = operation[1], operation[2]
                    now = time.time()

                    # Update items table
                    cursor.execute("""
                    UPDATE items
                    SET type = ?, updated_at = ?, metadata = ?
                    WHERE id = ?
                    """, (
                        item.metadata.get("type", "unknown"),
                        now,
                        json.dumps(item.metadata),
                        item_id
                    ))

                    # Update FTS table
                    cursor.execute("""
                    UPDATE items_fts
                    SET content = ?, type = ?, metadata = ?
                    WHERE id = ?
                    """, (
                        item.content,
                        item.metadata.get("type", "unknown"),
                        json.dumps(item.metadata),
                        item_id
                    ))

                elif operation[0] == "delete":
                    item_id = operation[1]

                    # Delete from items table
                    cursor.execute("DELETE FROM items WHERE id = ?", (item_id,))

                    # Delete from FTS table
                    cursor.execute(
                        "DELETE FROM items_fts WHERE id = ?", (item_id,))

            # Commit transaction
            self.conn.commit()

            # Clear buffer
            self.write_buffer.clear()

        except Exception as e:
            # Rollback on error
            self.conn.rollback()
            logger.error(f"Error flushing buffer: {e}")
            raise

    async def clear(self) -> bool:
        """Clear the keyword store.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Flush any remaining items in the buffer
            await self._flush_buffer()

            # Delete all items from the database
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM items")
            cursor.execute("DELETE FROM items_fts")
            self.conn.commit()

            # Clear query cache
            self.query_cache.clear()

            return True
        except Exception as e:
            logger.error(f"Error clearing keyword store: {e}")
            return False

    async def close(self):
        """Close the keyword store."""
        # Flush any remaining items in the buffer
        await self._flush_buffer()

        # Close the database connection
        if self.conn:
            self.conn.close()
            self.conn = None
