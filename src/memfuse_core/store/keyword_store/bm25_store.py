"""BM25 keyword store implementation."""

import os
import json
import uuid
import sqlite3
import re
import math
from typing import List, Optional
from collections import Counter

from ...models.core import Item, QueryResult
from .base import KeywordStore
from ...utils.path_manager import PathManager


class BM25Store(KeywordStore):
    """BM25-based keyword store implementation.

    This implementation uses the BM25 algorithm for keyword-based retrieval.
    It provides efficient text search with minimal dependencies.
    """

    def __init__(
        self,
        data_dir: str,
        cache_size: int = 100,
        k1: float = 1.5,
        b: float = 0.75,
        **kwargs
    ):
        """Initialize the BM25 store.

        Args:
            data_dir: Directory to store data
            cache_size: Size of the query cache
            k1: BM25 parameter for term frequency scaling
            b: BM25 parameter for document length normalization
            **kwargs: Additional arguments
        """
        super().__init__(data_dir, cache_size=cache_size, **kwargs)
        self.k1 = k1
        self.b = b

        # Create BM25 store directory
        self.bm25_dir = os.path.join(data_dir, "bm25_store")

        # Initialize index
        self.documents = {}
        self.document_lengths = {}
        self.term_document_freq = {}
        self.total_documents = 0
        self.avg_document_length = 0

        # Initialize SQLite database
        self.db_path = os.path.join(self.bm25_dir, "bm25_store.db")
        self.conn = None
        self.cursor = None

    async def initialize(self) -> bool:
        """Initialize the BM25 store.

        Returns:
            True if successful, False otherwise
        """
        # Create directory if it doesn't exist
        PathManager.ensure_directory(self.bm25_dir)

        # Initialize SQLite database
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()

        # Create tables
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            metadata TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS terms (
            term TEXT NOT NULL,
            document_id TEXT NOT NULL,
            frequency INTEGER NOT NULL,
            PRIMARY KEY (term, document_id),
            FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
        )
        ''')

        self.cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_terms_term ON terms(term)
        ''')

        self.cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_terms_document_id ON terms(document_id)
        ''')

        # Load index from database
        await self._load_index()

        self.initialized = True
        return True

    async def _load_index(self):
        """Load index from database."""
        # Load documents
        self.cursor.execute('SELECT id, content, metadata FROM documents')
        rows = self.cursor.fetchall()

        for row in rows:
            document_id = row['id']
            content = row['content']
            metadata = json.loads(row['metadata'])

            self.documents[document_id] = Item(
                id=document_id,
                content=content,
                metadata=metadata
            )

            # Calculate document length (number of terms)
            terms = self._tokenize(content)
            self.document_lengths[document_id] = len(terms)

        # Load term frequencies
        self.cursor.execute('''
        SELECT term, document_id, frequency FROM terms
        ''')
        rows = self.cursor.fetchall()

        for row in rows:
            term = row['term']
            document_id = row['document_id']
            frequency = row['frequency']

            if term not in self.term_document_freq:
                self.term_document_freq[term] = {}

            self.term_document_freq[term][document_id] = frequency

        # Update statistics
        self.total_documents = len(self.documents)

        if self.total_documents > 0:
            self.avg_document_length = sum(
                self.document_lengths.values()) / self.total_documents
        else:
            self.avg_document_length = 0

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into terms.

        Args:
            text: Text to tokenize

        Returns:
            List of terms
        """
        # Convert to lowercase
        text = text.lower()

        # Remove punctuation and split into terms
        terms = re.findall(r'\w+', text)

        return terms

    def _index_document(self, document_id: str, content: str):
        """Index a document.

        Args:
            document_id: Document ID
            content: Document content
        """
        # Tokenize content
        terms = self._tokenize(content)

        # Calculate term frequencies
        term_freq = Counter(terms)

        # Update document length
        self.document_lengths[document_id] = len(terms)

        # Update term-document frequencies
        for term, freq in term_freq.items():
            if term not in self.term_document_freq:
                self.term_document_freq[term] = {}

            self.term_document_freq[term][document_id] = freq

            # Update database
            self.cursor.execute('''
            INSERT OR REPLACE INTO terms (term, document_id, frequency)
            VALUES (?, ?, ?)
            ''', (term, document_id, freq))

        # Update statistics
        self.total_documents = len(self.documents)

        if self.total_documents > 0:
            self.avg_document_length = sum(
                self.document_lengths.values()) / self.total_documents
        else:
            self.avg_document_length = 0

        # Commit changes
        self.conn.commit()

    def _calculate_bm25_score(self, query_terms: List[str], document_id: str) -> float:
        """Calculate BM25 score for a document.

        Args:
            query_terms: Query terms
            document_id: Document ID

        Returns:
            BM25 score
        """
        score = 0.0

        # Get document length
        doc_length = self.document_lengths.get(document_id, 0)

        if doc_length == 0:
            return 0.0

        # Calculate score for each query term
        for term in query_terms:
            if term not in self.term_document_freq:
                continue

            # Calculate IDF (Inverse Document Frequency)
            df = len(self.term_document_freq[term])
            idf = math.log((self.total_documents - df + 0.5) / (df + 0.5) + 1.0)

            # Get term frequency in document
            tf = self.term_document_freq[term].get(document_id, 0)

            # Calculate BM25 score for term
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * \
                (1 - self.b + self.b * doc_length / self.avg_document_length)

            score += idf * numerator / denominator

        return score

    async def add_document(self, item: Item) -> str:
        """Add a document to the store.

        Args:
            item: Item to add

        Returns:
            ID of the added document
        """
        # Ensure initialized
        await self.ensure_initialized()

        # Generate ID if not provided
        if not item.id:
            item.id = str(uuid.uuid4())

        # Add document to database
        self.cursor.execute('''
        INSERT OR REPLACE INTO documents (id, content, metadata)
        VALUES (?, ?, ?)
        ''', (item.id, item.content, json.dumps(item.metadata)))

        # Add document to in-memory store
        self.documents[item.id] = item

        # Index document
        self._index_document(item.id, item.content)

        # Invalidate query cache
        self.query_cache.clear()

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

        # Ensure initialized
        await self.ensure_initialized()

        # Generate IDs if not provided
        for item in items:
            if not item.id:
                item.id = str(uuid.uuid4())

        # Add documents to database
        self.cursor.executemany('''
        INSERT OR REPLACE INTO documents (id, content, metadata)
        VALUES (?, ?, ?)
        ''', [(item.id, item.content, json.dumps(item.metadata)) for item in items])

        # Add documents to in-memory store and index them
        for item in items:
            self.documents[item.id] = item
            self._index_document(item.id, item.content)

        # Invalidate query cache
        self.query_cache.clear()

        return [item.id for item in items]

    async def get_document(self, item_id: str) -> Optional[Item]:
        """Get a document by ID.

        Args:
            item_id: ID of the document to get

        Returns:
            Document if found, None otherwise
        """
        # Ensure initialized
        await self.ensure_initialized()

        # Check in-memory store first
        if item_id in self.documents:
            return self.documents[item_id]

        # Query database
        self.cursor.execute('''
        SELECT id, content, metadata FROM documents
        WHERE id = ?
        ''', (item_id,))

        row = self.cursor.fetchone()

        if row:
            # Create item
            item = Item(
                id=row['id'],
                content=row['content'],
                metadata=json.loads(row['metadata'])
            )

            # Add to in-memory store
            self.documents[item_id] = item

            return item

        return None

    async def get_documents(self, item_ids: List[str]) -> List[Optional[Item]]:
        """Get multiple documents by ID.

        Args:
            item_ids: IDs of the documents to get

        Returns:
            List of documents (None for documents not found)
        """
        if not item_ids:
            return []

        # Ensure initialized
        await self.ensure_initialized()

        # Get documents
        results = []

        for item_id in item_ids:
            item = await self.get_document(item_id)
            results.append(item)

        return results

    async def update_document(self, item_id: str, item: Item) -> bool:
        """Update a document.

        Args:
            item_id: ID of the document to update
            item: New document data

        Returns:
            True if successful, False otherwise
        """
        # Ensure initialized
        await self.ensure_initialized()

        # Check if document exists
        if item_id not in self.documents:
            return False

        # Update document ID
        item.id = item_id

        # Update database
        self.cursor.execute('''
        UPDATE documents
        SET content = ?, metadata = ?
        WHERE id = ?
        ''', (item.content, json.dumps(item.metadata), item_id))

        # Update in-memory store
        self.documents[item_id] = item

        # Re-index document
        self._index_document(item_id, item.content)

        # Invalidate query cache
        self.query_cache.clear()

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

        # Ensure initialized
        await self.ensure_initialized()

        # Update documents
        results = []

        for item_id, item in zip(item_ids, items):
            success = await self.update_document(item_id, item)
            results.append(success)

        return results

    async def delete_document(self, item_id: str) -> bool:
        """Delete a document.

        Args:
            item_id: ID of the document to delete

        Returns:
            True if successful, False otherwise
        """
        # Ensure initialized
        await self.ensure_initialized()

        # Check if document exists
        if item_id not in self.documents:
            return False

        # Delete from database
        self.cursor.execute('''
        DELETE FROM documents
        WHERE id = ?
        ''', (item_id,))

        # Delete from in-memory store
        del self.documents[item_id]

        # Delete from document lengths
        if item_id in self.document_lengths:
            del self.document_lengths[item_id]

        # Delete from term-document frequencies
        for term in list(self.term_document_freq.keys()):
            if item_id in self.term_document_freq[term]:
                del self.term_document_freq[term][item_id]

                # Remove term if no documents left
                if not self.term_document_freq[term]:
                    del self.term_document_freq[term]

        # Update statistics
        self.total_documents = len(self.documents)

        if self.total_documents > 0:
            self.avg_document_length = sum(
                self.document_lengths.values()) / self.total_documents
        else:
            self.avg_document_length = 0

        # Commit changes
        self.conn.commit()

        # Invalidate query cache
        self.query_cache.clear()

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

        # Ensure initialized
        await self.ensure_initialized()

        # Delete documents
        results = []

        for item_id in item_ids:
            success = await self.delete_document(item_id)
            results.append(success)

        return results

    async def search(self, query_text: str, top_k: int = 5, user_id: Optional[str] = None) -> List[QueryResult]:
        """Search the store.

        Args:
            query_text: Query text
            top_k: Number of results to return
            user_id: User ID to filter by (optional)

        Returns:
            List of query results
        """
        # Ensure initialized
        await self.ensure_initialized()

        # Tokenize query
        query_terms = self._tokenize(query_text)

        # Calculate scores for all documents
        scores = {}

        for document_id in self.documents:
            # Filter by user_id if provided
            if user_id and self.documents[document_id].metadata.get("user_id") != user_id:
                continue

            score = self._calculate_bm25_score(query_terms, document_id)

            if score > 0:
                scores[document_id] = score

        # Sort by score
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Get top-k results
        results = []

        for document_id, score in sorted_scores[:top_k]:
            document = self.documents[document_id]

            result = QueryResult(
                id=document_id,
                content=document.content,
                metadata=document.metadata,
                score=score,
                store_type=self.store_type
            )

            results.append(result)

        return results

    async def clear(self) -> bool:
        """Clear the store.

        Returns:
            True if successful, False otherwise
        """
        # Ensure initialized
        await self.ensure_initialized()

        # Clear database
        self.cursor.execute('DELETE FROM documents')
        self.cursor.execute('DELETE FROM terms')

        # Clear in-memory store
        self.documents.clear()
        self.document_lengths.clear()
        self.term_document_freq.clear()

        # Reset statistics
        self.total_documents = 0
        self.avg_document_length = 0

        # Commit changes
        self.conn.commit()

        # Invalidate query cache
        self.query_cache.clear()

        return True
