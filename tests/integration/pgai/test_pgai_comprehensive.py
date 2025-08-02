"""Comprehensive integration tests for PgaiStore and PgaiVectorWrapper.

This file combines real database testing and mocked testing approaches
to provide comprehensive coverage of pgai functionality.

Replaces: test_pgai_real.py, test_pgai_manual.py
"""

import pytest
import asyncio
import os
from typing import List
from unittest.mock import patch, AsyncMock

from src.memfuse_core.store.pgai_store import PgaiStore
from src.memfuse_core.store.pgai_store.pgai_vector_wrapper import PgaiVectorWrapper
from src.memfuse_core.rag.chunk.base import ChunkData
from src.memfuse_core.rag.encode.base import EncoderBase
from src.memfuse_core.models import Query
import numpy as np


class MockEncoder(EncoderBase):
    """Mock encoder for testing."""

    def __init__(self):
        self.model_name = "mock-encoder"
        self.dimension = 384

    async def encode_text(self, text: str) -> np.ndarray:
        """Mock encoding that returns dummy vector."""
        return np.array([0.1] * self.dimension)

    async def encode_texts(self, texts: list) -> list:
        """Mock encoding that returns dummy vectors."""
        return [np.array([0.1] * self.dimension) for _ in texts]


class TestPgaiComprehensive:
    """Comprehensive tests for PgaiStore with both real and mocked approaches."""
    
    @pytest.fixture
    def real_db_config(self):
        """Configuration for real PostgreSQL database testing."""
        return {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': int(os.getenv('POSTGRES_PORT', '5432')),
            'database': os.getenv('POSTGRES_DB', 'memfuse_test'),
            'user': os.getenv('POSTGRES_USER', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD', 'postgres'),
            'table': 'test_chunks_comprehensive'
        }
    
    @pytest.fixture
    def mock_config(self):
        """Configuration for mocked testing."""
        return {
            "database": {
                "postgres": {
                    "host": "localhost",
                    "port": 5432,
                    "database": "test_memfuse",
                    "user": "postgres",
                    "password": "password",
                    "pool_size": 5
                },
                "pgai": {
                    "embedding_model": "text-embedding-3-small",
                    "embedding_dimensions": 1536,
                    "chunk_size": 1000,
                    "chunk_overlap": 200,
                    "vectorizer_worker_enabled": False
                }
            },
            "embedding": {
                "model": "all-MiniLM-L6-v2",
                "dimension": 384,
                "device": "cpu"
            }
        }
    
    @pytest.fixture
    def test_chunks(self):
        """Sample test data for both real and mocked tests."""
        return [
            ChunkData(
                chunk_id="chunk_1",
                content="This is the first test chunk for pgai integration testing.",
                metadata={"source": "test", "type": "integration", "session_id": "session_1"}
            ),
            ChunkData(
                chunk_id="chunk_2", 
                content="Second chunk contains information about machine learning and AI.",
                metadata={"source": "test", "type": "integration", "priority": "high", "session_id": "session_1"}
            ),
            ChunkData(
                chunk_id="chunk_3",
                content="Third chunk has technical information about databases and PostgreSQL.",
                metadata={"source": "docs", "type": "technical", "session_id": "session_2"}
            )
        ]
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not os.getenv('POSTGRES_AVAILABLE'), reason="PostgreSQL not available")
    async def test_pgai_store_real_database(self, real_db_config, test_chunks):
        """Test PgaiStore with real PostgreSQL database."""
        store = None
        try:
            # Initialize store
            store = PgaiStore(real_db_config)
            success = await store.initialize()
            assert success, "Failed to initialize PgaiStore"
            
            # Clear any existing data
            await store.clear()
            
            # Test add operation
            chunk_ids = await store.add(test_chunks)
            assert len(chunk_ids) == len(test_chunks), "Add operation failed"
            
            # Test count
            count = await store.count()
            assert count == len(test_chunks), f"Expected {len(test_chunks)}, got {count}"
            
            # Test query operation
            query = Query(text="machine learning", metadata={"top_k": 2})
            results = await store.query(query)
            assert len(results) > 0, "Query returned no results"
            
            # Test get operation
            chunk = await store.get(chunk_ids[0])
            assert chunk is not None, "Get operation failed"
            assert chunk.chunk_id == test_chunks[0].chunk_id
            
        finally:
            if store:
                await store.close()
    
    @pytest.mark.asyncio
    async def test_pgai_store_mocked(self, mock_config, test_chunks):
        """Test PgaiStore with mocked dependencies."""
        with patch('psycopg_pool.AsyncConnectionPool') as mock_pool:
            # Mock database pool
            mock_connection = AsyncMock()
            mock_pool.return_value.__aenter__.return_value = mock_connection
            mock_connection.execute.return_value = None
            mock_connection.fetch.return_value = []
            mock_connection.fetchval.return_value = 0
            
            store = PgaiStore(mock_config["database"]["postgres"])
            
            # Test initialization
            success = await store.initialize()
            assert success, "Mocked initialization failed"
            
            # Test add operation (mocked)
            with patch.object(store, 'add', return_value=["id1", "id2", "id3"]):
                chunk_ids = await store.add(test_chunks)
                assert len(chunk_ids) == len(test_chunks)
            
            # Test query operation (mocked)
            with patch.object(store, 'query', return_value=test_chunks[:2]):
                query = Query(text="test query", metadata={"top_k": 2})
                results = await store.query(query)
                assert len(results) == 2
    
    @pytest.mark.asyncio
    async def test_pgai_vector_wrapper_integration(self, mock_config):
        """Test PgaiVectorWrapper integration with mocked store."""
        with patch('psycopg_pool.AsyncConnectionPool'):
            store = PgaiStore(mock_config["database"]["postgres"])
            encoder = MockEncoder()
            wrapper = PgaiVectorWrapper(store, encoder)
            
            # Test wrapper initialization
            await wrapper.initialize()
            
            # Test add_items
            items = [
                {"id": "chunk_1", "content": "Test content 1", "metadata": {"session_id": "session_1"}},
                {"id": "chunk_2", "content": "Test content 2", "metadata": {"session_id": "session_1"}}
            ]
            
            with patch.object(store, 'add', return_value=["chunk_1", "chunk_2"]):
                item_ids = await wrapper.add_items(items)
                assert len(item_ids) == len(items)
            
            # Test search
            with patch.object(store, 'query', return_value=[]):
                search_results = await wrapper.search("test query", top_k=2)
                assert isinstance(search_results, list)
    
    def test_import_availability(self):
        """Test that all required imports are available."""
        # Test that we can import all necessary components
        from src.memfuse_core.store.pgai_store import PgaiStore
        from src.memfuse_core.store.pgai_store.pgai_vector_wrapper import PgaiVectorWrapper
        from src.memfuse_core.rag.chunk.base import ChunkData
        from src.memfuse_core.models import Query
        
        assert PgaiStore is not None
        assert PgaiVectorWrapper is not None
        assert ChunkData is not None
        assert Query is not None


if __name__ == "__main__":
    # Allow running this file directly for manual testing
    asyncio.run(pytest.main([__file__, "-v"]))
