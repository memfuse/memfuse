"""Unit tests for PgaiStore implementation."""

import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any

from src.memfuse_core.store.pgai_store import PgaiStore
from src.memfuse_core.rag.chunk.base import ChunkData
from src.memfuse_core.models import Query


class MockAsyncContextManager:
    """Mock async context manager for testing."""

    def __init__(self, return_value):
        self.return_value = return_value

    async def __aenter__(self):
        return self.return_value

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None


class TestPgaiStore:
    """Test cases for PgaiStore."""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing."""
        return {
            "database": {
                "postgres": {
                    "host": "localhost",
                    "port": 5432,
                    "database": "test_memfuse",
                    "user": "test_user",
                    "password": "test_password",
                    "pool_size": 5
                },
                "pgai": {
                    "embedding_model": "text-embedding-3-small",
                    "embedding_dimensions": 1536,
                    "chunk_size": 1000,
                    "chunk_overlap": 200,
                    "vectorizer_worker_enabled": False  # Disable for testing
                }
            }
        }
    
    @pytest.fixture
    def sample_chunks(self):
        """Sample chunk data for testing."""
        return [
            ChunkData(
                content="This is the first test chunk",
                chunk_id="chunk-1",
                metadata={"session_id": "session-1", "user_id": "user-1"}
            ),
            ChunkData(
                content="This is the second test chunk",
                chunk_id="chunk-2", 
                metadata={"session_id": "session-1", "user_id": "user-1"}
            ),
            ChunkData(
                content="This is a chunk from different session",
                chunk_id="chunk-3",
                metadata={"session_id": "session-2", "user_id": "user-2"}
            )
        ]
    
    @pytest.fixture
    async def pgai_store(self, mock_config):
        """Create a PgaiStore instance for testing."""
        with patch('src.memfuse_core.store.pgai_store.config_manager') as mock_config_manager:
            mock_config_manager.get_config.return_value = mock_config

            with patch('src.memfuse_core.store.pgai_store.PGAI_AVAILABLE', True):
                store = PgaiStore(table_name="test_m0_raw")

                # Create proper async context manager mocks
                mock_cursor = AsyncMock()
                mock_connection = AsyncMock()
                mock_pool = AsyncMock()

                # Setup connection methods
                mock_connection.commit = AsyncMock()
                mock_connection.rollback = AsyncMock()
                mock_connection.cursor = lambda: MockAsyncContextManager(mock_cursor)

                # Setup pool connection method to return the context manager directly
                mock_pool.connection = lambda: MockAsyncContextManager(mock_connection)

                # Assign the properly mocked pool
                store.pool = mock_pool
                store.initialized = True

                yield store
    
    @pytest.mark.asyncio
    async def test_initialization(self, mock_config):
        """Test store initialization."""
        with patch('src.memfuse_core.store.pgai_store.config_manager') as mock_config_manager:
            mock_config_manager.get_config.return_value = mock_config
            
            with patch('src.memfuse_core.store.pgai_store.PGAI_AVAILABLE', True):
                with patch('src.memfuse_core.store.pgai_store.pgai') as mock_pgai:
                    with patch('src.memfuse_core.store.pgai_store.AsyncConnectionPool') as mock_pool:
                        store = PgaiStore()
                        
                        # Mock pool and connection with proper async context managers
                        mock_cursor = AsyncMock()
                        mock_connection = AsyncMock()
                        mock_pool_instance = AsyncMock()

                        # Setup connection methods
                        mock_connection.commit = AsyncMock()
                        mock_connection.rollback = AsyncMock()
                        mock_connection.cursor = lambda: MockAsyncContextManager(mock_cursor)

                        # Setup pool connection method
                        mock_pool_instance.connection = lambda: MockAsyncContextManager(mock_connection)
                        mock_pool.return_value = mock_pool_instance
                        
                        result = await store.initialize()
                        
                        assert result is True
                        assert store.initialized is True
                        mock_pgai.install.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_add_chunks(self, pgai_store, sample_chunks):
        """Test adding chunks to the store."""
        # Execute the add operation (mocks are set up in fixture)
        result = await pgai_store.add(sample_chunks)

        # Verify results
        assert len(result) == 3
        assert result == ["chunk-1", "chunk-2", "chunk-3"]

        # Basic verification that the operation completed successfully
        # (Detailed mock verification would require accessing fixture internals)
    
    @pytest.mark.asyncio
    async def test_read_chunks(self, pgai_store, sample_chunks):
        """Test reading chunks by IDs."""
        # Mock database responses
        mock_cursor = AsyncMock()
        mock_connection = AsyncMock()
        mock_connection.cursor.return_value.__aenter__.return_value = mock_cursor
        pgai_store.pool.connection.return_value.__aenter__.return_value = mock_connection
        
        # Mock fetchone to return sample data
        mock_cursor.fetchone.side_effect = [
            ("chunk-1", "This is the first test chunk", '{"session_id": "session-1", "user_id": "user-1"}', None, None),
            ("chunk-2", "This is the second test chunk", '{"session_id": "session-1", "user_id": "user-1"}', None, None),
            None  # chunk-3 not found
        ]
        
        # Execute the read operation
        result = await pgai_store.read(["chunk-1", "chunk-2", "chunk-3"])
        
        # Verify results
        assert len(result) == 3
        assert result[0] is not None
        assert result[0].chunk_id == "chunk-1"
        assert result[0].content == "This is the first test chunk"
        assert result[1] is not None
        assert result[1].chunk_id == "chunk-2"
        assert result[2] is None  # Not found
    
    @pytest.mark.asyncio
    async def test_query_chunks(self, pgai_store):
        """Test querying chunks with text search."""
        # Mock database responses
        mock_cursor = AsyncMock()
        mock_connection = AsyncMock()
        mock_connection.cursor.return_value.__aenter__.return_value = mock_cursor
        pgai_store.pool.connection.return_value.__aenter__.return_value = mock_connection
        
        # Mock async iteration
        mock_cursor.__aiter__.return_value = [
            ("chunk-1", "This is a test chunk", '{"session_id": "session-1"}', None, None)
        ]
        
        # Create a mock query
        query = Query(text="test chunk")
        
        # Execute the query
        result = await pgai_store.query(query, top_k=5)
        
        # Verify results
        assert len(result) == 1
        assert result[0].chunk_id == "chunk-1"
        assert result[0].content == "This is a test chunk"
        
        # Verify database call
        mock_cursor.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_count_chunks(self, pgai_store):
        """Test counting chunks in the store."""
        # Mock database response
        mock_cursor = AsyncMock()
        mock_connection = AsyncMock()
        mock_connection.cursor.return_value.__aenter__.return_value = mock_cursor
        pgai_store.pool.connection.return_value.__aenter__.return_value = mock_connection
        
        mock_cursor.fetchone.return_value = (42,)
        
        # Execute count
        result = await pgai_store.count()
        
        # Verify result
        assert result == 42
        mock_cursor.execute.assert_called_once_with("SELECT COUNT(*) FROM test_m0_raw")
    
    @pytest.mark.asyncio
    async def test_get_chunks_by_session(self, pgai_store):
        """Test getting chunks by session ID."""
        # Mock database responses
        mock_cursor = AsyncMock()
        mock_connection = AsyncMock()
        mock_connection.cursor.return_value.__aenter__.return_value = mock_cursor
        pgai_store.pool.connection.return_value.__aenter__.return_value = mock_connection
        
        # Mock async iteration
        mock_cursor.__aiter__.return_value = [
            ("chunk-1", "First chunk", '{"session_id": "session-1"}', None, None),
            ("chunk-2", "Second chunk", '{"session_id": "session-1"}', None, None)
        ]
        
        # Execute the query
        result = await pgai_store.get_chunks_by_session("session-1")
        
        # Verify results
        assert len(result) == 2
        assert all(chunk.metadata.get("session_id") == "session-1" for chunk in result)
    
    @pytest.mark.asyncio
    async def test_delete_chunks(self, pgai_store):
        """Test deleting chunks by IDs."""
        # Mock database responses
        mock_cursor = AsyncMock()
        mock_connection = AsyncMock()
        mock_connection.cursor.return_value.__aenter__.return_value = mock_cursor
        pgai_store.pool.connection.return_value.__aenter__.return_value = mock_connection
        
        # Mock rowcount for successful deletions
        mock_cursor.rowcount = 1
        
        # Execute delete
        result = await pgai_store.delete(["chunk-1", "chunk-2"])
        
        # Verify results
        assert result == [True, True]
        assert mock_cursor.execute.call_count == 2
        mock_connection.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_clear_store(self, pgai_store):
        """Test clearing all chunks from the store."""
        # Mock database responses
        mock_cursor = AsyncMock()
        mock_connection = AsyncMock()
        mock_connection.cursor.return_value.__aenter__.return_value = mock_cursor
        pgai_store.pool.connection.return_value.__aenter__.return_value = mock_connection
        
        # Execute clear
        result = await pgai_store.clear()
        
        # Verify result
        assert result is True
        mock_cursor.execute.assert_called_once_with("DELETE FROM test_m0_raw")
        mock_connection.commit.assert_called_once()
    
    def test_database_url_construction(self, mock_config):
        """Test database URL construction from config."""
        with patch('src.memfuse_core.store.pgai_store.config_manager') as mock_config_manager:
            mock_config_manager.get_config.return_value = mock_config
            
            with patch('src.memfuse_core.store.pgai_store.PGAI_AVAILABLE', True):
                store = PgaiStore()
                expected_url = "postgresql://test_user:test_password@localhost:5432/test_memfuse"
                assert store.db_url == expected_url
    
    def test_pgai_not_available(self):
        """Test error when pgai dependencies are not available."""
        with patch('src.memfuse_core.store.pgai_store.PGAI_AVAILABLE', False):
            with pytest.raises(ImportError, match="pgai dependencies required"):
                PgaiStore()
