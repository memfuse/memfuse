"""Integration tests for PgaiStore with M0 layer."""

import pytest
import asyncio
import os
from typing import List
from unittest.mock import patch, AsyncMock

from src.memfuse_core.store.pgai_store import PgaiStore
from src.memfuse_core.hierarchy.layers import M0EpisodicLayer
from src.memfuse_core.rag.chunk.base import ChunkData
from src.memfuse_core.models.schema import MessageRecord


class TestPgaiIntegration:
    """Integration tests for PgaiStore with MemFuse components."""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for integration testing."""
        return {
            "database": {
                "postgres": {
                    "host": os.getenv("POSTGRES_HOST", "localhost"),
                    "port": int(os.getenv("POSTGRES_PORT", "5432")),
                    "database": os.getenv("POSTGRES_DB", "test_memfuse"),
                    "user": os.getenv("POSTGRES_USER", "postgres"),
                    "password": os.getenv("POSTGRES_PASSWORD", "password"),
                    "pool_size": 5
                },
                "pgai": {
                    "embedding_model": "text-embedding-3-small",
                    "embedding_dimensions": 1536,
                    "chunk_size": 500,
                    "chunk_overlap": 100,
                    "vectorizer_worker_enabled": False  # Disable for testing
                }
            },
            "store": {
                "backend": "pgai"
            },
            "memory": {
                "m0": {
                    "enabled": True,
                    "chunk_strategy": "simple"
                }
            }
        }
    
    @pytest.fixture
    def sample_messages(self):
        """Sample message data for testing."""
        return [
            MessageRecord(
                session_id="test-session-1",
                role="user",
                content="Hello, I need help with my project planning.",
                metadata={"timestamp": "2024-01-01T10:00:00Z"}
            ),
            MessageRecord(
                session_id="test-session-1", 
                role="assistant",
                content="I'd be happy to help you with project planning. What specific aspects would you like to focus on?",
                metadata={"timestamp": "2024-01-01T10:01:00Z"}
            ),
            MessageRecord(
                session_id="test-session-1",
                role="user", 
                content="I need to organize tasks and set deadlines for a software development project.",
                metadata={"timestamp": "2024-01-01T10:02:00Z"}
            )
        ]
    
    @pytest.fixture
    async def pgai_store_integration(self, mock_config):
        """Create a PgaiStore for integration testing."""
        with patch('src.memfuse_core.store.pgai_store.config_manager') as mock_config_manager:
            mock_config_manager.get_config.return_value = mock_config
            
            # Mock pgai availability and components for integration testing
            with patch('src.memfuse_core.store.pgai_store.PGAI_AVAILABLE', True):
                with patch('src.memfuse_core.store.pgai_store.pgai') as mock_pgai:
                    with patch('src.memfuse_core.store.pgai_store.AsyncConnectionPool') as mock_pool:
                        store = PgaiStore(table_name="integration_test_messages")
                        
                        # Mock successful initialization
                        mock_pool_instance = AsyncMock()
                        mock_pool.return_value = mock_pool_instance
                        store.pool = mock_pool_instance
                        store.initialized = True
                        
                        yield store
    
    @pytest.fixture
    async def m0_layer_with_pgai(self, pgai_store_integration, mock_config):
        """Create M0 layer with PgaiStore."""
        with patch('src.memfuse_core.utils.config.config_manager') as mock_config_manager:
            mock_config_manager.get_config.return_value = mock_config
            
            # Mock the storage manager to use our pgai store
            with patch('src.memfuse_core.hierarchy.layers.UnifiedStorageManager') as mock_storage_manager:
                mock_manager_instance = AsyncMock()
                mock_storage_manager.return_value = mock_manager_instance
                
                # Configure the mock to use pgai store
                mock_manager_instance.write_to_all.return_value = {
                    "pgai": ["chunk-1", "chunk-2", "chunk-3"]
                }
                
                layer = M0EpisodicLayer()
                layer.storage_manager = mock_manager_instance
                
                yield layer, mock_manager_instance
    
    @pytest.mark.asyncio
    async def test_pgai_store_with_chunking(self, pgai_store_integration, sample_messages):
        """Test PgaiStore integration with chunking process."""
        # Convert messages to chunks (simulate chunking process)
        chunks = []
        for i, message in enumerate(sample_messages):
            chunk = ChunkData(
                content=message.content,
                chunk_id=f"chunk-{i+1}",
                metadata={
                    "session_id": message.session_id,
                    "role": message.role,
                    "timestamp": message.metadata.get("timestamp"),
                    "message_index": i
                }
            )
            chunks.append(chunk)
        
        # Mock database operations
        mock_cursor = AsyncMock()
        mock_connection = AsyncMock()
        mock_connection.cursor.return_value.__aenter__.return_value = mock_cursor
        pgai_store_integration.pool.connection.return_value.__aenter__.return_value = mock_connection
        
        # Test adding chunks
        result = await pgai_store_integration.add(chunks)
        
        # Verify chunks were added
        assert len(result) == 3
        assert result == ["chunk-1", "chunk-2", "chunk-3"]
        
        # Verify database interactions
        assert mock_cursor.execute.call_count == 3
        mock_connection.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_m0_layer_pgai_integration(self, m0_layer_with_pgai, sample_messages):
        """Test M0 layer integration with PgaiStore."""
        layer, mock_storage_manager = m0_layer_with_pgai
        
        # Process messages through M0 layer
        result = await layer.process_data(sample_messages, {"user_id": "test-user"})
        
        # Verify the layer processed the data
        assert result is not None
        
        # Verify storage manager was called
        mock_storage_manager.write_to_all.assert_called_once()
        
        # Get the chunks that were passed to storage
        call_args = mock_storage_manager.write_to_all.call_args
        chunks_passed = call_args[0][0]  # First argument (chunks)
        metadata_passed = call_args[0][1]  # Second argument (metadata)
        
        # Verify chunks were created correctly
        assert len(chunks_passed) > 0
        assert all(isinstance(chunk, ChunkData) for chunk in chunks_passed)
        
        # Verify metadata was passed correctly
        assert metadata_passed["user_id"] == "test-user"
    
    @pytest.mark.asyncio
    async def test_pgai_store_session_queries(self, pgai_store_integration):
        """Test session-based queries with PgaiStore."""
        # Mock database responses for session queries
        mock_cursor = AsyncMock()
        mock_connection = AsyncMock()
        mock_connection.cursor.return_value.__aenter__.return_value = mock_cursor
        pgai_store_integration.pool.connection.return_value.__aenter__.return_value = mock_connection
        
        # Mock session query results
        mock_cursor.__aiter__.return_value = [
            ("chunk-1", "Hello, I need help with my project planning.", 
             '{"session_id": "test-session-1", "role": "user"}', None, None),
            ("chunk-2", "I'd be happy to help you with project planning.", 
             '{"session_id": "test-session-1", "role": "assistant"}', None, None)
        ]
        
        # Test session-based retrieval
        result = await pgai_store_integration.get_chunks_by_session("test-session-1")
        
        # Verify results
        assert len(result) == 2
        assert all(chunk.metadata.get("session_id") == "test-session-1" for chunk in result)
        assert result[0].metadata.get("role") == "user"
        assert result[1].metadata.get("role") == "assistant"
    
    @pytest.mark.asyncio
    async def test_pgai_store_statistics(self, pgai_store_integration):
        """Test statistics collection from PgaiStore."""
        # Mock database responses for statistics
        mock_cursor = AsyncMock()
        mock_connection = AsyncMock()
        mock_connection.cursor.return_value.__aenter__.return_value = mock_cursor
        pgai_store_integration.pool.connection.return_value.__aenter__.return_value = mock_connection
        
        # Mock statistics queries
        mock_cursor.fetchone.side_effect = [(10,)]  # Total count
        mock_cursor.__aiter__.side_effect = [
            [("test-session-1", 5), ("test-session-2", 3)],  # By session
            [("simple", 8), ("advanced", 2)],  # By strategy
            [("user-1", 6), ("user-2", 4)]  # By user
        ]
        
        # Get statistics
        stats = await pgai_store_integration.get_chunks_stats()
        
        # Verify statistics structure
        assert stats["total_chunks"] == 10
        assert "by_session" in stats
        assert "by_strategy" in stats
        assert "by_user" in stats
        assert stats["by_session"]["test-session-1"] == 5
        assert stats["by_strategy"]["simple"] == 8
        assert stats["by_user"]["user-1"] == 6
    
    @pytest.mark.asyncio
    async def test_pgai_vectorizer_configuration(self, mock_config):
        """Test pgai vectorizer configuration."""
        with patch('src.memfuse_core.store.pgai_store.config_manager') as mock_config_manager:
            mock_config_manager.get_config.return_value = mock_config
            
            with patch('src.memfuse_core.store.pgai_store.PGAI_AVAILABLE', True):
                with patch('src.memfuse_core.store.pgai_store.pgai') as mock_pgai:
                    with patch('src.memfuse_core.store.pgai_store.AsyncConnectionPool') as mock_pool:
                        store = PgaiStore()
                        
                        # Mock pool and cursor for vectorizer creation
                        mock_pool_instance = AsyncMock()
                        mock_cursor = AsyncMock()
                        mock_connection = AsyncMock()
                        mock_connection.cursor.return_value.__aenter__.return_value = mock_cursor
                        mock_pool_instance.connection.return_value.__aenter__.return_value = mock_connection
                        mock_pool.return_value = mock_pool_instance
                        
                        # Initialize store (this should create vectorizer)
                        await store.initialize()
                        
                        # Verify vectorizer creation was attempted
                        assert mock_cursor.execute.called
                        
                        # Check that vectorizer configuration uses config values
                        vectorizer_call = None
                        for call in mock_cursor.execute.call_args_list:
                            if "create_vectorizer" in str(call):
                                vectorizer_call = str(call)
                                break
                        
                        assert vectorizer_call is not None
                        assert "text-embedding-3-small" in vectorizer_call
                        assert "1536" in vectorizer_call
                        assert "500" in vectorizer_call  # chunk_size from config
    
    @pytest.mark.asyncio
    async def test_error_handling(self, pgai_store_integration):
        """Test error handling in PgaiStore operations."""
        # Mock database error
        mock_cursor = AsyncMock()
        mock_connection = AsyncMock()
        mock_connection.cursor.return_value.__aenter__.return_value = mock_cursor
        pgai_store_integration.pool.connection.return_value.__aenter__.return_value = mock_connection
        
        # Make cursor.execute raise an exception
        mock_cursor.execute.side_effect = Exception("Database connection error")
        
        # Test that errors are properly handled
        chunks = [ChunkData(content="test", chunk_id="test-1")]
        
        with pytest.raises(Exception, match="Database connection error"):
            await pgai_store_integration.add(chunks)
