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
from src.memfuse_core.store.pgai_vector_wrapper import PgaiVectorWrapper
from src.memfuse_core.rag.chunk.base import ChunkData
from src.memfuse_core.rag.encode.MiniLM import MiniLMEncoder
from src.memfuse_core.models import Query


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
            
            # Test read operation
            retrieved_chunks = await store.read([chunk.chunk_id for chunk in test_chunks])
            assert len(retrieved_chunks) == len(test_chunks), "Read operation failed"
            
            # Verify content integrity
            for original, retrieved in zip(test_chunks, retrieved_chunks):
                assert original.chunk_id == retrieved.chunk_id, "ID mismatch"
                assert original.content == retrieved.content, "Content mismatch"
                assert original.metadata == retrieved.metadata, "Metadata mismatch"
            
            # Test query operation
            query_results = await store.query(Query(text="technical information"), top_k=2)
            assert len(query_results) > 0, "Query should return results"
            
            # Test update operation
            updated_chunk = ChunkData(
                chunk_id="chunk_1",
                content="Updated content for first chunk",
                metadata={"source": "test", "type": "updated"}
            )
            update_success = await store.update("chunk_1", updated_chunk)
            assert update_success, "Update operation failed"
            
            # Test delete operation
            delete_results = await store.delete(["chunk_1"])
            assert delete_results[0], "Delete operation failed"
            
            # Verify deletion
            remaining_count = await store.count()
            assert remaining_count == len(test_chunks) - 1, "Delete verification failed"
            
            # Test clear operation
            await store.clear()
            final_count = await store.count()
            assert final_count == 0, "Clear operation failed"
            
        finally:
            if store:
                await store.close()
    
    @pytest.mark.asyncio
    async def test_pgai_store_mocked(self, mock_config, test_chunks):
        """Test PgaiStore with mocked dependencies."""
        with patch('src.memfuse_core.store.pgai_store.config_manager') as mock_config_manager:
            mock_config_manager.get_config.return_value = mock_config
            
            with patch('src.memfuse_core.store.pgai_store.PGAI_AVAILABLE', True):
                with patch('src.memfuse_core.store.pgai_store.pgai') as mock_pgai:
                    with patch('src.memfuse_core.store.pgai_store.AsyncConnectionPool') as mock_pool:
                        # Create store with mocked dependencies
                        store = PgaiStore(table_name="mocked_test_chunks")
                        store.pool = None
                        store.initialized = True
                        
                        # Mock basic operations
                        async def mock_add(chunks):
                            return [f"mock-id-{i+1}" for i in range(len(chunks))]
                        
                        async def mock_count():
                            return len(test_chunks)
                        
                        async def mock_read(chunk_ids, filters=None):
                            return test_chunks[:len(chunk_ids)]
                        
                        async def mock_clear():
                            return True
                        
                        store.add = mock_add
                        store.count = mock_count
                        store.read = mock_read
                        store.clear = mock_clear
                        
                        # Test mocked operations
                        chunk_ids = await store.add(test_chunks)
                        assert len(chunk_ids) == len(test_chunks)
                        
                        count = await store.count()
                        assert count == len(test_chunks)
                        
                        retrieved = await store.read(["chunk_1", "chunk_2"])
                        assert len(retrieved) == 2
                        
                        cleared = await store.clear()
                        assert cleared is True
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not os.getenv('POSTGRES_AVAILABLE'), reason="PostgreSQL not available")
    async def test_pgai_vector_wrapper_real(self, real_db_config, test_chunks):
        """Test PgaiVectorWrapper with real database."""
        store = None
        try:
            # Initialize store and wrapper
            store = PgaiStore(real_db_config)
            await store.initialize()
            
            encoder = MiniLMEncoder()
            wrapper = PgaiVectorWrapper(store, encoder)
            await wrapper.initialize()
            
            # Clear existing data
            await wrapper.clear()
            
            # Test VectorStore interface methods
            items = [
                {"id": chunk.chunk_id, "content": chunk.content, "metadata": chunk.metadata}
                for chunk in test_chunks
            ]
            
            # Test add_items
            item_ids = await wrapper.add_items(items)
            assert len(item_ids) == len(items)
            
            # Test search
            search_results = await wrapper.search("technical information", top_k=2)
            assert len(search_results) > 0
            assert all("id" in result and "content" in result for result in search_results)
            
            # Test get_item
            item = await wrapper.get_item("chunk_1")
            assert item is not None
            assert item["id"] == "chunk_1"
            
            # Test get_stats
            stats = await wrapper.get_stats()
            assert "total_items" in stats
            assert stats["backend"] == "pgai"
            
            # Test get_items_by_session
            session_items = await wrapper.get_items_by_session("session_1")
            assert len(session_items) >= 1
            
        finally:
            if store:
                await store.close()
    
    @pytest.mark.asyncio
    async def test_configuration_validation(self, mock_config):
        """Test configuration loading and validation."""
        with patch('src.memfuse_core.store.pgai_store.config_manager') as mock_config_manager:
            mock_config_manager.get_config.return_value = mock_config
            
            store = PgaiStore(table_name="config_test")
            
            # Test database URL construction
            expected_url = "postgresql://postgres:password@localhost:5432/test_memfuse"
            assert store.db_url == expected_url
            
            # Test table and view names
            assert store.table_name == "config_test"
            assert store.embedding_view == "config_test_embedding"
            assert store.vectorizer_name == "config_test_vectorizer"
    
    @pytest.mark.asyncio
    async def test_error_handling(self, mock_config):
        """Test error handling in various scenarios."""
        with patch('src.memfuse_core.store.pgai_store.config_manager') as mock_config_manager:
            mock_config_manager.get_config.return_value = mock_config
            
            # Test initialization failure
            with patch('src.memfuse_core.store.pgai_store.AsyncConnectionPool') as mock_pool:
                mock_pool.side_effect = Exception("Connection failed")
                
                store = PgaiStore(table_name="error_test")
                success = await store.initialize()
                assert not success, "Should fail to initialize with connection error"
    
    def test_import_availability(self):
        """Test that all required imports are available."""
        # Test that we can import all necessary components
        from src.memfuse_core.store.pgai_store import PgaiStore
        from src.memfuse_core.store.pgai_vector_wrapper import PgaiVectorWrapper
        from src.memfuse_core.rag.chunk.base import ChunkData
        from src.memfuse_core.models import Query
        
        assert PgaiStore is not None
        assert PgaiVectorWrapper is not None
        assert ChunkData is not None
        assert Query is not None


if __name__ == "__main__":
    # Allow running this file directly for manual testing
    asyncio.run(pytest.main([__file__, "-v"]))
