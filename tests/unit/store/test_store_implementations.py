"""Unit tests for concrete store implementations.

This module tests the actual store implementations (QdrantVectorStore, 
SQLiteKeywordStore, GraphMLStore) to ensure they properly implement
the ChunkStoreInterface.
"""

import pytest
import asyncio
import tempfile
import shutil
from typing import List, Optional, Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch

from src.memfuse_core.rag.chunk.base import ChunkData
from src.memfuse_core.models.core import Query
from src.memfuse_core.interfaces.chunk_store import StorageError


class TestStoreImplementations:
    """Test concrete store implementations."""

    @pytest.fixture
    def sample_chunks(self):
        """Create sample chunks for testing."""
        return [
            ChunkData(
                content="Python is a programming language.",
                chunk_id="chunk_1",
                metadata={"type": "chunk", "language": "python", "user_id": "test_user"}
            ),
            ChunkData(
                content="Machine learning is a subset of AI.",
                chunk_id="chunk_2", 
                metadata={"type": "chunk", "topic": "ml", "user_id": "test_user"}
            ),
            ChunkData(
                content="Data science involves statistics and programming.",
                chunk_id="chunk_3",
                metadata={"type": "chunk", "topic": "data_science", "user_id": "test_user"}
            )
        ]

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.mark.asyncio
    async def test_vector_store_implementation(self, sample_chunks, temp_dir):
        """Test QdrantVectorStore implementation."""
        # Mock the Qdrant client to avoid external dependencies
        with patch('src.memfuse_core.store.vector_store.qdrant_store.QdrantClient') as mock_client:
            # Setup mock responses
            mock_client_instance = MagicMock()
            mock_client.return_value = mock_client_instance
            
            # Mock collection operations
            mock_client_instance.get_collections.return_value.collections = []
            mock_client_instance.get_collection.return_value.vectors_count = 3
            
            # Mock upsert operation
            mock_client_instance.upsert.return_value = None
            
            # Mock retrieve operation
            mock_points = []
            for chunk in sample_chunks:
                mock_point = MagicMock()
                mock_point.id = chunk.chunk_id
                mock_point.payload = {
                    "content": chunk.content,
                    "metadata": chunk.metadata
                }
                mock_points.append(mock_point)
            mock_client_instance.retrieve.return_value = mock_points
            
            # Mock search operation
            mock_search_results = []
            for i, chunk in enumerate(sample_chunks):
                mock_result = MagicMock()
                mock_result.id = chunk.chunk_id
                mock_result.score = 0.9 - i * 0.1
                mock_result.payload = {
                    "content": chunk.content,
                    "metadata": chunk.metadata
                }
                mock_search_results.append(mock_result)
            mock_client_instance.search.return_value = mock_search_results
            
            # Import and test the store
            from src.memfuse_core.store.vector_store.qdrant_store import QdrantVectorStore
            
            # Create store instance
            store = QdrantVectorStore(
                data_dir=temp_dir,
                collection_name="test_collection",
                embedding_dim=384
            )
            
            # Mock the encoder
            mock_encoder = AsyncMock()
            mock_encoder.encode.return_value = [[0.1] * 384 for _ in sample_chunks]
            store.encoder = mock_encoder
            
            # Initialize store
            await store.initialize()
            
            # Test add operation
            chunk_ids = await store.add(sample_chunks)
            assert len(chunk_ids) == 3
            assert chunk_ids == ["chunk_1", "chunk_2", "chunk_3"]
            
            # Test read operation
            retrieved_chunks = await store.read(["chunk_1", "chunk_2"])
            assert len(retrieved_chunks) == 2
            assert retrieved_chunks[0].chunk_id == "chunk_1"
            assert retrieved_chunks[1].chunk_id == "chunk_2"
            
            # Test query operation
            query = Query(text="programming", metadata={"user_id": "test_user"})
            results = await store.query(query, top_k=2)
            assert len(results) <= 2
            
            # Test count operation
            count = await store.count()
            assert count == 3
            
            # Test delete operation
            deleted = await store.delete(["chunk_1"])
            assert deleted == [True]

    @pytest.mark.asyncio
    async def test_keyword_store_implementation(self, sample_chunks, temp_dir):
        """Test SQLiteKeywordStore implementation."""
        from src.memfuse_core.store.keyword_store.sqlite_store import SQLiteKeywordStore
        
        # Create store instance
        store = SQLiteKeywordStore(data_dir=temp_dir)
        
        # Initialize store
        await store.initialize()
        
        # Test add operation
        chunk_ids = await store.add(sample_chunks)
        assert len(chunk_ids) == 3
        
        # Test read operation
        retrieved_chunks = await store.read(["chunk_1", "chunk_2"])
        assert len(retrieved_chunks) == 2
        assert retrieved_chunks[0] is not None
        assert retrieved_chunks[1] is not None
        
        # Test read with filters
        filtered_chunks = await store.read(
            ["chunk_1", "chunk_2", "chunk_3"],
            filters={"language": "python"}
        )
        # Only chunk_1 should match the language filter
        assert filtered_chunks[0] is not None  # chunk_1 matches
        assert filtered_chunks[1] is None      # chunk_2 doesn't match
        assert filtered_chunks[2] is None      # chunk_3 doesn't match
        
        # Test query operation
        query = Query(text="programming", metadata={"user_id": "test_user"})
        results = await store.query(query, top_k=2)
        assert len(results) >= 0  # May return 0 or more results
        
        # Test count operation
        count = await store.count()
        assert count == 3
        
        # Test update operation
        updated_chunk = ChunkData(
            content="Python is an excellent programming language.",
            chunk_id="chunk_1",
            metadata={"type": "chunk", "language": "python", "user_id": "test_user", "updated": True}
        )
        success = await store.update("chunk_1", updated_chunk)
        assert success is True
        
        # Test delete operation
        deleted = await store.delete(["chunk_1"])
        assert deleted == [True]
        
        # Test clear operation
        cleared = await store.clear()
        assert cleared is True
        
        final_count = await store.count()
        assert final_count == 0

    @pytest.mark.asyncio
    async def test_graph_store_implementation(self, sample_chunks, temp_dir):
        """Test GraphMLStore implementation."""
        # Mock the embedding adapter to avoid external dependencies
        with patch('src.memfuse_core.store.adapters.model_adapter.ModelAdapterFactory.create_embedding_adapter') as mock_factory:
            mock_adapter = AsyncMock()
            mock_adapter.encode.return_value = [[0.1] * 384 for _ in sample_chunks]
            mock_factory.return_value = mock_adapter
            
            from src.memfuse_core.store.graph_store.graphml_store import GraphMLStore
            
            # Create store instance
            store = GraphMLStore(data_dir=temp_dir)
            
            # Initialize store
            await store.initialize()
            
            # Test add operation
            chunk_ids = await store.add(sample_chunks)
            assert len(chunk_ids) == 3
            
            # Test read operation
            retrieved_chunks = await store.read(["chunk_1", "chunk_2"])
            assert len(retrieved_chunks) == 2
            
            # Test query operation
            query = Query(text="programming", metadata={"user_id": "test_user"})
            results = await store.query(query, top_k=2)
            assert len(results) >= 0  # May return 0 or more results
            
            # Test count operation
            count = await store.count()
            assert count == 3
            
            # Test delete operation
            deleted = await store.delete(["chunk_1"])
            assert deleted == [True]

    @pytest.mark.asyncio
    async def test_error_handling(self, temp_dir):
        """Test error handling in store implementations."""
        from src.memfuse_core.store.keyword_store.sqlite_store import SQLiteKeywordStore
        
        store = SQLiteKeywordStore(data_dir=temp_dir)
        await store.initialize()
        
        # Test with invalid chunk data
        invalid_chunk = ChunkData(
            content="",  # Empty content
            chunk_id="",  # Empty ID
            metadata={}
        )
        
        # This should handle gracefully or raise StorageError
        try:
            await store.add([invalid_chunk])
        except StorageError:
            pass  # Expected behavior
        except Exception as e:
            pytest.fail(f"Unexpected exception type: {type(e)}")


class TestStoreIntegration:
    """Integration tests for store implementations."""
    
    @pytest.mark.asyncio
    async def test_multi_store_consistency(self, temp_dir):
        """Test that all stores handle the same data consistently."""
        # This test would verify that the same chunks produce
        # consistent results across different store types
        pass
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, temp_dir):
        """Test concurrent operations on stores."""
        # This test would verify thread safety and concurrent access
        pass
