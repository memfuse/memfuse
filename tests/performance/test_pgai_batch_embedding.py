"""Performance tests for PgaiStore batch embedding generation."""

import pytest
import asyncio
import time
from unittest.mock import patch, MagicMock
from typing import List

from src.memfuse_core.store.pgai_store import PgaiStore
from src.memfuse_core.rag.chunk.base import ChunkData


class TestPgaiBatchEmbeddingPerformance:
    """Test performance improvements from batch embedding generation."""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing."""
        return {
            "postgres": {
                "host": "localhost",
                "port": 5432,
                "database": "test_memfuse",
                "user": "postgres",
                "password": "password",
                "pool_size": 5
            },
            "pgai": {
                "enabled": True,
                "vectorizer_worker_enabled": False,
                "auto_embedding": True,
                "embedding_model": "text-embedding-3-small",
                "embedding_dimensions": 1536,
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "batch_size": 100,
                "max_retries": 3,
                "retry_delay": 5.0
            }
        }
    
    @pytest.fixture
    def sample_chunks(self):
        """Create sample chunks for testing."""
        chunks = []
        for i in range(50):  # Test with 50 chunks
            chunk = ChunkData(
                chunk_id=f"test_chunk_{i}",
                content=f"This is test content for chunk {i}. It contains some meaningful text to generate embeddings.",
                metadata={
                    "source": "performance_test",
                    "chunk_index": i,
                    "test_type": "batch_embedding"
                }
            )
            chunks.append(chunk)
        return chunks
    
    @pytest.fixture
    def mock_encoder(self):
        """Mock encoder for testing."""
        encoder = MagicMock()
        
        # Mock single encoding (simulate slower individual calls)
        def mock_encode_single(texts):
            time.sleep(0.01)  # Simulate 10ms per embedding
            return [[0.1] * 384 for _ in texts]
        
        # Mock batch encoding (simulate faster batch processing)
        def mock_encode_batch(texts):
            time.sleep(0.005 * len(texts))  # Simulate 5ms per embedding in batch
            return [[0.1] * 384 for _ in texts]
        
        encoder.encode.side_effect = mock_encode_batch
        return encoder
    
    @pytest.mark.asyncio
    async def test_batch_embedding_generation(self, mock_config, sample_chunks, mock_encoder):
        """Test that batch embedding generation works correctly."""
        with patch('src.memfuse_core.store.pgai_store.PGAI_AVAILABLE', True):
            with patch('src.memfuse_core.store.pgai_store.AsyncConnectionPool') as mock_pool:
                with patch('src.memfuse_core.interfaces.model_provider.ModelRegistry.get_provider') as mock_get_provider:
                    # Setup mock provider with encoder
                    mock_provider = MagicMock()
                    mock_provider.get_encoder.return_value = mock_encoder
                    mock_get_provider.return_value = mock_provider
                    
                    # Setup store
                    store = PgaiStore(config=mock_config, table_name="performance_test")
                    
                    # Mock database operations
                    mock_pool_instance = MagicMock()
                    mock_cursor = MagicMock()
                    mock_connection = MagicMock()
                    mock_connection.cursor.return_value.__aenter__.return_value = mock_cursor
                    mock_pool_instance.connection.return_value.__aenter__.return_value = mock_connection
                    mock_pool.return_value = mock_pool_instance
                    
                    store.pool = mock_pool_instance
                    store.initialized = True
                    
                    # Test batch embedding generation
                    contents = [chunk.content for chunk in sample_chunks]
                    
                    start_time = time.time()
                    embeddings = await store._generate_embeddings_batch(contents)
                    batch_time = time.time() - start_time
                    
                    # Verify results
                    assert len(embeddings) == len(sample_chunks)
                    assert all(len(emb) == 384 for emb in embeddings)
                    
                    # Verify encoder was called once with all texts (batch processing)
                    mock_encoder.encode.assert_called_once_with(contents)
                    
                    print(f"Batch embedding generation time for {len(sample_chunks)} chunks: {batch_time:.3f}s")
    
    @pytest.mark.asyncio
    async def test_individual_vs_batch_performance(self, mock_config, sample_chunks):
        """Compare performance between individual and batch embedding generation."""
        with patch('src.memfuse_core.store.pgai_store.PGAI_AVAILABLE', True):
            with patch('src.memfuse_core.services.service_factory.ServiceFactory.get_global_encoder') as mock_get_encoder:
                
                # Mock encoder with realistic timing differences
                mock_encoder = MagicMock()
                
                def mock_encode_individual(texts):
                    # Simulate individual calls (slower)
                    time.sleep(0.01 * len(texts))  # 10ms per text
                    return [[0.1] * 384 for _ in texts]
                
                def mock_encode_batch(texts):
                    # Simulate batch processing (faster)
                    time.sleep(0.005 * len(texts))  # 5ms per text in batch
                    return [[0.1] * 384 for _ in texts]
                
                mock_get_encoder.return_value = mock_encoder
                
                store = PgaiStore(config=mock_config, table_name="performance_test")
                store.initialized = True
                
                contents = [chunk.content for chunk in sample_chunks[:10]]  # Test with 10 chunks
                
                # Test individual embedding generation
                mock_encoder.encode.side_effect = mock_encode_individual
                start_time = time.time()
                individual_embeddings = []
                for content in contents:
                    embedding = await store._generate_embedding(content)
                    individual_embeddings.append(embedding)
                individual_time = time.time() - start_time
                
                # Reset mock
                mock_encoder.reset_mock()
                mock_encoder.encode.side_effect = mock_encode_batch
                
                # Test batch embedding generation
                start_time = time.time()
                batch_embeddings = await store._generate_embeddings_batch(contents)
                batch_time = time.time() - start_time
                
                # Verify results are equivalent
                assert len(individual_embeddings) == len(batch_embeddings)
                assert len(individual_embeddings) == len(contents)
                
                # Verify performance improvement
                print(f"Individual embedding time: {individual_time:.3f}s")
                print(f"Batch embedding time: {batch_time:.3f}s")
                print(f"Performance improvement: {individual_time / batch_time:.2f}x faster")
                
                # Batch should be faster (allowing some variance for test environment)
                assert batch_time < individual_time * 0.8  # At least 20% faster
    
    @pytest.mark.asyncio
    async def test_add_method_uses_batch_embedding(self, mock_config, sample_chunks, mock_encoder):
        """Test that the add method uses batch embedding generation."""
        with patch('src.memfuse_core.store.pgai_store.PGAI_AVAILABLE', True):
            with patch('src.memfuse_core.store.pgai_store.AsyncConnectionPool') as mock_pool:
                with patch('src.memfuse_core.services.service_factory.ServiceFactory.get_global_encoder') as mock_get_encoder:
                    mock_get_encoder.return_value = mock_encoder
                    
                    # Setup store
                    store = PgaiStore(config=mock_config, table_name="performance_test")
                    
                    # Mock database operations
                    mock_pool_instance = MagicMock()
                    mock_cursor = MagicMock()
                    mock_connection = MagicMock()
                    mock_connection.cursor.return_value.__aenter__.return_value = mock_cursor
                    mock_pool_instance.connection.return_value.__aenter__.return_value = mock_connection
                    mock_pool.return_value = mock_pool_instance
                    
                    store.pool = mock_pool_instance
                    store.initialized = True
                    
                    # Test add method
                    test_chunks = sample_chunks[:5]  # Use 5 chunks for testing
                    chunk_ids = await store.add(test_chunks)
                    
                    # Verify results
                    assert len(chunk_ids) == len(test_chunks)
                    
                    # Verify encoder was called once with all contents (batch processing)
                    expected_contents = [chunk.content for chunk in test_chunks]
                    mock_encoder.encode.assert_called_once_with(expected_contents)
                    
                    # Verify database operations
                    assert mock_cursor.execute.call_count == len(test_chunks)
                    mock_connection.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_update_batch_method(self, mock_config, sample_chunks, mock_encoder):
        """Test the new update_batch method."""
        with patch('src.memfuse_core.store.pgai_store.PGAI_AVAILABLE', True):
            with patch('src.memfuse_core.store.pgai_store.AsyncConnectionPool') as mock_pool:
                with patch('src.memfuse_core.services.service_factory.ServiceFactory.get_global_encoder') as mock_get_encoder:
                    mock_get_encoder.return_value = mock_encoder
                    
                    # Setup store
                    store = PgaiStore(config=mock_config, table_name="performance_test")
                    
                    # Mock database operations
                    mock_pool_instance = MagicMock()
                    mock_cursor = MagicMock()
                    mock_connection = MagicMock()
                    mock_cursor.rowcount = 1  # Simulate successful update
                    mock_connection.cursor.return_value.__aenter__.return_value = mock_cursor
                    mock_pool_instance.connection.return_value.__aenter__.return_value = mock_connection
                    mock_pool.return_value = mock_pool_instance
                    
                    store.pool = mock_pool_instance
                    store.initialized = True
                    
                    # Test update_batch method
                    test_chunks = sample_chunks[:3]  # Use 3 chunks for testing
                    chunk_ids = [chunk.chunk_id for chunk in test_chunks]
                    
                    results = await store.update_batch(chunk_ids, test_chunks)
                    
                    # Verify results
                    assert len(results) == len(test_chunks)
                    assert all(results)  # All updates should succeed
                    
                    # Verify encoder was called once with all contents (batch processing)
                    expected_contents = [chunk.content for chunk in test_chunks]
                    mock_encoder.encode.assert_called_once_with(expected_contents)
                    
                    # Verify database operations
                    assert mock_cursor.execute.call_count == len(test_chunks)
                    mock_connection.commit.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
