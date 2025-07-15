"""Test real embedding functionality through service startup simulation."""

import pytest
import asyncio
from typing import Optional

from src.memfuse_core.services.service_initializer import ServiceInitializer
from src.memfuse_core.services.model_service import ModelService
from src.memfuse_core.interfaces.model_provider import ModelRegistry
from src.memfuse_core.store.factory import StoreFactory
from src.memfuse_core.models import StoreBackend
from src.memfuse_core.rag.chunk.base import ChunkData
from src.memfuse_core.utils.config import config_manager
from omegaconf import DictConfig, OmegaConf


class TestRealServiceEmbedding:
    """Test real embedding functionality through service startup simulation."""
    
    @pytest.fixture
    def service_config(self):
        """Create service configuration similar to actual startup."""
        config = {
            "embedding": {
                "model": "all-MiniLM-L6-v2",
                "dimension": 384,
                "cache_size": 1000,
                "implementation": "minilm"
            },
            "retrieval": {
                "use_rerank": False
            },
            "store": {
                "backend": "pgai",
                "buffer_size": 10,
                "cache_size": 100,
                "top_k": 5,
                "similarity_threshold": 0.3
            },
            "postgres": {
                "host": "localhost",
                "port": 5432,
                "database": "test_memfuse",
                "user": "postgres",
                "password": "password",
                "pool_size": 5
            }
        }
        return OmegaConf.create(config)
    
    @pytest.fixture
    def sample_chunks(self):
        """Create sample chunks for testing."""
        return [
            ChunkData(
                chunk_id="chunk_1",
                content="This is the first test document about artificial intelligence and machine learning.",
                metadata={"source": "test_doc_1", "type": "ai"}
            ),
            ChunkData(
                chunk_id="chunk_2", 
                content="Python programming language is widely used for data science and web development.",
                metadata={"source": "test_doc_2", "type": "programming"}
            ),
            ChunkData(
                chunk_id="chunk_3",
                content="Database systems provide efficient storage and retrieval of structured information.",
                metadata={"source": "test_doc_3", "type": "database"}
            )
        ]
    
    @pytest.mark.asyncio
    async def test_full_service_embedding_workflow(self, service_config, sample_chunks):
        """Test the complete embedding workflow as it would happen in real service."""
        # Step 1: Initialize model service (simulates service startup)
        config_manager.set_config(service_config)
        
        model_service = ModelService()
        success = await model_service.initialize(service_config)
        assert success, "Model service initialization failed"
        
        # Register globally (simulates service registration)
        ModelRegistry.set_provider(model_service)
        
        # Verify global encoder is available
        global_encoder = model_service.get_encoder()
        assert global_encoder is not None
        assert global_encoder.model_name == "all-MiniLM-L6-v2"
        
        print(f"✅ Global encoder initialized: {global_encoder.model_name}")
        print(f"✅ Model instance ID: {id(global_encoder.model)}")
        
        # Step 2: Create vector store through factory (simulates store creation)
        store = await StoreFactory.create_vector_store(
            backend=StoreBackend.PGAI,
            data_dir="test_data",
            table_name="test_service_embedding"
        )
        
        assert store is not None
        print(f"✅ Vector store created: {type(store).__name__}")
        
        # Verify store has access to encoder
        assert hasattr(store, 'pgai_store'), "Store should have pgai_store attribute"
        assert hasattr(store.pgai_store, 'encoder'), "PgaiStore should have encoder attribute"
        assert store.pgai_store.encoder is not None, "PgaiStore encoder should not be None"
        
        # Verify it's the same encoder instance (global reuse)
        store_encoder = store.pgai_store.encoder
        assert id(store_encoder.model) == id(global_encoder.model), "Store should reuse global encoder model"
        
        print(f"✅ Store encoder reuses global model: {id(store_encoder.model) == id(global_encoder.model)}")
        
        # Step 3: Test embedding generation through store
        test_text = "Test embedding generation through real service workflow."
        embedding = await store.pgai_store._generate_embedding(test_text)
        
        # Verify embedding
        assert embedding is not None
        assert isinstance(embedding, list)
        assert len(embedding) == 384
        assert not all(x == 0 for x in embedding)
        
        print(f"✅ Store embedding generation successful")
        print(f"✅ Embedding dimension: {len(embedding)}")
        print(f"✅ Embedding range: [{min(embedding):.4f}, {max(embedding):.4f}]")
        
        # Step 4: Test batch embedding generation
        contents = [chunk.content for chunk in sample_chunks]
        embeddings = await store.pgai_store._generate_embeddings_batch(contents)
        
        # Verify batch embeddings
        assert len(embeddings) == len(sample_chunks)
        assert all(len(emb) == 384 for emb in embeddings)
        assert all(not all(x == 0 for x in emb) for emb in embeddings)
        
        print(f"✅ Batch embedding generation successful")
        print(f"✅ Batch size: {len(embeddings)}")
        
        # Step 5: Test add operation with real embeddings (mock database)
        from unittest.mock import patch, MagicMock, AsyncMock
        
        with patch('src.memfuse_core.store.pgai_store.AsyncConnectionPool') as mock_pool:
            # Mock database operations
            mock_pool_instance = MagicMock()
            mock_cursor = AsyncMock()
            mock_connection = MagicMock()
            mock_connection.cursor.return_value.__aenter__.return_value = mock_cursor
            mock_connection.commit = AsyncMock()
            mock_pool_instance.connection.return_value.__aenter__.return_value = mock_connection
            mock_pool.return_value = mock_pool_instance
            
            store.pgai_store.pool = mock_pool_instance
            store.pgai_store.initialized = True
            
            # Test add operation
            chunk_ids = await store.pgai_store.add(sample_chunks)
            
            # Verify results
            assert len(chunk_ids) == len(sample_chunks)
            assert chunk_ids == [chunk.chunk_id for chunk in sample_chunks]
            
            # Verify database operations were called
            assert mock_cursor.execute.call_count == len(sample_chunks)
            mock_connection.commit.assert_called_once()
            
            print(f"✅ Add operation with real embeddings successful")
            print(f"✅ Added {len(chunk_ids)} chunks")
    
    @pytest.mark.asyncio
    async def test_embedding_consistency_across_service_restarts(self, service_config):
        """Test that embeddings are consistent across simulated service restarts."""
        test_text = "Consistency test across service restarts."
        embeddings = []
        
        # Simulate multiple service startups
        for i in range(3):
            print(f"🔄 Simulating service startup #{i+1}")
            
            # Initialize model service
            model_service = ModelService()
            await model_service.initialize(service_config)
            ModelRegistry.set_provider(model_service)
            
            # Create store
            store = await StoreFactory.create_vector_store(
                backend=StoreBackend.PGAI,
                data_dir="test_data",
                table_name=f"test_consistency_{i}"
            )
            
            # Generate embedding
            embedding = await store.pgai_store._generate_embedding(test_text)
            embeddings.append(embedding)
            
            print(f"✅ Service startup #{i+1} completed")
        
        # Verify all embeddings are identical
        import numpy as np
        for i in range(1, len(embeddings)):
            assert np.allclose(embeddings[0], embeddings[i], rtol=1e-6), f"Embedding {i} differs from first"
        
        print(f"✅ Embedding consistency verified across {len(embeddings)} service restarts")
        print(f"✅ Max difference: {max(np.max(np.abs(np.array(embeddings[0]) - np.array(emb))) for emb in embeddings[1:]):.8f}")
    
    @pytest.mark.asyncio
    async def test_concurrent_embedding_generation(self, service_config):
        """Test concurrent embedding generation with global encoder."""
        # Initialize service
        model_service = ModelService()
        await model_service.initialize(service_config)
        ModelRegistry.set_provider(model_service)
        
        # Create multiple stores
        stores = []
        for i in range(3):
            store = await StoreFactory.create_vector_store(
                backend=StoreBackend.PGAI,
                data_dir="test_data",
                table_name=f"test_concurrent_{i}"
            )
            stores.append(store)
        
        # Test concurrent embedding generation
        test_texts = [f"Concurrent test text {i}" for i in range(5)]
        
        async def generate_embeddings(store, texts):
            return await store.pgai_store._generate_embeddings_batch(texts)
        
        # Run concurrent embedding generation
        import time
        start_time = time.time()
        
        tasks = [generate_embeddings(store, test_texts) for store in stores]
        results = await asyncio.gather(*tasks)
        
        concurrent_time = time.time() - start_time
        
        # Verify results
        assert len(results) == len(stores)
        assert all(len(result) == len(test_texts) for result in results)
        assert all(all(len(emb) == 384 for emb in result) for result in results)
        
        # Verify all stores produced identical embeddings (same model, same texts)
        import numpy as np
        for i in range(1, len(results)):
            for j in range(len(test_texts)):
                assert np.allclose(results[0][j], results[i][j], rtol=1e-6), f"Store {i} text {j} differs"
        
        print(f"✅ Concurrent embedding generation successful")
        print(f"✅ {len(stores)} stores, {len(test_texts)} texts each")
        print(f"✅ Total time: {concurrent_time:.2f}s")
        print(f"✅ All results identical: True")
    
    @pytest.mark.asyncio
    async def test_memory_efficiency_with_global_encoder(self, service_config):
        """Test memory efficiency of global encoder reuse."""
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Initialize service
        model_service = ModelService()
        await model_service.initialize(service_config)
        ModelRegistry.set_provider(model_service)
        
        after_model_memory = process.memory_info().rss / 1024 / 1024  # MB
        model_memory_usage = after_model_memory - initial_memory
        
        # Create multiple stores (should reuse same model)
        stores = []
        for i in range(5):
            store = await StoreFactory.create_vector_store(
                backend=StoreBackend.PGAI,
                data_dir="test_data",
                table_name=f"test_memory_{i}"
            )
            stores.append(store)
        
        after_stores_memory = process.memory_info().rss / 1024 / 1024  # MB
        stores_memory_usage = after_stores_memory - after_model_memory
        
        # Generate embeddings with all stores
        test_text = "Memory efficiency test text."
        for store in stores:
            await store.pgai_store._generate_embedding(test_text)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"✅ Memory usage analysis:")
        print(f"  Initial memory: {initial_memory:.1f} MB")
        print(f"  After model load: {after_model_memory:.1f} MB (+{model_memory_usage:.1f} MB)")
        print(f"  After {len(stores)} stores: {after_stores_memory:.1f} MB (+{stores_memory_usage:.1f} MB)")
        print(f"  Final memory: {final_memory:.1f} MB")
        
        # Verify memory efficiency
        # Creating multiple stores should not significantly increase memory
        # (allowing some overhead for store objects themselves)
        assert stores_memory_usage < model_memory_usage * 0.5, f"Stores use too much memory: {stores_memory_usage:.1f} MB"
        
        print(f"✅ Memory efficiency verified: stores overhead < 50% of model size")


if __name__ == "__main__":
    pytest.main([__file__])
