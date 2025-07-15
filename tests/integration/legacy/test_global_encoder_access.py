"""Test global encoder access through service initialization."""

import pytest
import asyncio
from typing import Optional

from src.memfuse_core.services.service_initializer import ServiceInitializer
from src.memfuse_core.services.model_service import ModelService
from src.memfuse_core.interfaces.model_provider import ModelRegistry
from src.memfuse_core.services.service_factory import ServiceFactory
from src.memfuse_core.utils.config import config_manager
from src.memfuse_core.store.pgai_store import PgaiStore
from src.memfuse_core.rag.chunk.base import ChunkData
from omegaconf import DictConfig, OmegaConf


class TestGlobalEncoderAccess:
    """Test global encoder access through service initialization."""
    
    @pytest.fixture
    def test_config(self):
        """Create test configuration."""
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
    
    @pytest.mark.asyncio
    async def test_service_initialization_loads_global_encoder(self, test_config):
        """Test that service initialization loads global encoder."""
        # Set configuration
        config_manager.set_config(test_config)
        
        # Initialize model service
        model_service = ModelService()
        success = await model_service.initialize(test_config)
        
        assert success, "Model service initialization failed"
        
        # Verify encoder is loaded
        encoder = model_service.get_encoder()
        assert encoder is not None, "No encoder loaded"
        assert encoder.model_name == "all-MiniLM-L6-v2", f"Wrong model: {encoder.model_name}"
        
        print(f"✅ Global encoder loaded: {encoder.model_name}")
        print(f"✅ Model instance ID: {id(encoder.model)}")
        
        # Verify encoder is registered globally
        ModelRegistry.set_provider(model_service)
        global_provider = ModelRegistry.get_provider()
        assert global_provider is not None, "No global provider registered"
        
        global_encoder = global_provider.get_encoder()
        assert global_encoder is not None, "No global encoder available"
        assert global_encoder.model_name == "all-MiniLM-L6-v2", "Wrong global encoder model"
        
        # Verify it's the same instance (model reuse)
        assert id(encoder.model) == id(global_encoder.model), "Different model instances - no reuse!"
        
        print(f"✅ Global encoder access verified")
        print(f"✅ Model reuse confirmed: {id(encoder.model) == id(global_encoder.model)}")
    
    @pytest.mark.asyncio
    async def test_pgai_store_uses_global_encoder(self, test_config):
        """Test that PgaiStore can access and use the global encoder."""
        # Set configuration
        config_manager.set_config(test_config)
        
        # Initialize model service and register globally
        model_service = ModelService()
        await model_service.initialize(test_config)
        ModelRegistry.set_provider(model_service)
        
        # Create PgaiStore
        store = PgaiStore(config=test_config, table_name="test_global_access")
        
        # Test embedding generation through global access
        test_text = "This is a test for global encoder access."
        embedding = await store._generate_embedding(test_text)
        
        # Verify embedding
        assert embedding is not None, "No embedding generated"
        assert isinstance(embedding, list), f"Wrong embedding type: {type(embedding)}"
        assert len(embedding) == 384, f"Wrong embedding dimension: {len(embedding)}"
        assert not all(x == 0 for x in embedding), "Zero embedding generated"
        
        print(f"✅ PgaiStore global encoder access successful")
        print(f"✅ Embedding dimension: {len(embedding)}")
        print(f"✅ Embedding range: [{min(embedding):.4f}, {max(embedding):.4f}]")
        
        # Test batch embedding generation
        test_texts = [
            "First test text for batch processing.",
            "Second test text for batch processing.",
            "Third test text for batch processing."
        ]
        
        embeddings = await store._generate_embeddings_batch(test_texts)
        
        # Verify batch embeddings
        assert len(embeddings) == len(test_texts), f"Wrong batch size: {len(embeddings)}"
        assert all(len(emb) == 384 for emb in embeddings), "Wrong batch embedding dimensions"
        assert all(not all(x == 0 for x in emb) for emb in embeddings), "Zero embeddings in batch"
        
        print(f"✅ PgaiStore batch global encoder access successful")
        print(f"✅ Batch size: {len(embeddings)}")
    
    @pytest.mark.asyncio
    async def test_multiple_stores_share_same_encoder(self, test_config):
        """Test that multiple PgaiStore instances share the same global encoder."""
        # Set configuration
        config_manager.set_config(test_config)
        
        # Initialize model service and register globally
        model_service = ModelService()
        await model_service.initialize(test_config)
        ModelRegistry.set_provider(model_service)
        
        # Get reference to the global encoder model
        global_encoder = model_service.get_encoder()
        global_model_id = id(global_encoder.model)
        
        # Create multiple PgaiStore instances
        store1 = PgaiStore(config=test_config, table_name="test_store1")
        store2 = PgaiStore(config=test_config, table_name="test_store2")
        
        # Test that both stores can generate embeddings
        test_text = "Test text for multiple stores."
        
        embedding1 = await store1._generate_embedding(test_text)
        embedding2 = await store2._generate_embedding(test_text)
        
        # Verify embeddings are generated
        assert embedding1 is not None and embedding2 is not None
        assert len(embedding1) == 384 and len(embedding2) == 384
        
        # Verify embeddings are identical (same model, same text)
        import numpy as np
        assert np.allclose(embedding1, embedding2, rtol=1e-6), "Different embeddings from same model"
        
        print(f"✅ Multiple stores share same encoder")
        print(f"✅ Global model ID: {global_model_id}")
        print(f"✅ Embeddings identical: {np.allclose(embedding1, embedding2, rtol=1e-6)}")
    
    @pytest.mark.asyncio
    async def test_encoder_performance_with_global_access(self, test_config):
        """Test encoder performance with global access."""
        import time
        
        # Set configuration
        config_manager.set_config(test_config)
        
        # Initialize model service and register globally
        model_service = ModelService()
        start_time = time.time()
        await model_service.initialize(test_config)
        init_time = time.time() - start_time
        
        ModelRegistry.set_provider(model_service)
        
        print(f"✅ Model initialization time: {init_time:.2f}s")
        
        # Create PgaiStore
        store = PgaiStore(config=test_config, table_name="test_performance")
        
        # Test single embedding performance
        test_text = "Performance test text for embedding generation."
        
        start_time = time.time()
        embedding = await store._generate_embedding(test_text)
        single_time = time.time() - start_time
        
        assert embedding is not None and len(embedding) == 384
        print(f"✅ Single embedding time: {single_time:.4f}s")
        
        # Test batch embedding performance
        test_texts = [f"Performance test text number {i}" for i in range(10)]
        
        start_time = time.time()
        embeddings = await store._generate_embeddings_batch(test_texts)
        batch_time = time.time() - start_time
        
        assert len(embeddings) == 10
        print(f"✅ Batch embedding time (10 texts): {batch_time:.4f}s")
        print(f"✅ Average time per text in batch: {batch_time/10:.4f}s")
        
        # Verify batch is more efficient
        estimated_individual_time = single_time * 10
        efficiency_ratio = estimated_individual_time / batch_time
        print(f"✅ Batch efficiency ratio: {efficiency_ratio:.2f}x")
        
        # Batch should be at least 1.5x more efficient
        assert efficiency_ratio > 1.5, f"Batch not efficient enough: {efficiency_ratio:.2f}x"
    
    @pytest.mark.asyncio
    async def test_encoder_consistency_across_sessions(self, test_config):
        """Test that encoder produces consistent results across different access patterns."""
        # Set configuration
        config_manager.set_config(test_config)
        
        # Initialize model service and register globally
        model_service = ModelService()
        await model_service.initialize(test_config)
        ModelRegistry.set_provider(model_service)
        
        test_text = "Consistency test text for encoder verification."
        
        # Method 1: Direct encoder access
        encoder = model_service.get_encoder()
        embedding1 = await encoder.encode_text(test_text)
        
        # Method 2: Through PgaiStore
        store = PgaiStore(config=test_config, table_name="test_consistency")
        embedding2 = await store._generate_embedding(test_text)
        
        # Method 3: Through global registry
        global_provider = ModelRegistry.get_provider()
        global_encoder = global_provider.get_encoder()
        embedding3 = await global_encoder.encode_text(test_text)
        
        # Convert to same format for comparison
        import numpy as np
        emb1_list = embedding1.tolist() if hasattr(embedding1, 'tolist') else embedding1
        emb2_list = embedding2
        emb3_list = embedding3.tolist() if hasattr(embedding3, 'tolist') else embedding3
        
        # Verify all embeddings are identical
        assert np.allclose(emb1_list, emb2_list, rtol=1e-6), "Direct vs Store access mismatch"
        assert np.allclose(emb1_list, emb3_list, rtol=1e-6), "Direct vs Global access mismatch"
        assert np.allclose(emb2_list, emb3_list, rtol=1e-6), "Store vs Global access mismatch"
        
        print(f"✅ Encoder consistency verified across all access methods")
        print(f"✅ Max difference (direct vs store): {np.max(np.abs(np.array(emb1_list) - np.array(emb2_list))):.8f}")
        print(f"✅ Max difference (direct vs global): {np.max(np.abs(np.array(emb1_list) - np.array(emb3_list))):.8f}")


if __name__ == "__main__":
    pytest.main([__file__])
