"""Real embedding tests using actual all-MiniLM-L6-v2 model."""

import pytest
import asyncio
import numpy as np
from typing import List, Optional

from src.memfuse_core.store.pgai_store import PgaiStore
from src.memfuse_core.store.pgai_vector_wrapper import PgaiVectorWrapper
from src.memfuse_core.rag.chunk.base import ChunkData
from src.memfuse_core.services.model_service import ModelService
from src.memfuse_core.interfaces.model_provider import ModelRegistry
from src.memfuse_core.rag.encode.MiniLM import MiniLMEncoder
from src.memfuse_core.utils.config import config_manager


class TestRealEmbedding:
    """Test real embedding generation using all-MiniLM-L6-v2 model."""
    
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
            }
        }
    
    @pytest.fixture
    def real_encoder(self):
        """Create a real MiniLM encoder with all-MiniLM-L6-v2 model."""
        encoder = MiniLMEncoder(
            model_name="all-MiniLM-L6-v2",
            cache_size=1000
        )
        # Model is loaded in constructor, no need to initialize
        return encoder
    
    @pytest.fixture
    async def model_service_with_real_encoder(self, real_encoder):
        """Create a model service with real encoder."""
        model_service = ModelService()
        # Set the real encoder as default
        model_service._default_encoder = real_encoder
        model_service._encoders["all-MiniLM-L6-v2"] = real_encoder
        
        # Register with ModelRegistry
        ModelRegistry.set_provider(model_service)
        
        return model_service
    
    @pytest.fixture
    def sample_texts(self):
        """Sample texts for embedding testing."""
        return [
            "This is a test document about machine learning.",
            "Python is a programming language used for data science.",
            "The weather is nice today with sunny skies.",
            "Database systems store and retrieve information efficiently.",
            "Natural language processing helps computers understand text."
        ]
    
    @pytest.mark.asyncio
    async def test_real_encoder_initialization(self, real_encoder):
        """Test that the real encoder initializes correctly."""
        assert real_encoder is not None
        assert real_encoder.model is not None
        assert real_encoder.model_name == "all-MiniLM-L6-v2"
        print(f"✅ Real encoder initialized: {real_encoder.model_name}")
    
    @pytest.mark.asyncio
    async def test_real_embedding_generation(self, real_encoder, sample_texts):
        """Test real embedding generation with all-MiniLM-L6-v2."""
        # Test single embedding
        text = sample_texts[0]
        embedding = await real_encoder.encode_text(text)
        
        # Verify embedding properties
        assert embedding is not None
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)  # all-MiniLM-L6-v2 produces 384-dim embeddings
        assert not np.allclose(embedding, 0)  # Should not be zero vector
        
        print(f"✅ Single embedding shape: {embedding.shape}")
        print(f"✅ Embedding norm: {np.linalg.norm(embedding):.4f}")
        
        # Test batch embedding
        embeddings = await real_encoder.encode_texts(sample_texts)
        
        # Verify batch embeddings
        assert len(embeddings) == len(sample_texts)
        assert all(isinstance(emb, np.ndarray) for emb in embeddings)
        assert all(emb.shape == (384,) for emb in embeddings)
        assert all(not np.allclose(emb, 0) for emb in embeddings)
        
        print(f"✅ Batch embeddings count: {len(embeddings)}")
        print(f"✅ All embeddings have correct shape: {all(emb.shape == (384,) for emb in embeddings)}")
        
        # Test that different texts produce different embeddings
        embedding1 = embeddings[0]
        embedding2 = embeddings[1]
        similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        print(f"✅ Similarity between different texts: {similarity:.4f}")
        assert similarity < 0.9  # Different texts should have different embeddings
    
    @pytest.mark.asyncio
    async def test_pgai_store_with_real_encoder(self, mock_config, model_service_with_real_encoder, sample_texts):
        """Test PgaiStore using real encoder through model service."""
        # Create PgaiStore
        store = PgaiStore(config=mock_config, table_name="test_real_embedding")
        
        # Test single embedding generation
        text = sample_texts[0]
        embedding = await store._generate_embedding(text)
        
        # Verify embedding
        assert embedding is not None
        assert isinstance(embedding, list)
        assert len(embedding) == 384
        assert not all(x == 0 for x in embedding)  # Should not be zero vector
        
        print(f"✅ PgaiStore single embedding length: {len(embedding)}")
        print(f"✅ Embedding values range: [{min(embedding):.4f}, {max(embedding):.4f}]")
        
        # Test batch embedding generation
        embeddings = await store._generate_embeddings_batch(sample_texts)
        
        # Verify batch embeddings
        assert len(embeddings) == len(sample_texts)
        assert all(isinstance(emb, list) for emb in embeddings)
        assert all(len(emb) == 384 for emb in embeddings)
        assert all(not all(x == 0 for x in emb) for emb in embeddings)
        
        print(f"✅ PgaiStore batch embeddings count: {len(embeddings)}")
        print(f"✅ All batch embeddings have correct length: {all(len(emb) == 384 for emb in embeddings)}")
    
    @pytest.mark.asyncio
    async def test_pgai_vector_wrapper_with_real_encoder(self, mock_config, real_encoder, sample_texts):
        """Test PgaiVectorWrapper with real encoder."""
        # Create PgaiStore and wrapper
        pgai_store = PgaiStore(config=mock_config, table_name="test_wrapper_real")
        wrapper = PgaiVectorWrapper(
            pgai_store=pgai_store,
            encoder=real_encoder,
            cache_size=1000
        )
        
        # Set encoder on pgai_store for direct access
        pgai_store.encoder = real_encoder
        
        # Test embedding generation through wrapper
        text = sample_texts[0]
        embedding = await pgai_store._generate_embedding(text)
        
        # Verify embedding
        assert embedding is not None
        assert isinstance(embedding, list)
        assert len(embedding) == 384
        assert not all(x == 0 for x in embedding)
        
        print(f"✅ Wrapper embedding length: {len(embedding)}")
        
        # Test that wrapper has access to encoder
        assert wrapper.encoder is not None
        assert wrapper.encoder.model_name == "all-MiniLM-L6-v2"
        
        print(f"✅ Wrapper encoder model: {wrapper.encoder.model_name}")
    
    @pytest.mark.asyncio
    async def test_embedding_consistency(self, real_encoder, sample_texts):
        """Test that the same text produces consistent embeddings."""
        text = sample_texts[0]
        
        # Generate embedding multiple times
        embedding1 = await real_encoder.encode_text(text)
        embedding2 = await real_encoder.encode_text(text)
        
        # Should be identical (or very close due to floating point precision)
        assert np.allclose(embedding1, embedding2, rtol=1e-6)
        
        print(f"✅ Embedding consistency verified")
        print(f"✅ Max difference: {np.max(np.abs(embedding1 - embedding2)):.8f}")
    
    @pytest.mark.asyncio
    async def test_batch_vs_individual_consistency(self, real_encoder, sample_texts):
        """Test that batch and individual embedding generation produce same results."""
        # Generate embeddings individually
        individual_embeddings = []
        for text in sample_texts:
            embedding = await real_encoder.encode_text(text)
            individual_embeddings.append(embedding)
        
        # Generate embeddings in batch
        batch_embeddings = await real_encoder.encode_texts(sample_texts)
        
        # Compare results
        assert len(individual_embeddings) == len(batch_embeddings)
        
        for i, (ind_emb, batch_emb) in enumerate(zip(individual_embeddings, batch_embeddings)):
            assert np.allclose(ind_emb, batch_emb, rtol=1e-6), f"Mismatch at index {i}"
        
        print(f"✅ Batch vs individual consistency verified for {len(sample_texts)} texts")
        
        # Calculate max differences
        max_diffs = [np.max(np.abs(ind - batch)) for ind, batch in zip(individual_embeddings, batch_embeddings)]
        print(f"✅ Max differences: {max_diffs}")
    
    @pytest.mark.asyncio
    async def test_model_reuse_verification(self, real_encoder):
        """Test that the model is reused and not reloaded."""
        # Get initial model reference
        initial_model = real_encoder.model
        initial_id = id(initial_model)
        
        # Generate some embeddings
        await real_encoder.encode_text("Test text 1")
        await real_encoder.encode_text("Test text 2")
        
        # Verify model is still the same instance
        current_model = real_encoder.model
        current_id = id(current_model)
        
        assert initial_id == current_id, "Model instance changed - possible reload!"
        assert initial_model is current_model, "Model reference changed!"
        
        print(f"✅ Model reuse verified - same instance ID: {initial_id}")
    
    @pytest.mark.asyncio
    async def test_global_encoder_access(self, model_service_with_real_encoder):
        """Test accessing encoder through global model registry."""
        # Get encoder through ModelRegistry
        model_provider = ModelRegistry.get_provider()
        assert model_provider is not None
        
        encoder = model_provider.get_encoder()
        assert encoder is not None
        assert encoder.model_name == "all-MiniLM-L6-v2"
        
        # Test embedding generation through global access
        text = "Test global encoder access"
        embedding = await encoder.encode_text(text)
        
        assert embedding is not None
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)
        
        print(f"✅ Global encoder access verified")
        print(f"✅ Global encoder model: {encoder.model_name}")


if __name__ == "__main__":
    pytest.main([__file__])
