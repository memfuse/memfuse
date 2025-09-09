"""
Unit tests for M2-related Pydantic models.

Tests the new models added for M2 fact extraction functionality including
Chunk, Fact, and M2FactStatus enum validation and serialization.
"""

import json
import pytest
import uuid
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any

from src.memfuse_core.models.core import M2Status, M2FactStatus, Chunk, Fact


@pytest.mark.unit
@pytest.mark.models
class TestM2FactStatus:
    """Test cases for M2FactStatus enum."""
    
    def test_m2_fact_status_enum_values(self):
        """Test M2FactStatus enum has correct values."""
        assert M2FactStatus.ACTIVE == "active"
        assert M2FactStatus.DEPRECATED == "deprecated"
    
    def test_m2_fact_status_enum_completeness(self):
        """Test M2FactStatus enum has all expected values and no extras."""
        expected_values = {"active", "deprecated"}
        actual_values = {status.value for status in M2FactStatus}
        assert actual_values == expected_values
    
    def test_m2_fact_status_enum_string_representation(self):
        """Test M2FactStatus enum string representation."""
        assert str(M2FactStatus.ACTIVE) == "M2FactStatus.ACTIVE"
        assert str(M2FactStatus.DEPRECATED) == "M2FactStatus.DEPRECATED"
    
    def test_m2_fact_status_enum_membership(self):
        """Test checking membership in M2FactStatus enum."""
        assert M2FactStatus.ACTIVE in M2FactStatus
        assert M2FactStatus.DEPRECATED in M2FactStatus
        
        # Test invalid values are not members
        assert "invalid_status" not in [s.value for s in M2FactStatus]
    
    def test_m2_fact_status_enum_iteration(self):
        """Test iterating over M2FactStatus enum."""
        statuses = list(M2FactStatus)
        assert len(statuses) == 2
        
        values = [status.value for status in statuses]
        assert "active" in values
        assert "deprecated" in values


@pytest.mark.unit
@pytest.mark.models
class TestChunkModel:
    """Test cases for Chunk Pydantic model."""
    
    def test_chunk_creation_valid_minimal(self):
        """Test creating a Chunk with minimal required fields."""
        chunk_id = str(uuid.uuid4())
        user_id = str(uuid.uuid4())
        now = datetime.now()
        
        chunk = Chunk(
            chunk_id=chunk_id,
            content="Test chunk content",
            token_count=50,
            user_id=user_id,
            created_at=now
        )
        
        # Test required fields
        assert chunk.chunk_id == chunk_id
        assert chunk.content == "Test chunk content"
        assert chunk.token_count == 50
        assert chunk.user_id == user_id
        assert chunk.created_at == now
        
        # Test default values
        assert chunk.session_id is None
        assert chunk.updated_at is None
        assert chunk.m2_status == M2Status.PENDING
        assert chunk.chunking_strategy is None
        assert chunk.m0_raw_ids == []
        assert chunk.metadata == {}
    
    def test_chunk_creation_valid_complete(self):
        """Test creating a Chunk with all fields."""
        chunk_id = str(uuid.uuid4())
        user_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4())
        created_at = datetime.now() - timedelta(hours=1)
        updated_at = datetime.now()
        
        chunk = Chunk(
            chunk_id=chunk_id,
            content="Complete chunk content with all fields",
            token_count=75,
            user_id=user_id,
            session_id=session_id,
            created_at=created_at,
            updated_at=updated_at,
            m2_status=M2Status.COMPLETED,
            chunking_strategy="contextual",
            m0_raw_ids=[str(uuid.uuid4()), str(uuid.uuid4())],
            metadata={"source": "test", "importance": "high"}
        )
        
        # Verify all fields
        assert chunk.chunk_id == chunk_id
        assert chunk.content == "Complete chunk content with all fields"
        assert chunk.token_count == 75
        assert chunk.user_id == user_id
        assert chunk.session_id == session_id
        assert chunk.created_at == created_at
        assert chunk.updated_at == updated_at
        assert chunk.m2_status == M2Status.COMPLETED
        assert chunk.chunking_strategy == "contextual"
        assert len(chunk.m0_raw_ids) == 2
        assert chunk.metadata == {"source": "test", "importance": "high"}
    
    def test_chunk_validation_empty_content(self):
        """Test Chunk accepts empty content (no min_length validation in current model)."""
        # The current Chunk model doesn't enforce min_length on content
        # This test verifies the current behavior
        chunk_id = str(uuid.uuid4())
        user_id = str(uuid.uuid4())
        now = datetime.now()
        
        chunk = Chunk(
            chunk_id=chunk_id,
            content="",  # Empty content is currently allowed
            token_count=0,
            user_id=user_id,
            created_at=now
        )
        
        # Should succeed with current model definition
        assert chunk.content == ""
        assert chunk.token_count == 0
    
    def test_chunk_validation_negative_token_count(self):
        """Test Chunk validation with negative token count."""
        chunk = Chunk(
            chunk_id=str(uuid.uuid4()),
            content="Test content",
            token_count=-5,  # Negative token count
            user_id=str(uuid.uuid4()),
            created_at=datetime.now()
        )
        
        # Model should accept negative values but we can validate business logic separately
        assert chunk.token_count == -5
    
    def test_chunk_json_serialization(self):
        """Test Chunk can be serialized to/from JSON."""
        chunk_id = str(uuid.uuid4())
        user_id = str(uuid.uuid4())
        created_at = datetime.now()
        
        original_chunk = Chunk(
            chunk_id=chunk_id,
            content="JSON serialization test",
            token_count=25,
            user_id=user_id,
            created_at=created_at,
            m2_status=M2Status.PROCESSING,
            metadata={"test": "json_serialization"}
        )
        
        # Serialize to JSON
        json_str = original_chunk.model_dump_json()
        json_dict = json.loads(json_str)
        
        # Verify JSON structure
        assert json_dict['chunk_id'] == chunk_id
        assert json_dict['content'] == "JSON serialization test"
        assert json_dict['token_count'] == 25
        assert json_dict['user_id'] == user_id
        assert json_dict['m2_status'] == "processing"
        assert json_dict['metadata'] == {"test": "json_serialization"}
        
        # Deserialize from JSON
        deserialized_chunk = Chunk.model_validate(json_dict)
        
        # Verify deserialized object matches original
        assert deserialized_chunk.chunk_id == original_chunk.chunk_id
        assert deserialized_chunk.content == original_chunk.content
        assert deserialized_chunk.token_count == original_chunk.token_count
        assert deserialized_chunk.user_id == original_chunk.user_id
        assert deserialized_chunk.m2_status == original_chunk.m2_status
        assert deserialized_chunk.metadata == original_chunk.metadata


@pytest.mark.unit
@pytest.mark.models
class TestFactModel:
    """Test cases for Fact Pydantic model."""
    
    def test_fact_creation_valid_minimal(self):
        """Test creating a Fact with minimal required fields."""
        fact_id = str(uuid.uuid4())
        user_id = str(uuid.uuid4())
        chunk_id = str(uuid.uuid4())
        created_at = datetime.now()
        
        fact = Fact(
            text="Machine learning is a subset of artificial intelligence.",
            confidence=0.85,
            chunk_ids=[chunk_id],
            user_id=user_id,
            created_at=created_at
        )
        
        # Test required fields
        assert fact.text == "Machine learning is a subset of artificial intelligence."
        assert fact.confidence == 0.85
        assert fact.chunk_ids == [chunk_id]
        assert fact.user_id == user_id
        assert fact.created_at == created_at
        
        # Test default values
        assert fact.hash is None
        assert fact.embedding is None
        assert fact.status == M2FactStatus.ACTIVE
        assert fact.policy_version is None
        assert fact.updated_at is None
        assert fact.embedding_generated_at is None
        assert fact.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
        assert fact.metadata == {}
    
    def test_fact_creation_valid_complete(self):
        """Test creating a Fact with all fields."""
        fact_id = str(uuid.uuid4())
        user_id = str(uuid.uuid4())
        chunk_ids = [str(uuid.uuid4()), str(uuid.uuid4())]
        created_at = datetime.now() - timedelta(hours=1)
        updated_at = datetime.now()
        embedding_generated_at = datetime.now() - timedelta(minutes=30)
        embedding = np.random.rand(384).astype(np.float32)
        
        fact = Fact(
            fact_id=fact_id,
            text="Deep learning uses neural networks with multiple layers.",
            hash="abc123def456",
            embedding=embedding,
            confidence=0.92,
            status=M2FactStatus.ACTIVE,
            chunk_ids=chunk_ids,
            user_id=user_id,
            policy_version="v1.5",
            created_at=created_at,
            updated_at=updated_at,
            embedding_generated_at=embedding_generated_at,
            embedding_model="custom-embedding-model",
            metadata={"domain": "ml", "confidence_source": "llm"}
        )
        
        # Verify all fields
        assert fact.fact_id == fact_id
        assert fact.text == "Deep learning uses neural networks with multiple layers."
        assert fact.hash == "abc123def456"
        assert np.array_equal(fact.embedding, embedding)
        assert fact.confidence == 0.92
        assert fact.status == M2FactStatus.ACTIVE
        assert fact.chunk_ids == chunk_ids
        assert fact.user_id == user_id
        assert fact.policy_version == "v1.5"
        assert fact.created_at == created_at
        assert fact.updated_at == updated_at
        assert fact.embedding_generated_at == embedding_generated_at
        assert fact.embedding_model == "custom-embedding-model"
        assert fact.metadata == {"domain": "ml", "confidence_source": "llm"}
    
    def test_fact_validation_empty_text(self):
        """Test Fact accepts empty text (no min_length validation in current model)."""
        # The current Fact model doesn't enforce min_length on text
        # This test verifies the current behavior
        user_id = str(uuid.uuid4())
        chunk_id = str(uuid.uuid4())
        
        fact = Fact(
            text="",  # Empty text is currently allowed
            confidence=0.8,
            chunk_ids=[chunk_id],
            user_id=user_id,
            created_at=datetime.now()
        )
        
        # Should succeed with current model definition
        assert fact.text == ""
        assert fact.confidence == 0.8
    
    def test_fact_validation_confidence_range(self):
        """Test Fact confidence validation range [0.0, 1.0]."""
        user_id = str(uuid.uuid4())
        chunk_id = str(uuid.uuid4())
        created_at = datetime.now()
        
        # Valid confidence values
        valid_confidences = [0.0, 0.5, 1.0]
        for confidence in valid_confidences:
            fact = Fact(
                text=f"Test fact with confidence {confidence}",
                confidence=confidence,
                chunk_ids=[chunk_id],
                user_id=user_id,
                created_at=created_at
            )
            assert fact.confidence == confidence
        
        # Invalid confidence values
        invalid_confidences = [-0.1, 1.1, 2.0]
        for confidence in invalid_confidences:
            with pytest.raises(ValueError):
                Fact(
                    text=f"Test fact with invalid confidence {confidence}",
                    confidence=confidence,
                    chunk_ids=[chunk_id],
                    user_id=user_id,
                    created_at=created_at
                )
    
    def test_fact_validation_chunk_ids_not_empty(self):
        """Test Fact validation requires non-empty chunk_ids list."""
        # The current Fact model does have min_length=1 validation on chunk_ids
        with pytest.raises(ValueError, match="at least 1 item"):
            Fact(
                text="Test fact with empty chunk_ids",
                confidence=0.8,
                chunk_ids=[],  # Empty chunk_ids should fail
                user_id=str(uuid.uuid4()),
                created_at=datetime.now()
            )
    
    def test_fact_embedding_numpy_array(self):
        """Test Fact can handle numpy array embeddings."""
        embedding = np.random.rand(384).astype(np.float32)
        
        fact = Fact(
            text="Test fact with numpy embedding",
            confidence=0.8,
            chunk_ids=[str(uuid.uuid4())],
            user_id=str(uuid.uuid4()),
            created_at=datetime.now(),
            embedding=embedding
        )
        
        # Verify embedding is preserved as numpy array
        assert isinstance(fact.embedding, np.ndarray)
        assert np.array_equal(fact.embedding, embedding)
        assert fact.embedding.dtype == np.float32
        assert fact.embedding.shape == (384,)
    
    def test_fact_json_serialization_without_embedding(self):
        """Test Fact JSON serialization without embedding (which can't be JSON serialized)."""
        fact_id = str(uuid.uuid4())
        user_id = str(uuid.uuid4())
        chunk_id = str(uuid.uuid4())
        created_at = datetime.now()
        
        fact = Fact(
            fact_id=fact_id,
            text="JSON serialization test fact",
            confidence=0.88,
            chunk_ids=[chunk_id],
            user_id=user_id,
            created_at=created_at,
            hash="test_hash_123",
            policy_version="v1.0",
            metadata={"test": "json"}
        )
        
        # Serialize to JSON (without embedding)
        json_dict = fact.model_dump(exclude={'embedding'})
        json_str = json.dumps(json_dict, default=str)  # Handle datetime serialization
        
        # Verify JSON structure
        parsed = json.loads(json_str)
        assert parsed['fact_id'] == fact_id
        assert parsed['text'] == "JSON serialization test fact"
        assert parsed['confidence'] == 0.88
        assert parsed['chunk_ids'] == [chunk_id]
        assert parsed['user_id'] == user_id
        assert parsed['hash'] == "test_hash_123"
        assert parsed['policy_version'] == "v1.0"
        assert parsed['metadata'] == {"test": "json"}
        assert 'embedding' not in parsed  # Should be excluded
    
    def test_fact_status_transitions(self):
        """Test Fact status field accepts M2FactStatus enum values."""
        user_id = str(uuid.uuid4())
        chunk_id = str(uuid.uuid4())
        created_at = datetime.now()
        
        # Test ACTIVE status
        active_fact = Fact(
            text="Active status fact",
            confidence=0.9,
            chunk_ids=[chunk_id],
            user_id=user_id,
            created_at=created_at,
            status=M2FactStatus.ACTIVE
        )
        assert active_fact.status == M2FactStatus.ACTIVE
        
        # Test DEPRECATED status
        deprecated_fact = Fact(
            text="Deprecated status fact",
            confidence=0.7,
            chunk_ids=[chunk_id],
            user_id=user_id,
            created_at=created_at,
            status=M2FactStatus.DEPRECATED
        )
        assert deprecated_fact.status == M2FactStatus.DEPRECATED
    
    def test_fact_multiple_chunk_ids(self):
        """Test Fact can handle multiple chunk IDs for lineage tracking."""
        user_id = str(uuid.uuid4())
        chunk_ids = [str(uuid.uuid4()) for _ in range(5)]
        
        fact = Fact(
            text="Fact extracted from multiple chunks",
            confidence=0.85,
            chunk_ids=chunk_ids,
            user_id=user_id,
            created_at=datetime.now()
        )
        
        assert len(fact.chunk_ids) == 5
        assert fact.chunk_ids == chunk_ids
        
        # Verify all chunk_ids are valid UUIDs
        for chunk_id in fact.chunk_ids:
            uuid.UUID(chunk_id)  # Should not raise exception
    
    def test_fact_metadata_flexibility(self):
        """Test Fact metadata field accepts various data structures."""
        user_id = str(uuid.uuid4())
        chunk_id = str(uuid.uuid4())
        
        # Test complex metadata structure
        complex_metadata = {
            "extraction_method": "llm_prompt",
            "source_confidence": 0.92,
            "tags": ["important", "verified"],
            "nested": {
                "model": "gpt-4",
                "temperature": 0.1
            },
            "extracted_at": datetime.now().isoformat()
        }
        
        fact = Fact(
            text="Fact with complex metadata",
            confidence=0.88,
            chunk_ids=[chunk_id],
            user_id=user_id,
            created_at=datetime.now(),
            metadata=complex_metadata
        )
        
        assert fact.metadata == complex_metadata
        assert fact.metadata["extraction_method"] == "llm_prompt"
        assert fact.metadata["tags"] == ["important", "verified"]
        assert fact.metadata["nested"]["model"] == "gpt-4"


@pytest.mark.unit
@pytest.mark.models
class TestModelIntegration:
    """Test integration between different M2 models."""
    
    def test_chunk_to_fact_relationship(self):
        """Test relationship between Chunk and Fact models via chunk_ids."""
        # Create a chunk
        chunk_id = str(uuid.uuid4())
        user_id = str(uuid.uuid4())
        
        chunk = Chunk(
            chunk_id=chunk_id,
            content="This chunk contains information about machine learning algorithms.",
            token_count=120,
            user_id=user_id,
            created_at=datetime.now(),
            m2_status=M2Status.COMPLETED
        )
        
        # Create facts extracted from this chunk
        facts = [
            Fact(
                text="Machine learning algorithms can be supervised or unsupervised.",
                confidence=0.95,
                chunk_ids=[chunk_id],  # Links back to chunk
                user_id=user_id,
                created_at=datetime.now(),
                policy_version="v1.0"
            ),
            Fact(
                text="Supervised learning requires labeled training data.",
                confidence=0.90,
                chunk_ids=[chunk_id],  # Links back to same chunk
                user_id=user_id,
                created_at=datetime.now(),
                policy_version="v1.0"
            )
        ]
        
        # Verify relationships
        assert chunk.chunk_id == chunk_id
        assert chunk.m2_status == M2Status.COMPLETED  # Chunk was processed
        
        for fact in facts:
            assert chunk_id in fact.chunk_ids
            assert fact.user_id == chunk.user_id  # Same user
            assert fact.status == M2FactStatus.ACTIVE  # Default status
    
    def test_fact_cross_chunk_extraction(self):
        """Test Fact model can reference multiple chunks for cross-chunk extraction."""
        user_id = str(uuid.uuid4())
        
        # Create multiple chunks
        chunk_ids = [str(uuid.uuid4()) for _ in range(3)]
        chunks = []
        
        for i, chunk_id in enumerate(chunk_ids):
            chunk = Chunk(
                chunk_id=chunk_id,
                content=f"Chunk {i} discussing neural networks and deep learning.",
                token_count=80 + i * 10,
                user_id=user_id,
                created_at=datetime.now() - timedelta(minutes=i * 10),
                m2_status=M2Status.COMPLETED
            )
            chunks.append(chunk)
        
        # Create a fact extracted from all chunks
        cross_chunk_fact = Fact(
            text="Neural networks are fundamental to deep learning architectures.",
            confidence=0.88,
            chunk_ids=chunk_ids,  # References all chunks
            user_id=user_id,
            created_at=datetime.now(),
            policy_version="v1.0",
            metadata={"extraction_type": "cross_chunk"}
        )
        
        # Verify cross-chunk relationship
        assert len(cross_chunk_fact.chunk_ids) == 3
        for chunk_id in chunk_ids:
            assert chunk_id in cross_chunk_fact.chunk_ids
        
        # Verify all chunks belong to same user
        for chunk in chunks:
            assert chunk.user_id == cross_chunk_fact.user_id
    
    def test_model_field_compatibility(self):
        """Test that model fields are compatible for M2 processing pipeline."""
        user_id = str(uuid.uuid4())
        chunk_id = str(uuid.uuid4())
        
        # Create chunk in pending state
        chunk = Chunk(
            chunk_id=chunk_id,
            content="Content for M2 processing pipeline test",
            token_count=95,
            user_id=user_id,
            created_at=datetime.now(),
            m2_status=M2Status.PENDING  # Ready for M2 processing
        )
        
        # Simulate M2 processing: chunk moves to processing state
        chunk.m2_status = M2Status.PROCESSING
        
        # Create fact from processed chunk
        extracted_fact = Fact(
            text="M2 processing pipeline successfully extracts facts from chunks.",
            confidence=0.92,
            chunk_ids=[chunk_id],
            user_id=user_id,
            created_at=datetime.now(),
            status=M2FactStatus.ACTIVE,
            policy_version="v1.0"
        )
        
        # Simulate completion: chunk moves to completed state
        chunk.m2_status = M2Status.COMPLETED
        
        # Verify pipeline state consistency
        assert chunk.m2_status == M2Status.COMPLETED
        assert extracted_fact.status == M2FactStatus.ACTIVE
        assert extracted_fact.chunk_ids[0] == chunk.chunk_id
        assert extracted_fact.user_id == chunk.user_id


if __name__ == '__main__':
    pytest.main([__file__, '-v'])