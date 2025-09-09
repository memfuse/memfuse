"""
Unit tests for semantic validation functionality.
"""

import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch

from src.memfuse_core.gateway.semantic_validation import (
    SemanticValidator,
    SemanticViolation,
    SemanticValidationResult,
    get_semantic_validator,
    initialize_semantic_validation
)
from src.memfuse_core.interfaces.gateway_interface import RequestContext


class TestSemanticValidator:
    """Test cases for SemanticValidator."""
    
    @pytest.fixture
    def validator(self):
        """Create a semantic validator for testing."""
        validator = SemanticValidator()
        validator.enabled = True
        validator.similarity_threshold = 0.8
        validator.relevance_threshold = 0.3
        validator.coherence_threshold = 0.4
        return validator
    
    @pytest.fixture
    def mock_encoder(self):
        """Mock encoder for testing."""
        encoder = MagicMock()
        encoder.encode = MagicMock(return_value=[[0.1, 0.2, 0.3, 0.4]])
        return encoder
    
    @pytest.mark.asyncio
    async def test_validator_initialization(self, validator):
        """Test validator initialization."""
        with patch('src.memfuse_core.gateway.semantic_validation.MiniLMEncoder') as mock_encoder_class:
            mock_encoder = AsyncMock()
            mock_encoder_class.return_value = mock_encoder
            
            await validator.initialize()
            
            assert validator.encoder is not None
            mock_encoder_class.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_disabled_validator_returns_passed(self, validator):
        """Test that disabled validator always returns passed."""
        validator.enabled = False
        
        result = await validator.validate_content("test content")
        
        assert result.passed is True
        assert len(result.violations) == 0
        assert result.metadata["enabled"] is False
    
    @pytest.mark.asyncio
    async def test_empty_content_returns_passed(self, validator):
        """Test that empty content returns passed."""
        result = await validator.validate_content("")
        
        assert result.passed is True
        assert len(result.violations) == 0
    
    @pytest.mark.asyncio
    async def test_cosine_similarity_calculation(self, validator):
        """Test cosine similarity calculation."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])
        vec3 = np.array([1.0, 0.0, 0.0])
        
        # Orthogonal vectors should have similarity 0
        similarity1 = validator._cosine_similarity(vec1, vec2)
        assert abs(similarity1 - 0.0) < 1e-6
        
        # Identical vectors should have similarity 1
        similarity2 = validator._cosine_similarity(vec1, vec3)
        assert abs(similarity2 - 1.0) < 1e-6
    
    @pytest.mark.asyncio
    async def test_similarity_violation_detection(self, validator, mock_encoder):
        """Test similarity violation detection."""
        validator.encoder = mock_encoder
        
        # Add a reference pattern with high similarity
        reference_embedding = np.array([0.1, 0.2, 0.3, 0.4])
        validator._reference_embeddings["test_pattern"] = reference_embedding
        
        with patch.object(validator, '_encode_text', return_value=reference_embedding):
            result = await validator.validate_content("similar content")
        
        # Should detect similarity violation
        assert result.passed is False
        assert len(result.violations) > 0
        assert result.violations[0].type == "similarity"
        assert result.violations[0].severity > validator.similarity_threshold
    
    @pytest.mark.asyncio
    async def test_relevance_scoring_with_context(self, validator, mock_encoder):
        """Test relevance scoring with request context."""
        validator.encoder = mock_encoder
        
        context = RequestContext(
            query="test query",
            user_id="user123",
            session_id="session456",
            request_metadata={"expected_topics": ["testing", "validation"]}
        )
        
        # Mock different embeddings for content and query
        content_embedding = np.array([0.1, 0.2, 0.3, 0.4])
        query_embedding = np.array([0.9, 0.8, 0.7, 0.6])  # Low similarity
        
        with patch.object(validator, '_encode_text') as mock_encode:
            mock_encode.side_effect = [content_embedding, query_embedding]
            
            relevance = await validator._calculate_relevance_score(
                "test content", content_embedding, context
            )
        
        # Should calculate relevance based on similarity
        assert 0.0 <= relevance <= 1.0
    
    @pytest.mark.asyncio
    async def test_coherence_scoring(self, validator, mock_encoder):
        """Test content coherence scoring."""
        validator.encoder = mock_encoder
        
        # Mock embeddings for sentences
        sentence_embeddings = [
            np.array([0.1, 0.2, 0.3, 0.4]),
            np.array([0.15, 0.25, 0.35, 0.45]),  # Similar to first
            np.array([0.9, 0.8, 0.7, 0.6])       # Different from others
        ]
        
        with patch.object(validator, '_encode_text') as mock_encode:
            mock_encode.side_effect = sentence_embeddings
            
            coherence = await validator._calculate_coherence_score(
                "First sentence. Second sentence. Third sentence.",
                sentence_embeddings[0]
            )
        
        assert 0.0 <= coherence <= 1.0
    
    @pytest.mark.asyncio
    async def test_semantic_conflict_detection(self, validator, mock_encoder):
        """Test semantic conflict detection."""
        validator.encoder = mock_encoder

        # Content with potential conflicts using opposing keywords
        content = "This statement is correct and true.\nThis statement is wrong and false."

        # Mock embeddings for segments - very different embeddings (low similarity)
        segment_embeddings = [
            np.array([1.0, 0.0, 0.0, 0.0]),  # First segment
            np.array([0.0, 1.0, 0.0, 0.0])   # Second segment (orthogonal = similarity ~0)
        ]

        with patch.object(validator, '_encode_text') as mock_encode:
            mock_encode.side_effect = segment_embeddings

            violations = await validator._detect_semantic_conflicts(
                content, segment_embeddings[0]
            )

        # Should detect potential conflict (low similarity + opposing keywords)
        assert len(violations) > 0
        assert violations[0].type == "conflict"
        assert "conflict" in violations[0].description.lower()
        assert violations[0].severity > 0.8  # High severity due to very low similarity
    
    def test_opposing_keywords_detection(self, validator):
        """Test opposing keywords detection."""
        text1 = "This is correct and true"
        text2 = "This is wrong and false"
        
        has_opposition = validator._contains_opposing_keywords(text1, text2)
        assert has_opposition is True
        
        text3 = "This is good"
        text4 = "This is also good"
        
        has_opposition2 = validator._contains_opposing_keywords(text3, text4)
        assert has_opposition2 is False
    
    @pytest.mark.asyncio
    async def test_complete_validation_workflow(self, validator, mock_encoder):
        """Test complete validation workflow."""
        validator.encoder = mock_encoder
        
        content = "This is test content for validation."
        context = RequestContext(
            query="test validation",
            user_id="user123",
            session_id="session456"
        )
        
        # Mock all encoding operations
        with patch.object(validator, '_encode_text') as mock_encode:
            mock_encode.return_value = np.array([0.1, 0.2, 0.3, 0.4])
            
            result = await validator.validate_content(content, context)
        
        assert isinstance(result, SemanticValidationResult)
        assert isinstance(result.passed, bool)
        assert isinstance(result.violations, list)
        assert isinstance(result.similarity_scores, dict)
        assert isinstance(result.relevance_score, float)
        assert isinstance(result.coherence_score, float)
        assert isinstance(result.metadata, dict)
    
    @pytest.mark.asyncio
    async def test_error_handling_in_validation(self, validator, mock_encoder):
        """Test error handling during validation."""
        validator.encoder = mock_encoder
        
        # Mock encoding to raise an exception
        with patch.object(validator, '_encode_text', side_effect=Exception("Encoding failed")):
            result = await validator.validate_content("test content")
        
        # Should fail open (return passed=True) for safety
        assert result.passed is True
        assert "error" in result.metadata
    
    @pytest.mark.asyncio
    async def test_reference_content_similarity(self, validator, mock_encoder):
        """Test similarity checking with reference content."""
        validator.encoder = mock_encoder
        
        reference_content = ["This is reference content", "Another reference"]
        
        # Mock embeddings
        content_embedding = np.array([0.1, 0.2, 0.3, 0.4])
        ref_embedding = np.array([0.1, 0.2, 0.3, 0.4])  # Identical
        
        with patch.object(validator, '_encode_text') as mock_encode:
            mock_encode.side_effect = [content_embedding, ref_embedding, ref_embedding]
            
            result = await validator.validate_content(
                "test content",
                reference_content=reference_content
            )
        
        # Should detect high similarity
        assert result.passed is False
        similarity_violations = [v for v in result.violations if v.type == "similarity"]
        assert len(similarity_violations) > 0


class TestSemanticValidationGlobals:
    """Test global semantic validation functions."""
    
    def test_get_semantic_validator_singleton(self):
        """Test that get_semantic_validator returns singleton."""
        validator1 = get_semantic_validator()
        validator2 = get_semantic_validator()
        
        assert validator1 is validator2
    
    @pytest.mark.asyncio
    async def test_initialize_semantic_validation(self):
        """Test semantic validation initialization."""
        with patch('src.memfuse_core.gateway.semantic_validation.get_semantic_validator') as mock_get:
            mock_validator = AsyncMock()
            mock_get.return_value = mock_validator
            
            await initialize_semantic_validation()
            
            mock_validator.initialize.assert_called_once()


class TestSemanticViolation:
    """Test SemanticViolation dataclass."""
    
    def test_violation_creation(self):
        """Test violation creation."""
        violation = SemanticViolation(
            type="similarity",
            severity=0.9,
            description="High similarity detected",
            confidence=0.8,
            metadata={"source": "test"}
        )
        
        assert violation.type == "similarity"
        assert violation.severity == 0.9
        assert violation.description == "High similarity detected"
        assert violation.confidence == 0.8
        assert violation.metadata["source"] == "test"


class TestSemanticValidationResult:
    """Test SemanticValidationResult dataclass."""
    
    def test_result_creation(self):
        """Test result creation."""
        violations = [
            SemanticViolation("similarity", 0.9, "test", 0.8, {})
        ]
        
        result = SemanticValidationResult(
            passed=False,
            violations=violations,
            similarity_scores={"test": 0.9},
            relevance_score=0.7,
            coherence_score=0.6,
            metadata={"test": True}
        )
        
        assert result.passed is False
        assert len(result.violations) == 1
        assert result.similarity_scores["test"] == 0.9
        assert result.relevance_score == 0.7
        assert result.coherence_score == 0.6
        assert result.metadata["test"] is True
