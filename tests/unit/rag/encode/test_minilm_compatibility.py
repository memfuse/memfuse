"""Test MiniLMEncoder interface compatibility.

This module tests that the MiniLMEncoder properly implements both
encode_text and encode methods for backward compatibility.
"""

import pytest
import numpy as np
from unittest.mock import Mock, AsyncMock, patch

from memfuse_core.rag.encode.MiniLM import MiniLMEncoder


class TestMiniLMEncoderCompatibility:
    """Test MiniLMEncoder interface compatibility."""

    @pytest.fixture
    def mock_sentence_transformer(self):
        """Create a mock SentenceTransformer."""
        mock_model = Mock()
        mock_model.encode = Mock(return_value=np.array([0.1, 0.2, 0.3, 0.4]))
        return mock_model

    @pytest.fixture
    def encoder(self, mock_sentence_transformer):
        """Create a MiniLMEncoder with mocked model."""
        with patch('memfuse_core.rag.encode.MiniLM.SentenceTransformer', return_value=mock_sentence_transformer):
            encoder = MiniLMEncoder(model_name="all-MiniLM-L6-v2")
            return encoder

    @pytest.mark.asyncio
    async def test_encode_text_method_exists(self, encoder):
        """Test that encode_text method exists and works."""
        test_text = "This is a test message"
        
        result = await encoder.encode_text(test_text)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (4,)  # Based on mock return value
        np.testing.assert_array_equal(result, np.array([0.1, 0.2, 0.3, 0.4]))

    @pytest.mark.asyncio
    async def test_encode_method_exists(self, encoder):
        """Test that encode method exists and works."""
        test_text = "This is a test message"
        
        result = await encoder.encode(test_text)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (4,)  # Based on mock return value
        np.testing.assert_array_equal(result, np.array([0.1, 0.2, 0.3, 0.4]))

    @pytest.mark.asyncio
    async def test_encode_methods_return_identical_results(self, encoder):
        """Test that encode and encode_text return identical results."""
        test_text = "This is a test message"
        
        result_encode_text = await encoder.encode_text(test_text)
        result_encode = await encoder.encode(test_text)
        
        # Both should return identical numpy arrays
        np.testing.assert_array_equal(result_encode_text, result_encode)
        assert result_encode_text.dtype == result_encode.dtype
        assert result_encode_text.shape == result_encode.shape

    @pytest.mark.asyncio
    async def test_encode_method_uses_same_caching(self, encoder):
        """Test that encode method uses the same caching as encode_text."""
        test_text = "This is a test message"
        
        # First call should hit the model
        result1 = await encoder.encode_text(test_text)
        
        # Second call with encode method should use cache
        result2 = await encoder.encode(test_text)
        
        # Results should be identical (from cache)
        np.testing.assert_array_equal(result1, result2)
        
        # Model should only be called once due to caching
        encoder.model.encode.assert_called_once()

    def test_encoder_has_both_methods(self, encoder):
        """Test that encoder has both required methods."""
        assert hasattr(encoder, 'encode_text')
        assert hasattr(encoder, 'encode')
        assert callable(getattr(encoder, 'encode_text'))
        assert callable(getattr(encoder, 'encode'))

    @pytest.mark.asyncio
    async def test_buffer_retrieval_compatibility(self, encoder):
        """Test compatibility with BufferRetrieval expectations."""
        # This simulates what BufferRetrieval does
        test_query = "test query"
        
        # Test the fallback path that BufferRetrieval uses
        if hasattr(encoder, 'encode_text'):
            result1 = await encoder.encode_text(test_query)
        elif hasattr(encoder, 'encode'):
            result1 = await encoder.encode(test_query)
        else:
            pytest.fail("Encoder has neither encode_text nor encode method")
        
        # Test direct encode method call
        if hasattr(encoder, 'encode'):
            result2 = await encoder.encode(test_query)
        else:
            pytest.fail("Encoder missing encode method")
        
        # Both paths should work and return identical results
        np.testing.assert_array_equal(result1, result2)

    @pytest.mark.asyncio
    async def test_encode_method_accepts_kwargs(self, encoder):
        """Test that encode method can handle additional keyword arguments."""
        test_text = "This is a test message"
        
        # Test that encode method can handle kwargs (like convert_to_numpy)
        result1 = await encoder.encode(test_text)
        result2 = await encoder.encode(test_text, convert_to_numpy=True)
        result3 = await encoder.encode(test_text, some_other_param="ignored")
        
        # All should return identical results since kwargs are ignored
        np.testing.assert_array_equal(result1, result2)
        np.testing.assert_array_equal(result1, result3)