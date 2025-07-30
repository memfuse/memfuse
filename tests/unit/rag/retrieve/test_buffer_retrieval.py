"""Tests for BufferRetrieval implementation.

This module tests the BufferRetrieval class which provides unified vector and text-based
retrieval for buffer layers (HybridBuffer, RoundBuffer).
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any

from src.memfuse_core.rag.retrieve.buffer import BufferRetrieval


class TestBufferRetrieval:
    """Test cases for BufferRetrieval class."""

    @pytest.fixture
    def buffer_retrieval(self):
        """Create a BufferRetrieval instance for testing."""
        return BufferRetrieval(
            encoder_name="minilm",
            similarity_threshold=0.1
        )

    @pytest.fixture
    def mock_hybrid_buffer(self):
        """Create a mock HybridBuffer for testing."""
        mock_buffer = Mock()
        mock_buffer._data_lock = asyncio.Lock()
        
        # Mock chunks
        mock_chunk1 = Mock()
        mock_chunk1.content = "Mars exploration is challenging due to radiation exposure"
        mock_chunk1.metadata = {"source": "test"}
        
        mock_chunk2 = Mock()
        mock_chunk2.content = "Space travel requires advanced life support systems"
        mock_chunk2.metadata = {"source": "test"}
        
        mock_buffer.chunks = [mock_chunk1, mock_chunk2]
        mock_buffer.embeddings = [
            [0.1, 0.2, 0.3, 0.4],  # Mock embedding for chunk1
            [0.2, 0.3, 0.4, 0.5]   # Mock embedding for chunk2
        ]
        
        return mock_buffer

    @pytest.fixture
    def mock_round_buffer(self):
        """Create a mock RoundBuffer for testing."""
        mock_buffer = Mock()
        mock_buffer._lock = asyncio.Lock()
        
        # Mock rounds with messages
        mock_buffer.rounds = [
            [
                {
                    "id": "msg1",
                    "content": "What are the challenges of Mars exploration?",
                    "role": "user",
                    "created_at": "2024-01-01T00:00:00Z",
                    "metadata": {}
                },
                {
                    "id": "msg2", 
                    "content": "Mars exploration faces radiation, atmosphere, and temperature challenges",
                    "role": "assistant",
                    "created_at": "2024-01-01T00:01:00Z",
                    "metadata": {}
                }
            ]
        ]
        
        return mock_buffer

    @pytest.fixture
    def mock_encoder(self):
        """Create a mock encoder for testing."""
        mock_encoder = AsyncMock()
        mock_encoder.encode_text.return_value = np.array([0.15, 0.25, 0.35, 0.45])
        return mock_encoder

    @pytest.mark.asyncio
    async def test_initialization(self, buffer_retrieval):
        """Test BufferRetrieval initialization."""
        assert buffer_retrieval.encoder_name == "minilm"
        assert buffer_retrieval.similarity_threshold == 0.1
        assert buffer_retrieval.encoder is None

    @pytest.mark.asyncio
    async def test_retrieve_from_hybrid_buffer_only(self, buffer_retrieval, mock_hybrid_buffer, mock_encoder):
        """Test retrieval from HybridBuffer only."""
        with patch.object(buffer_retrieval, '_get_encoder', return_value=mock_encoder):
            results = await buffer_retrieval.retrieve(
                query="Mars exploration challenges",
                top_k=5,
                hybrid_buffer=mock_hybrid_buffer,
                round_buffer=None
            )
            
            assert len(results) > 0
            assert all('hybrid_vector_' in result['id'] for result in results)
            assert all(result['metadata']['source'] == 'hybrid_buffer_vector' for result in results)

    @pytest.mark.asyncio
    async def test_retrieve_from_round_buffer_only(self, buffer_retrieval, mock_round_buffer):
        """Test retrieval from RoundBuffer only."""
        results = await buffer_retrieval.retrieve(
            query="Mars exploration challenges",
            top_k=5,
            hybrid_buffer=None,
            round_buffer=mock_round_buffer
        )
        
        assert len(results) > 0
        assert all('round_' in result['id'] for result in results)
        assert all(result['metadata']['source'] == 'round_buffer' for result in results)

    @pytest.mark.asyncio
    async def test_retrieve_from_both_buffers(self, buffer_retrieval, mock_hybrid_buffer, mock_round_buffer, mock_encoder):
        """Test retrieval from both HybridBuffer and RoundBuffer."""
        with patch.object(buffer_retrieval, '_get_encoder', return_value=mock_encoder):
            results = await buffer_retrieval.retrieve(
                query="Mars exploration challenges",
                top_k=5,
                hybrid_buffer=mock_hybrid_buffer,
                round_buffer=mock_round_buffer
            )
            
            # Should have results from both sources
            hybrid_results = [r for r in results if 'hybrid_vector_' in r['id']]
            round_results = [r for r in results if 'round_' in r['id']]
            
            assert len(hybrid_results) > 0
            assert len(round_results) > 0

    @pytest.mark.asyncio
    async def test_cosine_similarity_calculation(self, buffer_retrieval):
        """Test cosine similarity calculation."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])
        vec3 = np.array([1.0, 0.0, 0.0])
        
        # Orthogonal vectors should have low similarity
        sim1 = buffer_retrieval._cosine_similarity(vec1, vec2)
        assert 0.0 <= sim1 <= 0.6  # Should be around 0.5 after normalization
        
        # Identical vectors should have high similarity
        sim2 = buffer_retrieval._cosine_similarity(vec1, vec3)
        assert sim2 > 0.9  # Should be close to 1.0

    @pytest.mark.asyncio
    async def test_deduplication(self, buffer_retrieval):
        """Test result deduplication."""
        # Create duplicate results
        results = [
            {"id": "1", "content": "Same content", "score": 0.8},
            {"id": "2", "content": "Different content", "score": 0.7},
            {"id": "3", "content": "Same content", "score": 0.6},  # Duplicate
        ]
        
        deduplicated = await buffer_retrieval._deduplicate_and_sort(results, top_k=5)
        
        # Should remove duplicate content
        assert len(deduplicated) == 2
        # Should be sorted by score (descending)
        assert deduplicated[0]["score"] >= deduplicated[1]["score"]

    @pytest.mark.asyncio
    async def test_empty_buffers(self, buffer_retrieval):
        """Test handling of empty buffers."""
        empty_hybrid = Mock()
        empty_hybrid._data_lock = asyncio.Lock()
        empty_hybrid.chunks = []
        empty_hybrid.embeddings = []
        
        empty_round = Mock()
        empty_round._lock = asyncio.Lock()
        empty_round.rounds = []
        
        results = await buffer_retrieval.retrieve(
            query="test query",
            top_k=5,
            hybrid_buffer=empty_hybrid,
            round_buffer=empty_round
        )
        
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_similarity_threshold_filtering(self, buffer_retrieval, mock_encoder):
        """Test that results below similarity threshold are filtered out."""
        # Set high threshold
        buffer_retrieval.similarity_threshold = 0.9
        
        mock_hybrid = Mock()
        mock_hybrid._data_lock = asyncio.Lock()
        
        mock_chunk = Mock()
        mock_chunk.content = "Completely unrelated content about cooking"
        mock_chunk.metadata = {"source": "test"}
        
        mock_hybrid.chunks = [mock_chunk]
        mock_hybrid.embeddings = [[0.9, 0.1, 0.0, 0.0]]  # Very different from query
        
        with patch.object(buffer_retrieval, '_get_encoder', return_value=mock_encoder):
            results = await buffer_retrieval.retrieve(
                query="Mars exploration challenges",
                top_k=5,
                hybrid_buffer=mock_hybrid,
                round_buffer=None
            )
            
            # Should filter out low similarity results
            assert len(results) == 0

    @pytest.mark.asyncio
    async def test_error_handling(self, buffer_retrieval):
        """Test error handling in retrieval methods."""
        # Test with None buffers
        results = await buffer_retrieval.retrieve(
            query="test query",
            top_k=5,
            hybrid_buffer=None,
            round_buffer=None
        )
        
        assert results == []

    @pytest.mark.asyncio
    async def test_encoder_initialization(self, buffer_retrieval):
        """Test encoder initialization and caching."""
        with patch('src.memfuse_core.rag.retrieve.buffer.EncoderRegistry') as mock_registry:
            mock_encoder_class = Mock()
            mock_encoder_instance = AsyncMock()
            mock_encoder_class.return_value = mock_encoder_instance
            mock_registry.get.return_value = mock_encoder_class
            
            # First call should initialize encoder
            encoder1 = await buffer_retrieval._get_encoder()
            assert encoder1 == mock_encoder_instance
            
            # Second call should return cached encoder
            encoder2 = await buffer_retrieval._get_encoder()
            assert encoder2 == mock_encoder_instance
            assert encoder1 is encoder2
            
            # Should only initialize once
            mock_encoder_class.assert_called_once()

    @pytest.mark.asyncio
    async def test_round_buffer_keyword_matching(self, buffer_retrieval, mock_round_buffer):
        """Test keyword matching logic in RoundBuffer."""
        # Query with specific keywords
        results = await buffer_retrieval.retrieve(
            query="radiation temperature atmosphere",
            top_k=5,
            hybrid_buffer=None,
            round_buffer=mock_round_buffer
        )
        
        # Should find matches based on keyword overlap
        assert len(results) > 0
        
        # Check that results have proper metadata
        for result in results:
            assert 'retrieval' in result['metadata']
            assert result['metadata']['retrieval']['method'] == 'keyword_overlap'
            assert 'overlap_score' in result['metadata']['retrieval']

    @pytest.mark.asyncio
    async def test_top_k_limiting(self, buffer_retrieval, mock_round_buffer):
        """Test that results are properly limited by top_k."""
        results = await buffer_retrieval.retrieve(
            query="Mars exploration challenges",
            top_k=1,  # Limit to 1 result
            hybrid_buffer=None,
            round_buffer=mock_round_buffer
        )
        
        assert len(results) <= 1
