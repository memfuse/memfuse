"""Integration tests for QueryBuffer vector retrieval functionality.

This module tests the complete integration of QueryBuffer with the new BufferRetrieval
system, ensuring that vector similarity search works end-to-end.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any

from src.memfuse_core.buffer.query_buffer import QueryBuffer
from src.memfuse_core.buffer.hybrid_buffer import HybridBuffer
from src.memfuse_core.buffer.round_buffer import RoundBuffer
from src.memfuse_core.rag.retrieve.buffer import BufferRetrieval


class TestQueryBufferVectorRetrieval:
    """Integration tests for QueryBuffer with vector retrieval."""

    @pytest.fixture
    async def query_buffer(self):
        """Create a QueryBuffer instance for testing."""
        buffer = QueryBuffer(
            max_size=10,
            cache_size=50,
            default_sort_by="score",
            default_order="desc"
        )
        return buffer

    @pytest.fixture
    async def hybrid_buffer_with_data(self):
        """Create a HybridBuffer with test data."""
        # Mock the embedding model to avoid loading actual models in tests
        with patch('src.memfuse_core.buffer.hybrid_buffer.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_model.encode.return_value = [[0.1, 0.2, 0.3, 0.4]]
            mock_st.return_value = mock_model
            
            buffer = HybridBuffer(
                max_tokens=1000,
                max_chunks=10,
                embedding_model_name="all-MiniLM-L6-v2"
            )
            
            # Add test chunks
            from src.memfuse_core.buffer.chunk import Chunk
            
            chunk1 = Chunk(
                content="Mars exploration faces significant challenges including radiation exposure from cosmic rays and solar particles.",
                metadata={"source": "test", "topic": "mars"}
            )
            
            chunk2 = Chunk(
                content="Space travel requires advanced life support systems to maintain breathable atmosphere and temperature control.",
                metadata={"source": "test", "topic": "space"}
            )
            
            # Manually add chunks and embeddings
            buffer.chunks = [chunk1, chunk2]
            buffer.embeddings = [
                [0.1, 0.2, 0.3, 0.4],  # Embedding for chunk1
                [0.2, 0.3, 0.4, 0.5]   # Embedding for chunk2
            ]
            
            return buffer

    @pytest.fixture
    async def round_buffer_with_data(self):
        """Create a RoundBuffer with test data."""
        buffer = RoundBuffer(max_tokens=800, max_size=5)
        
        # Add test messages
        test_messages = [
            {
                "id": "msg1",
                "content": "What are the main challenges of living on Mars?",
                "role": "user",
                "created_at": "2024-01-01T00:00:00Z",
                "metadata": {"session_id": "test_session"}
            },
            {
                "id": "msg2",
                "content": "Living on Mars presents several major challenges: radiation exposure due to thin atmosphere, extreme temperature variations, lack of breathable air, and psychological isolation.",
                "role": "assistant", 
                "created_at": "2024-01-01T00:01:00Z",
                "metadata": {"session_id": "test_session"}
            }
        ]
        
        await buffer.add_messages(test_messages, session_id="test_session")
        return buffer

    @pytest.mark.asyncio
    async def test_query_buffer_initialization_with_buffer_retrieval(self, query_buffer):
        """Test that QueryBuffer properly initializes BufferRetrieval."""
        assert query_buffer.buffer_retrieval is not None
        assert isinstance(query_buffer.buffer_retrieval, BufferRetrieval)
        assert query_buffer.buffer_retrieval.encoder_name == "minilm"
        assert query_buffer.buffer_retrieval.similarity_threshold == 0.1

    @pytest.mark.asyncio
    async def test_vector_search_integration(self, query_buffer, hybrid_buffer_with_data):
        """Test vector search integration with HybridBuffer."""
        # Set the hybrid buffer
        query_buffer.set_hybrid_buffer(hybrid_buffer_with_data)
        
        # Mock the encoder to return predictable results
        with patch.object(query_buffer.buffer_retrieval, '_get_encoder') as mock_get_encoder:
            mock_encoder = AsyncMock()
            mock_encoder.encode_text.return_value = [0.15, 0.25, 0.35, 0.45]  # Similar to chunk1
            mock_get_encoder.return_value = mock_encoder
            
            # Perform query
            results = await query_buffer.query(
                query_text="radiation exposure challenges",
                top_k=5
            )
            
            # Should return results from vector search
            assert len(results) > 0
            
            # Check result format
            for result in results:
                assert 'id' in result
                assert 'content' in result
                assert 'score' in result
                assert 'metadata' in result
                assert result['metadata']['source'] == 'hybrid_buffer_vector'

    @pytest.mark.asyncio
    async def test_round_buffer_text_search_integration(self, query_buffer, round_buffer_with_data):
        """Test text search integration with RoundBuffer."""
        # Set the round buffer
        query_buffer.set_round_buffer(round_buffer_with_data)
        
        # Perform query
        results = await query_buffer.query(
            query_text="Mars challenges radiation temperature",
            top_k=5
        )
        
        # Should return results from text search
        assert len(results) > 0
        
        # Check result format
        for result in results:
            assert 'id' in result
            assert 'content' in result
            assert 'score' in result
            assert 'metadata' in result
            assert result['metadata']['source'] == 'round_buffer'

    @pytest.mark.asyncio
    async def test_combined_buffer_search(self, query_buffer, hybrid_buffer_with_data, round_buffer_with_data):
        """Test combined search across both HybridBuffer and RoundBuffer."""
        # Set both buffers
        query_buffer.set_hybrid_buffer(hybrid_buffer_with_data)
        query_buffer.set_round_buffer(round_buffer_with_data)
        
        # Mock the encoder
        with patch.object(query_buffer.buffer_retrieval, '_get_encoder') as mock_get_encoder:
            mock_encoder = AsyncMock()
            mock_encoder.encode_text.return_value = [0.15, 0.25, 0.35, 0.45]
            mock_get_encoder.return_value = mock_encoder
            
            # Perform query
            results = await query_buffer.query(
                query_text="Mars radiation challenges",
                top_k=10
            )
            
            # Should return results from both sources
            hybrid_results = [r for r in results if r['metadata']['source'] == 'hybrid_buffer_vector']
            round_results = [r for r in results if r['metadata']['source'] == 'round_buffer']
            
            assert len(hybrid_results) > 0
            assert len(round_results) > 0
            
            # Results should be sorted by score
            scores = [r['score'] for r in results]
            assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_empty_query_bug_fix(self, query_buffer):
        """Test that the original empty query bug is fixed."""
        # This should not return empty results when no data is available
        # but should handle gracefully
        results = await query_buffer.query(
            query_text="test query",
            top_k=5
        )
        
        # Should return empty list gracefully, not crash
        assert isinstance(results, list)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_query_caching_behavior(self, query_buffer, round_buffer_with_data):
        """Test query result caching behavior."""
        query_buffer.set_round_buffer(round_buffer_with_data)
        
        # First query
        results1 = await query_buffer.query(
            query_text="Mars challenges",
            top_k=5
        )
        
        # Second identical query should use cache
        results2 = await query_buffer.query(
            query_text="Mars challenges", 
            top_k=5
        )
        
        # Results should be identical
        assert len(results1) == len(results2)
        if results1:  # If we have results
            assert results1[0]['content'] == results2[0]['content']

    @pytest.mark.asyncio
    async def test_top_k_limiting(self, query_buffer, round_buffer_with_data):
        """Test that top_k parameter properly limits results."""
        query_buffer.set_round_buffer(round_buffer_with_data)
        
        # Query with small top_k
        results = await query_buffer.query(
            query_text="Mars challenges radiation",
            top_k=1
        )
        
        # Should respect top_k limit
        assert len(results) <= 1

    @pytest.mark.asyncio
    async def test_score_sorting(self, query_buffer, round_buffer_with_data):
        """Test that results are properly sorted by score."""
        query_buffer.set_round_buffer(round_buffer_with_data)
        
        results = await query_buffer.query(
            query_text="Mars challenges radiation temperature",
            top_k=5,
            sort_by="score",
            order="desc"
        )
        
        if len(results) > 1:
            # Scores should be in descending order
            for i in range(len(results) - 1):
                assert results[i]['score'] >= results[i + 1]['score']

    @pytest.mark.asyncio
    async def test_error_handling_with_invalid_buffers(self, query_buffer):
        """Test error handling with invalid buffer configurations."""
        # Set invalid buffer references
        query_buffer.hybrid_buffer = None
        query_buffer.round_buffer = None
        
        # Should handle gracefully
        results = await query_buffer.query(
            query_text="test query",
            top_k=5
        )
        
        assert isinstance(results, list)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_metadata_preservation(self, query_buffer, round_buffer_with_data):
        """Test that metadata is properly preserved in results."""
        query_buffer.set_round_buffer(round_buffer_with_data)
        
        results = await query_buffer.query(
            query_text="Mars challenges",
            top_k=5
        )
        
        for result in results:
            assert 'metadata' in result
            assert 'retrieval' in result['metadata']
            assert 'source' in result['metadata']['retrieval']
            assert 'method' in result['metadata']['retrieval']

    @pytest.mark.asyncio
    async def test_similarity_threshold_filtering(self, query_buffer, hybrid_buffer_with_data):
        """Test that similarity threshold properly filters results."""
        # Set high threshold to filter out most results
        query_buffer.buffer_retrieval.similarity_threshold = 0.95
        query_buffer.set_hybrid_buffer(hybrid_buffer_with_data)
        
        with patch.object(query_buffer.buffer_retrieval, '_get_encoder') as mock_get_encoder:
            mock_encoder = AsyncMock()
            # Return embedding very different from stored embeddings
            mock_encoder.encode_text.return_value = [0.9, 0.8, 0.7, 0.6]
            mock_get_encoder.return_value = mock_encoder
            
            results = await query_buffer.query(
                query_text="completely unrelated query about cooking",
                top_k=5
            )
            
            # Should filter out low similarity results
            assert len(results) == 0
