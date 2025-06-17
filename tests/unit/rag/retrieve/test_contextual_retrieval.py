"""Unit tests for advanced contextual retrieval functionality."""

import pytest
from unittest.mock import AsyncMock

from src.memfuse_core.rag.retrieve.hybrid import HybridRetrieval, ContextualRetrievalResult
from src.memfuse_core.rag.chunk.base import ChunkData
from src.memfuse_core.models import Query
from tests.mocks.llm import MockProvider


class TestContextualRetrieval:
    """Test cases for advanced contextual retrieval."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create mock stores
        self.mock_vector_store = AsyncMock()
        self.mock_keyword_store = AsyncMock()
        
        # Create hybrid retrieval with contextual capabilities
        self.retrieval = HybridRetrieval(
            vector_store=self.mock_vector_store,
            keyword_store=self.mock_keyword_store,
            graph_store=None,  # Not using graph for now
            vector_weight=0.6,
            keyword_weight=0.4,
            fusion_strategy="simple"
        )
        
        # Sample chunks with contextual descriptions
        self.sample_chunks = [
            ChunkData(
                content="[USER]: What is machine learning?\n\n[ASSISTANT]: ML is AI subset.",
                chunk_id="chunk_1",
                metadata={
                    "session_id": "test_session",
                    "contextual_description": "Introduction to machine learning concepts",
                    "gpt_enhanced": True,
                    "similarity": 0.9
                }
            ),
            ChunkData(
                content="[USER]: Neural networks?\n\n[ASSISTANT]: Networks of neurons.",
                chunk_id="chunk_2", 
                metadata={
                    "session_id": "test_session",
                    "contextual_description": "Basic neural network explanation",
                    "gpt_enhanced": True,
                    "similarity": 0.8
                }
            ),
            ChunkData(
                content="[USER]: Deep learning?\n\n[ASSISTANT]: Deep learning uses neural networks.",
                chunk_id="chunk_3",
                metadata={
                    "session_id": "test_session",
                    "contextual_description": "Deep learning and neural network relationships",
                    "gpt_enhanced": True,
                    "similarity": 0.7
                }
            )
        ]
    
    @pytest.mark.asyncio
    async def test_contextual_retrieve_success(self):
        """Test successful advanced contextual retrieval."""
        # Setup mock vector store
        self.mock_vector_store.query.return_value = self.sample_chunks
        
        # Execute contextual retrieval
        result = await self.retrieval.contextual_retrieve(
            query="What are neural networks?",
            session_id="test_session",
            top_chunks=3,
            top_contextual=3
        )
        
        # Verify result structure
        assert isinstance(result, ContextualRetrievalResult)
        assert result.total_pieces > 0
        assert len(result.similar_chunks) <= 3
        assert len(result.similar_contextual) <= 3
        assert result.formatted_context
        assert "session_id" in result.retrieval_stats
        
        # Verify contextual formatting
        assert "SIMILAR CHUNKS" in result.formatted_context
        assert "CONTEXTUAL CHUNKS" in result.formatted_context
    
    @pytest.mark.asyncio
    async def test_find_similar_chunks(self):
        """Test Layer 1: Similar chunks retrieval."""
        # Setup mock
        self.mock_vector_store.query.return_value = self.sample_chunks
        
        # Execute
        similar_chunks = await self.retrieval._find_similar_chunks(
            query="neural networks",
            session_id="test_session",
            top_k=2,
            similarity_threshold=0.0
        )
        
        # Verify
        assert len(similar_chunks) <= 2
        assert all(isinstance(chunk, ChunkData) for chunk in similar_chunks)
        
        # Verify query was called with correct parameters
        self.mock_vector_store.query.assert_called()
        call_args = self.mock_vector_store.query.call_args
        query_obj = call_args[0][0]
        assert query_obj.text == "neural networks"
        assert query_obj.metadata["session_id"] == "test_session"
    
    @pytest.mark.asyncio
    async def test_find_connected_contextual(self):
        """Test Layer 2: Connected contextual descriptions."""
        # Setup mock to return chunks for both calls
        self.mock_vector_store.query.return_value = self.sample_chunks
        
        # Execute
        connected = await self.retrieval._find_connected_contextual(
            query="neural networks",
            session_id="test_session",
            top_k=2
        )
        
        # Verify
        assert isinstance(connected, list)
        # Should find chunks with contextual descriptions
        for chunk in connected:
            assert chunk.metadata.get('contextual_description')
    
    @pytest.mark.asyncio
    async def test_find_similar_contextual(self):
        """Test Layer 3: Similar contextual descriptions."""
        # Setup mock
        self.mock_vector_store.query.return_value = self.sample_chunks
        
        # Execute
        similar_contextual = await self.retrieval._find_similar_contextual(
            query="neural networks",
            session_id="test_session",
            top_k=2,
            similarity_threshold=0.0
        )
        
        # Verify
        assert isinstance(similar_contextual, list)
        for chunk in similar_contextual:
            assert chunk.metadata.get('contextual_description')
    
    def test_format_contextual_results(self):
        """Test contextual results formatting."""
        # Test with all three types of chunks
        formatted = self.retrieval._format_contextual_results(
            similar_chunks=self.sample_chunks[:1],
            connected_contextual=self.sample_chunks[1:2],
            similar_contextual=self.sample_chunks[2:]
        )
        
        # Verify contextual formatting
        assert "=== SIMILAR CHUNKS (Original Content) ===" in formatted
        assert "=== CONNECTED CONTEXTUAL CHUNKS ===" in formatted
        assert "=== SIMILAR CONTEXTUAL CHUNKS ===" in formatted
        assert "machine learning" in formatted.lower()
        assert "neural network" in formatted.lower()
    
    @pytest.mark.asyncio
    async def test_answer_with_context(self):
        """Test question answering with contextual retrieval."""
        # Setup mock vector store
        self.mock_vector_store.query.return_value = self.sample_chunks
        
        # Create mock LLM provider
        mock_llm = MockProvider({
            "custom_responses": {
                "neural": "Neural networks are computing systems inspired by biological neural networks."
            }
        })
        
        # Execute
        answer = await self.retrieval.answer_with_context(
            question="What are neural networks?",
            session_id="test_session",
            llm_provider=mock_llm,
            top_chunks=3,
            top_contextual=3
        )
        
        # Verify
        assert isinstance(answer, str)
        assert len(answer) > 0
        # Should contain relevant information
        assert "neural" in answer.lower()
    
    @pytest.mark.asyncio
    async def test_answer_with_context_no_llm(self):
        """Test question answering without LLM provider (fallback)."""
        # Setup mock vector store
        self.mock_vector_store.query.return_value = self.sample_chunks
        
        # Execute without LLM provider
        answer = await self.retrieval.answer_with_context(
            question="What are neural networks?",
            session_id="test_session",
            llm_provider=None
        )
        
        # Verify fallback behavior
        assert isinstance(answer, str)
        assert "Context retrieved" in answer
        assert "pieces" in answer
    
    @pytest.mark.asyncio
    async def test_contextual_retrieve_no_chunks(self):
        """Test contextual retrieval when no chunks are found."""
        # Setup empty vector store
        self.mock_vector_store.query.return_value = []
        
        # Execute
        result = await self.retrieval.contextual_retrieve(
            query="nonexistent topic",
            session_id="empty_session"
        )
        
        # Verify empty result
        assert result.total_pieces == 0
        assert len(result.similar_chunks) == 0
        assert len(result.connected_contextual) == 0
        assert len(result.similar_contextual) == 0
    
    @pytest.mark.asyncio
    async def test_similarity_threshold_filtering(self):
        """Test similarity threshold filtering."""
        # Setup mock with chunks having different similarities
        chunks_with_scores = []
        for i, chunk in enumerate(self.sample_chunks):
            chunk.metadata['similarity'] = 0.5 + i * 0.2  # 0.5, 0.7, 0.9
            chunks_with_scores.append(chunk)
        
        self.mock_vector_store.query.return_value = chunks_with_scores
        
        # Execute with high threshold
        similar_chunks = await self.retrieval._find_similar_chunks(
            query="test",
            session_id="test_session",
            top_k=5,
            similarity_threshold=0.8  # Should filter out first chunk (0.5)
        )
        
        # Verify filtering
        for chunk in similar_chunks:
            assert chunk.metadata.get('similarity', 0) >= 0.8
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in contextual retrieval."""
        # Setup vector store to raise exception
        self.mock_vector_store.query.side_effect = Exception("Database error")

        # Execute
        result = await self.retrieval.contextual_retrieve(
            query="test query",
            session_id="error_session"
        )

        # Should return empty result (errors are handled gracefully)
        assert result.total_pieces == 0
        assert result.retrieval_stats["similar_chunks_count"] == 0
        assert result.retrieval_stats["connected_contextual_count"] == 0
        assert result.retrieval_stats["similar_contextual_count"] == 0
    
    @pytest.mark.asyncio
    async def test_no_vector_store(self):
        """Test behavior when no vector store is available."""
        retrieval = HybridRetrieval(
            vector_store=None,
            keyword_store=self.mock_keyword_store,
            fusion_strategy="simple"
        )
        
        result = await retrieval.contextual_retrieve(
            query="test query",
            session_id="test_session"
        )
        
        # Should return empty result
        assert result.total_pieces == 0


class TestContextualRetrievalResult:
    """Test cases for ContextualRetrievalResult dataclass."""
    
    def test_contextual_retrieval_result_creation(self):
        """Test ContextualRetrievalResult creation."""
        chunks = [ChunkData(content="test", metadata={})]
        stats = {"test": "value"}
        
        result = ContextualRetrievalResult(
            similar_chunks=chunks,
            connected_contextual=[],
            similar_contextual=chunks,
            total_pieces=2,
            formatted_context="test context",
            retrieval_stats=stats
        )
        
        assert result.similar_chunks == chunks
        assert result.connected_contextual == []
        assert result.similar_contextual == chunks
        assert result.total_pieces == 2
        assert result.formatted_context == "test context"
        assert result.retrieval_stats == stats
