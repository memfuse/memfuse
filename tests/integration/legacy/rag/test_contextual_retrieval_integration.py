"""Integration tests for advanced contextual retrieval functionality."""

import asyncio
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from memfuse_core.rag.retrieve.hybrid import HybridRetrieval
from memfuse_core.rag.chunk.base import ChunkData
from memfuse_core.llm.providers.openai import OpenAIProvider


class MockVectorStore:
    """Mock vector store with contextual chunks for integration testing."""
    
    def __init__(self):
        # Sample conversation with contextual descriptions
        self.chunks = [
            ChunkData(
                content="[USER]: What is machine learning?\n\n[ASSISTANT]: Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed for every task.",
                chunk_id="chunk_1",
                metadata={
                    "session_id": "ai_learning_session",
                    "chunk_index": 0,
                    "contextual_description": "This chunk introduces the fundamental concept of machine learning as a subset of AI, explaining its core capability of learning from data without explicit programming.",
                    "gpt_enhanced": True,
                    "similarity": 0.95
                }
            ),
            ChunkData(
                content="[USER]: Can you give me examples of machine learning?\n\n[ASSISTANT]: Sure! Common examples include email spam detection, recommendation systems like Netflix suggestions, image recognition in photos, voice assistants like Siri, and predictive text on your phone.",
                chunk_id="chunk_2", 
                metadata={
                    "session_id": "ai_learning_session",
                    "chunk_index": 1,
                    "contextual_description": "This chunk provides practical, real-world examples of machine learning applications that users encounter daily, making the concept more tangible and relatable.",
                    "gpt_enhanced": True,
                    "similarity": 0.85
                }
            ),
            ChunkData(
                content="[USER]: How does deep learning differ from traditional machine learning?\n\n[ASSISTANT]: Deep learning uses neural networks with multiple layers to automatically discover patterns in data, while traditional machine learning often requires manual feature engineering and uses simpler algorithms like decision trees or linear regression.",
                chunk_id="chunk_3",
                metadata={
                    "session_id": "ai_learning_session",
                    "chunk_index": 2,
                    "contextual_description": "This chunk explains the technical distinction between deep learning and traditional ML, focusing on neural networks' automatic feature discovery versus manual feature engineering.",
                    "gpt_enhanced": True,
                    "similarity": 0.90
                }
            ),
            ChunkData(
                content="[USER]: What are neural networks exactly?\n\n[ASSISTANT]: Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) organized in layers that process information by passing signals through weighted connections.",
                chunk_id="chunk_4",
                metadata={
                    "session_id": "ai_learning_session",
                    "chunk_index": 3,
                    "contextual_description": "This chunk provides a foundational explanation of neural networks, drawing the biological inspiration and describing the basic architecture of nodes and weighted connections.",
                    "gpt_enhanced": True,
                    "similarity": 0.88
                }
            ),
            ChunkData(
                content="[USER]: Are there different types of neural networks?\n\n[ASSISTANT]: Yes! There are many types: Convolutional Neural Networks (CNNs) for image processing, Recurrent Neural Networks (RNNs) for sequential data like text, Transformer networks for language understanding, and Generative Adversarial Networks (GANs) for creating new content.",
                chunk_id="chunk_5",
                metadata={
                    "session_id": "ai_learning_session",
                    "chunk_index": 4,
                    "contextual_description": "This chunk categorizes different neural network architectures by their specialized applications, from image processing to text generation, providing a comprehensive overview of the field.",
                    "gpt_enhanced": True,
                    "similarity": 0.92
                }
            )
        ]
    
    async def query(self, query_obj, top_k):
        """Mock query method that filters by session_id and returns relevant chunks."""
        session_id = query_obj.metadata.get("session_id")
        
        # Filter by session_id
        filtered_chunks = [
            chunk for chunk in self.chunks 
            if chunk.metadata.get("session_id") == session_id
        ]
        
        # If query text is empty, return all session chunks (for session-based retrieval)
        if not query_obj.text.strip():
            return filtered_chunks[:top_k]
        
        # Simple keyword matching for testing
        query_words = set(query_obj.text.lower().split())
        scored_chunks = []
        
        for chunk in filtered_chunks:
            # Score based on content and contextual description
            content_words = set(chunk.content.lower().split())
            description_words = set(chunk.metadata.get('contextual_description', '').lower().split())
            
            content_overlap = len(query_words.intersection(content_words))
            description_overlap = len(query_words.intersection(description_words))
            
            total_score = content_overlap + description_overlap * 0.8  # Weight contextual descriptions
            
            if total_score > 0:
                chunk.metadata['similarity'] = min(total_score / len(query_words), 1.0)
                scored_chunks.append(chunk)
        
        # Sort by similarity and return top_k
        scored_chunks.sort(key=lambda x: x.metadata.get('similarity', 0), reverse=True)
        return scored_chunks[:top_k]


class TestContextualRetrievalIntegration:
    """Integration tests for contextual retrieval functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.vector_store = MockVectorStore()
        self.retrieval = HybridRetrieval(
            vector_store=self.vector_store,
            keyword_store=None,
            graph_store=None,
            vector_weight=1.0,
            fusion_strategy="simple"
        )
    
    @pytest.mark.asyncio
    async def test_contextual_retrieval_integration(self):
        """Test complete contextual retrieval integration."""
        query = "What are the different types of neural networks and their applications?"
        session_id = "ai_learning_session"
        
        result = await self.retrieval.contextual_retrieve(
            query=query,
            session_id=session_id,
            top_chunks=3,
            top_contextual=3
        )
        
        # Verify result structure
        assert result.total_pieces > 0
        assert len(result.similar_chunks) <= 3
        assert len(result.similar_contextual) <= 3
        assert result.formatted_context
        
        # Verify context formatting
        assert "SIMILAR CHUNKS" in result.formatted_context
        assert "CONTEXTUAL CHUNKS" in result.formatted_context
        
        # Verify retrieval stats
        stats = result.retrieval_stats
        assert "similar_chunks_count" in stats
        assert "connected_contextual_count" in stats
        assert "similar_contextual_count" in stats
        assert stats["session_id"] == session_id
    
    @pytest.mark.asyncio
    async def test_contextual_qa_integration(self):
        """Test contextual question answering integration."""
        question = "Which type of neural network should I use for image recognition tasks?"
        session_id = "ai_learning_session"
        
        # Test without LLM provider (fallback mode)
        answer = await self.retrieval.answer_with_context(
            question=question,
            session_id=session_id,
            llm_provider=None
        )
        
        # Should return formatted context as fallback
        assert isinstance(answer, str)
        assert "Context retrieved" in answer
        assert "pieces" in answer
    
    @pytest.mark.asyncio
    async def test_session_based_filtering(self):
        """Test session-based chunk filtering."""
        # Test with correct session
        result = await self.retrieval.contextual_retrieve(
            query="machine learning",
            session_id="ai_learning_session",
            top_chunks=5
        )
        assert result.total_pieces > 0
        
        # Test with non-existent session
        result = await self.retrieval.contextual_retrieve(
            query="machine learning",
            session_id="non_existent_session",
            top_chunks=5
        )
        assert result.total_pieces == 0
    
    @pytest.mark.asyncio
    async def test_contextual_description_utilization(self):
        """Test that contextual descriptions are properly utilized."""
        result = await self.retrieval.contextual_retrieve(
            query="neural network architectures",
            session_id="ai_learning_session",
            top_chunks=3,
            top_contextual=3
        )
        
        # Verify contextual descriptions are included
        for chunk in result.similar_contextual:
            assert chunk.metadata.get('contextual_description')
            assert chunk.metadata.get('gpt_enhanced')
        
        # Verify context formatting includes descriptions
        assert "This chunk" in result.formatted_context  # Common phrase in descriptions
    
    @pytest.mark.asyncio
    async def test_three_layer_parallel_execution(self):
        """Test that all three layers execute and return results."""
        result = await self.retrieval.contextual_retrieve(
            query="deep learning neural networks",
            session_id="ai_learning_session",
            top_chunks=2,
            top_contextual=2
        )
        
        # All three layers should contribute
        assert len(result.similar_chunks) > 0
        assert len(result.similar_contextual) > 0
        
        # Total pieces should be sum of all layers
        expected_total = (len(result.similar_chunks) + 
                         len(result.connected_contextual) + 
                         len(result.similar_contextual))
        assert result.total_pieces == expected_total
        
        # Verify stats match actual results
        stats = result.retrieval_stats
        assert stats["similar_chunks_count"] == len(result.similar_chunks)
        assert stats["connected_contextual_count"] == len(result.connected_contextual)
        assert stats["similar_contextual_count"] == len(result.similar_contextual)
        assert stats["total_pieces"] == result.total_pieces


if __name__ == "__main__":
    # Allow running as standalone script for manual testing
    async def run_integration_tests():
        test_instance = TestContextualRetrievalIntegration()
        test_instance.setup_method()
        
        print("Running contextual retrieval integration tests...")
        
        await test_instance.test_contextual_retrieval_integration()
        print("✅ Contextual retrieval integration test passed")
        
        await test_instance.test_contextual_qa_integration()
        print("✅ Contextual QA integration test passed")
        
        await test_instance.test_session_based_filtering()
        print("✅ Session-based filtering test passed")
        
        await test_instance.test_contextual_description_utilization()
        print("✅ Contextual description utilization test passed")
        
        await test_instance.test_three_layer_parallel_execution()
        print("✅ Three-layer parallel execution test passed")
        
        print("\n🎉 All integration tests passed!")
    
    asyncio.run(run_integration_tests())
