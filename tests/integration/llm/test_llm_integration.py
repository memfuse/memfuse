"""Integration tests for LLM provider functionality."""

import asyncio
import pytest
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from memfuse_core.llm.providers.openai import OpenAIProvider
from memfuse_core.llm.base import LLMRequest
from memfuse_core.rag.chunk.message_character import MessageCharacterChunkStrategy
from tests.mocks.llm import MockProvider


class TestLLMIntegration:
    """Integration tests for LLM provider functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Use mock provider for reliable testing
        self.mock_provider = MockProvider({
            "custom_responses": {
                "contextual": "This chunk discusses machine learning fundamentals and applications.",
                "neural": "Neural networks are computing systems inspired by biological networks.",
                "conversation": "This conversation covers AI and machine learning topics."
            }
        })
        
        # Real provider for environment-based testing
        self.real_provider = None
        if os.getenv("OPENAI_API_KEY"):
            self.real_provider = OpenAIProvider({
                "api_key": os.getenv("OPENAI_API_KEY"),
                "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                "timeout": 30.0
            })
    
    @pytest.mark.asyncio
    async def test_mock_provider_integration(self):
        """Test mock provider integration with chunking strategy."""
        strategy = MessageCharacterChunkStrategy(
            enable_contextual=True,
            llm_provider=self.mock_provider
        )
        
        # Test message batch
        message_batch_list = [
            [
                {"role": "user", "content": "What is machine learning?"},
                {"role": "assistant", "content": "Machine learning is a subset of AI."}
            ]
        ]
        
        chunks = await strategy.create_chunks(message_batch_list)
        
        # Verify chunks were created
        assert len(chunks) > 0
        
        # Verify LLM enhancement
        chunk = chunks[0]
        assert chunk.metadata.get("strategy") == "message_character"
        
        # If contextual description was generated, verify it
        if chunk.metadata.get("contextual_description"):
            assert isinstance(chunk.metadata["contextual_description"], str)
            assert chunk.metadata.get("gpt_enhanced") is True
    
    @pytest.mark.asyncio
    async def test_llm_request_response_cycle(self):
        """Test complete LLM request-response cycle."""
        request = LLMRequest(
            messages=[
                {"role": "user", "content": "Explain neural networks briefly."}
            ],
            model="grok-3-mini",
            temperature=0.1,
            max_tokens=100
        )
        
        response = await self.mock_provider.generate(request)
        
        # Verify response structure
        assert response.success is True
        assert response.content
        assert response.error is None
        assert response.usage is not None
        
        # Verify content quality
        assert len(response.content) > 0
        assert "neural" in response.content.lower()
    
    @pytest.mark.asyncio
    async def test_chunking_with_llm_enhancement(self):
        """Test chunking strategy with LLM enhancement."""
        strategy = MessageCharacterChunkStrategy(
            enable_contextual=True,
            llm_provider=self.mock_provider,
            max_words_per_group=100
        )
        
        # Test with multiple message groups
        message_batch_list = [
            [
                {"role": "user", "content": "What are neural networks?"},
                {"role": "assistant", "content": "Neural networks are computing systems."}
            ],
            [
                {"role": "user", "content": "How do they work?"},
                {"role": "assistant", "content": "They process information through layers."}
            ]
        ]
        
        chunks = await strategy.create_chunks(message_batch_list)
        
        # Verify multiple chunks created
        assert len(chunks) >= 2
        
        # Verify each chunk has proper metadata
        for i, chunk in enumerate(chunks):
            assert chunk.metadata.get("strategy") == "message_character"
            assert chunk.metadata.get("chunk_index") == i
            assert chunk.content
            
            # Check for LLM enhancement
            if chunk.metadata.get("gpt_enhanced"):
                assert chunk.metadata.get("contextual_description")
    
    @pytest.mark.asyncio
    async def test_error_handling_integration(self):
        """Test error handling in LLM integration."""
        # Create provider that will fail
        failing_provider = MockProvider({
            "should_fail": True,
            "failure_message": "API rate limit exceeded"
        })
        
        strategy = MessageCharacterChunkStrategy(
            enable_contextual=True,
            llm_provider=failing_provider
        )
        
        message_batch_list = [
            [
                {"role": "user", "content": "Test message"},
                {"role": "assistant", "content": "Test response"}
            ]
        ]
        
        # Should not raise exception, but handle gracefully
        chunks = await strategy.create_chunks(message_batch_list)
        
        # Verify chunks were still created without LLM enhancement
        assert len(chunks) > 0
        chunk = chunks[0]
        assert chunk.content
        assert chunk.metadata.get("strategy") == "message_character"
        
        # Should not have LLM enhancement due to failure
        assert chunk.metadata.get("gpt_enhanced") is not True
    
    @pytest.mark.asyncio
    async def test_contextual_description_quality(self):
        """Test quality of generated contextual descriptions."""
        strategy = MessageCharacterChunkStrategy(
            enable_contextual=True,
            llm_provider=self.mock_provider
        )
        
        # Test with meaningful conversation
        message_batch_list = [
            [
                {
                    "role": "user", 
                    "content": "Can you explain the difference between supervised and unsupervised learning?"
                },
                {
                    "role": "assistant", 
                    "content": "Supervised learning uses labeled data to train models, while unsupervised learning finds patterns in unlabeled data. Examples include classification vs clustering."
                }
            ]
        ]
        
        chunks = await strategy.create_chunks(message_batch_list)
        
        # Verify contextual description quality
        chunk = chunks[0]
        if chunk.metadata.get("contextual_description"):
            description = chunk.metadata["contextual_description"]
            
            # Should be meaningful text
            assert len(description) > 10
            assert isinstance(description, str)
            
            # Should relate to the content
            content_lower = chunk.content.lower()
            description_lower = description.lower()
            
            # Check for topic relevance (flexible matching)
            relevant_terms = ["learning", "machine", "data", "model", "algorithm"]
            content_has_terms = any(term in content_lower for term in relevant_terms)
            description_has_terms = any(term in description_lower for term in relevant_terms)
            
            if content_has_terms:
                # If content has ML terms, description should too
                assert description_has_terms or len(description) > 20  # Or be substantial
    
    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="No OpenAI API key available")
    @pytest.mark.asyncio
    async def test_real_llm_provider_integration(self):
        """Test integration with real LLM provider (requires API key)."""
        if not self.real_provider:
            pytest.skip("Real LLM provider not available")
        
        request = LLMRequest(
            messages=[
                {"role": "user", "content": "What is machine learning in one sentence?"}
            ],
            model="grok-3-mini",
            temperature=0.1,
            max_tokens=50
        )
        
        response = await self.real_provider.generate(request)
        
        # Verify real response
        assert response.success is True
        assert response.content
        assert len(response.content) > 10
        assert "machine learning" in response.content.lower() or "ml" in response.content.lower()
    
    @pytest.mark.asyncio
    async def test_parallel_llm_processing(self):
        """Test parallel LLM processing in chunking."""
        strategy = MessageCharacterChunkStrategy(
            enable_contextual=True,
            llm_provider=self.mock_provider
        )
        
        # Create multiple message batches
        message_batch_list = [
            [{"role": "user", "content": f"Question {i}"}, 
             {"role": "assistant", "content": f"Answer {i}"}]
            for i in range(5)
        ]
        
        # Measure processing time (should be parallel)
        import time
        start_time = time.time()
        
        chunks = await strategy.create_chunks(message_batch_list)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Verify all chunks processed
        assert len(chunks) == 5
        
        # Verify parallel processing efficiency (should be much faster than sequential)
        # With mock provider, this should be very fast
        assert processing_time < 5.0  # Should complete quickly with mock
        
        # Verify all chunks have consistent metadata
        for chunk in chunks:
            assert chunk.metadata.get("strategy") == "message_character"
            assert chunk.content


if __name__ == "__main__":
    # Allow running as standalone script for manual testing
    async def run_llm_integration_tests():
        test_instance = TestLLMIntegration()
        test_instance.setup_method()
        
        print("Running LLM integration tests...")
        
        await test_instance.test_mock_provider_integration()
        print("✅ Mock provider integration test passed")
        
        await test_instance.test_llm_request_response_cycle()
        print("✅ LLM request-response cycle test passed")
        
        await test_instance.test_chunking_with_llm_enhancement()
        print("✅ Chunking with LLM enhancement test passed")
        
        await test_instance.test_error_handling_integration()
        print("✅ Error handling integration test passed")
        
        await test_instance.test_contextual_description_quality()
        print("✅ Contextual description quality test passed")
        
        await test_instance.test_parallel_llm_processing()
        print("✅ Parallel LLM processing test passed")
        
        print("\n🎉 All LLM integration tests passed!")
    
    asyncio.run(run_llm_integration_tests())
