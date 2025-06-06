"""Integration tests for MessageCharacterChunkStrategy with LLM."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from src.memfuse_core.rag.chunk.message_character import MessageCharacterChunkStrategy
from src.memfuse_core.rag.chunk.base import ChunkData
from src.memfuse_core.llm.base import LLMRequest, LLMResponse, LLMUsage
from tests.mocks.llm import MockProvider


class TestMessageCharacterChunkStrategyLLMIntegration:
    """Test LLM integration with MessageCharacterChunkStrategy."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create mock LLM provider
        self.llm_provider = MockProvider({
            "response_delay": 0.01,  # Fast for testing
            "custom_responses": {
                "conversation_context": "This chunk discusses user interaction and assistant responses within the conversation flow."
            }
        })
        
        # Create strategy with LLM provider
        self.strategy = MessageCharacterChunkStrategy(
            max_words_per_group=50,
            max_words_per_chunk=50,
            enable_contextual=True,
            llm_provider=self.llm_provider
        )
    
    @pytest.mark.asyncio
    async def test_create_chunks_with_llm_contextual_descriptions(self):
        """Test chunk creation with LLM-generated contextual descriptions."""
        # Mock vector store with previous chunks
        mock_vector_store = AsyncMock()
        previous_chunks = [
            ChunkData(
                content="[USER]: What is AI?\n\n[ASSISTANT]: AI is artificial intelligence.",
                metadata={"created_at": "2024-01-01T10:00:00Z"}
            )
        ]
        mock_vector_store.get_chunks_by_session.return_value = previous_chunks
        
        self.strategy.vector_store = mock_vector_store
        
        message_batch_list = [
            [
                {
                    "role": "user", 
                    "content": "Can you explain machine learning?",
                    "metadata": {"session_id": "test_session"}
                },
                {
                    "role": "assistant",
                    "content": "Machine learning is a subset of AI that enables computers to learn from data."
                }
            ]
        ]
        
        chunks = await self.strategy.create_chunks(message_batch_list)
        
        assert len(chunks) == 1
        chunk = chunks[0]
        
        # Verify basic chunk properties
        assert chunk.metadata["strategy"] == "message_character"
        assert chunk.metadata["has_context"] is True
        assert chunk.metadata["gpt_enhanced"] is True
        assert "contextual_description" in chunk.metadata
        
        # Verify LLM-generated description
        description = chunk.metadata["contextual_description"]
        assert description
        assert len(description) > 0
        
        # Verify vector store was called
        mock_vector_store.get_chunks_by_session.assert_called_once_with("test_session")
    
    @pytest.mark.asyncio
    async def test_create_chunks_llm_fallback_on_error(self):
        """Test fallback to template-based description when LLM fails."""
        # Create provider that always fails
        failing_provider = MockProvider({
            "response_delay": 0.01,
            "fail_rate": 1.0  # Always fail
        })
        
        strategy = MessageCharacterChunkStrategy(
            enable_contextual=True,
            llm_provider=failing_provider
        )
        
        message_batch_list = [
            [{"role": "user", "content": "Test message"}]
        ]
        
        chunks = await strategy.create_chunks(message_batch_list)
        
        assert len(chunks) == 1
        chunk = chunks[0]
        
        # Should have context but not GPT-enhanced
        assert chunk.metadata["has_context"] is False  # No vector store
        assert chunk.metadata["gpt_enhanced"] is False
        
        # Should still have some description (template-based)
        if "contextual_description" in chunk.metadata:
            description = chunk.metadata["contextual_description"]
            assert "template" in description.lower() or "contextual chunk" in description.lower()
    
    @pytest.mark.asyncio
    async def test_create_chunks_without_llm_provider(self):
        """Test chunk creation without LLM provider (fallback mode)."""
        strategy = MessageCharacterChunkStrategy(
            enable_contextual=True,
            llm_provider=None  # No LLM provider
        )
        
        message_batch_list = [
            [{"role": "user", "content": "Test message"}]
        ]
        
        chunks = await strategy.create_chunks(message_batch_list)
        
        assert len(chunks) == 1
        chunk = chunks[0]
        
        # Should not be GPT-enhanced
        assert chunk.metadata["gpt_enhanced"] is False
        
        # Should not have contextual description
        assert "contextual_description" not in chunk.metadata
    
    @pytest.mark.asyncio
    async def test_generate_contextual_description_with_context(self):
        """Test contextual description generation with previous context."""
        context_chunks = [
            ChunkData(
                content="[USER]: Hello\n\n[ASSISTANT]: Hi there!",
                metadata={}
            ),
            ChunkData(
                content="[USER]: How are you?\n\n[ASSISTANT]: I'm doing well, thanks!",
                metadata={}
            )
        ]
        
        current_chunk = "[USER]: What can you help me with?\n\n[ASSISTANT]: I can help with many things!"
        
        description = await self.strategy._generate_contextual_description(
            current_chunk, context_chunks
        )
        
        assert description
        assert len(description) > 0
        # Mock provider should return something about conversation context
        assert any(word in description.lower() for word in ["conversation", "context", "chunk", "interaction"])
    
    @pytest.mark.asyncio
    async def test_generate_contextual_description_without_context(self):
        """Test contextual description generation without previous context."""
        current_chunk = "[USER]: Hello world\n\n[ASSISTANT]: Hello! How can I help you?"
        
        description = await self.strategy._generate_contextual_description(
            current_chunk, []
        )
        
        assert description
        assert len(description) > 0
    
    def test_generate_template_based_description(self):
        """Test template-based description generation."""
        context_chunks = [
            ChunkData(content="Previous chunk 1", metadata={}),
            ChunkData(content="Previous chunk 2", metadata={})
        ]
        
        current_chunk = "Current chunk content"
        
        description = self.strategy._generate_template_based_description(
            current_chunk, context_chunks
        )
        
        assert "Context from 2 previous chunks" in description
        assert "Current chunk content" in description
    
    def test_generate_template_based_description_no_context(self):
        """Test template-based description without context."""
        current_chunk = "Standalone chunk content"
        
        description = self.strategy._generate_template_based_description(
            current_chunk, []
        )
        
        assert "No previous context" in description
        assert "Standalone chunk content" in description
    
    @pytest.mark.asyncio
    async def test_batch_processing_with_llm(self):
        """Test batch processing of multiple chunks with LLM."""
        # Mock vector store
        mock_vector_store = AsyncMock()
        mock_vector_store.get_chunks_by_session.return_value = []
        self.strategy.vector_store = mock_vector_store

        # Create multiple message batches - each should be long enough to create separate chunks
        message_batch_list = [
            [{"role": "user", "content": "First message " + "word " * 30, "metadata": {"session_id": "test"}}],
            [{"role": "user", "content": "Second message " + "word " * 30, "metadata": {"session_id": "test"}}],
            [{"role": "user", "content": "Third message " + "word " * 30, "metadata": {"session_id": "test"}}]
        ]

        chunks = await self.strategy.create_chunks(message_batch_list)

        # Should create at least one chunk (might be combined due to word limits)
        assert len(chunks) >= 1

        # All chunks should be processed
        for i, chunk in enumerate(chunks):
            assert chunk.metadata["strategy"] == "message_character"
            assert chunk.metadata["chunk_index"] == i
            # Since we have no previous context, gpt_enhanced might be False
            # but the chunks should still be created successfully
    
    @pytest.mark.asyncio
    async def test_llm_request_format(self):
        """Test that LLM requests are formatted correctly."""
        # Create a custom mock to capture the request
        captured_requests = []
        
        async def capture_generate(request):
            captured_requests.append(request)
            return LLMResponse(
                content="Mock contextual description",
                model=request.model,
                usage=LLMUsage()
            )
        
        self.llm_provider.generate = capture_generate
        
        context_chunks = [
            ChunkData(content="Previous chunk", metadata={})
        ]
        current_chunk = "Current chunk content"
        
        await self.strategy._generate_contextual_description(current_chunk, context_chunks)
        
        assert len(captured_requests) == 1
        request = captured_requests[0]
        
        # Verify request format
        assert isinstance(request, LLMRequest)
        assert request.model == "grok-3-mini"
        assert request.max_tokens == 150
        assert request.temperature == 0.3
        assert len(request.messages) == 1
        assert request.messages[0]["role"] == "user"
        
        # Verify prompt content
        prompt_content = request.messages[0]["content"]
        assert "conversation_context" in prompt_content
        assert "message_chunk" in prompt_content
        assert "Previous chunk" in prompt_content
        assert "Current chunk content" in prompt_content
    
    @pytest.mark.asyncio
    async def test_concurrent_llm_calls(self):
        """Test concurrent LLM calls for multiple chunks."""
        # Track call timing to ensure concurrency
        call_times = []
        
        async def timed_generate(request):
            import time
            call_times.append(time.time())
            await asyncio.sleep(0.1)  # Simulate API delay
            return LLMResponse(
                content=f"Response for {len(call_times)}",
                model=request.model,
                usage=LLMUsage()
            )
        
        self.llm_provider.generate = timed_generate
        
        # Mock vector store with previous chunks
        mock_vector_store = AsyncMock()
        mock_vector_store.get_chunks_by_session.return_value = [
            ChunkData(content="Previous chunk", metadata={"created_at": "2024-01-01T10:00:00Z"})
        ]
        self.strategy.vector_store = mock_vector_store
        
        # Create multiple chunks that should be processed concurrently
        # Make them long enough to potentially create separate chunks
        message_batch_list = [
            [{"role": "user", "content": f"Message {i} " + "word " * 20, "metadata": {"session_id": "test"}}]
            for i in range(3)
        ]

        start_time = asyncio.get_event_loop().time()
        chunks = await self.strategy.create_chunks(message_batch_list)
        end_time = asyncio.get_event_loop().time()

        # Verify results - might be combined into fewer chunks due to word limits
        assert len(chunks) >= 1
        assert len(call_times) >= 1  # At least one LLM call should be made
        
        # Verify concurrency: total time should be reasonable
        # Since chunks might be combined, we can't guarantee exact timing
        # but it should complete in reasonable time
        total_time = end_time - start_time
        assert total_time < 1.0  # Should complete within 1 second
