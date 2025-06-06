"""Unit tests for LLM providers."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from src.memfuse_core.llm.base import LLMRequest, LLMResponse, LLMUsage
from tests.mocks.llm import MockProvider


class TestMockProvider:
    """Test cases for MockProvider."""
    
    def test_init(self):
        """Test MockProvider initialization."""
        provider = MockProvider()
        assert provider.response_delay == 0.1
        assert provider.fail_rate == 0.0
        assert provider.custom_responses == {}
        assert provider._request_counter == 0
    
    def test_init_with_config(self):
        """Test MockProvider initialization with config."""
        config = {
            "response_delay": 0.5,
            "fail_rate": 0.1,
            "custom_responses": {"test": "response"}
        }
        provider = MockProvider(config)
        assert provider.response_delay == 0.5
        assert provider.fail_rate == 0.1
        assert provider.custom_responses == {"test": "response"}
    
    @pytest.mark.asyncio
    async def test_generate_success(self):
        """Test successful response generation."""
        provider = MockProvider({"response_delay": 0.01})  # Fast for testing
        request = LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            model="mock-model"
        )
        
        response = await provider.generate(request)
        
        assert response.success
        assert response.content
        assert response.model == "mock-model"
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens > 0
        assert "mock_request_id" in response.metadata
    
    @pytest.mark.asyncio
    async def test_generate_with_failure(self):
        """Test response generation with simulated failure."""
        provider = MockProvider({
            "response_delay": 0.01,
            "fail_rate": 1.0  # Always fail
        })
        request = LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            model="mock-model"
        )
        
        response = await provider.generate(request)
        
        assert not response.success
        assert "simulated failure" in response.error
        assert response.content == ""
    
    @pytest.mark.asyncio
    async def test_generate_with_custom_response(self):
        """Test response generation with custom response."""
        provider = MockProvider({
            "response_delay": 0.01,
            "custom_responses": {"hello": "Custom hello response"}
        })
        request = LLMRequest(
            messages=[{"role": "user", "content": "Hello world"}],
            model="mock-model"
        )
        
        response = await provider.generate(request)
        
        assert response.success
        assert response.content == "Custom hello response"
    
    @pytest.mark.asyncio
    async def test_generate_contextual_chunking(self):
        """Test contextual chunking response generation."""
        provider = MockProvider({"response_delay": 0.01})
        request = LLMRequest(
            messages=[{
                "role": "user", 
                "content": "<conversation_context>Previous messages</conversation_context>\n<message_chunk>Current chunk</message_chunk>"
            }],
            model="mock-model"
        )
        
        response = await provider.generate(request)
        
        assert response.success
        assert "contextual description" in response.content.lower()
    
    @pytest.mark.asyncio
    async def test_generate_invalid_request(self):
        """Test response generation with invalid request."""
        provider = MockProvider()
        request = LLMRequest(messages=[], model="mock-model")  # Invalid: no messages
        
        response = await provider.generate(request)
        
        assert not response.success
        assert "Invalid request" in response.error
    
    @pytest.mark.asyncio
    async def test_generate_stream(self):
        """Test streaming response generation."""
        provider = MockProvider({"response_delay": 0.01})
        request = LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            model="mock-model"
        )
        
        chunks = []
        async for chunk in provider.generate_stream(request):
            chunks.append(chunk)
        
        assert len(chunks) > 0
        full_response = "".join(chunks)
        assert "mock response" in full_response.lower()
    
    @pytest.mark.asyncio
    async def test_generate_stream_invalid_request(self):
        """Test streaming with invalid request."""
        provider = MockProvider()
        request = LLMRequest(messages=[], model="mock-model")  # Invalid
        
        chunks = []
        async for chunk in provider.generate_stream(request):
            chunks.append(chunk)
        
        assert len(chunks) == 1
        assert "Error: Invalid request" in chunks[0]
    
    @pytest.mark.asyncio
    async def test_is_available(self):
        """Test availability check."""
        provider = MockProvider()
        available = await provider.is_available()
        assert available
    
    def test_get_default_model(self):
        """Test getting default model."""
        provider = MockProvider()
        assert provider.get_default_model() == "mock-model"
    
    def test_get_supported_models(self):
        """Test getting supported models."""
        provider = MockProvider()
        models = provider.get_supported_models()
        assert "mock-model" in models
        assert "mock-gpt-4" in models
        assert "mock-claude" in models
    
    def test_set_custom_response(self):
        """Test setting custom response."""
        provider = MockProvider()
        provider.set_custom_response("test", "custom response")
        assert provider.custom_responses["test"] == "custom response"
    
    def test_set_fail_rate(self):
        """Test setting fail rate."""
        provider = MockProvider()
        
        provider.set_fail_rate(0.5)
        assert provider.fail_rate == 0.5
        
        # Test bounds
        provider.set_fail_rate(-0.1)
        assert provider.fail_rate == 0.0
        
        provider.set_fail_rate(1.5)
        assert provider.fail_rate == 1.0
    
    def test_reset_counter(self):
        """Test resetting request counter."""
        provider = MockProvider()
        provider._request_counter = 10
        provider.reset_counter()
        assert provider._request_counter == 0


@pytest.mark.skipif(
    True,  # Skip OpenAI tests by default since they require API key
    reason="OpenAI tests require API key and make real API calls"
)
class TestOpenAIProvider:
    """Test cases for OpenAIProvider (requires API key)."""
    
    def test_init_without_openai(self):
        """Test initialization when OpenAI library is not available."""
        with patch('src.memfuse_core.llm.providers.openai.OPENAI_AVAILABLE', False):
            from src.memfuse_core.llm.providers.openai import OpenAIProvider
            from src.memfuse_core.llm.base import LLMProviderError
            
            with pytest.raises(LLMProviderError):
                OpenAIProvider()
    
    @patch('src.memfuse_core.llm.providers.openai.OPENAI_AVAILABLE', True)
    @patch('src.memfuse_core.llm.providers.openai.AsyncOpenAI')
    def test_init_with_config(self, mock_openai):
        """Test initialization with configuration."""
        from src.memfuse_core.llm.providers.openai import OpenAIProvider
        
        config = {
            "api_key": "test-key",
            "base_url": "https://api.test.com",
            "timeout": 60
        }
        
        provider = OpenAIProvider(config)
        
        mock_openai.assert_called_once_with(
            api_key="test-key",
            base_url="https://api.test.com",
            timeout=60
        )
        assert provider.config == config
    
    def test_get_default_model(self):
        """Test getting default model."""
        with patch('src.memfuse_core.llm.providers.openai.OPENAI_AVAILABLE', True):
            with patch('src.memfuse_core.llm.providers.openai.AsyncOpenAI'):
                from src.memfuse_core.llm.providers.openai import OpenAIProvider
                provider = OpenAIProvider()
                assert provider.get_default_model() == "gpt-4o-mini"
    
    def test_get_supported_models(self):
        """Test getting supported models."""
        with patch('src.memfuse_core.llm.providers.openai.OPENAI_AVAILABLE', True):
            with patch('src.memfuse_core.llm.providers.openai.AsyncOpenAI'):
                from src.memfuse_core.llm.providers.openai import OpenAIProvider
                provider = OpenAIProvider()
                models = provider.get_supported_models()
                assert "gpt-4o-mini" in models
                assert "gpt-4" in models
