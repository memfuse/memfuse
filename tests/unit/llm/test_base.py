"""Unit tests for LLM base classes."""

import pytest
import asyncio
from unittest.mock import AsyncMock

from src.memfuse_core.llm.base import (
    LLMProvider, LLMRequest, LLMResponse, LLMUsage,
    LLMProviderError, LLMRateLimitError, LLMAuthenticationError
)


class TestLLMRequest:
    """Test cases for LLMRequest."""
    
    def test_init(self):
        """Test LLMRequest initialization."""
        request = LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            model="gpt-4o-mini"
        )
        
        assert request.messages == [{"role": "user", "content": "Hello"}]
        assert request.model == "gpt-4o-mini"
        assert request.max_tokens is None
        assert request.temperature == 0.3
        assert not request.stream
        assert request.metadata == {}
    
    def test_init_with_all_params(self):
        """Test LLMRequest initialization with all parameters."""
        metadata = {"test": "value"}
        request = LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            model="gpt-4",
            max_tokens=100,
            temperature=0.7,
            stream=True,
            metadata=metadata
        )
        
        assert request.max_tokens == 100
        assert request.temperature == 0.7
        assert request.stream
        assert request.metadata == metadata


class TestLLMResponse:
    """Test cases for LLMResponse."""
    
    def test_init(self):
        """Test LLMResponse initialization."""
        usage = LLMUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        response = LLMResponse(
            content="Hello world",
            model="gpt-4o-mini",
            usage=usage
        )
        
        assert response.content == "Hello world"
        assert response.model == "gpt-4o-mini"
        assert response.usage == usage
        assert response.metadata == {}
        assert response.success
        assert response.error is None
    
    def test_init_with_error(self):
        """Test LLMResponse initialization with error."""
        usage = LLMUsage()
        response = LLMResponse(
            content="",
            model="gpt-4o-mini",
            usage=usage,
            success=False,
            error="API error"
        )
        
        assert not response.success
        assert response.error == "API error"


class TestLLMUsage:
    """Test cases for LLMUsage."""
    
    def test_init(self):
        """Test LLMUsage initialization."""
        usage = LLMUsage()
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0
    
    def test_init_with_values(self):
        """Test LLMUsage initialization with values."""
        usage = LLMUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 20
        assert usage.total_tokens == 30


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing."""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.generate_mock = AsyncMock()
        self.generate_stream_mock = AsyncMock()
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        return await self.generate_mock(request)
    
    async def generate_stream(self, request: LLMRequest):
        async for chunk in self.generate_stream_mock(request):
            yield chunk
    
    def get_default_model(self) -> str:
        return "mock-model"


class TestLLMProvider:
    """Test cases for LLMProvider base class."""
    
    def test_init(self):
        """Test LLMProvider initialization."""
        provider = MockLLMProvider()
        assert provider.config == {}
        assert provider.name == "MockLLMProvider"
    
    def test_init_with_config(self):
        """Test LLMProvider initialization with config."""
        config = {"api_key": "test", "timeout": 60}
        provider = MockLLMProvider(config)
        assert provider.config == config
    
    @pytest.mark.asyncio
    async def test_generate_batch_success(self):
        """Test successful batch generation."""
        provider = MockLLMProvider()
        
        # Mock successful responses
        response1 = LLMResponse("Response 1", "mock-model", LLMUsage())
        response2 = LLMResponse("Response 2", "mock-model", LLMUsage())
        provider.generate_mock.side_effect = [response1, response2]
        
        requests = [
            LLMRequest([{"role": "user", "content": "Hello 1"}], "mock-model"),
            LLMRequest([{"role": "user", "content": "Hello 2"}], "mock-model")
        ]
        
        responses = await provider.generate_batch(requests)
        
        assert len(responses) == 2
        assert responses[0].content == "Response 1"
        assert responses[1].content == "Response 2"
        assert provider.generate_mock.call_count == 2
    
    @pytest.mark.asyncio
    async def test_generate_batch_with_error(self):
        """Test batch generation with error."""
        provider = MockLLMProvider()
        
        # Mock one success and one error
        response1 = LLMResponse("Response 1", "mock-model", LLMUsage())
        provider.generate_mock.side_effect = [response1, Exception("API error")]
        
        requests = [
            LLMRequest([{"role": "user", "content": "Hello 1"}], "mock-model"),
            LLMRequest([{"role": "user", "content": "Hello 2"}], "mock-model")
        ]
        
        responses = await provider.generate_batch(requests)
        
        assert len(responses) == 2
        assert responses[0].content == "Response 1"
        assert responses[0].success
        assert not responses[1].success
        assert "API error" in responses[1].error
    
    @pytest.mark.asyncio
    async def test_is_available_success(self):
        """Test availability check success."""
        provider = MockLLMProvider()
        provider.generate_mock.return_value = LLMResponse("test", "mock-model", LLMUsage(), success=True)
        
        available = await provider.is_available()
        assert available
    
    @pytest.mark.asyncio
    async def test_is_available_failure(self):
        """Test availability check failure."""
        provider = MockLLMProvider()
        provider.generate_mock.side_effect = Exception("Connection error")
        
        available = await provider.is_available()
        assert not available
    
    def test_validate_request_valid(self):
        """Test request validation with valid request."""
        provider = MockLLMProvider()
        request = LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            model="mock-model"
        )
        
        assert provider.validate_request(request)
    
    def test_validate_request_invalid_no_messages(self):
        """Test request validation with no messages."""
        provider = MockLLMProvider()
        request = LLMRequest(messages=[], model="mock-model")
        
        assert not provider.validate_request(request)
    
    def test_validate_request_invalid_no_model(self):
        """Test request validation with no model."""
        provider = MockLLMProvider()
        request = LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            model=""
        )
        
        assert not provider.validate_request(request)
    
    def test_validate_request_invalid_message_format(self):
        """Test request validation with invalid message format."""
        provider = MockLLMProvider()
        request = LLMRequest(
            messages=[{"content": "Hello"}],  # Missing role
            model="mock-model"
        )
        
        assert not provider.validate_request(request)
    
    def test_get_supported_models(self):
        """Test getting supported models."""
        provider = MockLLMProvider()
        models = provider.get_supported_models()
        assert models == ["mock-model"]


class TestLLMExceptions:
    """Test cases for LLM exceptions."""
    
    def test_llm_provider_error(self):
        """Test LLMProviderError."""
        error = LLMProviderError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)
    
    def test_llm_rate_limit_error(self):
        """Test LLMRateLimitError."""
        error = LLMRateLimitError("Rate limit exceeded")
        assert str(error) == "Rate limit exceeded"
        assert isinstance(error, LLMProviderError)
    
    def test_llm_authentication_error(self):
        """Test LLMAuthenticationError."""
        error = LLMAuthenticationError("Auth failed")
        assert str(error) == "Auth failed"
        assert isinstance(error, LLMProviderError)
