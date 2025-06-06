"""Base classes and interfaces for LLM providers."""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, AsyncGenerator, Union
from enum import Enum

logger = logging.getLogger(__name__)


class LLMModelType(Enum):
    """Supported LLM model types."""
    OPENAI_GPT4 = "gpt-4"
    OPENAI_GPT4_TURBO = "gpt-4-turbo"
    OPENAI_GPT4O = "gpt-4o"
    OPENAI_GPT4O_MINI = "gpt-4o-mini"
    ANTHROPIC_CLAUDE3_SONNET = "claude-3-sonnet-20240229"
    ANTHROPIC_CLAUDE3_HAIKU = "claude-3-haiku-20240307"


@dataclass
class LLMUsage:
    """Token usage information."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class LLMResponse:
    """Response from an LLM provider."""
    content: str
    model: str
    usage: LLMUsage
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None


@dataclass
class LLMRequest:
    """Request to an LLM provider."""
    messages: List[Dict[str, str]]
    model: str
    max_tokens: Optional[int] = None
    temperature: float = 0.3
    stream: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the LLM provider.
        
        Args:
            config: Provider-specific configuration
        """
        self.config = config or {}
        self.name = self.__class__.__name__
        
    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response from LLM.
        
        Args:
            request: LLM request object
            
        Returns:
            LLM response object
        """
        pass
    
    async def generate_batch(self, requests: List[LLMRequest]) -> List[LLMResponse]:
        """Generate responses for multiple requests.
        
        Default implementation processes requests sequentially.
        Providers can override for true batch processing.
        
        Args:
            requests: List of LLM requests
            
        Returns:
            List of LLM responses
        """
        responses = []
        for request in requests:
            try:
                response = await self.generate(request)
                responses.append(response)
            except Exception as e:
                logger.error(f"Error processing batch request: {e}")
                error_response = LLMResponse(
                    content="",
                    model=request.model,
                    usage=LLMUsage(),
                    success=False,
                    error=str(e)
                )
                responses.append(error_response)
        return responses
    
    @abstractmethod
    async def generate_stream(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        """Generate streaming response.
        
        Args:
            request: LLM request object
            
        Yields:
            Streaming content chunks
        """
        pass
    
    async def is_available(self) -> bool:
        """Check if the provider is available.
        
        Returns:
            True if provider is available, False otherwise
        """
        try:
            # Simple test request
            test_request = LLMRequest(
                messages=[{"role": "user", "content": "test"}],
                model=self.get_default_model(),
                max_tokens=1
            )
            response = await self.generate(test_request)
            return response.success
        except Exception as e:
            logger.warning(f"Provider {self.name} availability check failed: {e}")
            return False
    
    @abstractmethod
    def get_default_model(self) -> str:
        """Get the default model for this provider.
        
        Returns:
            Default model name
        """
        pass
    
    def get_supported_models(self) -> List[str]:
        """Get list of supported models.
        
        Returns:
            List of supported model names
        """
        return [self.get_default_model()]
    
    def validate_request(self, request: LLMRequest) -> bool:
        """Validate an LLM request.
        
        Args:
            request: LLM request to validate
            
        Returns:
            True if request is valid, False otherwise
        """
        if not request.messages:
            return False
        
        if not request.model:
            return False
            
        for message in request.messages:
            if "role" not in message or "content" not in message:
                return False
                
        return True


class LLMProviderError(Exception):
    """Base exception for LLM provider errors."""
    pass


class LLMRateLimitError(LLMProviderError):
    """Exception raised when rate limit is exceeded."""
    pass


class LLMAuthenticationError(LLMProviderError):
    """Exception raised when authentication fails."""
    pass


class LLMModelNotFoundError(LLMProviderError):
    """Exception raised when model is not found."""
    pass
