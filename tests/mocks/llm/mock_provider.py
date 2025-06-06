"""Mock provider implementation for testing."""

import asyncio
import logging
from typing import List, Dict, Any, Optional, AsyncGenerator

from src.memfuse_core.llm.base import LLMProvider, LLMRequest, LLMResponse, LLMUsage

logger = logging.getLogger(__name__)


class MockProvider(LLMProvider):
    """Mock LLM provider for testing and development."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize mock provider.
        
        Args:
            config: Provider configuration
        """
        super().__init__(config)
        
        # Mock configuration
        self.response_delay = self.config.get("response_delay", 0.1)
        self.fail_rate = self.config.get("fail_rate", 0.0)  # 0.0 = never fail, 1.0 = always fail
        self.custom_responses = self.config.get("custom_responses", {})
        
        # Counter for generating unique responses
        self._request_counter = 0
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate mock response.
        
        Args:
            request: LLM request object
            
        Returns:
            Mock LLM response
        """
        if not self.validate_request(request):
            return LLMResponse(
                content="",
                model=request.model,
                usage=LLMUsage(),
                success=False,
                error="Invalid request"
            )
        
        # Simulate API delay
        await asyncio.sleep(self.response_delay)
        
        # Simulate failures
        import random
        if random.random() < self.fail_rate:
            return LLMResponse(
                content="",
                model=request.model,
                usage=LLMUsage(),
                success=False,
                error="Mock provider simulated failure"
            )
        
        self._request_counter += 1
        
        # Generate mock response content
        content = self._generate_mock_content(request)
        
        # Mock usage statistics
        prompt_tokens = sum(len(msg.get("content", "").split()) for msg in request.messages)
        completion_tokens = len(content.split())
        
        return LLMResponse(
            content=content,
            model=request.model,
            usage=LLMUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            ),
            metadata={
                "mock_request_id": f"mock_{self._request_counter}",
                "mock_provider": True,
            },
            success=True
        )
    
    def _generate_mock_content(self, request: LLMRequest) -> str:
        """Generate mock content based on request."""
        # Check for custom responses
        last_message = request.messages[-1].get("content", "") if request.messages else ""
        
        # Look for custom response patterns
        for pattern, response in self.custom_responses.items():
            if pattern.lower() in last_message.lower():
                return response
        
        # Check for contextual chunking pattern
        if "conversation_context" in last_message and "message_chunk" in last_message:
            return self._generate_contextual_description(last_message)
        
        # Default responses based on content
        if "hello" in last_message.lower():
            return "Hello! This is a mock response from the MockProvider."
        
        if "test" in last_message.lower():
            return "This is a test response from the mock LLM provider."
        
        if "context" in last_message.lower():
            return "Mock contextual description: This chunk discusses the main topic with relevant background information."
        
        # Generic response
        return f"Mock response #{self._request_counter}: This is a simulated response to your message about '{last_message[:50]}...'"
    
    def _generate_contextual_description(self, prompt: str) -> str:
        """Generate mock contextual description for chunking."""
        # Extract some keywords from the prompt for more realistic responses
        keywords = []
        if "user" in prompt.lower():
            keywords.append("user interaction")
        if "assistant" in prompt.lower():
            keywords.append("assistant response")
        if "question" in prompt.lower():
            keywords.append("question-answer")
        if "help" in prompt.lower():
            keywords.append("help request")
        
        if keywords:
            context = f"This chunk contains {', '.join(keywords)} within the conversation flow."
        else:
            context = "This chunk represents a conversational exchange between participants."
        
        return f"Mock contextual description: {context} The content relates to the ongoing discussion and provides relevant information for retrieval purposes."
    
    async def generate_stream(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        """Generate mock streaming response.
        
        Args:
            request: LLM request object
            
        Yields:
            Mock streaming content chunks
        """
        if not self.validate_request(request):
            yield "Error: Invalid request"
            return
        
        # Generate full response first
        full_response = self._generate_mock_content(request)
        words = full_response.split()
        
        # Stream word by word
        for word in words:
            await asyncio.sleep(0.05)  # Small delay between words
            yield word + " "
        
        # Final newline
        yield "\n"
    
    def get_default_model(self) -> str:
        """Get the default mock model."""
        return "mock-model"
    
    def get_supported_models(self) -> List[str]:
        """Get list of supported mock models."""
        return ["mock-model", "mock-gpt-4", "mock-claude"]
    
    async def is_available(self) -> bool:
        """Mock provider is always available."""
        return True
    
    def set_custom_response(self, pattern: str, response: str):
        """Set a custom response for a specific pattern.
        
        Args:
            pattern: Text pattern to match in requests
            response: Custom response to return
        """
        self.custom_responses[pattern] = response
    
    def set_fail_rate(self, fail_rate: float):
        """Set the failure rate for testing error handling.
        
        Args:
            fail_rate: Failure rate between 0.0 and 1.0
        """
        self.fail_rate = max(0.0, min(1.0, fail_rate))
    
    def reset_counter(self):
        """Reset the request counter."""
        self._request_counter = 0
