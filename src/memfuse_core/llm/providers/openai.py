"""OpenAI provider implementation."""

import asyncio
import logging
from typing import List, Dict, Any, Optional, AsyncGenerator

from ..base import LLMProvider, LLMRequest, LLMResponse, LLMUsage
from ..base import LLMProviderError, LLMRateLimitError, LLMAuthenticationError, LLMModelNotFoundError

logger = logging.getLogger(__name__)

try:
    import openai
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning(
        "OpenAI library not available. Install with: pip install openai")


class OpenAIProvider(LLMProvider):
    """OpenAI provider implementation."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize OpenAI provider.

        Args:
            config: Provider configuration including api_key, base_url, etc.
        """
        super().__init__(config)

        if not OPENAI_AVAILABLE:
            raise LLMProviderError(
                "OpenAI library not available. Install with: pip install openai")

        # Extract API config - support both OpenAI and x.ai
        api_key = self.config.get("api_key")
        base_url = self.config.get("base_url")
        timeout = self.config.get("timeout", 30.0)

        # Auto-detect x.ai usage and prefer XAI_API_KEY if available
        if not api_key and base_url and "x.ai" in base_url:
            import os
            api_key = os.getenv("XAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        elif not api_key:
            import os
            api_key = os.getenv("OPENAI_API_KEY")

        # Initialize OpenAI client
        client_kwargs = {"timeout": timeout}
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = AsyncOpenAI(**client_kwargs)

        # Supported models (including x.ai models)
        self.supported_models = [
            "gpt-4-turbo",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-3.5-turbo",
            # x.ai models
            "grok-beta",
            "grok-vision-beta",
            "grok-3-mini",
        ]

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response from OpenAI.

        Args:
            request: LLM request object

        Returns:
            LLM response object
        """
        if not self.validate_request(request):
            return LLMResponse(
                content="",
                model=request.model,
                usage=LLMUsage(),
                success=False,
                error="Invalid request"
            )

        try:
            # Prepare OpenAI request
            openai_kwargs = {
                "model": request.model,
                "messages": request.messages,
                "temperature": request.temperature,
                "stream": request.stream,
            }

            if request.max_tokens:
                openai_kwargs["max_tokens"] = request.max_tokens

            # Make API call
            response = await self.client.chat.completions.create(**openai_kwargs)

            # Extract response data
            content = response.choices[0].message.content or ""
            usage = LLMUsage(
                prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                completion_tokens=response.usage.completion_tokens if response.usage else 0,
                total_tokens=response.usage.total_tokens if response.usage else 0,
            )

            return LLMResponse(
                content=content,
                model=response.model,
                usage=usage,
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                    "response_id": response.id,
                },
                success=True
            )

        except openai.RateLimitError as e:
            logger.warning(f"OpenAI rate limit exceeded: {e}")
            raise LLMRateLimitError(f"Rate limit exceeded: {e}")

        except openai.AuthenticationError as e:
            logger.error(f"OpenAI authentication failed: {e}")
            raise LLMAuthenticationError(f"Authentication failed: {e}")

        except openai.NotFoundError as e:
            logger.error(f"OpenAI model not found: {e}")
            raise LLMModelNotFoundError(f"Model not found: {e}")

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return LLMResponse(
                content="",
                model=request.model,
                usage=LLMUsage(),
                success=False,
                error=str(e)
            )

    async def generate_batch(self, requests: List[LLMRequest]) -> List[LLMResponse]:
        """Generate responses for multiple requests with rate limiting.

        Args:
            requests: List of LLM requests

        Returns:
            List of LLM responses
        """
        batch_size = self.config.get("batch_size", 5)
        batch_delay = self.config.get("batch_delay", 1.0)

        responses = []

        for i in range(0, len(requests), batch_size):
            batch = requests[i:i + batch_size]

            # Process batch concurrently
            batch_tasks = [self.generate(request) for request in batch]
            batch_responses = await asyncio.gather(*batch_tasks, return_exceptions=True)

            # Handle exceptions
            for j, response in enumerate(batch_responses):
                if isinstance(response, Exception):
                    logger.error(f"Error in batch request {i+j}: {response}")
                    error_response = LLMResponse(
                        content="",
                        model=batch[j].model,
                        usage=LLMUsage(),
                        success=False,
                        error=str(response)
                    )
                    responses.append(error_response)
                else:
                    responses.append(response)

            # Rate limiting delay between batches
            if i + batch_size < len(requests):
                await asyncio.sleep(batch_delay)

        return responses

    async def generate_stream(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        """Generate streaming response from OpenAI.

        Args:
            request: LLM request object

        Yields:
            Streaming content chunks
        """
        if not self.validate_request(request):
            return

        try:
            # Prepare streaming request
            openai_kwargs = {
                "model": request.model,
                "messages": request.messages,
                "temperature": request.temperature,
                "stream": True,
            }

            if request.max_tokens:
                openai_kwargs["max_tokens"] = request.max_tokens

            # Make streaming API call
            stream = await self.client.chat.completions.create(**openai_kwargs)

            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            yield f"Error: {e}"

    def get_default_model(self) -> str:
        """Get the default model (using x.ai grok-3-mini)."""
        return "grok-3-mini"

    def get_supported_models(self) -> List[str]:
        """Get list of supported OpenAI models."""
        return self.supported_models.copy()

    def validate_request(self, request: LLMRequest) -> bool:
        """Validate OpenAI-specific request."""
        if not super().validate_request(request):
            return False

        # Check if model is supported
        if request.model not in self.supported_models:
            logger.warning(
                f"Model {request.model} not in supported models: {self.supported_models}")
            # Don't fail validation, let OpenAI API handle it

        return True
