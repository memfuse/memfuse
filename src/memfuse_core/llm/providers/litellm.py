"""LiteLLM provider implementation.

This provider uses the `litellm` package as a compatibility layer to call a
wide range of OpenAI-compatible chat models while keeping the same interface
as our other providers.

Notes:
- If `litellm` is unavailable, the provider will raise on init. Callers may
  catch and fallback to another provider (e.g., OpenAIProvider).
- `generate_structured` mirrors the behavior of the OpenAI provider:
  1) Attempt native structured output via `response_format` (JSON schema).
  2) Fallback to a plain JSON instruction + post-parse.
"""

import logging
from typing import Any, AsyncGenerator, Dict, List, Optional, Type

from ..base import (
    LLMProvider,
    LLMRequest,
    LLMResponse,
    LLMUsage,
    LLMProviderError,
    LLMRateLimitError,
    LLMAuthenticationError,
    LLMModelNotFoundError,
)

logger = logging.getLogger(__name__)

try:
    # LiteLLM async completion API
    from litellm import acompletion
    LITELLM_AVAILABLE = True
except Exception:  # pragma: no cover - defensive import
    LITELLM_AVAILABLE = False
    logger.warning(
        "LiteLLM library not available. Install with: pip install litellm"
    )

try:
    from pydantic import BaseModel
    PYDANTIC_AVAILABLE = True
except Exception:  # pragma: no cover - defensive import
    BaseModel = object  # type: ignore[assignment]
    PYDANTIC_AVAILABLE = False


class LiteLLMProvider(LLMProvider):
    """LiteLLM provider implementation."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        if not LITELLM_AVAILABLE:
            raise LLMProviderError(
                "LiteLLM library not available. Install with: pip install litellm"
            )

        # Supported models (assumes OpenAI-compatible across providers)
        self.supported_models = [
            "gpt-5",
            "gpt-5-mini",
            "gpt-5-nano",
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-4.1-nano",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-3.5-turbo",
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            # x.ai / compatible
            "grok-beta",
            "grok-vision-beta",
            "grok-3-mini",
        ]

    def _acompletion_kwargs(self, request: LLMRequest) -> Dict[str, Any]:
        """Build kwargs for `litellm.acompletion`, mapping config fields.

        Supports passing `api_key`, `api_base`/`base_url`, and `api_version`
        where appropriate (e.g., Azure OpenAI).
        """
        kwargs: Dict[str, Any] = {
            "model": request.model,
            "messages": request.messages,
            "temperature": request.temperature,
        }
        if request.max_tokens:
            kwargs["max_tokens"] = request.max_tokens

        api_key = self.config.get("api_key")
        if api_key:
            kwargs["api_key"] = api_key

        base_url = self.config.get("base_url") or self.config.get("api_base")
        if base_url:
            kwargs["api_base"] = base_url

        api_version = self.config.get("api_version")
        if api_version:
            kwargs["api_version"] = api_version

        timeout = self.config.get("timeout")
        if timeout:
            kwargs["timeout"] = timeout

        # Force OpenAI-style provider when hitting a generic OpenAI-compatible endpoint
        custom_provider = self.config.get("custom_llm_provider")
        if not custom_provider:
            if base_url:
                # If a custom api_base is provided, default to OpenAI provider semantics
                custom_provider = "openai"
        if custom_provider:
            kwargs["custom_llm_provider"] = custom_provider

        return kwargs

    async def generate(self, request: LLMRequest) -> LLMResponse:
        if not self.validate_request(request):
            return LLMResponse(
                content="",
                model=request.model,
                usage=LLMUsage(),
                success=False,
                error="Invalid request",
            )

        try:
            kwargs = self._acompletion_kwargs(request)

            # liteLLM returns an OpenAI-like response object (dict or pydantic)
            response = await acompletion(**kwargs)

            # Normalize content and usage
            # Handle both dict-like and attribute-like surfaces
            choices = getattr(response, "choices", None) or response["choices"]
            choice0 = choices[0]
            message = getattr(choice0, "message", None) or choice0["message"]
            content = getattr(message, "content", None) or message.get("content", "")
            finish_reason = getattr(choice0, "finish_reason", None) or choice0.get(
                "finish_reason"
            )
            model_name = getattr(response, "model", None) or response.get("model", request.model)
            response_id = getattr(response, "id", None) or response.get("id")

            usage_obj = getattr(response, "usage", None) or response.get("usage")
            usage = LLMUsage(
                prompt_tokens=getattr(usage_obj, "prompt_tokens", 0)
                if usage_obj
                else usage_obj.get("prompt_tokens", 0)
                if usage_obj
                else 0,
                completion_tokens=getattr(usage_obj, "completion_tokens", 0)
                if usage_obj
                else usage_obj.get("completion_tokens", 0)
                if usage_obj
                else 0,
                total_tokens=getattr(usage_obj, "total_tokens", 0)
                if usage_obj
                else usage_obj.get("total_tokens", 0)
                if usage_obj
                else 0,
            )

            return LLMResponse(
                content=content or "",
                model=model_name,
                usage=usage,
                metadata={
                    "finish_reason": finish_reason,
                    "response_id": response_id,
                },
                success=True,
            )

        except Exception as e:
            # Map common errors, be conservative
            msg = str(e)
            if "rate" in msg.lower():
                raise LLMRateLimitError(msg)
            if "auth" in msg.lower() or "401" in msg:
                raise LLMAuthenticationError(msg)
            if "model" in msg.lower() and "not" in msg.lower():
                raise LLMModelNotFoundError(msg)
            logger.error(f"LiteLLM API error: {e}")
            return LLMResponse(
                content="",
                model=request.model,
                usage=LLMUsage(),
                success=False,
                error=msg,
            )

    async def generate_structured(
        self, request: LLMRequest, response_format: Type[BaseModel]
    ) -> LLMResponse:
        """Generate structured output via LiteLLM.

        Strategy:
        - If model likely supports OpenAI-style JSON Schema `response_format`,
          pass it directly.
        - Otherwise, add a JSON-only instruction and parse the result.
        """
        if not PYDANTIC_AVAILABLE:
            logger.error("Pydantic not available for structured outputs")
            return LLMResponse(
                content="",
                model=request.model,
                usage=LLMUsage(),
                success=False,
                error="Pydantic not available for structured outputs",
            )

        if not self.validate_request(request):
            return LLMResponse(
                content="",
                model=request.model,
                usage=LLMUsage(),
                success=False,
                error="Invalid request",
            )

        structured_output_models = {
            "gpt-5",
            "gpt-5-mini",
            "gpt-5-nano",
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-4.1-nano",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4o-2024-08-06",
            "gpt-4o-2024-11-20",
            "gpt-4o-mini-2024-07-18",
        }

        supports_structured = request.model in structured_output_models

        # Many OpenAI-compatible proxies don't support "response_format"/structured APIs.
        # Default behavior: if using a custom base_url (non-official OpenAI), skip native
        # structured parsing and rely on JSON-instruction fallback.
        base_url = self.config.get("base_url") or self.config.get("api_base") or ""
        enable_structured_api = self.config.get("enable_structured_api")
        if enable_structured_api is None:
            # Enable only for the official OpenAI endpoint by default
            enable_structured_api = "api.openai.com" in base_url or base_url == ""
        if not enable_structured_api:
            supports_structured = False

        try:
            kwargs = self._acompletion_kwargs(request)

            if supports_structured:
                try:
                    # LiteLLM supports passing the Pydantic class directly;
                    # it will construct the provider-specific schema.
                    kwargs["response_format"] = response_format

                    response = await acompletion(**kwargs)

                    choices = getattr(response, "choices", None) or response["choices"]
                    choice0 = choices[0]
                    message = getattr(choice0, "message", None) or choice0["message"]
                    content = getattr(message, "content", None) or message.get(
                        "content", ""
                    )
                    # Some providers return tool or json_content fields. Prefer parsed JSON.
                    parsed_data = None
                    try:
                        import json
                        # If content is a JSON string, parse and validate
                        if isinstance(content, str) and content.strip().startswith("{"):
                            parsed_data = response_format.model_validate(
                                json.loads(content)
                            )
                    except Exception:
                        parsed_data = None

                    model_name = getattr(response, "model", None) or response.get(
                        "model", request.model
                    )
                    response_id = getattr(response, "id", None) or response.get("id")
                    finish_reason = getattr(choice0, "finish_reason", None) or choice0.get(
                        "finish_reason"
                    )
                    usage_obj = getattr(response, "usage", None) or response.get("usage")
                    usage = LLMUsage(
                        prompt_tokens=getattr(usage_obj, "prompt_tokens", 0)
                        if usage_obj
                        else usage_obj.get("prompt_tokens", 0)
                        if usage_obj
                        else 0,
                        completion_tokens=getattr(usage_obj, "completion_tokens", 0)
                        if usage_obj
                        else usage_obj.get("completion_tokens", 0)
                        if usage_obj
                        else 0,
                        total_tokens=getattr(usage_obj, "total_tokens", 0)
                        if usage_obj
                        else usage_obj.get("total_tokens", 0)
                        if usage_obj
                        else 0,
                    )

                    return LLMResponse(
                        content=content or "",
                        model=model_name,
                        usage=usage,
                        parsed_data=parsed_data,
                        metadata={
                            "finish_reason": finish_reason,
                            "response_id": response_id,
                            "structured_parsing": True,
                        },
                        success=True,
                    )
                except Exception as parse_err:
                    logger.warning(
                        f"LiteLLM structured parsing failed, using JSON fallback: {parse_err}"
                    )

            # Fallback path: add instruction and parse JSON
            import copy
            import json

            json_instruction = (
                "Please respond with valid JSON that matches this schema:\n"
                f"{response_format.model_json_schema()}\n\n"
                "Your response should be pure JSON with no additional text."
            )
            modified_messages = copy.deepcopy(request.messages)
            if modified_messages and modified_messages[-1].get("role") == "user":
                modified_messages[-1]["content"] = (
                    modified_messages[-1].get("content", "") + f"\n\n{json_instruction}"
                )
            else:
                modified_messages.append({"role": "system", "content": json_instruction})

            kwargs["messages"] = modified_messages
            response = await acompletion(**kwargs)

            choices = getattr(response, "choices", None) or response["choices"]
            choice0 = choices[0]
            message = getattr(choice0, "message", None) or choice0["message"]
            content = getattr(message, "content", None) or message.get("content", "")

            usage_obj = getattr(response, "usage", None) or response.get("usage")
            usage = LLMUsage(
                prompt_tokens=getattr(usage_obj, "prompt_tokens", 0)
                if usage_obj
                else usage_obj.get("prompt_tokens", 0)
                if usage_obj
                else 0,
                completion_tokens=getattr(usage_obj, "completion_tokens", 0)
                if usage_obj
                else usage_obj.get("completion_tokens", 0)
                if usage_obj
                else 0,
                total_tokens=getattr(usage_obj, "total_tokens", 0)
                if usage_obj
                else usage_obj.get("total_tokens", 0)
                if usage_obj
                else 0,
            )

            parsed_data = None
            try:
                content_stripped = (content or "").strip()
                if content_stripped.startswith("{") and content_stripped.endswith("}"):
                    parsed_data = response_format.model_validate(json.loads(content_stripped))
                else:
                    # Try to extract JSON substring
                    start_idx = content_stripped.find("{")
                    end_idx = content_stripped.rfind("}")
                    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                        parsed_data = response_format.model_validate(
                            json.loads(content_stripped[start_idx : end_idx + 1])
                        )
            except Exception as json_error:  # pragma: no cover - defensive
                logger.warning(f"LiteLLM JSON parse fallback failed: {json_error}")

            model_name = getattr(response, "model", None) or response.get(
                "model", request.model
            )
            response_id = getattr(response, "id", None) or response.get("id")
            finish_reason = getattr(choice0, "finish_reason", None) or choice0.get(
                "finish_reason"
            )

            return LLMResponse(
                content=content or "",
                model=model_name,
                usage=usage,
                parsed_data=parsed_data,
                metadata={
                    "finish_reason": finish_reason,
                    "response_id": response_id,
                    "structured_parsing": False,
                    "json_fallback": True,
                },
                success=True,
            )

        except Exception as e:
            logger.error(f"LiteLLM structured generation error: {e}")
            return LLMResponse(
                content="",
                model=request.model,
                usage=LLMUsage(),
                success=False,
                error=str(e),
            )

    async def generate_stream(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        """Streaming via LiteLLM using async iterator when available."""
        try:
            kwargs = self._acompletion_kwargs(request)
            kwargs["stream"] = True
            stream = await acompletion(**kwargs)
            async for chunk in stream:
                text: Optional[str] = None
                # dict-like form
                if isinstance(chunk, dict):
                    try:
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        text = delta.get("content") or ""
                    except Exception:
                        text = None
                else:
                    # object-like form
                    try:
                        delta_obj = chunk.choices[0].delta  # type: ignore[attr-defined]
                        text = getattr(delta_obj, "content", None)
                    except Exception:
                        text = None
                if text:
                    yield text
        except Exception as e:
            logger.error(f"LiteLLM streaming error: {e}")
            yield f"Error: {e}"

    def get_default_model(self) -> str:
        return "grok-3-mini"

    def get_supported_models(self) -> List[str]:
        return self.supported_models.copy()

    def validate_request(self, request: LLMRequest) -> bool:
        if not super().validate_request(request):
            return False
        # Allow unknown models; upstream may still handle them.
        return True
