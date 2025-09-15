"""High-level chat interface for LLM interactions."""

import json
import logging
import os
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Try to import OpenAI
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class ChatLLM:
    """High-level chat interface for LLM interactions.
    
    This implementation follows the pattern from memfuse_mvp/memfuse/llm.py
    but adapts to work with the current MemFuse core architecture.
    """
    
    def __init__(self, settings=None, **kwargs):
        """Initialize ChatLLM with settings or configuration.
        
        Args:
            settings: Settings object (memfuse_mvp style) or None for default config
            **kwargs: Additional configuration parameters
        """
        # Handle different initialization patterns
        if settings is not None:
            # memfuse_mvp style initialization
            self._init_from_settings(settings)
        else:
            # New style initialization with config
            self._init_from_config(**kwargs)
    
    def _init_from_settings(self, settings):
        """Initialize from settings object (memfuse_mvp style)."""
        try:
            if OPENAI_AVAILABLE:
                # Only pass base_url if provided to allow library defaults
                if getattr(settings, 'openai_base_url', None):
                    self.client = OpenAI(
                        api_key=settings.openai_api_key, 
                        base_url=settings.openai_base_url
                    )
                else:
                    self.client = OpenAI(api_key=settings.openai_api_key)
                
                self.model = settings.openai_model
                self.system_prompt_text = getattr(settings, 'system_prompt', '')
                self.assistant_role_target = getattr(settings, "openai_assistant_role", "assistant")
                # Expose a thin completion API for extractor usage
                self._raw_client = self.client
            else:
                logger.warning("OpenAI not available, using mock client")
                self.client = MockOpenAIClient()
                self._raw_client = self.client
                self.model = getattr(settings, 'openai_model', 'mock-model')
                self.system_prompt_text = getattr(settings, 'system_prompt', '')
                self.assistant_role_target = "assistant"
        except Exception as e:
            logger.warning(f"Failed to initialize from settings: {e}, using mock client")
            self._init_mock()
    
    def _init_from_config(self, **kwargs):
        """Initialize from configuration parameters."""
        try:
            if OPENAI_AVAILABLE:
                api_key = kwargs.get('api_key') or os.getenv('OPENAI_API_KEY', '')
                base_url = kwargs.get('base_url') or os.getenv('OPENAI_BASE_URL')
                
                if not api_key:
                    logger.warning("No OpenAI API key provided, using mock client")
                    self._init_mock()
                    return
                
                if base_url:
                    self.client = OpenAI(api_key=api_key, base_url=base_url)
                else:
                    self.client = OpenAI(api_key=api_key)
                
                self.model = kwargs.get('model') or os.getenv('OPENAI_COMPATIBLE_MODEL', 'gpt-3.5-turbo')
                self.system_prompt_text = kwargs.get('system_prompt', '')
                self.assistant_role_target = kwargs.get('assistant_role', 'assistant')
                self._raw_client = self.client
            else:
                self._init_mock()
        except Exception as e:
            logger.warning(f"Failed to initialize from config: {e}, using mock client")
            self._init_mock()
    
    def _init_mock(self):
        """Initialize with mock client for testing/offline scenarios."""
        self.client = MockOpenAIClient()
        self._raw_client = self.client
        self.model = 'mock-model'
        self.system_prompt_text = ''
        self.assistant_role_target = 'assistant'
    
    def chat(self, system_prompt: str, messages: List[Dict[str, str]]) -> str:
        """Chat interface following memfuse_mvp pattern.
        
        Args:
            system_prompt: System prompt
            messages: List of chat messages with role and content
            
        Returns:
            Generated response text
        """
        try:
            # messages: list of {role, content}
            # Some OpenAI-compatible backends (e.g., Google Gemini proxy) use role "model" instead of "assistant".
            # Normalize roles to backend expectations based on env setting.
            normalized: List[Dict[str, str]] = []
            
            # prepend system prompt
            normalized.append({"role": "system", "content": system_prompt})
            
            for m in messages:
                role = m.get("role", "")
                content = m.get("content", "")
                if role in ("assistant", "ai"):
                    # Map to target role per backend requirements (assistant or model)
                    role = self.assistant_role_target
                elif role == "user":
                    role = "user"
                elif role == "system":
                    role = "system"
                else:
                    # fall back to user if unknown
                    role = "user"
                normalized.append({"role": role, "content": content})
            
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=normalized,
            )
            return completion.choices[0].message.content or ""
            
        except Exception as e:
            logger.error(f"Chat request failed: {e}")
            return f"Error: {str(e)}"
    
    async def chat_async(self, system: str, messages: List[Dict[str, str]], **kwargs) -> str:
        """Async chat interface for compatibility.
        
        Args:
            system: System prompt
            messages: List of chat messages
            **kwargs: Additional parameters for the LLM request
            
        Returns:
            Generated response text
        """
        return self.chat(system, messages)
    
    def completion_json(self, system_prompt: str, user_prompt: str) -> str:
        """JSON completion following memfuse_mvp pattern.
        
        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            
        Returns:
            Generated JSON response as string
        """
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            completion = self._raw_client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
            )
            return completion.choices[0].message.content or "{}"
        except Exception as e:
            logger.error(f"JSON completion failed: {e}")
            # Return a default structure that matches expected format
            return '{"steps": []}'
    
    async def generate_with_context(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """Generate response with additional context.
        
        Args:
            prompt: Main prompt
            context: Additional context information
            **kwargs: Additional parameters
            
        Returns:
            Generated response
        """
        # Build system prompt with context
        system_prompt = "You are a helpful AI assistant."
        if context:
            system_prompt += f"\n\nContext: {json.dumps(context, indent=2)}"
        
        messages = [{"role": "user", "content": prompt}]
        return await self.chat(system_prompt, messages, **kwargs)
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the current provider.
        
        Returns:
            Provider information dictionary
        """
        return {
            "provider_name": self.provider.name,
            "default_model": self.provider.get_default_model(),
            "supported_models": self.provider.get_supported_models(),
            "config": self.config.to_dict() if hasattr(self.config, 'to_dict') else {}
        }


class MockChoice:
    """Mock choice for OpenAI response."""
    def __init__(self, content: str):
        self.message = MockMessage(content)


class MockMessage:
    """Mock message for OpenAI response."""
    def __init__(self, content: str):
        self.content = content


class MockCompletion:
    """Mock completion for OpenAI response."""
    def __init__(self, content: str):
        self.choices = [MockChoice(content)]


class MockChatCompletions:
    """Mock chat completions for OpenAI API."""
    
    def create(self, model: str, messages: List[Dict[str, str]], **kwargs) -> MockCompletion:
        """Create mock completion."""
        # Extract the last user message for context
        user_messages = [msg for msg in messages if msg.get("role") == "user"]
        last_message = user_messages[-1]["content"] if user_messages else ""
        
        # Check if JSON format is requested
        response_format = kwargs.get("response_format", {})
        if response_format.get("type") == "json_object":
            # Generate contextual JSON response
            if "steps" in last_message.lower() or "plan" in last_message.lower():
                content = '{"steps": [{"agent": "RAGQueryAgent", "input": {"query": "mock query"}}, {"agent": "ReportGenerationAgent", "input": {}}]}'
            else:
                content = '{"result": "mock json response"}'
        else:
            # Generate regular text response
            if "report" in last_message.lower():
                content = "This is a mock report generated for testing purposes."
            else:
                content = f"This is a mock response to: {last_message[:100]}..."
        
        return MockCompletion(content)


class MockChat:
    """Mock chat for OpenAI API."""
    def __init__(self):
        self.completions = MockChatCompletions()


class MockOpenAIClient:
    """Mock OpenAI client for testing and offline scenarios."""
    
    def __init__(self):
        self.chat = MockChat()


# Convenience function for creating ChatLLM instances
def create_chat_llm(**config_kwargs) -> ChatLLM:
    """Create a ChatLLM instance with configuration.
    
    Args:
        **config_kwargs: Configuration parameters
        
    Returns:
        ChatLLM instance
    """
    return ChatLLM(**config_kwargs)