"""Configuration management for LLM providers."""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class LLMConfig:
    """Configuration for LLM providers."""
    
    # Provider selection
    provider: str = "openai"  # openai, anthropic, local, mock
    
    # Model configuration
    model: str = "grok-3-mini"
    max_tokens: Optional[int] = 150
    temperature: float = 0.3
    
    # API configuration
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    
    # Rate limiting
    max_requests_per_minute: int = 60
    max_tokens_per_minute: int = 40000
    
    # Retry configuration
    max_retries: int = 3
    retry_delay: float = 1.0
    backoff_factor: float = 2.0
    
    # Batch processing
    batch_size: int = 5
    batch_delay: float = 1.0
    
    # Caching
    enable_cache: bool = True
    cache_ttl: int = 3600  # 1 hour
    
    # Timeout
    timeout: float = 30.0
    
    # Additional provider-specific config
    provider_config: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Create configuration from environment variables."""
        return cls(
            provider=os.getenv("MEMFUSE_LLM_PROVIDER", "openai"),
            model=os.getenv("MEMFUSE_LLM_MODEL", "grok-3-mini"),
            max_tokens=int(os.getenv("MEMFUSE_LLM_MAX_TOKENS", "150")),
            temperature=float(os.getenv("MEMFUSE_LLM_TEMPERATURE", "0.3")),
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("MEMFUSE_LLM_BASE_URL"),
            max_requests_per_minute=int(os.getenv("MEMFUSE_LLM_MAX_REQUESTS_PER_MINUTE", "60")),
            max_tokens_per_minute=int(os.getenv("MEMFUSE_LLM_MAX_TOKENS_PER_MINUTE", "40000")),
            max_retries=int(os.getenv("MEMFUSE_LLM_MAX_RETRIES", "3")),
            retry_delay=float(os.getenv("MEMFUSE_LLM_RETRY_DELAY", "1.0")),
            batch_size=int(os.getenv("MEMFUSE_LLM_BATCH_SIZE", "5")),
            batch_delay=float(os.getenv("MEMFUSE_LLM_BATCH_DELAY", "1.0")),
            enable_cache=os.getenv("MEMFUSE_LLM_ENABLE_CACHE", "true").lower() == "true",
            cache_ttl=int(os.getenv("MEMFUSE_LLM_CACHE_TTL", "3600")),
            timeout=float(os.getenv("MEMFUSE_LLM_TIMEOUT", "30.0")),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "provider": self.provider,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "api_key": "***" if self.api_key else None,  # Mask API key
            "base_url": self.base_url,
            "max_requests_per_minute": self.max_requests_per_minute,
            "max_tokens_per_minute": self.max_tokens_per_minute,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "backoff_factor": self.backoff_factor,
            "batch_size": self.batch_size,
            "batch_delay": self.batch_delay,
            "enable_cache": self.enable_cache,
            "cache_ttl": self.cache_ttl,
            "timeout": self.timeout,
            "provider_config": self.provider_config,
        }
    
    def validate(self) -> bool:
        """Validate configuration."""
        if not self.provider:
            return False
            
        if not self.model:
            return False
            
        if self.temperature < 0 or self.temperature > 2:
            return False
            
        if self.max_tokens is not None and self.max_tokens <= 0:
            return False
            
        if self.max_retries < 0:
            return False
            
        if self.batch_size <= 0:
            return False
            
        return True


# Default configurations for different providers
DEFAULT_OPENAI_CONFIG = LLMConfig(
    provider="openai",
    model="grok-3-mini",
    max_tokens=150,
    temperature=0.3,
    max_requests_per_minute=60,
    max_tokens_per_minute=40000,
)

DEFAULT_ANTHROPIC_CONFIG = LLMConfig(
    provider="anthropic",
    model="claude-3-haiku-20240307",
    max_tokens=150,
    temperature=0.3,
    max_requests_per_minute=50,
    max_tokens_per_minute=30000,
)

DEFAULT_MOCK_CONFIG = LLMConfig(
    provider="mock",
    model="mock-model",
    max_tokens=150,
    temperature=0.3,
    max_requests_per_minute=1000,
    max_tokens_per_minute=100000,
)
