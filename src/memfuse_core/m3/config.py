"""M3 configuration management."""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from loguru import logger
import yaml
import os


@dataclass
class M3Config:
    """Configuration for M3 system."""
    
    # Workflow reuse settings
    workflow_reuse_threshold: float = 0.85
    max_workflow_reuse_candidates: int = 5
    
    # Agent settings
    default_agent_timeout: int = 300  # seconds
    max_agent_retries: int = 3
    
    # Embedding settings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    
    # Database settings
    max_workflow_history: int = 1000
    max_lesson_history: int = 1000
    
    # Orchestrator settings
    enable_workflow_reuse: bool = True
    enable_lesson_learning: bool = True
    enable_parallel_execution: bool = True
    # Query-time guidance enrichment
    enable_query_guidance: bool = False
    
    # Agent configurations
    agent_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "M3Config":
        """Create M3Config from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "M3Config":
        """Load M3Config from YAML file."""
        try:
            with open(yaml_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            return cls.from_dict(config_dict.get('m3', {}))
        except Exception as e:
            logger.warning(f"Failed to load M3 config from {yaml_path}: {e}")
            return cls()
    
    @classmethod
    def from_env(cls) -> "M3Config":
        """Load M3Config from environment variables."""
        config = cls()
        
        # Check for environment variable overrides
        if threshold := os.getenv("M3_WORKFLOW_REUSE_THRESHOLD"):
            try:
                config.workflow_reuse_threshold = float(threshold)
            except ValueError:
                logger.warning(f"Invalid M3_WORKFLOW_REUSE_THRESHOLD: {threshold}")
        
        if timeout := os.getenv("M3_AGENT_TIMEOUT"):
            try:
                config.default_agent_timeout = int(timeout)
            except ValueError:
                logger.warning(f"Invalid M3_AGENT_TIMEOUT: {timeout}")
        
        if model := os.getenv("M3_EMBEDDING_MODEL"):
            config.embedding_model = model
        
        if reuse := os.getenv("M3_ENABLE_WORKFLOW_REUSE"):
            config.enable_workflow_reuse = reuse.lower() in ("true", "1", "yes")
        
        if learning := os.getenv("M3_ENABLE_LESSON_LEARNING"):
            config.enable_lesson_learning = learning.lower() in ("true", "1", "yes")
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert M3Config to dictionary."""
        return {
            "workflow_reuse_threshold": self.workflow_reuse_threshold,
            "max_workflow_reuse_candidates": self.max_workflow_reuse_candidates,
            "default_agent_timeout": self.default_agent_timeout,
            "max_agent_retries": self.max_agent_retries,
            "embedding_model": self.embedding_model,
            "embedding_dim": self.embedding_dim,
            "max_workflow_history": self.max_workflow_history,
            "max_lesson_history": self.max_lesson_history,
            "enable_workflow_reuse": self.enable_workflow_reuse,
            "enable_lesson_learning": self.enable_lesson_learning,
            "enable_parallel_execution": self.enable_parallel_execution,
            "enable_query_guidance": self.enable_query_guidance,
            "agent_configs": self.agent_configs,
        }


class M3ConfigManager:
    """Manager for M3 configuration."""
    
    _instance: Optional["M3ConfigManager"] = None
    _config: Optional[M3Config] = None
    
    def __init__(self):
        self._config_paths = [
            "config/m3/default.yaml",
            "config/m3.yaml",
            "/etc/memfuse/m3.yaml",
        ]
    
    @classmethod
    def get_instance(cls) -> "M3ConfigManager":
        """Get singleton instance of M3ConfigManager."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def get_config(self) -> M3Config:
        """Get current M3 configuration."""
        if self._config is None:
            self._config = self._load_config()
        return self._config
    
    def reload_config(self) -> M3Config:
        """Reload M3 configuration from sources."""
        self._config = self._load_config()
        return self._config
    
    def _load_config(self) -> M3Config:
        """Load configuration from various sources."""
        config = M3Config()
        
        # Try to load from YAML files
        for config_path in self._config_paths:
            if os.path.exists(config_path):
                try:
                    config = M3Config.from_yaml(config_path)
                    logger.info(f"Loaded M3 config from {config_path}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load M3 config from {config_path}: {e}")
        
        # Apply environment variable overrides
        env_config = M3Config.from_env()
        
        # Merge configurations (environment variables take precedence)
        merged_config = M3Config(
            workflow_reuse_threshold=env_config.workflow_reuse_threshold if os.getenv("M3_WORKFLOW_REUSE_THRESHOLD") else config.workflow_reuse_threshold,
            max_workflow_reuse_candidates=config.max_workflow_reuse_candidates,
            default_agent_timeout=env_config.default_agent_timeout if os.getenv("M3_AGENT_TIMEOUT") else config.default_agent_timeout,
            max_agent_retries=config.max_agent_retries,
            embedding_model=env_config.embedding_model if os.getenv("M3_EMBEDDING_MODEL") else config.embedding_model,
            embedding_dim=config.embedding_dim,
            max_workflow_history=config.max_workflow_history,
            max_lesson_history=config.max_lesson_history,
            enable_workflow_reuse=env_config.enable_workflow_reuse if os.getenv("M3_ENABLE_WORKFLOW_REUSE") else config.enable_workflow_reuse,
            enable_lesson_learning=env_config.enable_lesson_learning if os.getenv("M3_ENABLE_LESSON_LEARNING") else config.enable_lesson_learning,
            enable_parallel_execution=config.enable_parallel_execution,
            enable_query_guidance=config.enable_query_guidance,
            agent_configs=config.agent_configs,
        )
        
        return merged_config
    
    def update_config(self, **kwargs) -> None:
        """Update configuration values."""
        if self._config is None:
            self._config = self._load_config()
        
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
            else:
                logger.warning(f"Unknown M3 config key: {key}")


# Global configuration instance
def get_m3_config() -> M3Config:
    """Get global M3 configuration."""
    return M3ConfigManager.get_instance().get_config()
