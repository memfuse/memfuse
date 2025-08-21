"""Configuration factory for buffer components.

This module provides a centralized configuration factory that allows each buffer
component to manage its own configuration autonomously while maintaining consistency
and providing sensible defaults.
"""

from typing import Dict, Any, Optional, Type
from loguru import logger


class ComponentConfigFactory:
    """Factory for creating component-specific configurations.
    
    This factory allows each component to define its own configuration schema
    and defaults while providing a unified interface for configuration management.
    """
    
    # Component configuration schemas with defaults
    COMPONENT_SCHEMAS = {
        'round_buffer': {
            'max_tokens': 800,
            'max_size': 5,
            'token_model': 'gpt-4o-mini',
            'enable_session_tracking': True,
            'auto_transfer_threshold': 0.8
        },
        'hybrid_buffer': {
            'max_size': 5,
            'chunk_strategy': 'message',
            'embedding_model': 'all-MiniLM-L6-v2',
            'enable_auto_flush': True,
            'auto_flush_interval': 60.0,
            'chunk_overlap': 0.1,
            'force_flush_timeout': 1800.0  # 30 minutes default
        },
        'query_buffer': {
            'max_size': 15,
            'cache_size': 100,
            'default_sort_by': 'score',
            'default_order': 'desc',
            'enable_session_queries': True,
            'cache_ttl': 300
        },
        'flush_manager': {
            'max_workers': 3,
            'max_queue_size': 100,
            'default_timeout': 30.0,
            'flush_interval': 60.0,
            'enable_auto_flush': True,
            'retry_attempts': 3,
            'retry_delay': 1.0
        },
        'speculative_buffer': {
            'max_size': 10,
            'context_window': 3,
            'prediction_strategy': 'semantic_similarity',
            'enable_learning': True,
            'prediction_threshold': 0.7
        },
        'write_buffer': {
            'enable_batch_optimization': True,
            'batch_size_threshold': 10,
            'enable_statistics': True
        }
    }
    
    @classmethod
    def create_component_config(cls, component_name: str, 
                               user_config: Optional[Dict[str, Any]] = None,
                               global_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create configuration for a specific component.
        
        Args:
            component_name: Name of the component
            user_config: User-provided configuration overrides
            global_config: Global configuration context
            
        Returns:
            Complete configuration dictionary for the component
        """
        if component_name not in cls.COMPONENT_SCHEMAS:
            logger.warning(f"Unknown component: {component_name}, using empty config")
            return user_config or {}
        
        # Start with component defaults
        config = cls.COMPONENT_SCHEMAS[component_name].copy()
        
        # Apply global configuration context if available
        if global_config:
            cls._apply_global_context(config, component_name, global_config)
        
        # Apply user overrides
        if user_config:
            cls._deep_merge(config, user_config)
        
        # Validate configuration
        cls._validate_component_config(component_name, config)
        
        logger.debug(f"Created config for {component_name}: {config}")
        return config
    
    @classmethod
    def create_write_buffer_config(cls, global_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create complete WriteBuffer configuration with all sub-components.
        
        Args:
            global_config: Global configuration dictionary
            
        Returns:
            Complete WriteBuffer configuration
        """
        buffer_config = global_config.get('buffer', {}) if global_config else {}
        
        # Create configurations for all WriteBuffer sub-components
        write_buffer_config = {
            'round_buffer': cls.create_component_config(
                'round_buffer',
                buffer_config.get('round_buffer'),
                global_config
            ),
            'hybrid_buffer': cls.create_component_config(
                'hybrid_buffer',
                buffer_config.get('hybrid_buffer'),
                global_config
            ),
            'flush_manager': cls.create_component_config(
                'flush_manager',
                cls._map_performance_to_flush_config(buffer_config.get('performance', {})),
                global_config
            ),
            'write_buffer': cls.create_component_config(
                'write_buffer',
                buffer_config.get('write_buffer'),
                global_config
            )
        }
        
        return write_buffer_config
    
    @classmethod
    def create_query_buffer_config(cls, global_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create QueryBuffer configuration.
        
        Args:
            global_config: Global configuration dictionary
            
        Returns:
            QueryBuffer configuration
        """
        buffer_config = global_config.get('buffer', {}) if global_config else {}
        query_config = buffer_config.get('query', {})
        
        return cls.create_component_config('query_buffer', query_config, global_config)
    
    @classmethod
    def create_speculative_buffer_config(cls, global_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create SpeculativeBuffer configuration.
        
        Args:
            global_config: Global configuration dictionary
            
        Returns:
            SpeculativeBuffer configuration
        """
        buffer_config = global_config.get('buffer', {}) if global_config else {}
        speculative_config = buffer_config.get('speculative_buffer', {})
        
        return cls.create_component_config('speculative_buffer', speculative_config, global_config)

    @classmethod
    def _map_performance_to_flush_config(cls, performance_config: Dict[str, Any]) -> Dict[str, Any]:
        """Map performance configuration to flush_manager configuration.

        Args:
            performance_config: Performance configuration dictionary

        Returns:
            Mapped flush_manager configuration
        """
        flush_config = {}

        # Map performance settings to flush_manager settings
        if 'max_flush_workers' in performance_config:
            flush_config['max_workers'] = performance_config['max_flush_workers']
        if 'max_flush_queue_size' in performance_config:
            flush_config['max_queue_size'] = performance_config['max_flush_queue_size']
        if 'flush_timeout' in performance_config:
            flush_config['default_timeout'] = performance_config['flush_timeout']
        if 'flush_interval' in performance_config:
            flush_config['flush_interval'] = performance_config['flush_interval']
        if 'enable_auto_flush' in performance_config:
            flush_config['enable_auto_flush'] = performance_config['enable_auto_flush']

        return flush_config
    
    @classmethod
    def _apply_global_context(cls, config: Dict[str, Any], component_name: str,
                             global_config: Dict[str, Any]) -> None:
        """Apply global configuration context to component config.

        Args:
            config: Component configuration to modify
            component_name: Name of the component
            global_config: Global configuration context
        """
        # Apply global model settings
        if 'model' in global_config:
            model_config = global_config['model']
            if component_name == 'round_buffer':
                if 'default_model' in model_config:
                    config['token_model'] = model_config['default_model']
            elif component_name == 'hybrid_buffer':
                if 'embedding_model' in model_config:
                    config['embedding_model'] = model_config['embedding_model']

        # Apply global performance settings
        if 'performance' in global_config:
            perf_config = global_config['performance']
            if component_name == 'flush_manager':
                if 'max_workers' in perf_config:
                    config['max_workers'] = perf_config['max_workers']
                if 'flush_interval' in perf_config:
                    config['flush_interval'] = perf_config['flush_interval']
            elif component_name == 'hybrid_buffer':
                if 'force_flush_timeout' in perf_config:
                    config['force_flush_timeout'] = perf_config['force_flush_timeout']
                if 'enable_auto_flush' in perf_config:
                    config['enable_auto_flush'] = perf_config['enable_auto_flush']
                if 'flush_interval' in perf_config:
                    config['auto_flush_interval'] = perf_config['flush_interval']
    
    @classmethod
    def _deep_merge(cls, base: Dict[str, Any], override: Dict[str, Any]) -> None:
        """Deep merge override configuration into base configuration.
        
        Args:
            base: Base configuration dictionary (modified in place)
            override: Override configuration dictionary
        """
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                cls._deep_merge(base[key], value)
            else:
                base[key] = value
    
    @classmethod
    def _validate_component_config(cls, component_name: str, config: Dict[str, Any]) -> None:
        """Validate component configuration.
        
        Args:
            component_name: Name of the component
            config: Configuration to validate
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Basic validation rules
        if component_name == 'round_buffer':
            if config.get('max_tokens', 0) <= 0:
                raise ValueError("round_buffer.max_tokens must be positive")
            if config.get('max_size', 0) <= 0:
                raise ValueError("round_buffer.max_size must be positive")
        
        elif component_name == 'hybrid_buffer':
            if config.get('max_size', 0) <= 0:
                raise ValueError("hybrid_buffer.max_size must be positive")
            valid_strategies = ['message', 'semantic', 'fixed_size']
            if config.get('chunk_strategy') not in valid_strategies:
                logger.warning(f"Unknown chunk_strategy: {config.get('chunk_strategy')}, using 'message'")
                config['chunk_strategy'] = 'message'
        
        elif component_name == 'query_buffer':
            if config.get('max_size', 0) <= 0:
                raise ValueError("query_buffer.max_size must be positive")
            if config.get('cache_size', 0) <= 0:
                raise ValueError("query_buffer.cache_size must be positive")
        
        elif component_name == 'flush_manager':
            if config.get('max_workers', 0) <= 0:
                raise ValueError("flush_manager.max_workers must be positive")
            if config.get('max_queue_size', 0) <= 0:
                raise ValueError("flush_manager.max_queue_size must be positive")


class BufferConfigManager:
    """High-level configuration manager for the entire buffer system.
    
    This manager coordinates configuration across all buffer components
    and provides a unified interface for buffer system configuration.
    """
    
    def __init__(self, global_config: Optional[Dict[str, Any]] = None):
        """Initialize the buffer configuration manager.
        
        Args:
            global_config: Global configuration dictionary
        """
        self.global_config = global_config or {}
        self.factory = ComponentConfigFactory()
    
    def get_buffer_service_config(self) -> Dict[str, Any]:
        """Get complete configuration for BufferService.
        
        Returns:
            Complete BufferService configuration
        """
        return {
            'write_buffer': self.factory.create_write_buffer_config(self.global_config),
            'query_buffer': self.factory.create_query_buffer_config(self.global_config),
            'speculative_buffer': self.factory.create_speculative_buffer_config(self.global_config),
            'retrieval': self.global_config.get('retrieval', {'use_rerank': True})
        }
    
    def get_component_config(self, component_name: str, 
                           user_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get configuration for a specific component.
        
        Args:
            component_name: Name of the component
            user_overrides: User-provided configuration overrides
            
        Returns:
            Component configuration
        """
        return self.factory.create_component_config(
            component_name, 
            user_overrides, 
            self.global_config
        )
    
    def validate_configuration(self) -> bool:
        """Validate the entire buffer configuration.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Validate all component configurations
            for component_name in ComponentConfigFactory.COMPONENT_SCHEMAS.keys():
                self.factory.create_component_config(component_name, None, self.global_config)
            
            logger.info("Buffer configuration validation passed")
            return True
        except Exception as e:
            logger.error(f"Buffer configuration validation failed: {e}")
            return False
