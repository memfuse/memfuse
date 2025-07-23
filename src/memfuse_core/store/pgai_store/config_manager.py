"""
Configuration management for multi-layer PgAI system.

This module provides centralized configuration handling, validation,
and default value management for all PgAI components.
"""

from typing import Dict, List, Any, Optional
from loguru import logger


class ConfigManager:
    """
    Configuration manager for multi-layer PgAI system.

    Provides centralized configuration handling, validation,
    and default value management.
    """
    
    # Default configuration values
    DEFAULT_CONFIG = {
        'memory_layers': {
            'm0': {
                'enabled': True,
                'table_name': 'm0_raw',
                'pgai': {
                    'auto_embedding': True,
                    'immediate_trigger': True,
                    'embedding_model': 'all-MiniLM-L6-v2',
                    'embedding_dimensions': 384,
                    'batch_size': 10,
                    'max_retries': 3,
                    'retry_delay': 5.0
                }
            },
            'm1': {
                'enabled': False,
                'table_name': 'm1_episodic',
                'pgai': {
                    'auto_embedding': True,
                    'immediate_trigger': True,
                    'embedding_model': 'all-MiniLM-L6-v2',
                    'embedding_dimensions': 384,
                    'batch_size': 5,
                    'max_retries': 3,
                    'retry_delay': 5.0
                },
                'fact_extraction': {
                    'enabled': True,
                    'llm_model': 'grok-3-mini',
                    'temperature': 0.3,
                    'max_tokens': 1000,
                    'max_facts_per_chunk': 10,
                    'min_confidence_threshold': 0.7,
                    'batch_size': 5,
                    'context_window': 2,
                    'classification_strategy': 'open',
                    'enable_auto_categorization': True,
                    'custom_fact_types': []
                }
            }
        }
    }
    
    @staticmethod
    def get_layer_config(config: Dict[str, Any], layer: str) -> Dict[str, Any]:
        """Get configuration for a specific memory layer.
        
        Args:
            config: Full configuration dictionary
            layer: Layer name ('m0' or 'm1')
            
        Returns:
            Layer-specific configuration with defaults applied
        """
        # Get layer config with defaults
        layer_config = config.get('memory_layers', {}).get(layer, {})
        default_layer_config = ConfigManager.DEFAULT_CONFIG['memory_layers'].get(layer, {})
        
        # Merge with defaults
        merged_config = ConfigManager._deep_merge(default_layer_config, layer_config)
        
        return merged_config
    
    @staticmethod
    def get_enabled_layers(config: Dict[str, Any]) -> List[str]:
        """Get list of enabled memory layers.
        
        Args:
            config: Full configuration dictionary
            
        Returns:
            List of enabled layer names
        """
        enabled_layers = []
        memory_layers = config.get('memory_layers', {})
        
        for layer_name, layer_config in memory_layers.items():
            if layer_config.get('enabled', False):
                enabled_layers.append(layer_name)
        
        # If no layers explicitly configured, use M0 as default
        if not enabled_layers:
            enabled_layers = ['m0']
            
        return enabled_layers
    
    @staticmethod
    def get_pgai_config(config: Dict[str, Any], layer: str) -> Dict[str, Any]:
        """Get PgAI configuration for a specific layer.
        
        Args:
            config: Full configuration dictionary
            layer: Layer name ('m0' or 'm1')
            
        Returns:
            PgAI configuration for the layer
        """
        layer_config = ConfigManager.get_layer_config(config, layer)
        return layer_config.get('pgai', {})
    
    @staticmethod
    def get_fact_extraction_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Get fact extraction configuration for M1 layer.
        
        Args:
            config: Full configuration dictionary
            
        Returns:
            Fact extraction configuration
        """
        m1_config = ConfigManager.get_layer_config(config, 'm1')
        return m1_config.get('fact_extraction', {})
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> bool:
        """Validate configuration structure and values.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            True if configuration is valid
        """
        try:
            # Check basic structure
            if 'memory_layers' not in config:
                logger.warning("No memory_layers configuration found, using defaults")
                return True
            
            memory_layers = config['memory_layers']
            if not isinstance(memory_layers, dict):
                logger.error("memory_layers must be a dictionary")
                return False
            
            # Validate each layer
            for layer_name, layer_config in memory_layers.items():
                if layer_name not in ['m0', 'm1']:
                    logger.warning(f"Unknown layer: {layer_name}")
                    continue
                
                if not isinstance(layer_config, dict):
                    logger.error(f"Layer {layer_name} config must be a dictionary")
                    return False
                
                # Validate layer-specific configuration
                if not ConfigManager._validate_layer_config(layer_name, layer_config):
                    return False
            
            logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    @staticmethod
    def _validate_layer_config(layer_name: str, layer_config: Dict[str, Any]) -> bool:
        """Validate configuration for a specific layer.
        
        Args:
            layer_name: Name of the layer
            layer_config: Layer configuration dictionary
            
        Returns:
            True if layer configuration is valid
        """
        # Check enabled flag
        enabled = layer_config.get('enabled', False)
        if not isinstance(enabled, bool):
            logger.error(f"Layer {layer_name} 'enabled' must be boolean")
            return False
        
        # Validate PgAI config if present
        pgai_config = layer_config.get('pgai', {})
        if pgai_config and not ConfigManager._validate_pgai_config(layer_name, pgai_config):
            return False
        
        # Validate fact extraction config for M1
        if layer_name == 'm1':
            fact_config = layer_config.get('fact_extraction', {})
            if fact_config and not ConfigManager._validate_fact_extraction_config(fact_config):
                return False
        
        return True
    
    @staticmethod
    def _validate_pgai_config(layer_name: str, pgai_config: Dict[str, Any]) -> bool:
        """Validate PgAI configuration."""
        required_fields = ['embedding_model', 'embedding_dimensions']
        
        for field in required_fields:
            if field not in pgai_config:
                logger.warning(f"Missing {field} in {layer_name} pgai config, using default")
        
        # Validate embedding dimensions
        dimensions = pgai_config.get('embedding_dimensions', 384)
        if not isinstance(dimensions, int) or dimensions <= 0:
            logger.error(f"Invalid embedding_dimensions for {layer_name}: {dimensions}")
            return False
        
        return True
    
    @staticmethod
    def _validate_fact_extraction_config(fact_config: Dict[str, Any]) -> bool:
        """Validate fact extraction configuration."""
        # Validate confidence threshold
        threshold = fact_config.get('min_confidence_threshold', 0.7)
        if not isinstance(threshold, (int, float)) or not 0.0 <= threshold <= 1.0:
            logger.error(f"Invalid min_confidence_threshold: {threshold}")
            return False
        
        # Validate classification strategy
        strategy = fact_config.get('classification_strategy', 'open')
        if strategy not in ['open', 'predefined', 'custom']:
            logger.error(f"Invalid classification_strategy: {strategy}")
            return False
        
        return True
    
    @staticmethod
    def _deep_merge(default: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries, with override taking precedence.
        
        Args:
            default: Default configuration dictionary
            override: Override configuration dictionary
            
        Returns:
            Merged configuration dictionary
        """
        result = default.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigManager._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    @staticmethod
    def get_table_name(config: Dict[str, Any], layer: str) -> str:
        """Get table name for a specific layer.
        
        Args:
            config: Full configuration dictionary
            layer: Layer name ('m0' or 'm1')
            
        Returns:
            Table name for the layer
        """
        layer_config = ConfigManager.get_layer_config(config, layer)
        return layer_config.get('table_name', f'{layer}_default')
    
    @staticmethod
    def apply_defaults(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply default configuration values to user config.
        
        Args:
            config: User configuration dictionary (can be None)
            
        Returns:
            Configuration with defaults applied
        """
        if config is None:
            config = {}
        
        return ConfigManager._deep_merge(ConfigManager.DEFAULT_CONFIG, config)
