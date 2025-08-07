"""Global Configuration Manager for MemFuse Performance Optimization.

This module implements a high-performance global configuration manager that:
1. Loads configuration once at server startup
2. Provides instant access to configuration values through caching
3. Eliminates repeated Hydra configuration parsing
4. Supports hot-reload for non-critical settings
"""

import asyncio
import time
from typing import Dict, Any, Optional, Union
from threading import Lock
from loguru import logger
from omegaconf import DictConfig, OmegaConf


class GlobalConfigManager:
    """High-performance global configuration manager with singleton pattern.
    
    This class implements the global singleton pattern for configuration management,
    providing instant access to configuration values and eliminating repeated
    Hydra configuration parsing overhead.
    
    Features:
    - One-time configuration loading at startup
    - Instant configuration access through caching
    - Thread-safe operations
    - Configuration hot-reload capability
    - Performance monitoring and metrics
    """
    
    _instance: Optional['GlobalConfigManager'] = None
    _lock = Lock()
    _initialized = False
    
    def __init__(self):
        """Initialize the global configuration manager."""
        if GlobalConfigManager._initialized:
            return
            
        # Core configuration storage
        self._config: Optional[Dict[str, Any]] = None
        self._config_cache: Dict[str, Any] = {}
        self._load_time: Optional[float] = None
        self._access_count = 0
        
        # Performance tracking
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Thread safety
        self._config_lock = Lock()
        
        GlobalConfigManager._initialized = True
        logger.info("GlobalConfigManager: Singleton instance created")
    
    def __new__(cls) -> 'GlobalConfigManager':
        """Ensure singleton pattern with thread safety."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def get_instance(cls) -> 'GlobalConfigManager':
        """Get the singleton instance."""
        return cls()
    
    async def initialize(self, config: Union[DictConfig, Dict[str, Any]]) -> None:
        """One-time initialization at server startup.
        
        This method should be called once during server startup to load
        and cache the configuration for optimal performance.
        
        Args:
            config: Configuration object (DictConfig or dict)
        """
        start_time = time.time()
        
        with self._config_lock:
            if self._config is not None:
                logger.warning("GlobalConfigManager: Already initialized, skipping")
                return
            
            # Convert DictConfig to dict for optimal access performance
            if hasattr(config, 'to_container'):
                self._config = OmegaConf.to_container(config, resolve=True)
            else:
                self._config = dict(config) if config else {}
            
            # Build performance-optimized cache
            self._build_cache()
            
            self._load_time = time.time() - start_time
            
        logger.info(f"GlobalConfigManager: Initialized in {self._load_time:.3f}s")
        logger.info(f"GlobalConfigManager: Cached {len(self._config_cache)} configuration keys")
    
    def _build_cache(self) -> None:
        """Build performance-optimized configuration cache.
        
        This method pre-computes commonly accessed configuration paths
        for instant access without dictionary traversal overhead.
        """
        if not self._config:
            return
        
        # Cache commonly accessed configuration paths
        cache_paths = [
            # Server configuration
            ("server.host", ["server", "host"]),
            ("server.port", ["server", "port"]),
            ("server.reload", ["server", "reload"]),
            
            # Database configuration
            ("database.type", ["database", "type"]),
            ("database.postgres.host", ["database", "postgres", "host"]),
            ("database.postgres.port", ["database", "postgres", "port"]),
            ("database.postgres.database", ["database", "postgres", "database"]),
            ("database.postgres.user", ["database", "postgres", "user"]),
            ("database.postgres.password", ["database", "postgres", "password"]),
            ("database.postgres.pool_size", ["database", "postgres", "pool_size"]),
            ("database.postgres.max_overflow", ["database", "postgres", "max_overflow"]),
            ("database.postgres.pool_timeout", ["database", "postgres", "pool_timeout"]),
            ("database.postgres.pool_recycle", ["database", "postgres", "pool_recycle"]),
            
            # Buffer configuration
            ("buffer.enabled", ["buffer", "enabled"]),
            ("buffer.round_buffer.max_tokens", ["buffer", "round_buffer", "max_tokens"]),
            ("buffer.round_buffer.max_size", ["buffer", "round_buffer", "max_size"]),
            ("buffer.hybrid_buffer.max_size", ["buffer", "hybrid_buffer", "max_size"]),
            ("buffer.query.max_size", ["buffer", "query", "max_size"]),
            ("buffer.query.cache_size", ["buffer", "query", "cache_size"]),
            
            # Memory configuration
            ("memory.memory_service.parallel_enabled", ["memory", "memory_service", "parallel_enabled"]),
            ("memory.layers.m0.enabled", ["memory", "layers", "m0", "enabled"]),
            ("memory.layers.m1.enabled", ["memory", "layers", "m1", "enabled"]),
            ("memory.layers.m2.enabled", ["memory", "layers", "m2", "enabled"]),
            
            # Store configuration
            ("store.backend", ["store", "backend"]),
            ("store.top_k", ["store", "top_k"]),
            ("store.similarity_threshold", ["store", "similarity_threshold"]),
            
            # Embedding configuration
            ("embedding.dimension", ["embedding", "dimension"]),
            ("embedding.model", ["embedding", "model"]),
            
            # Data directory
            ("data_dir", ["data_dir"]),
        ]
        
        for cache_key, path in cache_paths:
            value = self._get_nested_value(self._config, path)
            if value is not None:
                self._config_cache[cache_key] = value
    
    def _get_nested_value(self, config: Dict[str, Any], path: list) -> Any:
        """Get nested configuration value by path.
        
        Args:
            config: Configuration dictionary
            path: List of keys representing the path
            
        Returns:
            Configuration value or None if not found
        """
        current = config
        for key in path:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        return current
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with instant access from cache.
        
        This method provides sub-millisecond access to commonly used
        configuration values through pre-computed caching.
        
        Args:
            key: Configuration key (dot-separated path)
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        self._access_count += 1
        
        # Fast path: check cache first
        if key in self._config_cache:
            self._cache_hits += 1
            return self._config_cache[key]
        
        # Slow path: traverse configuration
        self._cache_misses += 1
        
        if not self._config:
            return default
        
        # Split key and traverse
        path = key.split('.')
        value = self._get_nested_value(self._config, path)
        
        # Cache the result for future access
        if value is not None:
            with self._config_lock:
                self._config_cache[key] = value
        
        return value if value is not None else default
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section.
        
        Args:
            section: Section name (e.g., 'database', 'buffer')
            
        Returns:
            Configuration section dictionary
        """
        if not self._config:
            return {}
        
        return self._config.get(section, {})
    
    def get_config(self) -> Dict[str, Any]:
        """Get the full configuration dictionary.
        
        Returns:
            Complete configuration dictionary
        """
        return self._config or {}
    
    def get_raw_config(self) -> Dict[str, Any]:
        """Get the raw configuration dictionary.
        
        Returns:
            Complete configuration dictionary
        """
        return self._config or {}
    
    def is_initialized(self) -> bool:
        """Check if the configuration manager is initialized.
        
        Returns:
            True if initialized, False otherwise
        """
        return self._config is not None
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for monitoring.
        
        Returns:
            Dictionary containing performance metrics
        """
        cache_hit_rate = (
            self._cache_hits / (self._cache_hits + self._cache_misses)
            if (self._cache_hits + self._cache_misses) > 0
            else 0.0
        )
        
        return {
            "initialized": self.is_initialized(),
            "load_time_seconds": self._load_time,
            "cached_keys": len(self._config_cache),
            "total_accesses": self._access_count,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": cache_hit_rate,
        }
    
    async def hot_reload(self, config: Union[DictConfig, Dict[str, Any]]) -> None:
        """Hot-reload configuration for non-critical settings.
        
        This method allows updating configuration without server restart
        for settings that don't require service reinitialization.
        
        Args:
            config: New configuration object
        """
        start_time = time.time()
        
        with self._config_lock:
            # Convert and update configuration
            if hasattr(config, 'to_container'):
                new_config = OmegaConf.to_container(config, resolve=True)
            else:
                new_config = dict(config) if config else {}
            
            # Update configuration
            self._config = new_config
            
            # Clear and rebuild cache
            self._config_cache.clear()
            self._build_cache()
            
            # Reset performance counters
            self._cache_hits = 0
            self._cache_misses = 0
        
        reload_time = time.time() - start_time
        logger.info(f"GlobalConfigManager: Hot-reloaded in {reload_time:.3f}s")


# Global singleton instance
_global_config_manager: Optional[GlobalConfigManager] = None


def get_global_config_manager() -> GlobalConfigManager:
    """Get the global configuration manager singleton instance."""
    global _global_config_manager
    if _global_config_manager is None:
        _global_config_manager = GlobalConfigManager.get_instance()
    return _global_config_manager


# Convenience functions for common configuration access
def get_config(key: str, default: Any = None) -> Any:
    """Get configuration value using global config manager.
    
    Args:
        key: Configuration key (dot-separated path)
        default: Default value if key not found
        
    Returns:
        Configuration value or default
    """
    return get_global_config_manager().get(key, default)


def get_config_section(section: str) -> Dict[str, Any]:
    """Get configuration section using global config manager.
    
    Args:
        section: Section name
        
    Returns:
        Configuration section dictionary
    """
    return get_global_config_manager().get_section(section)


def is_config_initialized() -> bool:
    """Check if global configuration is initialized.
    
    Returns:
        True if initialized, False otherwise
    """
    return get_global_config_manager().is_initialized()