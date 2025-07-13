"""
Store factory for creating appropriate pgai store instances based on configuration.

This factory provides backward compatibility and automatic selection between
polling-based and event-driven pgai stores.
"""

from typing import Dict, Any, Optional
import logging

from .pgai_store import PgaiStore
from .simplified_event_driven_store import SimplifiedEventDrivenPgaiStore

logger = logging.getLogger(__name__)


class PgaiStoreFactory:
    """Factory for creating pgai store instances with appropriate configuration."""
    
    @staticmethod
    def create_store(config: Optional[Dict[str, Any]] = None, table_name: str = "m0_episodic") -> PgaiStore:
        """
        Create appropriate pgai store instance based on configuration.
        
        Args:
            config: Optional configuration dictionary
            table_name: Database table name
            
        Returns:
            PgaiStore instance (either traditional or event-driven)
        """
        try:
            # Get configuration
            if config:
                pgai_config = config.get("pgai", {})
            else:
                from ..utils.config import config_manager
                full_config = config_manager.get_config()
                pgai_config = full_config.get("database", {}).get("pgai", {}) if full_config else {}
            
            # Determine which store type to create
            immediate_trigger = pgai_config.get("immediate_trigger", False)
            auto_embedding = pgai_config.get("auto_embedding", False)
            
            if auto_embedding and immediate_trigger:
                logger.info(f"Creating SimplifiedEventDrivenPgaiStore for table: {table_name}")
                return SimplifiedEventDrivenPgaiStore(config, table_name)
            else:
                logger.info(f"Creating traditional PgaiStore for table: {table_name}")
                return PgaiStore(config, table_name)
                
        except Exception as e:
            logger.error(f"Failed to create pgai store: {e}")
            logger.info(f"Falling back to traditional PgaiStore for table: {table_name}")
            return PgaiStore(config, table_name)
    
    @staticmethod
    def get_store_type(config: Optional[Dict[str, Any]] = None) -> str:
        """
        Get the store type that would be created with given configuration.
        
        Args:
            config: Optional configuration dictionary
            
        Returns:
            Store type string: "event_driven" or "traditional"
        """
        try:
            if config:
                pgai_config = config.get("pgai", {})
            else:
                from ..utils.config import config_manager
                full_config = config_manager.get_config()
                pgai_config = full_config.get("database", {}).get("pgai", {}) if full_config else {}
            
            immediate_trigger = pgai_config.get("immediate_trigger", False)
            auto_embedding = pgai_config.get("auto_embedding", False)
            
            if auto_embedding and immediate_trigger:
                return "event_driven"
            else:
                return "traditional"
                
        except Exception:
            return "traditional"
    
    @staticmethod
    def validate_configuration(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize pgai configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Validated and normalized configuration
        """
        pgai_config = config.get("pgai", {})
        
        # Set defaults
        defaults = {
            "enabled": True,
            "auto_embedding": True,
            "immediate_trigger": False,
            "use_polling_fallback": True,
            "max_retries": 3,
            "retry_interval": 5.0,
            "retry_timeout": 300,
            "worker_count": 3,
            "queue_size": 1000,
            "batch_processing": False,
            "enable_metrics": True,
            "log_level": "INFO",
            "embedding_model": "all-MiniLM-L6-v2",
            "embedding_dimensions": 384
        }
        
        # Apply defaults for missing keys
        for key, default_value in defaults.items():
            if key not in pgai_config:
                pgai_config[key] = default_value
        
        # Validate configuration
        errors = []
        
        # Validate worker count
        if not isinstance(pgai_config["worker_count"], int) or pgai_config["worker_count"] < 1:
            errors.append("worker_count must be a positive integer")
            pgai_config["worker_count"] = 3
        
        # Validate queue size
        if not isinstance(pgai_config["queue_size"], int) or pgai_config["queue_size"] < 1:
            errors.append("queue_size must be a positive integer")
            pgai_config["queue_size"] = 1000
        
        # Validate retry configuration
        if not isinstance(pgai_config["max_retries"], int) or pgai_config["max_retries"] < 0:
            errors.append("max_retries must be a non-negative integer")
            pgai_config["max_retries"] = 3
        
        if not isinstance(pgai_config["retry_interval"], (int, float)) or pgai_config["retry_interval"] < 0:
            errors.append("retry_interval must be a non-negative number")
            pgai_config["retry_interval"] = 5.0
        
        # Validate embedding dimensions
        if not isinstance(pgai_config["embedding_dimensions"], int) or pgai_config["embedding_dimensions"] < 1:
            errors.append("embedding_dimensions must be a positive integer")
            pgai_config["embedding_dimensions"] = 384
        
        # Log validation errors
        if errors:
            logger.warning(f"Configuration validation errors: {errors}")
        
        # Update config
        config["pgai"] = pgai_config
        
        return config


class BackwardCompatibilityManager:
    """Manages backward compatibility for pgai store configurations."""
    
    @staticmethod
    def migrate_legacy_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Migrate legacy configuration to new format.
        
        Args:
            config: Legacy configuration
            
        Returns:
            Migrated configuration
        """
        pgai_config = config.get("pgai", {})
        
        # Map legacy keys to new keys
        legacy_mappings = {
            "retry_delay": "retry_interval",
            "vectorizer_worker_enabled": "auto_embedding"
        }
        
        for legacy_key, new_key in legacy_mappings.items():
            if legacy_key in pgai_config and new_key not in pgai_config:
                pgai_config[new_key] = pgai_config[legacy_key]
                logger.info(f"Migrated legacy config: {legacy_key} -> {new_key}")
        
        # Handle special cases
        if "immediate_trigger" not in pgai_config:
            # Default to false for backward compatibility
            pgai_config["immediate_trigger"] = False
            
        if "use_polling_fallback" not in pgai_config:
            # Enable fallback by default for safety
            pgai_config["use_polling_fallback"] = True
        
        config["pgai"] = pgai_config
        return config
    
    @staticmethod
    def check_compatibility(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check configuration compatibility and provide warnings.
        
        Args:
            config: Configuration to check
            
        Returns:
            Compatibility report
        """
        pgai_config = config.get("pgai", {})
        report = {
            "compatible": True,
            "warnings": [],
            "recommendations": []
        }
        
        # Check for conflicting settings
        if pgai_config.get("immediate_trigger", False) and not pgai_config.get("auto_embedding", True):
            report["warnings"].append("immediate_trigger requires auto_embedding to be enabled")
            report["compatible"] = False
        
        # Check for performance implications
        if pgai_config.get("worker_count", 3) > 10:
            report["warnings"].append("High worker count may impact database performance")
        
        if pgai_config.get("queue_size", 1000) > 10000:
            report["warnings"].append("Large queue size may consume significant memory")
        
        # Provide recommendations
        if pgai_config.get("immediate_trigger", False):
            report["recommendations"].append("Consider enabling metrics for monitoring immediate trigger performance")
        
        if not pgai_config.get("use_polling_fallback", True):
            report["recommendations"].append("Consider keeping polling fallback enabled for reliability")
        
        return report


# Convenience function for backward compatibility
def create_pgai_store(config: Optional[Dict[str, Any]] = None, table_name: str = "m0_episodic") -> PgaiStore:
    """
    Create pgai store instance with automatic configuration handling.
    
    This is the main entry point for creating pgai stores with full
    backward compatibility and automatic mode selection.
    """
    return PgaiStoreFactory.create_store(config, table_name)
