"""Base service interface for MemFuse services.

This module provides a common interface for all services in the MemFuse framework.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional
from omegaconf import DictConfig
from loguru import logger


class BaseService(ABC):
    """Base class for all MemFuse services.
    
    This class provides a common interface for service initialization,
    configuration, and lifecycle management.
    """
    
    def __init__(self, name: str):
        """Initialize the base service.
        
        Args:
            name: Name of the service
        """
        self.name = name
        self._initialized = False
        self._config: Optional[DictConfig] = None
    
    @abstractmethod
    async def initialize(self, cfg: Optional[DictConfig] = None) -> bool:
        """Initialize the service.
        
        Args:
            cfg: Configuration for the service
            
        Returns:
            True if initialization was successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def shutdown(self) -> bool:
        """Shutdown the service gracefully.
        
        Returns:
            True if shutdown was successful, False otherwise
        """
        pass
    
    def is_initialized(self) -> bool:
        """Check if the service is initialized.
        
        Returns:
            True if the service is initialized, False otherwise
        """
        return self._initialized
    
    def get_config(self) -> Optional[DictConfig]:
        """Get the service configuration.
        
        Returns:
            Service configuration or None if not set
        """
        return self._config
    
    def set_config(self, cfg: DictConfig) -> None:
        """Set the service configuration.
        
        Args:
            cfg: Configuration to set
        """
        self._config = cfg
        logger.debug(f"{self.name} service configuration updated")
    
    def _mark_initialized(self) -> None:
        """Mark the service as initialized."""
        self._initialized = True
        logger.info(f"{self.name} service initialized successfully")
    
    def _mark_shutdown(self) -> None:
        """Mark the service as shutdown."""
        self._initialized = False
        logger.info(f"{self.name} service shutdown successfully")


class ServiceRegistry:
    """Registry for managing service instances."""
    
    _services: dict[str, BaseService] = {}
    
    @classmethod
    def register(cls, service: BaseService) -> None:
        """Register a service.
        
        Args:
            service: Service to register
        """
        cls._services[service.name] = service
        logger.debug(f"Registered service: {service.name}")
    
    @classmethod
    def get(cls, name: str) -> Optional[BaseService]:
        """Get a service by name.
        
        Args:
            name: Name of the service
            
        Returns:
            Service instance or None if not found
        """
        return cls._services.get(name)
    
    @classmethod
    def get_all(cls) -> dict[str, BaseService]:
        """Get all registered services.
        
        Returns:
            Dictionary of all registered services
        """
        return cls._services.copy()
    
    @classmethod
    def unregister(cls, name: str) -> bool:
        """Unregister a service.
        
        Args:
            name: Name of the service to unregister
            
        Returns:
            True if service was unregistered, False if not found
        """
        if name in cls._services:
            del cls._services[name]
            logger.debug(f"Unregistered service: {name}")
            return True
        return False
    
    @classmethod
    def clear(cls) -> None:
        """Clear all registered services."""
        cls._services.clear()
        logger.debug("Cleared all registered services")
