"""Service interface for MemFuse services."""

from abc import ABC, abstractmethod
from typing import Optional, Any
from omegaconf import DictConfig


class ServiceInterface(ABC):
    """Base interface for all MemFuse services."""
    
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
    
    @abstractmethod
    def is_initialized(self) -> bool:
        """Check if the service is initialized.
        
        Returns:
            True if the service is initialized, False otherwise
        """
        pass
