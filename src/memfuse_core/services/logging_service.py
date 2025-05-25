"""Logging service for MemFuse server.

This module provides centralized logging configuration and management.
"""

import sys
from typing import Optional
from omegaconf import DictConfig
from loguru import logger

from .base_service import BaseService


class LoggingService(BaseService):
    """Service for managing logging configuration."""
    
    def __init__(self):
        """Initialize the logging service."""
        super().__init__("logging")
        self._configured = False
    
    async def initialize(self, cfg: Optional[DictConfig] = None) -> bool:
        """Initialize the logging service.
        
        Args:
            cfg: Configuration for the service
            
        Returns:
            True if initialization was successful, False otherwise
        """
        try:
            if cfg is not None:
                self.set_config(cfg)
            
            # Configure logging
            self._configure_logging(cfg or DictConfig({}))
            
            self._mark_initialized()
            return True
        except Exception as e:
            logger.error(f"Failed to initialize logging service: {e}")
            return False
    
    async def shutdown(self) -> bool:
        """Shutdown the logging service.
        
        Returns:
            True if shutdown was successful, False otherwise
        """
        try:
            # Remove all handlers
            logger.remove()
            self._configured = False
            self._mark_shutdown()
            return True
        except Exception as e:
            logger.error(f"Failed to shutdown logging service: {e}")
            return False
    
    def _configure_logging(self, cfg: DictConfig) -> None:
        """Configure logging with loguru.
        
        Args:
            cfg: Configuration from Hydra
        """
        # Determine log level from configuration
        log_level = "DEBUG" if cfg.get("server", {}).get("debug", False) else "INFO"
        
        # Remove default handler if not already configured
        if not self._configured:
            logger.remove()
        
        # Define log formats
        console_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )
        
        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name}:{function}:{line} - "
            "{message}"
        )
        
        # Add colorized console output
        logger.add(
            sys.stderr,
            level=log_level,
            format=console_format,
            colorize=True
        )
        
        # Add file logging with rotation
        logger.add(
            "logs/memfuse_core.log",
            rotation="10 MB",
            retention="1 week",
            level=log_level,
            format=file_format,
            backtrace=True,
            diagnose=True
        )
        
        self._configured = True
        logger.info("Logging system configured successfully")


# Global logging service instance
_logging_service: Optional[LoggingService] = None


def get_logging_service() -> LoggingService:
    """Get the global logging service instance.
    
    Returns:
        LoggingService instance
    """
    global _logging_service
    if _logging_service is None:
        _logging_service = LoggingService()
    return _logging_service
