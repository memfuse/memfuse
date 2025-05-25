"""Application service for MemFuse server.

This module provides the FastAPI application creation and configuration service.
"""

from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from omegaconf import DictConfig
from loguru import logger

from .base_service import BaseService
from ..utils.config import config_manager


class AppService(BaseService):
    """Service for managing FastAPI application configuration and setup."""
    
    def __init__(self):
        """Initialize the app service."""
        super().__init__("app")
        self._app: Optional[FastAPI] = None
    
    async def initialize(self, cfg: Optional[DictConfig] = None) -> bool:
        """Initialize the app service.
        
        Args:
            cfg: Configuration for the service
            
        Returns:
            True if initialization was successful, False otherwise
        """
        try:
            if cfg is not None:
                self.set_config(cfg)
            
            # Create the FastAPI application
            self._app = self._create_app()
            
            self._mark_initialized()
            return True
        except Exception as e:
            logger.error(f"Failed to initialize app service: {e}")
            return False
    
    async def shutdown(self) -> bool:
        """Shutdown the app service.
        
        Returns:
            True if shutdown was successful, False otherwise
        """
        try:
            self._app = None
            self._mark_shutdown()
            return True
        except Exception as e:
            logger.error(f"Failed to shutdown app service: {e}")
            return False
    
    def get_app(self) -> Optional[FastAPI]:
        """Get the FastAPI application.
        
        Returns:
            FastAPI application or None if not initialized
        """
        return self._app
    
    def _create_app(self) -> FastAPI:
        """Create and configure the FastAPI application.
        
        Returns:
            Configured FastAPI application
        """
        # Import API routers here to avoid circular imports
        from ..api import health, users, agents, sessions, messages, knowledge, api_keys
        from ..utils.auth import RateLimitMiddleware
        
        # Get configuration
        config_dict = config_manager.get_config()
        cfg = config_dict if config_dict else {}
        
        # Create FastAPI app
        app = FastAPI(
            title="MemFuse Server API",
            description=(
                "API for MemFuse Server: a lightning-fast, open-source memory "
                "layer for LLMs"
            ),
            version="0.0.1",
            redirect_slashes=False,
        )
        
        # Configure CORS
        cors_origins = cfg.get("server", {}).get("cors_origins", ["*"])
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Add rate limiting middleware
        rate_limit_per_minute = cfg.get("server", {}).get("rate_limit_per_minute", 60)
        app.add_middleware(
            RateLimitMiddleware,
            rate_limit_per_minute=rate_limit_per_minute
        )
        
        # Include RESTful API routers
        self._register_routes(app)
        
        logger.info("FastAPI application created and configured")
        return app
    
    def _register_routes(self, app: FastAPI) -> None:
        """Register API routes with the FastAPI application.
        
        Args:
            app: FastAPI application to register routes with
        """
        from ..api import health, users, agents, sessions, messages, knowledge, api_keys
        
        # Register all API routers
        app.include_router(health.router, prefix="/api/v1/health", tags=["health"])
        app.include_router(users.router, prefix="/api/v1/users", tags=["users"])
        app.include_router(agents.router, prefix="/api/v1/agents", tags=["agents"])
        app.include_router(sessions.router, prefix="/api/v1/sessions", tags=["sessions"])
        app.include_router(
            messages.router,
            prefix="/api/v1/sessions/{session_id}/messages",
            tags=["messages"]
        )
        app.include_router(
            knowledge.router,
            prefix="/api/v1/users/{user_id}/knowledge",
            tags=["knowledge"]
        )
        app.include_router(
            api_keys.router,
            prefix="/api/v1/users/{user_id}/api-keys",
            tags=["api-keys"]
        )
        
        logger.debug("API routes registered successfully")


# Global app service instance
_app_service: Optional[AppService] = None


def get_app_service() -> AppService:
    """Get the global app service instance.
    
    Returns:
        AppService instance
    """
    global _app_service
    if _app_service is None:
        _app_service = AppService()
    return _app_service
