"""
Global Service Manager for MemFuse.

This module manages global singleton services that can be shared across all users,
optimizing memory usage and initialization time for high-scale deployments.

Design Philosophy:
- Global Singletons: Services that are user-agnostic (LLM, Models, Conflict Detection)
- User-Scoped: Services that require user-specific data (MemoryService, Storage)
- Session Context: Passed as parameters, not stored in instances
"""

from typing import Optional, Dict, Any, Type
from loguru import logger
import asyncio
from threading import Lock

from ..llm.base import LLMProvider
from ..llm.providers.openai import OpenAIProvider


class GlobalServiceManager:
    """
    Manages global singleton services for optimal resource utilization.
    
    Global Services (One instance for all users):
    - LLM Provider (OpenAI, etc.)
    - Advanced LLM Service (fact extraction, conflict detection)
    - Conflict Detection Engine
    - Embedding Models
    - Rerank Models
    
    User-Scoped Services (One per user):
    - Memory Service (user-specific storage)
    - Facts Database (user-specific data)
    - Storage Managers (user-specific directories)
    """
    
    _instance: Optional['GlobalServiceManager'] = None
    _lock = Lock()
    
    def __new__(cls) -> 'GlobalServiceManager':
        """Ensure singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the global service manager."""
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        # Global singleton services
        self._llm_provider: Optional[LLMProvider] = None
        self._advanced_llm_service: Optional[Any] = None
        self._conflict_detection_engine: Optional[Any] = None
        
        # Global models (already managed by ServiceFactory)
        self._embedding_model: Optional[Any] = None
        self._rerank_model: Optional[Any] = None
        self._reranker_instance: Optional[Any] = None
        
        # Configuration
        self._config: Dict[str, Any] = {}
        
        # Initialization state
        self._services_initialized = False
        self._initialization_lock = asyncio.Lock()
        
        self._initialized = True
        logger.info("GlobalServiceManager: Singleton instance created")
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize all global services.
        
        Args:
            config: Global configuration dictionary
            
        Returns:
            True if initialization successful, False otherwise
        """
        if self._services_initialized:
            logger.debug("GlobalServiceManager: Services already initialized")
            return True
            
        async with self._initialization_lock:
            if self._services_initialized:  # Double-check after acquiring lock
                return True
                
            try:
                logger.info("GlobalServiceManager: Initializing global services...")
                self._config = config
                
                # Initialize LLM Provider (global singleton)
                await self._initialize_llm_provider()
                
                # Initialize Advanced LLM Service (global singleton)
                await self._initialize_advanced_llm_service()
                
                # Initialize Conflict Detection Engine (global singleton)
                await self._initialize_conflict_detection_engine()
                
                self._services_initialized = True
                logger.info("GlobalServiceManager: All global services initialized successfully")
                return True
                
            except Exception as e:
                logger.error(f"GlobalServiceManager: Failed to initialize services: {e}")
                return False
    
    async def _initialize_llm_provider(self):
        """Initialize global LLM provider."""
        if self._llm_provider is not None:
            return
            
        try:
            # Create global LLM provider instance
            llm_config = self._config.get("llm", {})
            self._llm_provider = OpenAIProvider(config=llm_config)
            logger.info("GlobalServiceManager: LLM Provider initialized (global singleton)")
            
        except Exception as e:
            logger.error(f"GlobalServiceManager: Failed to initialize LLM Provider: {e}")
            raise
    
    async def _initialize_advanced_llm_service(self):
        """Initialize global Advanced LLM Service."""
        if self._advanced_llm_service is not None:
            return
            
        try:
            from ..hierarchy.llm_service import AdvancedLLMService
            
            # Use global LLM provider
            l1_config = self._config.get("l1", {}).get("extraction", {})
            self._advanced_llm_service = AdvancedLLMService(
                llm_provider=self._llm_provider,
                config=l1_config
            )
            logger.info("GlobalServiceManager: Advanced LLM Service initialized (global singleton)")
            
        except Exception as e:
            logger.error(f"GlobalServiceManager: Failed to initialize Advanced LLM Service: {e}")
            raise
    
    async def _initialize_conflict_detection_engine(self):
        """Initialize global Conflict Detection Engine."""
        if self._conflict_detection_engine is not None:
            return
            
        try:
            from ..hierarchy.conflict_detection import ConflictDetectionEngine
            
            # Use global Advanced LLM Service
            conflict_config = self._config.get("l1", {}).get("conflict_detection", {})
            self._conflict_detection_engine = ConflictDetectionEngine(
                llm_service=self._advanced_llm_service,
                config=conflict_config
            )
            logger.info("GlobalServiceManager: Conflict Detection Engine initialized (global singleton)")
            
        except Exception as e:
            logger.error(f"GlobalServiceManager: Failed to initialize Conflict Detection Engine: {e}")
            raise
    
    def get_llm_provider(self) -> Optional[LLMProvider]:
        """Get global LLM provider instance."""
        return self._llm_provider
    
    def get_advanced_llm_service(self) -> Optional[Any]:
        """Get global Advanced LLM Service instance."""
        return self._advanced_llm_service
    
    def get_conflict_detection_engine(self) -> Optional[Any]:
        """Get global Conflict Detection Engine instance."""
        return self._conflict_detection_engine
    
    def set_global_models(
        self,
        embedding_model: Optional[Any] = None,
        rerank_model: Optional[Any] = None,
        reranker_instance: Optional[Any] = None
    ):
        """
        Set global model instances.
        
        Args:
            embedding_model: Global embedding model
            rerank_model: Global rerank model
            reranker_instance: Global reranker instance
        """
        if embedding_model is not None:
            self._embedding_model = embedding_model
            logger.debug("GlobalServiceManager: Set global embedding model")
            
        if rerank_model is not None:
            self._rerank_model = rerank_model
            logger.debug("GlobalServiceManager: Set global rerank model")
            
        if reranker_instance is not None:
            self._reranker_instance = reranker_instance
            logger.debug("GlobalServiceManager: Set global reranker instance")
    
    def get_embedding_model(self) -> Optional[Any]:
        """Get global embedding model."""
        return self._embedding_model
    
    def get_rerank_model(self) -> Optional[Any]:
        """Get global rerank model."""
        return self._rerank_model
    
    def get_reranker_instance(self) -> Optional[Any]:
        """Get global reranker instance."""
        return self._reranker_instance
    
    async def shutdown(self):
        """Shutdown all global services."""
        logger.info("GlobalServiceManager: Shutting down global services...")
        
        # Reset services
        self._llm_provider = None
        self._advanced_llm_service = None
        self._conflict_detection_engine = None
        self._services_initialized = False
        
        logger.info("GlobalServiceManager: Global services shutdown complete")


# Global instance
_global_service_manager: Optional[GlobalServiceManager] = None


def get_global_service_manager() -> GlobalServiceManager:
    """Get the global service manager instance."""
    global _global_service_manager
    if _global_service_manager is None:
        _global_service_manager = GlobalServiceManager()
    return _global_service_manager


async def initialize_global_services(config: Dict[str, Any]) -> bool:
    """
    Initialize all global services.
    
    Args:
        config: Global configuration
        
    Returns:
        True if successful, False otherwise
    """
    manager = get_global_service_manager()
    return await manager.initialize(config)


async def shutdown_global_services():
    """Shutdown all global services."""
    global _global_service_manager
    if _global_service_manager is not None:
        await _global_service_manager.shutdown()
        _global_service_manager = None
