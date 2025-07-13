"""
Simplified error handling for immediate trigger system.

This module provides essential error handling without over-engineering.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional
from functools import wraps

logger = logging.getLogger(__name__)


def with_retry(max_retries: int = 3, base_delay: float = 1.0):
    """Simple retry decorator with exponential backoff."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Function {func.__name__} failed after {max_retries} attempts: {e}")
                        raise
                    
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"Function {func.__name__} failed (attempt {attempt + 1}/{max_retries}), retrying in {delay}s: {e}")
                    await asyncio.sleep(delay)
        
        return wrapper
    return decorator


class SimpleHealthChecker:
    """Simple health monitoring."""
    
    def __init__(self):
        self.components = {}
        
    def register_component(self, name: str, health_func):
        """Register a component for health checking."""
        self.components[name] = health_func
        
    async def check_health(self) -> Dict[str, Any]:
        """Check health of all registered components."""
        health_status = {
            "timestamp": time.time(),
            "overall_healthy": True,
            "components": {}
        }
        
        for name, health_func in self.components.items():
            try:
                if asyncio.iscoroutinefunction(health_func):
                    component_health = await health_func()
                else:
                    component_health = health_func()
                
                health_status["components"][name] = component_health
                
                # Check if component is healthy
                if isinstance(component_health, dict):
                    is_healthy = component_health.get("healthy", True)
                else:
                    is_healthy = bool(component_health)
                
                if not is_healthy:
                    health_status["overall_healthy"] = False
                    
            except Exception as e:
                logger.error(f"Health check failed for component {name}: {e}")
                health_status["components"][name] = {
                    "healthy": False,
                    "error": str(e)
                }
                health_status["overall_healthy"] = False
        
        return health_status


class SimpleTaskManager:
    """Simple async task management."""
    
    def __init__(self):
        self.tasks = []
        
    def add_task(self, coro, name: Optional[str] = None):
        """Add a task with tracking."""
        task = asyncio.create_task(coro, name=name)
        self.tasks.append(task)
        task.add_done_callback(lambda t: self._remove_task(t))
        return task
    
    def _remove_task(self, task):
        """Remove completed task."""
        try:
            self.tasks.remove(task)
        except ValueError:
            pass
    
    async def cleanup_all(self):
        """Cleanup all tasks."""
        for task in self.tasks:
            if not task.done():
                task.cancel()
        
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        
        self.tasks.clear()


# Global instances
simple_health_checker = SimpleHealthChecker()
simple_task_manager = SimpleTaskManager()


def initialize_simple_error_handling():
    """Initialize simple error handling."""
    logger.info("Simple error handling initialized")


async def cleanup_simple_error_handling():
    """Cleanup simple error handling."""
    await simple_task_manager.cleanup_all()
    logger.info("Simple error handling cleaned up")
