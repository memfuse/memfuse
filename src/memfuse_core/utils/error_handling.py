"""Error handling utilities for MemFuse server."""

from loguru import logger
import functools
from typing import Callable, TypeVar, Any, Awaitable

from fastapi import HTTPException
from ..models.core import ApiResponse, ErrorDetail


T = TypeVar('T')


def handle_api_errors(operation_name: str) -> Callable[[Callable[..., Awaitable[ApiResponse]]], Callable[..., Awaitable[ApiResponse]]]:
    """Decorator to handle API errors.

    This decorator catches exceptions and converts them to appropriate API responses.

    Args:
        operation_name: Name of the operation for logging purposes

    Returns:
        The decorated function
    """
    def decorator(func: Callable[..., Awaitable[ApiResponse]]) -> Callable[..., Awaitable[ApiResponse]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> ApiResponse:
            try:
                return await func(*args, **kwargs)
            except HTTPException as e:
                # Let FastAPI handle HTTP exceptions
                raise e
            except Exception as e:
                # Log the error
                logger.error(f"Failed to {operation_name}: {str(e)}")

                # Return an error response
                return ApiResponse.error(
                    message=f"Failed to {operation_name}",
                    errors=[ErrorDetail(field="general", message=str(e))],
                )
        return wrapper
    return decorator
