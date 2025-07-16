"""Error handling utilities for MemFuse server."""

from loguru import logger
import functools
import inspect
from typing import Callable, TypeVar, Any, Awaitable, get_type_hints, Union

from fastapi import HTTPException
from ..models.core import ApiResponse, ErrorDetail


T = TypeVar('T')


class ApiError(Exception):
    """Custom exception that carries ApiResponse data."""
    
    def __init__(self, api_response: ApiResponse):
        self.api_response = api_response
        self.status_code = api_response.code
        super().__init__(api_response.message)


def handle_api_errors(operation_name: str) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
    """Decorator to handle API errors.

    This decorator catches exceptions and converts them to appropriate API responses.
    For endpoints that return ApiResponse objects, it returns ApiResponse.error().
    For endpoints that return None (like delete endpoints), it raises HTTPException.

    Args:
        operation_name: Name of the operation for logging purposes

    Returns:
        The decorated function
    """
    def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except ApiError:
                # Re-raise ApiError so it can be handled by the exception handler
                raise
            except HTTPException as e:
                # Let FastAPI handle HTTP exceptions
                raise e
            except Exception as e:
                # Log the error
                logger.error(f"Failed to {operation_name}: {str(e)}")

                # Check if this is a database constraint error
                error_msg = str(e).lower()
                if 'constraint' in error_msg or 'foreign key' in error_msg:
                    status_code = 409  # Conflict
                    message = f"Failed to {operation_name}: operation conflicts with existing data"
                else:
                    status_code = 500  # Internal Server Error
                    message = f"Failed to {operation_name}"

                # Determine the return type of the function
                type_hints = get_type_hints(func)
                return_type = type_hints.get('return', None)
                
                # Check if function returns None (like delete endpoints)
                if return_type is None or return_type == type(None):
                    # Raise HTTPException for endpoints that don't return ApiResponse
                    raise HTTPException(
                        status_code=status_code,
                        detail={
                            "status": "error",
                            "code": status_code,
                            "message": message,
                            "errors": [{"field": "general", "message": str(e)}]
                        }
                    )
                else:
                    # Return ApiResponse.error() for endpoints that return ApiResponse
                    return ApiResponse.error(
                        message=message,
                        code=status_code,
                        errors=[ErrorDetail(field="general", message=str(e))],
                    )
        return wrapper
    return decorator
