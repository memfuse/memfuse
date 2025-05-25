"""Authentication middleware for MemFuse server."""

import time
from fastapi import Request, HTTPException, Depends
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from typing import Dict, Any

from ..services.database_service import DatabaseService
from .config import config_manager, get_api_key_header

# Get API key header name from configuration
API_KEY_HEADER_NAME = get_api_key_header()

# API key header
API_KEY_HEADER = APIKeyHeader(name=API_KEY_HEADER_NAME, auto_error=False)


async def validate_api_key(api_key_header: str = Depends(API_KEY_HEADER)) -> Dict[str, Any]:
    """Validate the API key from the Authorization header.

    Args:
        api_key_header: The Authorization header value

    Returns:
        The API key data including user_id

    Raises:
        HTTPException: If the API key is invalid and validation is enabled
    """
    # Check if API key validation is enabled
    config = config_manager.get_config()
    api_key_validation = config.get(
        "server", {}).get("api_key_validation", False)

    # If API key validation is disabled, return a default API key data
    if not api_key_validation:
        # Return a default API key data with admin permissions
        return {
            "id": "00000000-0000-0000-0000-000000000000",
            "user_id": "00000000-0000-0000-0000-000000000000",
            "key": "default",
            "name": "Default API Key",
            "permissions": "admin",
            "created_at": None,
            "expires_at": None,
            "is_admin": True
        }

    # API key validation is enabled, proceed with validation
    if not api_key_header:
        raise HTTPException(
            status_code=401,
            detail={
                "status": "error",
                "code": 401,
                "message": "Authentication failed: Missing API key",
                "errors": [
                    {
                        "field": "api_key",
                        "message": "API key is required"
                    }
                ]
            },
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Extract the API key from the header
    # Format: "Bearer {api_key}"
    parts = api_key_header.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=401,
            detail={
                "status": "error",
                "code": 401,
                "message": "Authentication failed: Invalid API key format",
                "errors": [
                    {
                        "field": "api_key",
                        "message": "API key format should be 'Bearer YOUR_API_KEY'"
                    }
                ]
            },
            headers={"WWW-Authenticate": "Bearer"},
        )

    api_key = parts[1]

    # Validate the API key
    db = DatabaseService.get_instance()
    api_key_data = db.validate_api_key(api_key)

    if api_key_data is None:
        raise HTTPException(
            status_code=401,
            detail={
                "status": "error",
                "code": 401,
                "message": "Authentication failed: Invalid API key",
                "errors": [
                    {
                        "field": "api_key",
                        "message": "The provided API key is invalid or has expired"
                    }
                ]
            },
            headers={"WWW-Authenticate": "Bearer"},
        )

    return api_key_data


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware for FastAPI."""

    def __init__(self, app, rate_limit_per_minute: int = 60):
        """Initialize the rate limit middleware.

        Args:
            app: FastAPI application
            rate_limit_per_minute: Maximum number of requests per minute
        """
        super().__init__(app)
        self.rate_limit = rate_limit_per_minute
        self.requests = {}  # {ip: (count, reset_time)}

    async def dispatch(self, request: Request, call_next):
        """Process the request.

        Args:
            request: FastAPI request
            call_next: Next middleware or endpoint

        Returns:
            FastAPI response
        """
        ip = request.client.host
        now = time.time()

        # Check if IP exists and if the reset time has passed
        if ip in self.requests:
            count, reset_time = self.requests[ip]
            if now > reset_time:
                # Reset counter if the minute has passed
                self.requests[ip] = (1, now + 60)
            else:
                # Increment counter
                count += 1
                if count > self.rate_limit:
                    # Rate limit exceeded
                    return JSONResponse(
                        content={
                            "status": "error",
                            "code": 429,
                            "message": "Rate limit exceeded",
                            "errors": [
                                {
                                    "field": "general",
                                    "message": "Too many requests, please try again later"
                                }
                            ]
                        },
                        status_code=429
                    )
                self.requests[ip] = (count, reset_time)
        else:
            # First request from this IP
            self.requests[ip] = (1, now + 60)

        return await call_next(request)
