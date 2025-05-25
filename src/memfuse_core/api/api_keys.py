"""API Key management endpoints."""

import logging
from fastapi import APIRouter, Depends

from ..models import (
    ApiKeyCreate,
    ApiResponse,
    ErrorDetail,
)
from ..services.database_service import DatabaseService
from ..utils.auth import validate_api_key
from ..utils import (
    validate_user_exists,
    handle_api_errors,
)


router = APIRouter()

# Configure logging
logger = logging.getLogger(__name__)


@router.post("/", response_model=ApiResponse)
@handle_api_errors("create API key")
async def create_api_key(
    request: ApiKeyCreate,
    api_key_data: dict = Depends(validate_api_key),
) -> ApiResponse:
    """Create a new API key for a user."""
    db = DatabaseService.get_instance()

    # Check if user exists
    user_id = request.user_id
    is_valid, error_response, user = validate_user_exists(db, user_id)
    if not is_valid:
        return error_response

    # Check if the requesting user has permission to create API keys for this user
    # Only the user themselves or an admin can create API keys
    if api_key_data["user_id"] != user_id and not api_key_data.get("is_admin", False):
        return ApiResponse.error(
            message="You do not have permission to create API keys for this user",
            code=403,
            errors=[
                ErrorDetail(
                    field="user_id",
                    message="You do not have permission to create API keys for this user"
                )
            ],
        )

    # Create the API key
    api_key_id = db.create_api_key(
        user_id=user_id,
        name=request.name,
        key=request.key,  # Allow custom key if provided
        permissions=request.permissions,
        expires_at=request.expires_at,
    )

    # Get the API key data
    api_key = db.get_api_key(api_key_id)

    return ApiResponse.success(
        data={"api_key": api_key},
        message="API key created successfully",
    )


@router.get("/", response_model=ApiResponse)
@handle_api_errors("list API keys")
async def list_api_keys(
    user_id: str,
    api_key_data: dict = Depends(validate_api_key),
) -> ApiResponse:
    """List all API keys for a user."""
    db = DatabaseService.get_instance()

    # Check if user exists
    is_valid, error_response, _ = validate_user_exists(db, user_id)
    if not is_valid:
        return error_response

    # Check if the requesting user has permission to list API keys for this user
    # Only the user themselves or an admin can list API keys
    if api_key_data["user_id"] != user_id and not api_key_data.get("is_admin", False):
        return ApiResponse.error(
            message="You do not have permission to list API keys for this user",
            code=403,
            errors=[
                ErrorDetail(
                    field="user_id",
                    message="You do not have permission to list API keys for this user"
                )
            ],
        )

    # Get the API keys
    api_keys = db.get_api_keys_by_user(user_id)

    # Remove the actual key value for security
    for api_key in api_keys:
        if "key" in api_key:
            api_key["key"] = "********"  # Mask the actual key

    return ApiResponse.success(
        data={"api_keys": api_keys},
        message="API keys retrieved successfully",
    )


@router.delete("/{key_id}", response_model=ApiResponse)
@handle_api_errors("delete API key")
async def delete_api_key(
    user_id: str,
    key_id: str,
    api_key_data: dict = Depends(validate_api_key),
) -> ApiResponse:
    """Delete an API key."""
    db = DatabaseService.get_instance()

    # Check if user exists
    is_valid, error_response, _ = validate_user_exists(db, user_id)
    if not is_valid:
        return error_response

    # Check if the API key exists and belongs to the user
    api_key = db.get_api_key(key_id)
    if not api_key or api_key["user_id"] != user_id:
        return ApiResponse.error(
            message=f"API key with ID '{key_id}' not found for user '{user_id}'",
            code=404,
            errors=[
                ErrorDetail(
                    field="key_id",
                    message=f"API key with ID '{key_id}' not found for user '{user_id}'"
                )
            ],
        )

    # Check if the requesting user has permission to delete this API key
    # Only the user themselves or an admin can delete API keys
    if api_key_data["user_id"] != user_id and not api_key_data.get("is_admin", False):
        return ApiResponse.error(
            message="You do not have permission to delete API keys for this user",
            code=403,
            errors=[
                ErrorDetail(
                    field="user_id",
                    message="You do not have permission to delete API keys for this user"
                )
            ],
        )

    # Delete the API key
    success = db.delete_api_key(key_id)

    if not success:
        return ApiResponse.error(
            message=f"Failed to delete API key with ID '{key_id}'",
            code=500,
            errors=[
                ErrorDetail(
                    field="key_id",
                    message=f"Failed to delete API key with ID '{key_id}'"
                )
            ],
        )

    return ApiResponse.success(
        data={"key_id": key_id},
        message="API key deleted successfully",
    )
