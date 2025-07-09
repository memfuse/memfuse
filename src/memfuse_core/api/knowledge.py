"""Knowledge API endpoints."""

import logging
from fastapi import APIRouter, Depends, status

from ..models import (
    KnowledgeAdd,
    KnowledgeRead,
    KnowledgeUpdate,
    KnowledgeDelete,
    ApiResponse,
    ErrorDetail,
)
from ..services.database_service import DatabaseService
from ..utils.auth import validate_api_key
from ..utils import (
    ensure_user_exists,
    handle_api_errors,
    raise_api_error,
)


router = APIRouter()

# Configure logging
logger = logging.getLogger(__name__)


@router.get("/", response_model=ApiResponse)
# Also handle path without trailing slash
@router.get("", response_model=ApiResponse)
@handle_api_errors("list knowledge")
async def list_knowledge(
    user_id: str,
    _: dict = Depends(validate_api_key),  # API key validation
) -> ApiResponse:
    """List all knowledge items for a user."""
    db = DatabaseService.get_instance()

    # Check if user exists
    user = ensure_user_exists(db, user_id)

    # Get knowledge items
    knowledge_items = db.get_knowledge_by_user(user_id)

    return ApiResponse.success(
        data={"knowledge": knowledge_items},
        message="Knowledge items retrieved successfully",
    )


@router.post("/", response_model=ApiResponse, status_code=status.HTTP_201_CREATED)
# Also handle path without trailing slash
@router.post("", response_model=ApiResponse, status_code=status.HTTP_201_CREATED)
@handle_api_errors("add knowledge")
async def add_knowledge(
    user_id: str,
    request: KnowledgeAdd,
    _: dict = Depends(validate_api_key),  # API key validation
) -> ApiResponse:
    """Add knowledge items for a user."""
    db = DatabaseService.get_instance()

    # Check if user exists
    user = ensure_user_exists(db, user_id)

    # Add knowledge items
    knowledge_ids = []
    for content in request.knowledge:
        knowledge_id = db.add_knowledge(
            user_id=user_id,
            content=content,
        )
        knowledge_ids.append(knowledge_id)

    return ApiResponse.success(
        data={"knowledge_ids": knowledge_ids},
        message="Knowledge items added successfully",
        code=201,
    )


@router.post("/read", response_model=ApiResponse)
# Also handle path with trailing slash
@router.post("/read/", response_model=ApiResponse)
@handle_api_errors("read knowledge")
async def read_knowledge(
    user_id: str,
    request: KnowledgeRead,
    _: dict = Depends(validate_api_key),  # API key validation
) -> ApiResponse:
    """Read specific knowledge items for a user."""
    db = DatabaseService.get_instance()

    # Check if user exists
    user = ensure_user_exists(db, user_id)

    # Read knowledge items
    knowledge_items = []
    for knowledge_id in request.knowledge_ids:
        knowledge = db.get_knowledge(knowledge_id)
        if knowledge and knowledge["user_id"] == user_id:
            knowledge_items.append(knowledge)

    return ApiResponse.success(
        data={"knowledge": knowledge_items},
        message="Knowledge items retrieved successfully",
    )


@router.put("/", response_model=ApiResponse)
# Also handle path without trailing slash
@router.put("", response_model=ApiResponse)
@handle_api_errors("update knowledge")
async def update_knowledge(
    user_id: str,
    request: KnowledgeUpdate,
    _: dict = Depends(validate_api_key),  # API key validation
) -> ApiResponse:
    """Update knowledge items for a user."""
    db = DatabaseService.get_instance()

    # Check if user exists
    user = ensure_user_exists(db, user_id)

    # Update knowledge items
    updated_ids = []
    for i, knowledge_id in enumerate(request.knowledge_ids):
        # Check if knowledge item exists and belongs to the user
        knowledge = db.get_knowledge(knowledge_id)
        if not knowledge or knowledge["user_id"] != user_id:
            continue

        # Update the knowledge item
        success = db.update_knowledge(
            knowledge_id=knowledge_id,
            content=request.new_knowledge[i],
        )

        if success:
            updated_ids.append(knowledge_id)

    return ApiResponse.success(
        data={"knowledge_ids": updated_ids},
        message="Knowledge items updated successfully",
    )


@router.delete("/", response_model=ApiResponse)
# Also handle path without trailing slash
@router.delete("", response_model=ApiResponse)
@handle_api_errors("delete knowledge")
async def delete_knowledge(
    user_id: str,
    request: KnowledgeDelete,
    _: dict = Depends(validate_api_key),  # API key validation
) -> ApiResponse:
    """Delete knowledge items for a user."""
    db = DatabaseService.get_instance()

    # Check if user exists
    user = ensure_user_exists(db, user_id)

    # Delete knowledge items
    deleted_ids = []
    for knowledge_id in request.knowledge_ids:
        # Check if knowledge item exists and belongs to the user
        knowledge = db.get_knowledge(knowledge_id)
        if not knowledge or knowledge["user_id"] != user_id:
            continue

        # Delete the knowledge item
        success = db.delete_knowledge(knowledge_id)

        if success:
            deleted_ids.append(knowledge_id)

    return ApiResponse.success(
        data={"knowledge_ids": deleted_ids},
        message="Knowledge items deleted successfully",
    )
