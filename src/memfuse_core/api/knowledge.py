"""Knowledge API endpoints."""

import logging
from fastapi import APIRouter, Depends

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


router = APIRouter()

# Configure logging
logger = logging.getLogger(__name__)


@router.get("/", response_model=ApiResponse)
# Also handle path without trailing slash
@router.get("", response_model=ApiResponse)
async def list_knowledge(
    user_id: str,
    api_key_data: dict = Depends(validate_api_key),
) -> ApiResponse:
    """List all knowledge items for a user."""
    try:
        db = DatabaseService.get_instance()

        # Check if user exists
        user = db.get_user(user_id)
        if not user:
            return ApiResponse.error(
                message=f"User with ID '{user_id}' not found",
                code=404,
                errors=[
                    ErrorDetail(
                        field="user_id",
                        message=f"User with ID '{user_id}' not found"
                    )
                ],
            )

        # Get knowledge items
        knowledge_items = db.get_knowledge_by_user(user_id)

        return ApiResponse.success(
            data={"knowledge": knowledge_items},
            message="Knowledge items retrieved successfully",
        )
    except Exception as e:
        logger.error(f"Failed to list knowledge items: {str(e)}")
        return ApiResponse.error(
            message="Failed to list knowledge items",
            errors=[ErrorDetail(field="general", message=str(e))],
        )


@router.post("/", response_model=ApiResponse)
# Also handle path without trailing slash
@router.post("", response_model=ApiResponse)
async def add_knowledge(
    user_id: str,
    request: KnowledgeAdd,
    api_key_data: dict = Depends(validate_api_key),
) -> ApiResponse:
    """Add knowledge items for a user."""
    try:
        db = DatabaseService.get_instance()

        # Check if user exists
        user = db.get_user(user_id)
        if not user:
            return ApiResponse.error(
                message=f"User with ID '{user_id}' not found",
                code=404,
                errors=[
                    ErrorDetail(
                        field="user_id",
                        message=f"User with ID '{user_id}' not found"
                    )
                ],
            )

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
        )
    except Exception as e:
        logger.error(f"Failed to add knowledge items: {str(e)}")
        return ApiResponse.error(
            message="Failed to add knowledge items",
            errors=[ErrorDetail(field="general", message=str(e))],
        )


@router.post("/read", response_model=ApiResponse)
# Also handle path with trailing slash
@router.post("/read/", response_model=ApiResponse)
async def read_knowledge(
    user_id: str,
    request: KnowledgeRead,
    api_key_data: dict = Depends(validate_api_key),
) -> ApiResponse:
    """Read specific knowledge items for a user."""
    try:
        db = DatabaseService.get_instance()

        # Check if user exists
        user = db.get_user(user_id)
        if not user:
            return ApiResponse.error(
                message=f"User with ID '{user_id}' not found",
                code=404,
                errors=[
                    ErrorDetail(
                        field="user_id",
                        message=f"User with ID '{user_id}' not found"
                    )
                ],
            )

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
    except Exception as e:
        logger.error(f"Failed to read knowledge items: {str(e)}")
        return ApiResponse.error(
            message="Failed to read knowledge items",
            errors=[ErrorDetail(field="general", message=str(e))],
        )


@router.put("/", response_model=ApiResponse)
# Also handle path without trailing slash
@router.put("", response_model=ApiResponse)
async def update_knowledge(
    user_id: str,
    request: KnowledgeUpdate,
    api_key_data: dict = Depends(validate_api_key),
) -> ApiResponse:
    """Update knowledge items for a user."""
    try:
        db = DatabaseService.get_instance()

        # Check if user exists
        user = db.get_user(user_id)
        if not user:
            return ApiResponse.error(
                message=f"User with ID '{user_id}' not found",
                code=404,
                errors=[
                    ErrorDetail(
                        field="user_id",
                        message=f"User with ID '{user_id}' not found"
                    )
                ],
            )

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
    except Exception as e:
        logger.error(f"Failed to update knowledge items: {str(e)}")
        return ApiResponse.error(
            message="Failed to update knowledge items",
            errors=[ErrorDetail(field="general", message=str(e))],
        )


@router.delete("/", response_model=ApiResponse)
# Also handle path without trailing slash
@router.delete("", response_model=ApiResponse)
async def delete_knowledge(
    user_id: str,
    request: KnowledgeDelete,
    api_key_data: dict = Depends(validate_api_key),
) -> ApiResponse:
    """Delete knowledge items for a user."""
    try:
        db = DatabaseService.get_instance()

        # Check if user exists
        user = db.get_user(user_id)
        if not user:
            return ApiResponse.error(
                message=f"User with ID '{user_id}' not found",
                code=404,
                errors=[
                    ErrorDetail(
                        field="user_id",
                        message=f"User with ID '{user_id}' not found"
                    )
                ],
            )

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
    except Exception as e:
        logger.error(f"Failed to delete knowledge items: {str(e)}")
        return ApiResponse.error(
            message="Failed to delete knowledge items",
            errors=[ErrorDetail(field="general", message=str(e))],
        )
