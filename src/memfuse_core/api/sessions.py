"""Session API endpoints."""

import logging
from fastapi import APIRouter, Depends, Query, status
from typing import Optional

from ..models import (
    SessionCreate,
    SessionUpdate,
    ApiResponse,
    ErrorDetail,
)
from ..services.database_service import DatabaseService
from ..utils.auth import validate_api_key
from ..utils import (
    ensure_user_exists,
    ensure_agent_exists,
    ensure_session_exists,
    ensure_session_by_name_exists,
    handle_api_errors,
    raise_api_error,
)


router = APIRouter()

# Configure logging
logger = logging.getLogger(__name__)


@router.get("/", response_model=ApiResponse)
# Also handle path without trailing slash
@router.get("", response_model=ApiResponse)
@handle_api_errors("list sessions")
async def list_sessions(
    user_id: Optional[str] = Query(None),
    agent_id: Optional[str] = Query(None),
    name: Optional[str] = Query(None),
    _: dict = Depends(validate_api_key),  # API key validation
) -> ApiResponse:
    """List sessions, optionally filtered by user, agent, or name."""
    db = await DatabaseService.get_instance()

    # If name is provided, get session by name
    if name:
        # CORRECT IMPLEMENTATION (2025-05-24): Support both user-scoped and global session lookup
        # This maintains the RESTful API design:
        # - GET /api/v1/sessions?name=xxx&user_id=yyy → user-scoped lookup (recommended)
        # - GET /api/v1/sessions?name=xxx → global lookup (returns all sessions with that name)
        # The real data isolation happens at the data storage and query level, not here
        session = await ensure_session_by_name_exists(db, name, user_id=user_id)
        return ApiResponse.success(
            data={"sessions": [session]},
            message="Session retrieved successfully",
        )

    # Validate user_id if provided
    if user_id:
        await ensure_user_exists(db, user_id)

    # Validate agent_id if provided
    if agent_id:
        await ensure_agent_exists(db, agent_id)

    # Get sessions with filters
    sessions = await db.get_sessions(user_id=user_id, agent_id=agent_id)

    return ApiResponse.success(
        data={"sessions": sessions},
        message="Sessions retrieved successfully",
    )


@router.post("/", response_model=ApiResponse, status_code=status.HTTP_201_CREATED)
# Also handle path without trailing slash
@router.post("", response_model=ApiResponse, status_code=status.HTTP_201_CREATED)
@handle_api_errors("create session")
async def create_session(
    request: SessionCreate,
    _: dict = Depends(validate_api_key),  # API key validation
) -> ApiResponse:
    """Create a new session."""
    db = await DatabaseService.get_instance()

    # Validate user_id and agent_id
    user = await ensure_user_exists(db, request.user_id)
    agent = await ensure_agent_exists(db, request.agent_id)

    # Generate a session name if not provided
    session_name = request.name
    if session_name is None:
        import uuid
        # Use the same pattern as in MemoryService
        session_name = f"{user['name']}-{agent['name']}-{str(uuid.uuid4())[:8]}"
        logger.info(f"Generated session name: {session_name}")

    # Check if a session with this name already exists for this user
    existing_session = await db.get_session_by_name(session_name, user_id=request.user_id)
    if existing_session:
        error_response = ApiResponse.error(
            message=f"Session with name '{session_name}' already exists for this user",
            code=400,
            errors=[
                ErrorDetail(
                    field="name",
                    message=f"Session with name '{session_name}' already exists. Please choose a different name."
                )
            ],
        )
        raise_api_error(error_response)

    # Check if there's already a session with null name for this user/agent pair
    # This is to prevent duplicate sessions with null names
    existing_sessions = await db.get_sessions(
        user_id=request.user_id, agent_id=request.agent_id)
    for existing_session in existing_sessions:
        if existing_session["name"] is None or existing_session["name"] == "":
            # Update the existing session with the new name
            logger.info(
                f"Found existing session with null name, updating with name: {session_name}")
            success = await db.update_session(
                session_id=existing_session["id"],
                name=session_name
            )
            
            if not success:
                error_response = ApiResponse.error(
                    message="Failed to update existing session",
                    errors=[ErrorDetail(field="general", message="Database update failed")],
                )
                raise_api_error(error_response)
            
            # Get the updated session
            updated_session = await db.get_session(existing_session["id"])

            # Initialize memory service for this updated session
            try:
                from ..services import MemoryService
                memory = MemoryService(
                    user=user["name"],
                    agent=agent["name"],
                    session_id=updated_session["id"]
                )
                # Initialize the memory instance
                await memory.initialize()
            except Exception as e:
                logger.error(f"Error initializing memory service: {str(e)}")
                # Even if initialization fails, still return the session

            return ApiResponse.success(
                data={"session": updated_session},
                message="Session updated with name",
            )

    # Create the session
    session_id = await db.create_session(
        user_id=request.user_id,
        agent_id=request.agent_id,
        name=session_name,  # Using generated or provided name
    )

    # Get the created session
    session = await db.get_session(session_id)

    # Initialize memory service for this session
    try:
        from ..services import MemoryService
        memory = MemoryService(
            user=user["name"],
            agent=agent["name"],
            session_id=session["id"]
        )
        # Initialize the memory instance
        await memory.initialize()
    except Exception as e:
        logger.error(f"Error initializing memory service: {str(e)}")
        # Even if initialization fails, still return the session

    return ApiResponse.success(
        data={"session": session},
        message="Session created successfully",
        code=201,
    )


@router.get("/{session_id}", response_model=ApiResponse)
# Also handle path with trailing slash
@router.get("/{session_id}/", response_model=ApiResponse)
@handle_api_errors("get session")
async def get_session(
    session_id: str,
    _: dict = Depends(validate_api_key),  # API key validation
) -> ApiResponse:
    """Get session details."""
    db = await DatabaseService.get_instance()

    # Check if session exists
    session = await ensure_session_exists(db, session_id)

    return ApiResponse.success(
        data={"session": session},
        message="Session retrieved successfully",
    )


@router.put("/{session_id}", response_model=ApiResponse)
# Also handle path with trailing slash
@router.put("/{session_id}/", response_model=ApiResponse)
@handle_api_errors("update session")
async def update_session(
    session_id: str,
    request: SessionUpdate,
    _: dict = Depends(validate_api_key),  # API key validation
) -> ApiResponse:
    """Update session details."""
    db = await DatabaseService.get_instance()

    # Check if session exists
    _ = await ensure_session_exists(db, session_id)

    # Update the session
    success = await db.update_session(
        session_id=session_id,
        name=request.name,  # Using name parameter
    )

    if not success:
        error_response = ApiResponse.error(
            message="Failed to update session",
            errors=[ErrorDetail(field="general", message="Database update failed")],
        )
        raise_api_error(error_response)

    # Get the updated session
    updated_session = await db.get_session(session_id)

    return ApiResponse.success(
        data={"session": updated_session},
        message="Session updated successfully",
    )


@router.delete("/{session_id}", response_model=ApiResponse)
# Also handle path with trailing slash
@router.delete("/{session_id}/", response_model=ApiResponse)
@handle_api_errors("delete session")
async def delete_session(
    session_id: str,
    _: dict = Depends(validate_api_key),  # API key validation
) -> ApiResponse:
    """Delete a session."""
    db = await DatabaseService.get_instance()

    # Check if session exists
    _ = await ensure_session_exists(db, session_id)

    # Delete the session
    success = await db.delete_session(session_id)

    if not success:
        error_response = ApiResponse.error(
            message="Failed to delete session",
            errors=[ErrorDetail(field="general", message="Database delete failed")],
        )
        raise_api_error(error_response)

    return ApiResponse.success(
        data={"session_id": session_id},
        message="Session deleted successfully",
    )
