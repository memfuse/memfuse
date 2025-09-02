"""Agent API endpoints."""

import logging
from fastapi import APIRouter, Depends, status
from typing import Optional

from ..models import (
    AgentCreate,
    AgentUpdate,
    ApiResponse,
    ErrorDetail,
)
from ..services.database_service import DatabaseService
from ..utils.auth import validate_api_key
from ..utils import (
    ensure_agent_exists,
    ensure_agent_by_name_exists,
    ensure_agent_name_available,
    handle_api_errors,
    raise_api_error,
)


router = APIRouter()

# Configure logging
logger = logging.getLogger(__name__)


@router.get("/", response_model=ApiResponse)
# Also handle path without trailing slash
@router.get("", response_model=ApiResponse)
@handle_api_errors("list agents")
async def list_agents(
    name: Optional[str] = None,
    _: dict = Depends(validate_api_key),  # API key validation
) -> ApiResponse:
    """List all agents or get an agent by name."""
    db = await DatabaseService.get_instance()

    # If name is provided, get agent by name
    if name:
        agent = await ensure_agent_by_name_exists(db, name)
        return ApiResponse.success(
            data={"agents": [agent]},
            message="Agent retrieved successfully",
        )

    # Otherwise, list all agents
    agents = await db.get_all_agents()
    return ApiResponse.success(
        data={"agents": agents},
        message="Agents retrieved successfully",
    )


@router.post("/", response_model=ApiResponse, status_code=status.HTTP_201_CREATED)
# Also handle path without trailing slash
@router.post("", response_model=ApiResponse, status_code=status.HTTP_201_CREATED)
@handle_api_errors("create agent")
async def create_agent(
    request: AgentCreate,
    _: dict = Depends(validate_api_key),  # API key validation
) -> ApiResponse:
    """Create a new agent."""
    db = await DatabaseService.get_instance()

    # Pre-check name availability to preserve existing 400 behavior
    await ensure_agent_name_available(db, request.name)

    # Create the agent with concurrency-safe error handling
    try:
        agent_id = await db.create_agent(
            name=request.name,
            description=request.description,
        )
    except Exception as e:
        # Concurrency-safe behavior: if unique violation occurs here due to a race
        # after the availability check, return the existing agent as success (200).
        msg = str(e).lower()
        if (
            'duplicate key' in msg or 'unique' in msg or 'already exists' in msg
        ):
            existing = await db.get_agent_by_name(request.name)
            if existing:
                return ApiResponse.success(
                    data={"agent": existing},
                    message="Agent already existed; returning existing record",
                    code=200,
                )
            # If not found unexpectedly, fallback to 400 error
            error_response = ApiResponse.error(
                message=f"Agent with name '{request.name}' already exists",
                code=400,
                errors=[ErrorDetail(field="name", message="already exists")],
            )
            raise_api_error(error_response)
        # Unknown error - let decorator handle
        raise

    # Get the created agent
    agent = await db.get_agent(agent_id)

    return ApiResponse.success(
        data={"agent": agent},
        message="Agent created successfully",
        code=201,
    )


@router.get("/{agent_id}", response_model=ApiResponse)
# Also handle path with trailing slash
@router.get("/{agent_id}/", response_model=ApiResponse)
@handle_api_errors("get agent")
async def get_agent(
    agent_id: str,
    _: dict = Depends(validate_api_key),  # API key validation
) -> ApiResponse:
    """Get agent details."""
    db = await DatabaseService.get_instance()

    # Validate agent exists
    agent = await ensure_agent_exists(db, agent_id)

    return ApiResponse.success(
        data={"agent": agent},
        message="Agent retrieved successfully",
    )


@router.put("/{agent_id}", response_model=ApiResponse)
# Also handle path with trailing slash
@router.put("/{agent_id}/", response_model=ApiResponse)
@handle_api_errors("update agent")
async def update_agent(
    agent_id: str,
    request: AgentUpdate,
    _: dict = Depends(validate_api_key),  # API key validation
) -> ApiResponse:
    """Update agent details."""
    db = await DatabaseService.get_instance()

    # Check if agent exists
    _ = await ensure_agent_exists(db, agent_id)

    # Update the agent
    success = await db.update_agent(
        agent_id=agent_id,
        name=request.name,
        description=request.description,
    )

    if not success:
        error_response = ApiResponse.error(
            message="Failed to update agent",
            errors=[ErrorDetail(
                field="general", message="Database update failed")],
        )
        raise_api_error(error_response)

    # Get the updated agent
    updated_agent = await db.get_agent(agent_id)

    return ApiResponse.success(
        data={"agent": updated_agent},
        message="Agent updated successfully",
    )


@router.delete("/{agent_id}", response_model=ApiResponse)
# Also handle path with trailing slash
@router.delete("/{agent_id}/", response_model=ApiResponse)
@handle_api_errors("delete agent")
async def delete_agent(
    agent_id: str,
    _: dict = Depends(validate_api_key),  # API key validation
) -> ApiResponse:
    """Delete an agent."""
    db = await DatabaseService.get_instance()

    # Check if agent exists
    _ = await ensure_agent_exists(db, agent_id)

    # Delete the agent
    success = await db.delete_agent(agent_id)

    if not success:
        error_response = ApiResponse.error(
            message="Failed to delete agent",
            errors=[ErrorDetail(
                field="general", message="Database delete failed")],
        )
        raise_api_error(error_response)

    return ApiResponse.success(
        data={"agent_id": agent_id},
        message="Agent deleted successfully",
    )
