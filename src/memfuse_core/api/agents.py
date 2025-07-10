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
    db = DatabaseService.get_instance()

    # If name is provided, get agent by name
    if name:
        agent = ensure_agent_by_name_exists(db, name)
        return ApiResponse.success(
            data={"agents": [agent]},
            message="Agent retrieved successfully",
        )

    # Otherwise, list all agents
    agents = db.get_all_agents()
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
    db = DatabaseService.get_instance()

    # Check if agent with the same name already exists
    ensure_agent_name_available(db, request.name)

    # Create the agent
    agent_id = db.create_agent(
        name=request.name,
        description=request.description,
    )

    # Get the created agent
    agent = db.get_agent(agent_id)

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
    db = DatabaseService.get_instance()

    # Validate agent exists
    agent = ensure_agent_exists(db, agent_id)

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
    db = DatabaseService.get_instance()

    # Check if agent exists
    _ = ensure_agent_exists(db, agent_id)

    # Update the agent
    success = db.update_agent(
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
    updated_agent = db.get_agent(agent_id)

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
    db = DatabaseService.get_instance()

    # Check if agent exists
    _ = ensure_agent_exists(db, agent_id)

    # Delete the agent
    success = db.delete_agent(agent_id)

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
