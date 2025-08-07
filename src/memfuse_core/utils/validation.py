"""Validation utilities for MemFuse server."""

from __future__ import annotations
from typing import Optional, Dict, Any, Tuple, TYPE_CHECKING

from fastapi import HTTPException
from ..models.core import ApiResponse, ErrorDetail

if TYPE_CHECKING:
    from ..database import Database


def raise_api_error(api_response: ApiResponse) -> None:
    """Raise HTTPException with ApiResponse format in detail."""
    raise HTTPException(
        status_code=api_response.code,
        detail=api_response.dict()
    )


# Convenience functions that raise exceptions instead of returning tuples
async def ensure_user_exists(db: "Database", user_id: str) -> Dict[str, Any]:
    """Validate that a user exists, raise HTTPException if not."""
    is_valid, error_response, user = await validate_user_exists(db, user_id)
    if not is_valid:
        raise_api_error(error_response)
    return user


async def ensure_user_by_name_exists(db: "Database", name: str) -> Dict[str, Any]:
    """Validate that a user with the given name exists, raise HTTPException if not."""
    is_valid, error_response, user = await validate_user_by_name_exists(db, name)
    if not is_valid:
        raise_api_error(error_response)
    return user


async def ensure_user_name_available(db: "Database", name: str) -> None:
    """Validate that a user name is available, raise HTTPException if not."""
    is_valid, error_response = await validate_user_name_available(db, name)
    if not is_valid:
        raise_api_error(error_response)


async def ensure_agent_exists(db: "Database", agent_id: str) -> Dict[str, Any]:
    """Validate that an agent exists, raise HTTPException if not."""
    is_valid, error_response, agent = await validate_agent_exists(db, agent_id)
    if not is_valid:
        raise_api_error(error_response)
    return agent


async def ensure_agent_by_name_exists(db: "Database", name: str) -> Dict[str, Any]:
    """Validate that an agent with the given name exists, raise HTTPException if not."""
    is_valid, error_response, agent = await validate_agent_by_name_exists(db, name)
    if not is_valid:
        raise_api_error(error_response)
    return agent


async def ensure_agent_name_available(db: "Database", name: str) -> None:
    """Validate that an agent name is available, raise HTTPException if not."""
    is_valid, error_response = await validate_agent_name_available(db, name)
    if not is_valid:
        raise_api_error(error_response)


async def validate_user_exists(db: "Database", user_id: str) -> Tuple[bool, Optional[ApiResponse], Optional[Dict[str, Any]]]:
    """Validate that a user exists.

    Args:
        db: Database instance
        user_id: User ID

    Returns:
        Tuple of (is_valid, error_response, user_data)
    """
    user = await db.get_user(user_id)
    if not user:
        return False, ApiResponse.error(
            message=f"User with ID '{user_id}' not found",
            code=404,
            errors=[
                ErrorDetail(
                    field="user_id",
                    message=f"User with ID '{user_id}' not found"
                )
            ],
        ), None
    return True, None, user


async def validate_agent_exists(db: "Database", agent_id: str) -> Tuple[bool, Optional[ApiResponse], Optional[Dict[str, Any]]]:
    """Validate that an agent exists.

    Args:
        db: Database instance
        agent_id: Agent ID

    Returns:
        Tuple of (is_valid, error_response, agent_data)
    """
    agent = await db.get_agent(agent_id)
    if not agent:
        return False, ApiResponse.error(
            message=f"Agent with ID '{agent_id}' not found",
            code=404,
            errors=[
                ErrorDetail(
                    field="agent_id",
                    message=f"Agent with ID '{agent_id}' not found"
                )
            ],
        ), None
    return True, None, agent


async def validate_session_exists(db: "Database", session_id: str) -> Tuple[bool, Optional[ApiResponse], Optional[Dict[str, Any]]]:
    """Validate that a session exists.

    Args:
        db: Database instance
        session_id: Session ID

    Returns:
        Tuple of (is_valid, error_response, session_data)
    """
    session = await db.get_session(session_id)
    if not session:
        return False, ApiResponse.error(
            message=f"Session with ID '{session_id}' not found",
            code=404,
            errors=[
                ErrorDetail(
                    field="session_id",
                    message=f"Session with ID '{session_id}' not found"
                )
            ],
        ), None
    return True, None, session


async def validate_user_by_name_exists(db: "Database", name: str) -> Tuple[bool, Optional[ApiResponse], Optional[Dict[str, Any]]]:
    """Validate that a user with the given name exists.

    Args:
        db: Database instance
        name: User name

    Returns:
        Tuple of (is_valid, error_response, user_data)
    """
    user = await db.get_user_by_name(name)
    if not user:
        return False, ApiResponse.error(
            message=f"User with name '{name}' not found",
            code=404,
            errors=[
                ErrorDetail(
                    field="name",
                    message=f"User with name '{name}' not found"
                )
            ],
        ), None
    return True, None, user


async def validate_agent_by_name_exists(db: "Database", name: str) -> Tuple[bool, Optional[ApiResponse], Optional[Dict[str, Any]]]:
    """Validate that an agent with the given name exists.

    Args:
        db: Database instance
        name: Agent name

    Returns:
        Tuple of (is_valid, error_response, agent_data)
    """
    agent = await db.get_agent_by_name(name)
    if not agent:
        return False, ApiResponse.error(
            message=f"Agent with name '{name}' not found",
            code=404,
            errors=[
                ErrorDetail(
                    field="name",
                    message=f"Agent with name '{name}' not found"
                )
            ],
        ), None
    return True, None, agent


async def validate_session_by_name_exists(db: "Database", name: str, user_id: Optional[str] = None) -> Tuple[bool, Optional[ApiResponse], Optional[Dict[str, Any]]]:
    """Validate that a session with the given name exists.

    Args:
        db: Database instance
        name: Session name
        user_id: User ID for scoped session lookup (optional)

    Returns:
        Tuple of (is_valid, error_response, session_data)
    """
    session = await db.get_session_by_name(name, user_id=user_id)
    if not session:
        scope_msg = f" for user '{user_id}'" if user_id else ""
        return False, ApiResponse.error(
            message=f"Session with name '{name}' not found{scope_msg}",
            code=404,
            errors=[
                ErrorDetail(
                    field="name",
                    message=f"Session with name '{name}' not found{scope_msg}"
                )
            ],
        ), None
    return True, None, session


async def validate_user_name_available(db: "Database", name: str) -> Tuple[bool, Optional[ApiResponse]]:
    """Validate that a user name is available.

    Args:
        db: Database instance
        name: User name

    Returns:
        Tuple of (is_valid, error_response)
    """
    existing_user = await db.get_user_by_name(name)
    if existing_user:
        return False, ApiResponse.error(
            message=f"User with name '{name}' already exists",
            code=400,
            errors=[
                ErrorDetail(
                    field="name",
                    message=f"User with name '{name}' already exists"
                )
            ],
        )
    return True, None


async def validate_agent_name_available(db: "Database", name: str) -> Tuple[bool, Optional[ApiResponse]]:
    """Validate that an agent name is available.

    Args:
        db: Database instance
        name: Agent name

    Returns:
        Tuple of (is_valid, error_response)
    """
    existing_agent = await db.get_agent_by_name(name)
    if existing_agent:
        return False, ApiResponse.error(
            message=f"Agent with name '{name}' already exists",
            code=400,
            errors=[
                ErrorDetail(
                    field="name",
                    message=f"Agent with name '{name}' already exists"
                )
            ],
        )
    return True, None


async def validate_session_name_available(db: "Database", name: str, user_id: Optional[str] = None) -> Tuple[bool, Optional[ApiResponse]]:
    """Validate that a session name is available.

    Args:
        db: Database instance
        name: Session name
        user_id: User ID for scoped session lookup (optional)

    Returns:
        Tuple of (is_valid, error_response)
    """
    existing_session = await db.get_session_by_name(name, user_id=user_id)
    if existing_session:
        scope_msg = f" for user '{user_id}'" if user_id else ""
        return False, ApiResponse.error(
            message=f"Session with name '{name}' already exists{scope_msg}",
            code=400,
            errors=[
                ErrorDetail(
                    field="name",
                    message=f"Session with name '{name}' already exists{scope_msg}"
                )
            ],
        )
    return True, None


async def ensure_session_exists(db: "Database", session_id: str) -> Dict[str, Any]:
    """Validate that a session exists, raise HTTPException if not."""
    is_valid, error_response, session = await validate_session_exists(db, session_id)
    if not is_valid:
        raise_api_error(error_response)
    return session


async def ensure_session_by_name_exists(db: "Database", name: str, user_id: Optional[str] = None) -> Dict[str, Any]:
    """Validate that a session with the given name exists, raise HTTPException if not."""
    is_valid, error_response, session = await validate_session_by_name_exists(db, name, user_id)
    if not is_valid:
        raise_api_error(error_response)
    return session


async def ensure_session_name_available(db: "Database", name: str, user_id: Optional[str] = None) -> None:
    """Validate that a session name is available, raise HTTPException if not."""
    is_valid, error_response = await validate_session_name_available(db, name, user_id)
    if not is_valid:
        raise_api_error(error_response)