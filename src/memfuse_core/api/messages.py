"""Message API endpoints."""

from typing import Dict, List, Optional, Tuple, Any, cast
from loguru import logger
from fastapi import APIRouter, Depends

from ..models import (
    MessageAdd,
    MessageRead,
    MessageUpdate,
    MessageDelete,
    ApiResponse,
    ErrorDetail,
    Message,
)
from ..utils.auth import validate_api_key


router = APIRouter()

ENABLE_BUFFER_SERVICE = True

# Create a dependency for API key validation
api_key_dependency = Depends(validate_api_key)

# Avoid circular imports by importing these functions when needed
# from ..services.service_factory import ServiceFactory
# from ..services.database_service import DatabaseService


async def validate_session(
    session_id: str
) -> Tuple[Optional[Dict[str, Any]], Optional[ApiResponse]]:
    """Validate that a session exists and return session data or error response.

    Args:
        session_id: The ID of the session to validate

    Returns:
        Tuple containing:
        - Session data if session exists, None otherwise
        - Error response if session doesn't exist, None otherwise
    """
    # Import here to avoid circular imports
    from ..services.database_service import DatabaseService

    db = DatabaseService.get_instance()
    session = db.get_session(session_id)

    if not session:
        error_response = ApiResponse.error(
            message=f"Session with ID '{session_id}' not found",
            code=404,
            errors=[
                ErrorDetail(
                    field="session_id",
                    message=f"Session with ID '{session_id}' not found"
                )
            ],
        )
        return None, error_response

    return session, None


async def get_service_for_session(
    session: Optional[Dict[str, Any]],
    session_id: str
):
    """Get the appropriate service (Buffer or Memory) for a session.

    Args:
        session: Session data dictionary (can be None)
        session_id: Session ID

    Returns:
        Service instance or None if session is invalid
    """
    # Import here to avoid circular imports
    from ..services.service_factory import ServiceFactory
    from ..services.database_service import DatabaseService

    # Get user, agent, and session information
    if session is None:
        logger.error("Session is None")
        return None

    db = DatabaseService.get_instance()
    user_id = session.get("user_id")
    agent_id = session.get("agent_id")

    if not user_id or not agent_id:
        logger.error("Invalid session data: missing user_id or agent_id")
        return None

    user = db.get_user(user_id)
    agent = db.get_agent(agent_id)

    if not user or not agent:
        logger.error(f"User or agent not found for session {session_id}")
        return None

    user_name = user["name"]
    agent_name = agent["name"]
    session_name = session.get("name", "default")

    if ENABLE_BUFFER_SERVICE:
        # If BufferService is enabled, use it
        logger.info("Using BufferService for message operations")
        service = await ServiceFactory.get_buffer_service(
            user=user_name,
            agent=agent_name,
            session=session_name,
            session_id=session_id,
        )
    else:
        # Otherwise, get a MemoryService instance for this user
        logger.info("BufferService disabled, using MemoryService instance")
        service = await ServiceFactory.get_memory_service(
            user=user_name,
            agent=agent_name,
            session=session_name,
            session_id=session_id,
        )

    return service


def convert_pydantic_to_dict(
    messages: List[Any]
) -> List[Dict[str, Any]]:
    """Convert Pydantic models to dictionaries if needed.

    Args:
        messages: List of message objects

    Returns:
        List of message dictionaries
    """
    result = []
    for message in messages:
        if hasattr(message, 'model_dump'):
            result.append(cast(Message, message).model_dump())
        else:
            result.append(cast(Dict[str, Any], message))
    return result


@router.post("/", response_model=ApiResponse)
# Also handle path without trailing slash
@router.post("", response_model=ApiResponse)
async def add_messages(
    session_id: str,
    request: MessageAdd,
    # Underscore prefix to indicate unused
    _api_key_data: dict = api_key_dependency,
) -> ApiResponse:
    """Add messages to a session."""
    try:
        # Validate session
        session, error_response = await validate_session(session_id)
        if error_response:
            return error_response

        # Get memory service
        if session is None:
            return ApiResponse.error(
                message="Invalid session data",
                code=500,
                errors=[
                    ErrorDetail(
                        field="session_id",
                        message="Session data is invalid"
                    )
                ]
            )

        memory = await get_service_for_session(session, session_id)
        if not memory:
            return ApiResponse.error(
                message="Failed to get service",
                code=500,
                errors=[
                    ErrorDetail(
                        field="general",
                        message="Memory or buffer service unavailable"
                    )
                ]
            )

        # Convert messages and add them
        messages = convert_pydantic_to_dict(request.messages)
        # P1 OPTIMIZATION: Pass session_id to add method
        result = await memory.add(messages, session_id=session_id)

        # Extract message IDs from result
        message_ids = []
        if (result and result.get("status") == "success"
                and result.get("data") is not None):
            message_ids = result["data"].get("message_ids", [])

        return ApiResponse.success(
            data={"message_ids": message_ids},
            message="Messages added successfully",
        )
    except Exception as e:
        logger.error(f"Failed to add messages: {str(e)}")
        return ApiResponse.error(
            message="Failed to add messages",
            errors=[ErrorDetail(field="general", message=str(e))],
        )


@router.get("/", response_model=ApiResponse)
# Also handle path without trailing slash
@router.get("", response_model=ApiResponse)
async def list_messages(
    session_id: str,
    limit: Optional[str] = "20",  # Changed to str to handle validation manually
    sort_by: str = "timestamp",
    order: str = "desc",
    # Underscore prefix to indicate unused
    _api_key_data: dict = api_key_dependency,
) -> ApiResponse:
    """List messages in a session with optional limit and sorting.

    Args:
        session_id: The ID of the session to get messages from
        limit: Maximum number of messages to return (default: 20, max: 100)
        sort_by: Field to sort messages by (allowed values: timestamp, id)
        order: Sort order (allowed values: asc, desc)
    """
    try:
        # Validate query parameters
        # Validate and convert limit
        limit_value = 20  # Default value
        if limit is not None:
            try:
                limit_value = int(limit)
                if limit_value <= 0:
                    return ApiResponse.error(
                        message="Invalid limit parameter",
                        code=400,
                        errors=[ErrorDetail(
                            field="limit", message="Limit must be greater than 0")],
                    )
                if limit_value > 100:
                    # Cap at 100
                    limit_value = 100
            except ValueError:
                return ApiResponse.error(
                    message="Invalid limit parameter",
                    code=400,
                    errors=[ErrorDetail(
                        field="limit", message="Limit must be an integer")],
                )

        # Validate sort_by
        allowed_sort_fields = ["timestamp", "id"]
        if sort_by not in allowed_sort_fields:
            return ApiResponse.error(
                message="Invalid sort_by parameter",
                code=400,
                errors=[ErrorDetail(
                    field="sort_by",
                    message=f"sort_by must be one of: {', '.join(allowed_sort_fields)}"
                )],
            )

        # Validate order
        allowed_orders = ["asc", "desc"]
        if order not in allowed_orders:
            return ApiResponse.error(
                message="Invalid order parameter",
                code=400,
                errors=[ErrorDetail(
                    field="order",
                    message=f"order must be one of: {', '.join(allowed_orders)}"
                )],
            )

        # Validate session
        _, error_response = await validate_session(session_id)
        if error_response:
            return error_response

        # Get messages directly from database with limit and sorting
        from ..services.database_service import DatabaseService
        db = DatabaseService.get_instance()
        messages = db.get_messages_by_session(
            session_id=session_id,
            limit=limit_value,
            sort_by=sort_by,
            order=order
        )

        return ApiResponse.success(
            data={"messages": messages},
            message="Messages retrieved successfully",
        )
    except Exception as e:
        logger.error(f"Failed to list messages: {str(e)}")
        return ApiResponse.error(
            message="Failed to list messages",
            errors=[ErrorDetail(field="general", message=str(e))],
        )


@router.post("/read", response_model=ApiResponse)
# Also handle path with trailing slash
@router.post("/read/", response_model=ApiResponse)
async def read_messages(
    session_id: str,
    request: MessageRead,
    # Underscore prefix to indicate unused
    _api_key_data: dict = api_key_dependency,
) -> ApiResponse:
    """Read specific messages from a session."""
    try:
        # Validate session
        session, error_response = await validate_session(session_id)
        if error_response:
            return error_response

        # Get memory service
        if session is None:
            return ApiResponse.error(
                message="Invalid session data",
                code=500,
                errors=[
                    ErrorDetail(
                        field="session_id",
                        message="Session data is invalid"
                    )
                ]
            )

        memory = await get_service_for_session(session, session_id)
        if not memory:
            return ApiResponse.error(
                message="Failed to get service",
                code=500,
                errors=[
                    ErrorDetail(
                        field="general",
                        message="Memory or buffer service unavailable"
                    )
                ]
            )

        # Read messages
        result = await memory.read(request.message_ids)

        return ApiResponse.success(
            data={"messages": result["data"]["messages"]},
            message="Messages read successfully",
        )
    except Exception as e:
        logger.error(f"Failed to read messages: {str(e)}")
        return ApiResponse.error(
            message="Failed to read messages",
            errors=[ErrorDetail(field="general", message=str(e))],
        )


@router.put("/", response_model=ApiResponse)
# Also handle path without trailing slash
@router.put("", response_model=ApiResponse)
async def update_messages(
    session_id: str,
    request: MessageUpdate,
    # Underscore prefix to indicate unused
    _api_key_data: dict = api_key_dependency,
) -> ApiResponse:
    """Update messages in a session."""
    try:
        # Validate session
        session, error_response = await validate_session(session_id)
        if error_response:
            return error_response

        # Get memory service
        if session is None:
            return ApiResponse.error(
                message="Invalid session data",
                code=500,
                errors=[
                    ErrorDetail(
                        field="session_id",
                        message="Session data is invalid"
                    )
                ]
            )

        memory = await get_service_for_session(session, session_id)
        if not memory:
            return ApiResponse.error(
                message="Failed to get service",
                code=500,
                errors=[
                    ErrorDetail(
                        field="general",
                        message="Memory or buffer service unavailable"
                    )
                ]
            )

        # Convert messages and update them
        new_messages = convert_pydantic_to_dict(request.new_messages)
        await memory.update(request.message_ids, new_messages)

        return ApiResponse.success(
            data={"message_ids": request.message_ids},
            message="Messages updated successfully",
        )
    except Exception as e:
        logger.error(f"Failed to update messages: {str(e)}")
        return ApiResponse.error(
            message="Failed to update messages",
            errors=[ErrorDetail(field="general", message=str(e))],
        )


@router.delete("/", response_model=ApiResponse)
# Also handle path without trailing slash
@router.delete("", response_model=ApiResponse)
async def delete_messages(
    session_id: str,
    request: MessageDelete,
    # Underscore prefix to indicate unused
    _api_key_data: dict = api_key_dependency,
) -> ApiResponse:
    """Delete messages from a session."""
    try:
        # Validate session
        session, error_response = await validate_session(session_id)
        if error_response:
            return error_response

        # Get memory service
        if session is None:
            return ApiResponse.error(
                message="Invalid session data",
                code=500,
                errors=[
                    ErrorDetail(
                        field="session_id",
                        message="Session data is invalid"
                    )
                ]
            )

        memory = await get_service_for_session(session, session_id)
        if not memory:
            return ApiResponse.error(
                message="Failed to get service",
                code=500,
                errors=[
                    ErrorDetail(
                        field="general",
                        message="Memory or buffer service unavailable"
                    )
                ]
            )

        # Delete messages
        result = await memory.delete(request.message_ids)

        # Check if any messages were not found
        if result.get("status") == "error" and result.get("code") == 404:
            return ApiResponse.error(
                message=result.get(
                    "message", "Some message IDs were not found"),
                code=404,
                errors=result.get("errors", []),
            )

        return ApiResponse.success(
            data={"message_ids": request.message_ids},
            message="Messages deleted successfully",
        )
    except Exception as e:
        logger.error(f"Failed to delete messages: {str(e)}")
        return ApiResponse.error(
            message="Failed to delete messages",
            errors=[ErrorDetail(field="general", message=str(e))],
        )
