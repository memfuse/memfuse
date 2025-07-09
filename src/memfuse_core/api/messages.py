"""Message API endpoints."""

from typing import Dict, List, Optional, Tuple, Any, cast
from loguru import logger
from fastapi import APIRouter, Depends, status

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
from ..utils import (
    ensure_session_exists,
    handle_api_errors,
    raise_api_error,
)
from ..services.database_service import DatabaseService
from ..services.service_factory import ServiceFactory


router = APIRouter()

ENABLE_BUFFER_SERVICE = True

# Create a dependency for API key validation
api_key_dependency = Depends(validate_api_key)

# Avoid circular imports by importing these functions when needed


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


@router.post("/", response_model=ApiResponse, status_code=status.HTTP_201_CREATED)
# Also handle path without trailing slash
@router.post("", response_model=ApiResponse, status_code=status.HTTP_201_CREATED)
@handle_api_errors("add messages")
async def add_messages(
    session_id: str,
    request: MessageAdd,
    # Underscore prefix to indicate unused
    _api_key_data: dict = api_key_dependency,
) -> ApiResponse:
    """Add messages to a session."""
    db = DatabaseService.get_instance()
    
    # Validate session exists
    session = ensure_session_exists(db, session_id)

    # Get memory service
    memory = await get_service_for_session(session, session_id)
    if not memory:
        error_response = ApiResponse.error(
            message="Failed to get service",
            code=500,
            errors=[
                ErrorDetail(
                    field="general",
                    message="Memory or buffer service unavailable"
                )
            ]
        )
        raise_api_error(error_response)

    # Convert messages and add them
    messages = convert_pydantic_to_dict(request.messages)
    # P1 OPTIMIZATION: Pass session_id to add method
    result = await memory.add(messages, session_id=session_id)

    # Extract message IDs from result
    message_ids = []
    if (result and result.get("status") == "success"
            and result.get("data") is not None):
        message_ids = result["data"].get("message_ids", [])

    # Create response data with message IDs
    response_data = {"message_ids": message_ids}

    # Add any additional fields from the service result (e.g., transfer_triggered)
    if result and result.get("status") == "success":
        # Include additional fields like transfer_triggered, total_messages, etc.
        for key, value in result.items():
            if key not in ["status", "code", "data", "message", "errors"]:
                response_data[key] = value

    return ApiResponse.success(
        data=response_data,
        message="Messages added successfully",
        code=201,
    )


@router.get("/", response_model=ApiResponse)
# Also handle path without trailing slash
@router.get("", response_model=ApiResponse)
@handle_api_errors("list messages")
async def list_messages(
    session_id: str,
    limit: Optional[str] = "20",  # Changed to str to handle validation manually
    sort_by: str = "timestamp",
    order: str = "desc",
    buffer_only: Optional[str] = None,  # Buffer support
    # Underscore prefix to indicate unused
    _api_key_data: dict = api_key_dependency,
) -> ApiResponse:
    """List messages in a session with optional limit and sorting.

    Args:
        session_id: The ID of the session to get messages from
        limit: Maximum number of messages to return (default: 20, max: 100)
        sort_by: Field to sort messages by (allowed values: timestamp, id)
        order: Sort order (allowed values: asc, desc)
        buffer_only: Buffer parameter - if "true", only return RoundBuffer data; if "false", return HybridBuffer + SQLite data excluding RoundBuffer
    """
    db = DatabaseService.get_instance()
    
    # Validate query parameters
    # Validate and convert limit
    limit_value = 20  # Default value
    if limit is not None:
        try:
            limit_value = int(limit)
            if limit_value <= 0:
                error_response = ApiResponse.error(
                    message="Invalid limit parameter",
                    code=400,
                    errors=[ErrorDetail(
                        field="limit", message="Limit must be greater than 0")],
                )
                raise_api_error(error_response)
            if limit_value > 100:
                # Cap at 100
                limit_value = 100
        except ValueError:
            error_response = ApiResponse.error(
                message="Invalid limit parameter",
                code=400,
                errors=[ErrorDetail(
                    field="limit", message="Limit must be an integer")],
            )
            raise_api_error(error_response)

    # Validate sort_by
    allowed_sort_fields = ["timestamp", "id"]
    if sort_by not in allowed_sort_fields:
        error_response = ApiResponse.error(
            message="Invalid sort_by parameter",
            code=400,
            errors=[ErrorDetail(
                field="sort_by",
                message=f"sort_by must be one of: {', '.join(allowed_sort_fields)}"
            )],
        )
        raise_api_error(error_response)

    # Validate order
    allowed_orders = ["asc", "desc"]
    if order not in allowed_orders:
        error_response = ApiResponse.error(
            message="Invalid order parameter",
            code=400,
            errors=[ErrorDetail(
                field="order",
                message=f"order must be one of: {', '.join(allowed_orders)}"
            )],
        )
        raise_api_error(error_response)

    # Validate and convert buffer_only parameter
    buffer_only_value = None
    if buffer_only is not None:
        if buffer_only.lower() == "true":
            buffer_only_value = True
        elif buffer_only.lower() == "false":
            buffer_only_value = False
        else:
            error_response = ApiResponse.error(
                message="Invalid buffer_only parameter",
                code=400,
                errors=[ErrorDetail(
                    field="buffer_only",
                    message="buffer_only must be 'true' or 'false'"
                )],
            )
            raise_api_error(error_response)

    # Validate session exists
    session = ensure_session_exists(db, session_id)

    # Get the appropriate service (Buffer or Memory) for this session
    service = await get_service_for_session(session, session_id)
    if not service:
        error_response = ApiResponse.error(
            message="Failed to get service",
            code=500,
            errors=[
                ErrorDetail(
                    field="general",
                    message="Memory or buffer service unavailable"
                )
            ]
        )
        raise_api_error(error_response)

    # Get messages through the service (which handles both buffer and direct database access)
    if hasattr(service, 'get_messages_by_session'):
        # Check if service supports buffer_only parameter
        if buffer_only_value is not None and hasattr(service, 'get_messages_by_session'):
            # Try to call with buffer_only parameter
            try:
                messages = await service.get_messages_by_session(
                    session_id=session_id,
                    limit=limit_value,
                    sort_by=sort_by,
                    order=order,
                    buffer_only=buffer_only_value
                )
            except TypeError:
                # Service doesn't support buffer_only parameter, call without it
                messages = await service.get_messages_by_session(
                    session_id=session_id,
                    limit=limit_value,
                    sort_by=sort_by,
                    order=order
                )
        else:
            # Call without buffer_only parameter
            messages = await service.get_messages_by_session(
                session_id=session_id,
                limit=limit_value,
                sort_by=sort_by,
                order=order
            )
    else:
        # Fallback to direct database access if service doesn't support the method
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


@router.post("/read", response_model=ApiResponse)
# Also handle path with trailing slash
@router.post("/read/", response_model=ApiResponse)
@handle_api_errors("read messages")
async def read_messages(
    session_id: str,
    request: MessageRead,
    # Underscore prefix to indicate unused
    _api_key_data: dict = api_key_dependency,
) -> ApiResponse:
    """Read specific messages from a session."""
    db = DatabaseService.get_instance()
    
    # Validate session exists
    session = ensure_session_exists(db, session_id)

    # Get memory service
    memory = await get_service_for_session(session, session_id)
    if not memory:
        error_response = ApiResponse.error(
            message="Failed to get service",
            code=500,
            errors=[
                ErrorDetail(
                    field="general",
                    message="Memory or buffer service unavailable"
                )
            ]
        )
        raise_api_error(error_response)

    # Read messages
    result = await memory.read(request.message_ids)

    return ApiResponse.success(
        data={"messages": result["data"]["messages"]},
        message="Messages read successfully",
    )


@router.put("/", response_model=ApiResponse)
# Also handle path without trailing slash
@router.put("", response_model=ApiResponse)
@handle_api_errors("update messages")
async def update_messages(
    session_id: str,
    request: MessageUpdate,
    # Underscore prefix to indicate unused
    _api_key_data: dict = api_key_dependency,
) -> ApiResponse:
    """Update messages in a session."""
    db = DatabaseService.get_instance()
    
    # Validate session exists
    session = ensure_session_exists(db, session_id)

    # Get memory service
    memory = await get_service_for_session(session, session_id)
    if not memory:
        error_response = ApiResponse.error(
            message="Failed to get service",
            code=500,
            errors=[
                ErrorDetail(
                    field="general",
                    message="Memory or buffer service unavailable"
                )
            ]
        )
        raise_api_error(error_response)

    # Convert messages and update them
    new_messages = convert_pydantic_to_dict(request.new_messages)
    await memory.update(request.message_ids, new_messages)

    return ApiResponse.success(
        data={"message_ids": request.message_ids},
        message="Messages updated successfully",
    )


@router.delete("/", response_model=ApiResponse)
# Also handle path without trailing slash
@router.delete("", response_model=ApiResponse)
@handle_api_errors("delete messages")
async def delete_messages(
    session_id: str,
    request: MessageDelete,
    # Underscore prefix to indicate unused
    _api_key_data: dict = api_key_dependency,
) -> ApiResponse:
    """Delete messages from a session."""
    db = DatabaseService.get_instance()
    
    # Validate session exists
    session = ensure_session_exists(db, session_id)

    # Get memory service
    memory = await get_service_for_session(session, session_id)
    if not memory:
        error_response = ApiResponse.error(
            message="Failed to get service",
            code=500,
            errors=[
                ErrorDetail(
                    field="general",
                    message="Memory or buffer service unavailable"
                )
            ]
        )
        raise_api_error(error_response)

    # Delete messages
    result = await memory.delete(request.message_ids)

    # Check if any messages were not found
    if result.get("status") == "error" and result.get("code") == 404:
        error_response = ApiResponse.error(
            message=result.get(
                "message", "Some message IDs were not found"),
            code=404,
            errors=result.get("errors", []),
        )
        raise_api_error(error_response)

    return ApiResponse.success(
        data={"message_ids": request.message_ids},
        message="Messages deleted successfully",
    )
