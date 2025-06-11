"""User API endpoints."""

import logging
from fastapi import APIRouter, Depends
from typing import Optional

from ..models import (
    UserCreate,
    UserUpdate,
    MemoryQuery,
    ApiResponse,
    ErrorDetail,
)
from ..services.database_service import DatabaseService
from ..utils.auth import validate_api_key
from ..utils import (
    validate_user_exists,
    validate_user_by_name_exists,
    validate_user_name_available,
    handle_api_errors,
    prepare_response_data,
)


router = APIRouter()

# Configure logging
logger = logging.getLogger(__name__)


@router.get("/", response_model=ApiResponse)
# Also handle path without trailing slash
@router.get("", response_model=ApiResponse)
@handle_api_errors("list users")
async def list_users(
    name: Optional[str] = None,
    _: dict = Depends(validate_api_key),  # API key validation
) -> ApiResponse:
    """List all users or get a user by name."""
    db = DatabaseService.get_instance()

    # If name is provided, get user by name
    if name:
        is_valid, error_response, user = validate_user_by_name_exists(db, name)
        if not is_valid:
            return error_response

        return ApiResponse.success(
            data={"users": [user]},
            message="User retrieved successfully",
        )

    # Otherwise, list all users
    users = db.get_all_users()
    return ApiResponse.success(
        data={"users": users},
        message="Users retrieved successfully",
    )


@router.post("/", response_model=ApiResponse)
# Also handle path without trailing slash
@router.post("", response_model=ApiResponse)
@handle_api_errors("create user")
async def create_user(
    request: UserCreate,
    _: dict = Depends(validate_api_key),  # API key validation
) -> ApiResponse:
    """Create a new user."""
    db = DatabaseService.get_instance()

    # Check if user with the same name already exists
    is_valid, error_response = validate_user_name_available(db, request.name)
    if not is_valid:
        return error_response

    # Create the user
    user_id = db.create_user(
        name=request.name,
        description=request.description,
    )

    # Get the created user
    user = db.get_user(user_id)

    return ApiResponse.success(
        data={"user": user},
        message="User created successfully",
    )


@router.get("/{user_id}", response_model=ApiResponse)
# Also handle path with trailing slash
@router.get("/{user_id}/", response_model=ApiResponse)
@handle_api_errors("get user")
async def get_user(
    user_id: str,
    _: dict = Depends(validate_api_key),  # API key validation
) -> ApiResponse:
    """Get user details."""
    db = DatabaseService.get_instance()

    # Validate user exists
    is_valid, error_response, user = validate_user_exists(db, user_id)
    if not is_valid:
        return error_response

    return ApiResponse.success(
        data={"user": user},
        message="User retrieved successfully",
    )


@router.put("/{user_id}", response_model=ApiResponse)
@handle_api_errors("update user")
async def update_user(
    user_id: str,
    request: UserUpdate,
    _: dict = Depends(validate_api_key),  # API key validation
) -> ApiResponse:
    """Update user details."""
    db = DatabaseService.get_instance()

    # Check if user exists
    is_valid, error_response, user = validate_user_exists(db, user_id)
    if not is_valid:
        return error_response

    # Update the user
    success = db.update_user(
        user_id=user_id,
        name=request.name,
        description=request.description,
    )

    if not success:
        return ApiResponse.error(
            message="Failed to update user",
            errors=[ErrorDetail(
                field="general", message="Database update failed")],
        )

    # Get the updated user
    updated_user = db.get_user(user_id)

    return ApiResponse.success(
        data={"user": updated_user},
        message="User updated successfully",
    )


@router.delete("/{user_id}", response_model=ApiResponse)
@handle_api_errors("delete user")
async def delete_user(
    user_id: str,
    _: dict = Depends(validate_api_key),  # API key validation
) -> ApiResponse:
    """Delete a user."""
    db = DatabaseService.get_instance()

    # Check if user exists
    is_valid, error_response, _ = validate_user_exists(db, user_id)
    if not is_valid:
        return error_response

    # Delete the user
    success = db.delete_user(user_id)

    if not success:
        return ApiResponse.error(
            message="Failed to delete user",
            errors=[ErrorDetail(
                field="general", message="Database delete failed")],
        )

    return ApiResponse.success(
        data={"user_id": user_id},
        message="User deleted successfully",
    )


@router.post("/{user_id}/query", response_model=ApiResponse)
# Also handle path with trailing slash
@router.post("/{user_id}/query/", response_model=ApiResponse)
@handle_api_errors("query memory")
async def query_memory(
    user_id: str,
    request: MemoryQuery,
    _: dict = Depends(validate_api_key),  # API key validation
) -> ApiResponse:
    """Query memory across all sessions for a user.

    This endpoint supports querying memory across all sessions for a user.
    If session_id is provided, results will be tagged with scope="in_session" or
    scope="cross_session" depending on whether they belong to the specified session.
    If session_id is not provided, all results will have scope=null.
    """
    from ..services.service_factory import ServiceFactory

    db = DatabaseService.get_instance()
    logger.info("Using BufferService for query operations")

    # Check if user exists
    is_valid, error_response, user = validate_user_exists(db, user_id)
    if not is_valid:
        return error_response

    # Validate session if provided
    if request.session_id:
        session = db.get_session(request.session_id)
        if not session or session["user_id"] != user_id:
            return ApiResponse.error(
                message=f"Session '{request.session_id}' not found for user '{user_id}'",
                code=404,
                errors=[
                    ErrorDetail(
                        field="session_id",
                        message=f"Session '{request.session_id}' not found for user '{user_id}'"
                    )
                ],
            )

    # Validate agent if provided
    if request.agent_id:
        agent = db.get_agent(request.agent_id)
        if not agent:
            return ApiResponse.error(
                message=f"Agent '{request.agent_id}' not found",
                code=404,
                errors=[
                    ErrorDetail(
                        field="agent_id",
                        message=f"Agent '{request.agent_id}' not found"
                    )
                ],
            )

    # Always query all sessions for the user
    sessions = db.get_sessions(user_id=user_id, agent_id=request.agent_id)

    if not sessions:
        return ApiResponse.success(
            data={"results": [], "total": 0},
            message="No sessions found for query",
        )

    # First, collect all results from all sessions
    all_session_results = []

    # We need to query each session separately to ensure proper metadata
    all_session_results = []

    for session in sessions:
        # Create a buffer service for this session (to ensure consistency with write operations)
        from ..api.messages import ENABLE_BUFFER_SERVICE

        if ENABLE_BUFFER_SERVICE:
            # Use BufferService for consistency with write operations
            memory = await ServiceFactory.get_buffer_service_for_user(user["name"])
        else:
            # Use MemoryService directly when buffer is disabled
            memory = await ServiceFactory.get_memory_service(
                user=user["name"],
                agent=db.get_agent(session["agent_id"])["name"],
                session_id=session["id"]
            )

        # Query this session
        logger.info(f"🔍 API LAYER: Querying session {session['id']} with query: {request.query[:50]}...")
        result = await memory.query(
            query=request.query,
            top_k=request.top_k,
            store_type=request.store_type,
            session_id=session["id"],
            include_messages=request.include_messages,
            include_knowledge=request.include_knowledge,
        )

        # Debug the raw result from BufferService
        logger.info(f"🔍 API LAYER: Raw result from BufferService: {type(result)}")
        logger.info(f"🔍 API LAYER: Result keys: {list(result.keys()) if isinstance(result, dict) else 'not dict'}")
        logger.info(f"🔍 API LAYER: Result status: {result.get('status') if isinstance(result, dict) else 'unknown'}")

        # Get results for this session
        session_results = result.get("data", {}).get("results", [])
        logger.info(f"🔍 API LAYER: Session {session['id']} returned {len(session_results)} results")

        if session_results:
            logger.info(f"🔍 API LAYER: First result from session: {session_results[0]}")
        else:
            logger.warning(f"🚨 API LAYER: Session {session['id']} returned NO results!")

        # Add session scope information
        for r in session_results:
            if r.get("metadata") and r.get("metadata").get("session_id"):
                if request.session_id:
                    if r["metadata"]["session_id"] == request.session_id:
                        r["metadata"]["scope"] = "in_session"
                    else:
                        r["metadata"]["scope"] = "cross_session"
                else:
                    r["metadata"]["scope"] = None

        # Add results from this session to the combined results
        all_session_results.extend(session_results)

    # Create a map to store the true metadata for each message ID
    # We'll use this to deduplicate results based on the actual message ID
    message_map = {}
    knowledge_map = {}

    # First, get all messages from the database to ensure we have the correct metadata
    logger.info(f"🔍 API LAYER: Processing {len(all_session_results)} raw results from BufferService")

    # Debug: Check if we got any results at all
    if not all_session_results:
        logger.warning("🚨 API LAYER: No results received from BufferService!")
        return ApiResponse.success(
            data={"results": [], "total": 0},
            message="No results found from BufferService",
        )

    for i, result in enumerate(all_session_results):
        logger.info(f"🔍 API LAYER: Processing raw result {i}: type={type(result)}, content preview: {str(result)[:100]}...")
        # Handle MessageList format (List of Messages from WriteBuffer)
        if isinstance(result, list):
            # This is a MessageList from WriteBuffer - process each message
            for message in result:
                if isinstance(message, dict):
                    # Generate a temporary ID for WriteBuffer messages
                    result_id = f"buffer_{hash(str(message))}"

                    # Create a standardized result format
                    standardized_result = {
                        "id": result_id,
                        "content": message.get("content", ""),
                        "score": 1.0,  # Default score for WriteBuffer results
                        "type": "message",
                        "role": message.get("role", "unknown"),
                        "created_at": None,  # WriteBuffer messages don't have timestamps yet
                        "updated_at": None,
                        "metadata": message.get("metadata", {})
                    }

                    # Store in message_map
                    if result_id not in message_map:
                        message_map[result_id] = standardized_result
            continue

        # Ensure result is a dictionary before processing
        if not isinstance(result, dict):
            logger.warning(f"Unexpected result format: {type(result)}, skipping")
            continue

        # Handle standard result format (from persistent storage)
        result_id = result.get("id")
        if not result_id:
            continue

        if result.get("type") == "message":
            # Check if this is a buffer result (from memory) vs database result
            is_buffer_result = result.get("metadata", {}).get("retrieval", {}).get("source") in ["write_buffer", "speculative_buffer", "query_buffer", "hybrid_buffer"]

            if is_buffer_result:
                # For buffer results, use the result as-is since they come from memory
                logger.info(f"Processing buffer result ID: {result_id}")
                if result_id not in message_map or result.get("score", 0) > message_map[result_id].get("score", 0):
                    message_map[result_id] = result
            else:
                # Get the message from the database to verify its true session and agent
                message = db.get_message(result_id)
                if message:
                    # Get the actual round and session for this message
                    round_data = db.get_round(message.get(
                        "round_id")) if message.get("round_id") else None
                    if round_data and round_data.get("session_id"):
                        actual_session_id = round_data.get("session_id")
                        actual_session = db.get_session(actual_session_id)

                        if actual_session:
                            # Store the message with its correct metadata
                            # If we've seen this message before, keep the one with the higher score
                            if result_id not in message_map or result.get("score", 0) > message_map[result_id].get("score", 0):
                                # Create a new result with the correct metadata
                                message_map[result_id] = {
                                    "id": result_id,
                                    "content": result.get("content"),
                                    "score": result.get("score", 0),
                                    "type": "message",
                                    "role": message.get("role"),
                                    "created_at": message.get("created_at"),
                                    "updated_at": message.get("updated_at"),
                                    "metadata": {
                                        "user_id": user_id,
                                        "agent_id": actual_session["agent_id"],
                                        "session_id": actual_session_id,
                                        "session_name": actual_session["name"],
                                        "scope": ("in_session" if actual_session_id == request.session_id
                                                  else "cross_session") if request.session_id else None,
                                        "level": 0,
                                        "retrieval": result.get("metadata", {}).get("retrieval", {})
                                    }
                                }
        elif result.get("type") == "knowledge":
            # For knowledge items, just store them by ID
            if result_id not in knowledge_map or result.get("score", 0) > knowledge_map[result_id].get("score", 0):
                knowledge_map[result_id] = {
                    "id": result_id,
                    "content": result.get("content"),
                    "score": result.get("score", 0),
                    "type": "knowledge",
                    "role": None,  # Knowledge items don't have roles
                    "created_at": result.get("created_at"),
                    "updated_at": result.get("updated_at"),
                    "metadata": {
                        "user_id": user_id,
                        "agent_id": None,  # Knowledge is not associated with agents
                        "session_id": None,  # Knowledge is not associated with sessions
                        "session_name": None,
                        "scope": None,
                        "level": 0,
                        "retrieval": result.get("metadata", {}).get("retrieval", {})
                    }
                }

    # Combine message and knowledge results
    all_results = list(message_map.values()) + list(knowledge_map.values())

    # Sort results by score (descending)
    all_results.sort(key=lambda x: x.get("score", 0), reverse=True)

    # Deduplicate results based on ID
    # This is a final safeguard against duplicate IDs
    unique_results = {}
    for result in all_results:
        result_id = result.get("id")
        if result_id not in unique_results or result.get("score", 0) > unique_results[result_id].get("score", 0):
            # Check if this is a buffer result (from memory) vs database result
            is_buffer_result = result.get("metadata", {}).get("retrieval", {}).get("source") in ["write_buffer", "speculative_buffer", "query_buffer", "hybrid_buffer"]

            # Debug logging
            logger.info(f"Processing result ID: {result_id}, is_buffer_result: {is_buffer_result}")
            logger.info(f"Result metadata: {result.get('metadata', {})}")

            # Get the message from the database to verify its true session and agent
            # Skip database verification for buffer results as they come from memory
            if result.get("type") == "message" and not is_buffer_result:
                logger.info(f"Checking database for message ID: {result_id}")
                message = db.get_message(result_id)
                if message:
                    logger.info(f"Found message in database: {message.get('id')}")
                    # Get the actual round and session for this message
                    round_data = db.get_round(message.get(
                        "round_id")) if message.get("round_id") else None
                    if round_data and round_data.get("session_id"):
                        actual_session_id = round_data.get("session_id")
                        actual_session = db.get_session(actual_session_id)

                        if actual_session:
                            # Update the metadata with the correct session and agent
                            result["metadata"]["user_id"] = user_id
                            result["metadata"]["agent_id"] = actual_session["agent_id"]
                            result["metadata"]["session_id"] = actual_session_id
                            result["metadata"]["session_name"] = actual_session["name"]
                            result["metadata"]["scope"] = ("in_session" if actual_session_id == request.session_id
                                                           else "cross_session") if request.session_id else None

                    # Add the result with database verification
                    unique_results[result_id] = result
                    logger.info(f"Added database result to unique_results: {result_id}")
                else:
                    logger.info(f"Message not found in database, skipping: {result_id}")
                # If message not found in database, skip this result (don't add to unique_results)
            else:
                # For buffer results or knowledge items, use the result as-is
                logger.info(f"Adding buffer/knowledge result to unique_results: {result_id}")
                unique_results[result_id] = result

    # Convert back to a list
    all_results = list(unique_results.values())

    # Sort again by score
    all_results.sort(key=lambda x: x.get("score", 0), reverse=True)

    # Limit to top_k results
    all_results = all_results[:request.top_k]

    # Convert NumPy types to Python native types
    all_results = prepare_response_data(all_results)

    return ApiResponse.success(
        data={
            "results": all_results,
            "total": len(all_results)
        },
        message=f"Found {len(all_results)} results",
    )
