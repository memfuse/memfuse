"""User API endpoints."""

from loguru import logger
from fastapi import APIRouter, Depends, status, Query
from typing import Optional

from ..models import (
    UserCreate,
    UserUpdate,
    MemoryQuery,
    ApiResponse,
    ErrorDetail,
)
from ..services.database_service import DatabaseService
from ..utils.global_config_manager import get_global_config_manager
from ..utils.auth import validate_api_key
from ..utils import (
    ensure_user_exists,
    ensure_user_by_name_exists,
    ensure_user_name_available,
    handle_api_errors,
    raise_api_error,
)


router = APIRouter()

@router.get("/", response_model=ApiResponse)
# Also handle path without trailing slash
@router.get("", response_model=ApiResponse)
@handle_api_errors("list users")
async def list_users(
    name: Optional[str] = None,
    _: dict = Depends(validate_api_key),  # API key validation
) -> ApiResponse:
    """List all users or get a user by name."""
    db = await DatabaseService.get_instance()

    # If name is provided, get user by name
    if name:
        user = await ensure_user_by_name_exists(db, name)
        return ApiResponse.success(
            data={"users": [user]},
            message="User retrieved successfully",
        )

    # Otherwise, list all users
    users = await db.get_all_users()
    return ApiResponse.success(
        data={"users": users},
        message="Users retrieved successfully",
    )


@router.post("/", response_model=ApiResponse, status_code=status.HTTP_201_CREATED)
# Also handle path without trailing slash
@router.post("", response_model=ApiResponse, status_code=status.HTTP_201_CREATED)
@handle_api_errors("create user")
async def create_user(
    request: UserCreate,
    _: dict = Depends(validate_api_key),  # API key validation
) -> ApiResponse:
    """Create a new user."""
    db = await DatabaseService.get_instance()

    # Check if user with the same name already exists
    await ensure_user_name_available(db, request.name)

    # Create the user
    user_id = await db.create_user(
        name=request.name,
        description=request.description,
    )

    # Get the created user
    user = await db.get_user(user_id)

    return ApiResponse.success(
        data={"user": user},
        message="User created successfully",
        code=201,
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
    db = await DatabaseService.get_instance()

    # Validate user exists
    user = await ensure_user_exists(db, user_id)

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
    db = await DatabaseService.get_instance()

    # Check if user exists
    _ = await ensure_user_exists(db, user_id)

    # Update the user
    success = await db.update_user(
        user_id=user_id,
        name=request.name,
        description=request.description,
    )

    if not success:
        error_response = ApiResponse.error(
            message="Failed to update user",
            errors=[ErrorDetail(
                field="general", message="Database update failed")],
        )
        raise_api_error(error_response)

    # Get the updated user
    updated_user = await db.get_user(user_id)

    return ApiResponse.success(
        data={"user": updated_user},
        message="User updated successfully",
    )


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
@handle_api_errors("delete user")
async def delete_user(
    user_id: str,
    _: dict = Depends(validate_api_key),  # API key validation
) -> None:
    """Delete a user and all associated resources (cascade deletion).

    When a user is deleted, all associated resources are also deleted:
    - All sessions where the user participates
    - All messages in the user's sessions
    """
    db = await DatabaseService.get_instance()

    # Check if user exists
    _ = await ensure_user_exists(db, user_id)

    logger.info(f"Deleting user {user_id} with cascade deletion")

    # Implement cascade deletion manually
    # Step 1: Get all sessions for this user
    user_sessions = await db.get_sessions(user_id=user_id)
    logger.info(f"Found {len(user_sessions)} sessions for user {user_id}")

    # Step 2: Delete all messages in each session
    total_messages_deleted = 0
    for session in user_sessions:
        session_id = session['id']
        # Get messages in this session
        messages = await db.get_messages_by_session(session_id)
        logger.info(f"Found {len(messages)} messages in session {session_id}")

        # Delete each message
        for message in messages:
            message_success = await db.delete_message(message['id'])
            if message_success:
                total_messages_deleted += 1
            else:
                logger.warning(f"Failed to delete message {message['id']}")

    # Step 3: Delete all sessions for this user
    sessions_deleted = 0
    for session in user_sessions:
        session_success = await db.delete_session(session['id'])
        if session_success:
            sessions_deleted += 1
        else:
            logger.warning(f"Failed to delete session {session['id']}")

    # Step 4: Delete the user
    user_success = await db.delete_user(user_id)

    if not user_success:
        error_response = ApiResponse.error(
            message="Failed to delete user",
            errors=[ErrorDetail(
                field="general", message="Database delete failed")],
        )
        raise_api_error(error_response)

    logger.info(f"User {user_id} deleted successfully: {sessions_deleted} sessions and {total_messages_deleted} messages removed")
    # Return 204 No Content (no response body)


@router.post("/{user_id}/query", response_model=ApiResponse)
# Also handle path with trailing slash
@router.post("/{user_id}/query/", response_model=ApiResponse)
@handle_api_errors("query memory")
async def query_memory(
    user_id: str,
    request: MemoryQuery,
    tag: str | None = Query(default=None, description="Compatibility: use 'm3' to route query to M3"),
    _: dict = Depends(validate_api_key),  # API key validation
) -> ApiResponse:
    """Query memory across all sessions for a user.

    This endpoint supports querying memory across all sessions for a user.
    If session_id is provided, results will be tagged with scope="in_session" or
    scope="cross_session" depending on whether they belong to the specified session.
    If session_id is not provided, all results will have scope=null.
    """
    from ..services.service_factory import ServiceFactory

    db = await DatabaseService.get_instance()
    logger.info("Using MemoryGateway for query operations")

    # Check if user exists (support both user_id and user_name)
    # Try to get user by ID first, then by name
    user = await db.get_user(user_id)
    if not user:
        user = await db.get_user_by_name(user_id)

    if not user:
        error_response = ApiResponse.error(
            message=f"User with ID or name '{user_id}' not found",
            code=404,
            errors=[
                ErrorDetail(
                    field="user_id",
                    message=f"User with ID or name '{user_id}' not found"
                )
            ],
        )
        raise_api_error(error_response)

    # Use the actual user ID for subsequent operations
    actual_user_id = user["id"]

    # Validate session if provided
    if request.session_id:
        session = await db.get_session(request.session_id)
        if not session or session["user_id"] != actual_user_id:
            error_response = ApiResponse.error(
                message=f"Session '{request.session_id}' not found for user '{user_id}'",
                code=404,
                errors=[
                    ErrorDetail(
                        field="session_id",
                        message=f"Session '{request.session_id}' not found for user '{user_id}'"
                    )
                ],
            )
            raise_api_error(error_response)

    # Validate agent if provided
    if request.agent_id:
        agent = await db.get_agent(request.agent_id)
        if not agent:
            error_response = ApiResponse.error(
                message=f"Agent '{request.agent_id}' not found",
                code=404,
                errors=[
                    ErrorDetail(
                        field="agent_id",
                        message=f"Agent '{request.agent_id}' not found"
                    )
                ],
            )
            raise_api_error(error_response)

    # M3-specific query path: prefer metadata.task presence; optionally allow legacy tag when enabled
    cfg = get_global_config_manager()
    legacy_tag_trigger = bool(cfg.get("memory.layers.m3.legacy_tag_trigger", False))
    md = request.metadata or {}
    task_name = None
    try:
        task_name = str(md.get("task") or "") or None
    except Exception:
        task_name = None
    meta_tag = None
    try:
        meta_tag = str(md.get("tag", "")).lower()
    except Exception:
        meta_tag = None

    route_m3 = bool(task_name)
    if not route_m3 and legacy_tag_trigger and ((tag or "").lower() == "m3" or meta_tag == "m3"):
        route_m3 = True

    if route_m3:
        from ..procedural.store import ProceduralStore
        from ..utils.embeddings import create_embedding

        store = ProceduralStore()
        emb = await create_embedding(request.query)
        top_k = max(1, request.top_k or 5)
        workflows = await store.query_procedural_similar(emb, top_k=top_k)
        # lessons: filter_agent may be specified; default None for global
        meta = request.metadata or {}
        filter_agent = meta.get("filter_agent") or None
        lessons = await store.query_lessons_similar(emb, agent=(filter_agent or None), top_k=top_k)

        include_workflows = True if meta.get("include_workflows", True) else False
        session_workflows = []
        if include_workflows and request.session_id:
            try:
                sw_limit = int(meta.get("session_workflows_limit", max(1, request.top_k or 50)))
            except Exception:
                sw_limit = max(1, request.top_k or 50)
            try:
                session_workflows = await store.query_message_workflows_for_session(request.session_id, limit=sw_limit)
            except Exception:
                session_workflows = []

        # Apply client-like filters server-side for convenience
        filter_workflow_id = meta.get("filter_workflow_id")
        try:
            min_score = float(meta.get("min_score")) if meta.get("min_score") is not None else None
        except Exception:
            min_score = None
        filter_status = meta.get("filter_status")
        filter_tags = meta.get("filter_tags") if isinstance(meta.get("filter_tags"), list) else None

        wf_list = [
            {"workflow_id": w, "score": s, "workflow": wf}
            for (w, wf, s) in workflows
            if (filter_workflow_id is None or w == filter_workflow_id)
            and (min_score is None or (s is not None and s >= min_score))
        ]

        lesson_list = [
            {"lesson_id": lid, "status": st, "score": sc, "working_params": wp, "fix_summary": fx}
            for (lid, st, fx, wp, sc) in lessons
            if (filter_status is None or st == filter_status)
            and (min_score is None or (sc is not None and sc >= min_score))
        ]

        sw_list = session_workflows
        # Filter by task/workflow name if provided
        if task_name:
            def _has_task(sw):
                try:
                    md = sw.get("metadata") or {}
                    return str(md.get("task", "")) == task_name
                except Exception:
                    return False
            sw_list = [sw for sw in sw_list if _has_task(sw)]
        if filter_workflow_id is not None:
            sw_list = [sw for sw in sw_list if (sw.get("workflow_id") == filter_workflow_id)]
        if filter_tags:
            want = set([str(t) for t in filter_tags])
            def _has_any(tags):
                try:
                    return bool(set(tags or []) & want)
                except Exception:
                    return False
            sw_list = [sw for sw in sw_list if _has_any(sw.get("tags"))]

        results = {
            "procedural_memory": wf_list,
            "lessons": lesson_list,
            "session_workflows": sw_list,
        }

        return ApiResponse.success(
            data={"results": results},
            message="M3 results retrieved",
        )

    # Create API Gateway instance
    from ..gateway.api_gateway import create_memory_gateway
    from ..interfaces.gateway_interface import OperationType

    buffer_service = await ServiceFactory.get_buffer_service_for_user(user["name"])
    gateway = create_memory_gateway(
        buffer_service=buffer_service,
        db_service=db
    )

    # Prepare request data for gateway
    request_data = {
        "query": request.query,
        "user_id": actual_user_id,
        "user_name": user.get("name") if user else None,
        "agent_id": request.agent_id,
        "session_id": request.session_id,
        "top_k": request.top_k,
        "store_type": request.store_type,
        "include_messages": request.include_messages,
        "include_knowledge": request.include_knowledge,
        "metadata": request.metadata or {}
    }

    # Process query through gateway
    response = await gateway.process_request(
        request_data=request_data,
        operation_type=OperationType.QUERY
    )

    # Return the gateway response directly (it's already in ApiResponse format)
    if isinstance(response, dict) and 'status' in response:
        return ApiResponse(**response)
    else:
        # Fallback if response format is unexpected
        return ApiResponse.success(
            data=response,
            message="Query completed"
        )
