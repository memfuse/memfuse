"""Message API endpoints."""

from typing import Dict, List, Optional, Any, cast
from loguru import logger
from fastapi import APIRouter, Depends, status, Query

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
# Avoid importing ServiceFactory at module load to prevent heavy side-effects


router = APIRouter()

# BufferService is now always used, with buffer enabled/disabled controlled by config/buffer/default.yaml



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

    db = await DatabaseService.get_instance()
    user_id = session.get("user_id")
    agent_id = session.get("agent_id")

    if not user_id or not agent_id:
        logger.error("Invalid session data: missing user_id or agent_id")
        return None

    user = await db.get_user(user_id)
    agent = await db.get_agent(agent_id)

    if not user or not agent:
        logger.error(f"User or agent not found for session {session_id}")
        return None

    user_name = user["name"]
    agent_name = agent["name"]
    session_name = session.get("name", "default")

    # Always use BufferService, which internally handles buffer enabled/disabled mode
    # based on config/buffer/default.yaml settings
    logger.info("Using BufferService for message operations")
    # Lazy import to avoid heavy import at module import time
    from ..services.service_factory import ServiceFactory
    service = await ServiceFactory.get_buffer_service(
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


def normalize_messages_response(messages: Any) -> List[Dict[str, Any]]:
    """Normalize messages response to ensure consistent format.

    This function ensures that the messages response is always a List[Dict[str, Any]]
    regardless of the underlying service implementation.

    Args:
        messages: Messages data from service (could be various formats)

    Returns:
        Normalized list of message dictionaries

    Raises:
        ValueError: If messages cannot be normalized to expected format
    """
    # Handle None or empty cases early
    if messages is None:
        return []

    # If already a list, validate and return
    if isinstance(messages, list):
        normalized = []
        for i, msg in enumerate(messages):
            if isinstance(msg, dict):
                normalized.append(msg)
            elif hasattr(msg, 'model_dump'):
                # Handle Pydantic models
                normalized.append(cast(Message, msg).model_dump())
            elif hasattr(msg, '__dict__'):
                # Handle objects with __dict__
                normalized.append(vars(msg))
            else:
                logger.warning(f"Unexpected message format at index {i}: {type(msg)}")
                # Skip invalid items rather than failing
                continue
        return normalized

    # If it's a dictionary, check for nested structures
    elif isinstance(messages, dict):
        if 'messages' in messages:
            # Recursive call to handle nested structure
            return normalize_messages_response(messages['messages'])
        elif 'data' in messages and isinstance(messages['data'], dict) and 'messages' in messages['data']:
            # Handle nested data.messages structure
            return normalize_messages_response(messages['data']['messages'])
        else:
            # Single message dictionary - wrap in list
            return [messages]

    # Handle other iterables (but not strings/bytes)
    elif hasattr(messages, '__iter__') and not isinstance(messages, (str, bytes)):
        try:
            return normalize_messages_response(list(messages))
        except Exception as e:
            logger.error(f"Failed to convert iterable to list: {e}")
            return []

    # Unexpected format
    else:
        logger.error(f"Unexpected messages format: {type(messages)}")
        raise ValueError(f"Cannot normalize messages of type {type(messages)} to List[Dict[str, Any]]")


@router.post("/", response_model=ApiResponse, status_code=status.HTTP_201_CREATED)
# Also handle path without trailing slash
@router.post("", response_model=ApiResponse, status_code=status.HTTP_201_CREATED)
@handle_api_errors("add messages")
async def add_messages(
    session_id: str,
    request: MessageAdd,
    tag: str | None = Query(default=None, description="Legacy compatibility only when enabled: tag=m3"),
    # Underscore prefix to indicate unused
    _api_key_data: dict = api_key_dependency,
) -> ApiResponse:
    """Add messages to a session."""
    try:
        db = await DatabaseService.get_instance()

        # Validate session exists
        session = await ensure_session_exists(db, session_id)

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

        # Convert messages first
        messages = convert_pydantic_to_dict(request.messages)

        # Precompute gateway decision for metadata-driven behaviors before any mutations
        from memfuse_core.gateway.message_metadata import MessageMetadataInterpreter as _MMI
        from memfuse_core.gateway.message_metadata import MessageAddDecision as _MAD
        try:
            # Keep API thin; avoid heavy imports here. Default legacy flag to False.
            decision = _MMI().interpret_add_messages(messages, tag=tag, legacy_tag_trigger=False)
        except Exception:
            decision = _MAD(False, None, None, None)
        # Fallback: minimal inline check if interpreter not available
        try:
            if not getattr(decision, 'trigger_m3', False):
                eos_msgs = [
                    m for m in messages
                    if isinstance(m, dict)
                    and str(m.get("role", "")) == "user"
                    and isinstance(m.get("metadata"), dict)
                    and bool(m.get("metadata", {}).get("task_eos", False))
                ]
                if eos_msgs:
                    last = eos_msgs[-1]
                    user_goal_fb = str(last.get("content") or "").strip() or None
                    wf_fb = str((last.get("metadata") or {}).get("task") or "").strip() or None
                    decision = _MAD(True, wf_fb, user_goal_fb, wf_fb)
        except Exception:
            pass

        # Determine trigger/workflow from EOS message before any mutations
        def _to_dict(obj):
            try:
                if hasattr(obj, 'model_dump'):
                    return obj.model_dump()
            except Exception:
                pass
            return obj if isinstance(obj, dict) else {}

        eos_msgs_pre = []
        for _m in (request.messages or []):
            dm = _to_dict(_m)
            if str(dm.get('role', '')) == 'user':
                md = dm.get('metadata') or {}
                try:
                    if bool(md.get('task_eos', False)):
                        eos_msgs_pre.append(dm)
                except Exception:
                    pass
        trigger_eos = bool(eos_msgs_pre)
        wf_pre = None
        goal_pre = None
        if trigger_eos:
            last_pre = eos_msgs_pre[-1]
            wf_pre = str((last_pre.get("metadata") or {}).get("task") or "").strip() or None
            goal_pre = str(last_pre.get("content") or "").strip() or None
        else:
            # Fallback: check converted dict messages
            try:
                for _m in (messages or []):
                    if isinstance(_m, dict) and str(_m.get('role','')) == 'user':
                        md = _m.get('metadata') or {}
                        if bool(md.get('task_eos', False)):
                            wf_pre = str(md.get('task') or '').strip() or wf_pre
                            goal_pre = str(_m.get('content') or '').strip() or goal_pre
                if wf_pre or goal_pre:
                    trigger_eos = True
            except Exception:
                pass

        # P1 OPTIMIZATION: Pass session_id to add method
        try:
            result = await memory.add(messages, session_id=session_id)
        except ModuleNotFoundError as e:
            # Allow missing optional vector backends in minimal environments (e.g., qdrant_client)
            if 'qdrant_client' in str(e):
                result = {"status": "success", "data": {"message_ids": []}}
            else:
                raise

        # Extract message IDs from result
        message_ids = []
        logger.info(f"Messages API: Service result: {result}")
        if (result and result.get("status") == "success"
                and result.get("data") is not None):
            message_ids = result["data"].get("message_ids", [])
            logger.info(f"Messages API: Extracted message_ids: {message_ids}")

        # Create response data with message IDs
        response_data = {"message_ids": message_ids}

        # Add any additional fields from the service result (e.g., transfer_triggered)
        if result and result.get("status") == "success":
            # Include additional fields like transfer_triggered, total_messages, etc.
            for key, value in result.items():
                if key not in ["status", "code", "data", "message", "errors"]:
                    response_data[key] = value

        # Delegate metadata interpretation to Gateway interpreter; API stays thin
        # Trigger only when a user message has metadata.task_eos == true
        # Also consider Gateway decision (metadata interpreter) as trigger source
        if not trigger_eos and getattr(decision, 'trigger_m3', False):
            trigger_eos = True
        # Final robust check: scan converted dict messages
        if not trigger_eos:
            try:
                for _m in (messages or []):
                    if isinstance(_m, dict) and str(_m.get('role','')) == 'user':
                        md = _m.get('metadata') or {}
                        if bool(md.get('task_eos', False)):
                            trigger_eos = True
                            break
            except Exception:
                pass
        eos_msgs = eos_msgs_pre if trigger_eos else []
        if trigger_eos:
            # Read config lazily; default to safe values if unavailable
            m3_enabled = True
            max_history_scan = 2000
            try:
                from memfuse_core.utils.global_config_manager import get_global_config_manager
                cfg = get_global_config_manager()
                m3_enabled = bool(cfg.get("memory.layers.m3.enabled", m3_enabled))
                max_history_scan = int(cfg.get("memory.layers.m3.max_history_scan", max_history_scan))
            except Exception:
                pass
            if not m3_enabled:
                eos_msgs = []

        if trigger_eos and m3_enabled:
            # Compute workflow and goal from last EOS message
            workflow_name = wf_pre or getattr(decision, 'workflow_name', None)
            user_goal = goal_pre or getattr(decision, 'user_goal', None)
            # Final fallback: derive workflow_name from any message metadata.task
            if not workflow_name:
                try:
                    for _m in (request.messages or []):
                        dm = _to_dict(_m)
                        md = dm.get('metadata') or {}
                        t = str(md.get('task') or '').strip()
                        if t:
                            workflow_name = t
                except Exception:
                    pass
            # As a last resort, re-run conversion and scan for EOS/task
            if (not workflow_name) or (not user_goal):
                try:
                    dm2 = convert_pydantic_to_dict(request.messages)
                    for _m in (dm2 or []):
                        if isinstance(_m, dict) and str(_m.get('role','')) == 'user':
                            md = _m.get('metadata') or {}
                            if bool(md.get('task_eos', False)):
                                workflow_name = workflow_name or str(md.get('task') or '').strip() or None
                                user_goal = user_goal or str(_m.get('content') or '').strip() or None
                except Exception:
                    pass

            # Build workflow-scoped history
            history: list[dict] = []
            # Prefer memory service for history to avoid heavy DB deps in thin API contexts
            history_all = []
            if hasattr(memory, 'get_messages_by_session'):
                try:
                    history_all = await memory.get_messages_by_session(
                        session_id=session_id,
                        limit=max_history_scan,
                        sort_by="timestamp",
                        order="asc",
                        buffer_only=None,
                    )
                except Exception:
                    history_all = []
            try:
                history_all = normalize_messages_response(history_all)
            except Exception:
                history_all = []
            if workflow_name:
                def _has_task(md):
                    try:
                        return isinstance(md, dict) and str(md.get('task', '')) == workflow_name
                    except Exception:
                        return False
                history = [h for h in history_all if _has_task(h.get('metadata'))]
            else:
                history = history_all

            from memfuse_core.m3.orchestrator import Orchestrator
            from memfuse_core.procedural.store import ProceduralStore
            import uuid as _uuid
            orch = Orchestrator()
            ai_text = await orch.handle_request(
                session_id,
                user_goal,
                workflow_name=workflow_name or None,
                history_messages=history or None,
            )

            assistant_msg = [{
                "role": "assistant",
                "content": ai_text,
                "metadata": {"m3_enabled": True, "source": "orchestrator", "task": workflow_name, "task_eos": True},
            }]
            ai_result = await memory.add(assistant_msg, session_id=session_id)
            ai_ids = []
            if (ai_result and ai_result.get("status") == "success" and ai_result.get("data")):
                ai_ids = ai_result["data"].get("message_ids", [])
            if ai_ids:
                response_data["assistant_message_id"] = ai_ids[0]

            try:
                store = ProceduralStore()
                workflow_id = getattr(orch, "last_workflow_id", None) or str(_uuid.uuid4())
                if ai_ids:
                    steps = getattr(orch, "last_plan_steps", None) or []
                    outcomes = getattr(orch, "last_step_outcomes", None) or []
                    if isinstance(steps, list) and steps:
                        for idx, st in enumerate(steps):
                            try:
                                agent_name = getattr(st, "agent", None) or (st.get("agent") if isinstance(st, dict) else None)
                            except Exception:
                                agent_name = None
                            meta = {
                                "m3_enabled": True,
                                "user_goal": user_goal,
                                "reused": getattr(orch, "last_reused", False),
                                "task": workflow_name,
                                "task_eos": True,
                            }
                            try:
                                if idx < len(outcomes):
                                    oc = outcomes[idx]
                                    if isinstance(oc, dict):
                                        if oc.get("success") is not None:
                                            meta["success"] = bool(oc.get("success"))
                                        if oc.get("attempts") is not None:
                                            meta["attempts"] = int(oc.get("attempts"))
                                        if oc.get("duration_ms") is not None:
                                            meta["duration_ms"] = int(oc.get("duration_ms"))
                                        if oc.get("error"):
                                            meta["error"] = str(oc.get("error"))[:200]
                            except Exception:
                                pass
                            if agent_name:
                                meta["agent"] = agent_name
                            await store.log_message_workflow(
                                message_id=ai_ids[0],
                                workflow_id=workflow_id,
                                step_index=idx,
                                tags=["m3", "workflow"],
                                metadata=meta,
                            )
                    else:
                        await store.log_message_workflow(
                            message_id=ai_ids[0],
                            workflow_id=workflow_id,
                            step_index=0,
                            tags=["m3", "workflow"],
                            metadata={
                                "m3_enabled": True,
                                "user_goal": user_goal,
                                "reused": getattr(orch, "last_reused", False),
                                "task": workflow_name,
                                "task_eos": True,
                            },
                        )
                response_data["workflow_id"] = workflow_id
            except Exception as e:
                logger.warning(f"Failed to log message_workflow: {e}")
        # else: not triggered or disabled; do nothing

        return ApiResponse.success(
            data=response_data,
            message="Messages added successfully",
            code=201,
        )
    except ModuleNotFoundError as e:
        # Allow environments without optional vector backends to still use API-level flows
        if 'qdrant_client' in str(e):
            return ApiResponse.success(
                data={"message_ids": []},
                message="Messages accepted (optional vector backend unavailable)",
                code=201,
            )
        raise


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
        buffer_only: Buffer parameter - if "true", only return RoundBuffer data; if "false" or omitted, return data from all sources (RoundBuffer + HybridBuffer + Database)
    """
    db = await DatabaseService.get_instance()
    
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
    session = await ensure_session_exists(db, session_id)

    # Get the appropriate service (Buffer or Memory) for this session
    service = await get_service_for_session(session, session_id)

    # Get messages through the service (which handles both buffer and direct database access)
    raw_messages = None

    if service and hasattr(service, 'get_messages_by_session'):
        # Always try to call with buffer_only parameter first
        try:
            raw_messages = await service.get_messages_by_session(
                session_id=session_id,
                limit=limit_value,
                sort_by=sort_by,
                order=order,
                buffer_only=buffer_only_value
            )
        except TypeError:
            # Service doesn't support buffer_only parameter, call without it
            try:
                raw_messages = await service.get_messages_by_session(
                    session_id=session_id,
                    limit=limit_value,
                    sort_by=sort_by,
                    order=order
                )
            except Exception as e:
                logger.error(f"Service error, falling back to database: {e}")
                raw_messages = await db.get_messages_by_session(
                    session_id=session_id,
                    limit=limit_value,
                    sort_by=sort_by,
                    order=order
                )
        except Exception as e:
            logger.error(f"Service error, falling back to database: {e}")
            raw_messages = await db.get_messages_by_session(
                session_id=session_id,
                limit=limit_value,
                sort_by=sort_by,
                order=order
            )
    else:
        # Fallback to direct database access if service doesn't support the method
        raw_messages = await db.get_messages_by_session(
            session_id=session_id,
            limit=limit_value,
            sort_by=sort_by,
            order=order
        )

    # Normalize the response format to ensure consistency
    try:
        normalized_messages = normalize_messages_response(raw_messages)
    except ValueError as e:
        logger.error(f"Failed to normalize messages response: {e}")
        # Return empty list rather than failing
        normalized_messages = []

    return ApiResponse.success(
        data={"messages": normalized_messages},
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
    db = await DatabaseService.get_instance()

    # Validate session exists
    session = await ensure_session_exists(db, session_id)

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

    # Check if any messages were not found
    if result.get("status") == "error" and result.get("code") == 404:
        error_response = ApiResponse.error(
            message=result.get("message", "Some message IDs were not found"),
            code=404,
            errors=result.get("errors", []),
        )
        raise_api_error(error_response)

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
    db = await DatabaseService.get_instance()

    # Validate session exists
    session = await ensure_session_exists(db, session_id)

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
    logger.info(f"Messages API: Updating messages {request.message_ids} with new content")
    result = await memory.update(request.message_ids, new_messages)
    logger.info(f"Messages API: Update result: {result}")

    # Check if any messages were not found
    if result.get("status") == "error" and result.get("code") == 404:
        error_response = ApiResponse.error(
            message=result.get("message", "Some message IDs were not found"),
            code=404,
            errors=result.get("errors", []),
        )
        raise_api_error(error_response)

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
    db = await DatabaseService.get_instance()

    # Validate session exists
    session = await ensure_session_exists(db, session_id)

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
