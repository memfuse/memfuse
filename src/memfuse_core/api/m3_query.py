"""M3-focused Query API (Phase A).

POST /api/v1/users/{user_id}/query
Body: {"query": str, "top_k": int?, "metadata": {"tag": "m3"}?}

When metadata.tag == 'm3', search procedural_memory and procedural_lessons.
This keeps Phase A read path simple and avoids touching existing messages schema.
"""

from typing import Optional, Dict, Any, List
from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, Field

from ..services.database_service import DatabaseService
from ..utils.auth import validate_api_key
from ..utils import ensure_user_exists, handle_api_errors
from ..models import ApiResponse
from ..procedural.store import ProceduralStore
from ..utils.embeddings import create_embedding


router = APIRouter()


class M3UserQuery(BaseModel):
    query: str = Field(..., description="Free-text user query")
    top_k: Optional[int] = Field(default=5)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    session_id: Optional[str] = Field(default=None, description="Optional session id to include session workflow logs")
    include_workflows: Optional[bool] = Field(default=True, description="Include message_workflows when session_id provided")
    # Optional filters
    filter_workflow_id: Optional[str] = Field(default=None, description="Filter results by workflow_id")
    filter_tags: Optional[List[str]] = Field(default=None, description="Filter session_workflows by tags (any match)")
    filter_agent: Optional[str] = Field(default=None, description="Filter lessons by agent (server-side when possible)")
    filter_status: Optional[str] = Field(default=None, description="Filter lessons by status: success|fail")
    min_score: Optional[float] = Field(default=None, description="Minimum cosine similarity score (0-1) for workflows/lessons")


@router.post("/{user_id}/query", response_model=ApiResponse)
@handle_api_errors("m3 user query")
async def m3_user_query(
    user_id: str,
    request: M3UserQuery,
    _: dict = Depends(validate_api_key),
) -> ApiResponse:
    db = await DatabaseService.get_instance()
    # Validate user exists
    await ensure_user_exists(db, user_id)

    # Only route to M3 when metadata.tag == 'm3'
    tag = str((request.metadata or {}).get("tag", "")).lower()
    if tag != "m3":
        return ApiResponse.success(
            data={"results": [], "note": "metadata.tag != 'm3'"},
            message="Query executed",
        )

    # Compute embedding for semantic similarity
    emb = await create_embedding(request.query)
    store = ProceduralStore()

    # Phase A: search procedural memory and lessons globally (no user scoping yet)
    workflows = await store.query_procedural_similar(emb, top_k=max(1, request.top_k or 5))
    # Lessons: use agent hint when provided
    lessons = await store.query_lessons_similar(emb, agent=(request.filter_agent or None), top_k=max(1, request.top_k or 5))
    session_workflows = []
    if request.include_workflows and request.session_id:
        try:
            session_workflows = await store.query_message_workflows_for_session(request.session_id, limit=max(1, request.top_k or 50))
        except Exception:
            session_workflows = []

    # Apply client-side filters
    min_score = request.min_score if request.min_score is not None else None
    wf_list = [
        {"workflow_id": w, "score": s, "workflow": wf}
        for (w, wf, s) in workflows
        if (request.filter_workflow_id is None or w == request.filter_workflow_id)
        and (min_score is None or (s is not None and s >= min_score))
    ]
    lesson_list = [
        {"lesson_id": lid, "status": st, "score": sc, "working_params": wp, "fix_summary": fx}
        for (lid, st, fx, wp, sc) in lessons
        if (request.filter_status is None or st == request.filter_status)
        and (min_score is None or (sc is not None and sc >= min_score))
    ]
    sw_list = session_workflows
    if request.filter_workflow_id is not None:
        sw_list = [sw for sw in sw_list if (sw.get("workflow_id") == request.filter_workflow_id)]
    if request.filter_tags:
        want = set([str(t) for t in request.filter_tags])
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
