"""M3-focused Query API (Phase A).

POST /api/v1/users/{user_id}/query
Body: {"query": str, "top_k": int?, "metadata": {"tag": "m3"}?}

When metadata.tag == 'm3', search procedural_memory and procedural_lessons.
This keeps Phase A read path simple and avoids touching existing messages schema.
"""

from typing import Optional, Dict, Any
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
    lessons = await store.query_lessons_similar(emb, agent=None, top_k=max(1, request.top_k or 5))

    results = {
        "procedural_memory": [
            {"workflow_id": w, "score": s, "workflow": wf}
            for (w, wf, s) in workflows
        ],
        "lessons": [
            {"lesson_id": lid, "status": st, "score": sc, "working_params": wp, "fix_summary": fx}
            for (lid, st, fx, wp, sc) in lessons
        ],
    }

    return ApiResponse.success(
        data={"results": results},
        message="M3 results retrieved",
    )

