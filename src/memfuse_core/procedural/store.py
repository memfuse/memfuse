from __future__ import annotations

"""Procedural memory store for Phase A.

This module provides async helpers to ensure Phase A tables exist and to
perform basic CRUD operations. It intentionally avoids modifying global
DB initialization; tables are created lazily on first use.
"""

import json
import uuid
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from ..services.database_service import DatabaseService


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS message_workflows (
    id TEXT PRIMARY KEY,
    message_id TEXT NOT NULL,
    workflow_id TEXT,
    step_index INT,
    tags TEXT[] DEFAULT '{}',
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS procedural_memory (
    workflow_id TEXT PRIMARY KEY,
    trigger_embedding VECTOR(384),
    trigger_pattern TEXT,
    successful_workflow JSONB NOT NULL,
    usage_count INT DEFAULT 1,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS procedural_lessons (
    lesson_id TEXT PRIMARY KEY,
    trigger_embedding VECTOR(384),
    goal_text TEXT,
    agent TEXT,
    status TEXT CHECK (status IN ('success', 'fail')),
    error TEXT,
    fix_summary TEXT,
    working_params JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
"""


class ProceduralStore:
    def __init__(self) -> None:
        self._initialized = False

    async def _ensure_tables(self) -> None:
        if self._initialized:
            return
        db = await DatabaseService.get_instance()
        try:
            await db.backend.execute(_SCHEMA_SQL, tuple())  # type: ignore[attr-defined]
            self._initialized = True
        except Exception as e:
            logger.error(f"Failed to init procedural tables: {e}")
            # Do not re-raise to allow offline tests to continue

    # ---------------------- message_workflows ----------------------
    async def log_message_workflow(
        self,
        message_id: str,
        workflow_id: Optional[str],
        step_index: Optional[int],
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        await self._ensure_tables()
        mw_id = str(uuid.uuid4())
        db = await DatabaseService.get_instance()
        try:
            query = (
                "INSERT INTO message_workflows (id, message_id, workflow_id, step_index, tags, metadata) "
                "VALUES (%s, %s, %s, %s, %s, %s)"
            )
            await db.execute(query, (mw_id, message_id, workflow_id, step_index, tags or [], json.dumps(metadata or {})))
        except Exception as e:
            logger.warning(f"log_message_workflow failed: {e}")
        return mw_id

    # ---------------------- procedural_memory ----------------------
    async def upsert_procedural_workflow(
        self,
        workflow_id: str,
        trigger_embedding: List[float],
        successful_workflow: Dict[str, Any],
        trigger_pattern: Optional[str] = None,
    ) -> None:
        await self._ensure_tables()
        db = await DatabaseService.get_instance()
        # Using UPSERT pattern with ON CONFLICT
        query = (
            "INSERT INTO procedural_memory (workflow_id, trigger_embedding, trigger_pattern, successful_workflow, usage_count) "
            "VALUES (%s, %s, %s, %s::jsonb, 1) "
            "ON CONFLICT (workflow_id) DO UPDATE SET "
            "trigger_embedding = EXCLUDED.trigger_embedding, "
            "trigger_pattern = EXCLUDED.trigger_pattern, "
            "successful_workflow = EXCLUDED.successful_workflow, "
            "usage_count = procedural_memory.usage_count + 1"
        )
        try:
            # asyncpg will accept Python list for vector when pgvector installed
            await db.execute(query, (workflow_id, trigger_embedding, trigger_pattern, json.dumps(successful_workflow)))
        except Exception as e:
            logger.warning(f"upsert_procedural_workflow failed: {e}")

    async def query_procedural_similar(self, query_embedding: List[float], top_k: int) -> List[Tuple[str, Dict[str, Any], float]]:
        await self._ensure_tables()
        db = await DatabaseService.get_instance()
        query = (
            "WITH q AS (SELECT %s::vector AS v) "
            "SELECT workflow_id, successful_workflow, 1 - (trigger_embedding <=> q.v) AS cosine_similarity "
            "FROM procedural_memory, q ORDER BY trigger_embedding <=> q.v ASC LIMIT %s"
        )
        try:
            rows = await db.execute(query, (query_embedding, top_k))
            results: List[Tuple[str, Dict[str, Any], float]] = []
            for r in rows:
                wid = r.get("workflow_id")
                wf = r.get("successful_workflow") or {}
                score = float(r.get("cosine_similarity") or 0.0)
                results.append((wid, wf, score))
            return results
        except Exception as e:
            logger.warning(f"query_procedural_similar failed: {e}")
            return []

    async def bump_procedural_usage(self, workflow_id: str, by: int = 1) -> int:
        await self._ensure_tables()
        db = await DatabaseService.get_instance()
        try:
            q = "UPDATE procedural_memory SET usage_count = usage_count + %s WHERE workflow_id = %s"
            count = await db.execute(q, (by, workflow_id))
            return int(count or 0)
        except Exception as e:
            logger.warning(f"bump_procedural_usage failed: {e}")
            return 0

    # ---------------------- procedural_lessons ----------------------
    async def insert_lesson(
        self,
        trigger_embedding: List[float],
        goal_text: str,
        agent: str,
        status: str,
        error: Optional[str],
        fix_summary: Optional[str],
        working_params: Optional[Dict[str, Any]],
    ) -> str:
        await self._ensure_tables()
        db = await DatabaseService.get_instance()
        lid = str(uuid.uuid4())
        try:
            q = (
                "INSERT INTO procedural_lessons (lesson_id, trigger_embedding, goal_text, agent, status, error, fix_summary, working_params) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)"
            )
            await db.execute(q, (lid, trigger_embedding, goal_text, agent, status, error, fix_summary, json.dumps(working_params or {})))
        except Exception as e:
            logger.warning(f"insert_lesson failed: {e}")
        return lid

    async def query_lessons_similar(self, trigger_embedding: List[float], agent: Optional[str], top_k: int = 5) -> List[Tuple[str, str, str, Dict[str, Any], float]]:
        await self._ensure_tables()
        db = await DatabaseService.get_instance()
        where = " WHERE 1=1"
        params: List[Any] = [trigger_embedding]
        if agent:
            where += " AND agent = %s"
        params.append(agent) if agent else None
        params.append(top_k)
        q = (
            "WITH q AS (SELECT %s::vector AS v) "
            f"SELECT lesson_id, status, fix_summary, working_params, 1 - (trigger_embedding <=> q.v) AS cosine_similarity "
            f"FROM procedural_lessons, q {where} ORDER BY trigger_embedding <=> q.v ASC LIMIT %s"
        )
        try:
            rows = await db.execute(q, tuple(params))
            out: List[Tuple[str, str, str, Dict[str, Any], float]] = []
            for r in rows:
                out.append(
                    (
                        r.get("lesson_id"),
                        r.get("status"),
                        r.get("fix_summary") or "",
                        r.get("working_params") or {},
                        float(r.get("cosine_similarity") or 0.0),
                    )
                )
            return out
        except Exception as e:
            logger.warning(f"query_lessons_similar failed: {e}")
            return []

