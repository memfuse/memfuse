"""Procedural memory store for Phase A.

This module provides async helpers to ensure Phase A tables exist and to
perform basic CRUD operations. It intentionally avoids modifying global
DB initialization; tables are created lazily on first use.
"""

from __future__ import annotations

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
    """Procedural memory store for M3 workflows and lessons."""
    
    def __init__(self, db: Optional[DatabaseService] = None) -> None:
        self._initialized = False
        self._db = db

    async def _ensure_tables(self) -> None:
        """Ensure M3 tables exist in the database."""
        if self._initialized:
            return
        
        db = self._db or await DatabaseService.get_instance()
        try:
            await db.backend.execute(_SCHEMA_SQL, tuple())  # type: ignore[attr-defined]
            
            # Create indexes best-effort (some ops like diskann may not be available)
            index_statements = [
                # message_workflows
                "CREATE INDEX IF NOT EXISTS idx_message_workflows_message_id ON message_workflows (message_id)",
                "CREATE INDEX IF NOT EXISTS idx_message_workflows_workflow_id ON message_workflows (workflow_id)",
                "CREATE INDEX IF NOT EXISTS idx_message_workflows_tags ON message_workflows USING GIN (tags)",
                "CREATE INDEX IF NOT EXISTS idx_message_workflows_metadata_gin ON message_workflows USING GIN (metadata)",
                # procedural_memory (vector)
                "CREATE INDEX IF NOT EXISTS idx_procedural_memory_trigger_embedding ON procedural_memory USING diskann (trigger_embedding vector_cosine_ops)",
                # procedural_lessons (vector + agent)
                "CREATE INDEX IF NOT EXISTS idx_procedural_lessons_trigger_embedding ON procedural_lessons USING diskann (trigger_embedding vector_cosine_ops)",
                "CREATE INDEX IF NOT EXISTS idx_procedural_lessons_agent ON procedural_lessons (agent)",
            ]
            
            for stmt in index_statements:
                try:
                    await db.backend.execute(stmt, tuple())  # type: ignore[attr-defined]
                except Exception as ie:
                    logger.debug(f"Skipping index creation due to: {ie}")
            
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
        """Log a message as part of an M3 workflow."""
        await self._ensure_tables()
        mw_id = str(uuid.uuid4())
        db = self._db or await DatabaseService.get_instance()
        
        try:
            query = (
                "INSERT INTO message_workflows (id, message_id, workflow_id, step_index, tags, metadata) "
                "VALUES (%s, %s, %s, %s, %s, %s)"
            )
            await db.backend.execute(query, (
                mw_id, message_id, workflow_id, step_index, 
                tags or [], json.dumps(metadata or {})
            ))
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
        """Store or update a successful workflow for reuse."""
        await self._ensure_tables()
        db = self._db or await DatabaseService.get_instance()
        
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
            await db.backend.execute(query, (
                workflow_id, trigger_embedding, trigger_pattern, json.dumps(successful_workflow)
            ))
        except Exception as e:
            logger.warning(f"upsert_procedural_workflow failed: {e}")

    async def query_procedural_similar(
        self, 
        query_embedding: List[float], 
        top_k: int
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        """Query similar workflows for reuse."""
        await self._ensure_tables()
        db = self._db or await DatabaseService.get_instance()
        
        # Use a simpler, more reliable query for pgvector
        query = (
            "SELECT workflow_id, successful_workflow, "
            "1 - (trigger_embedding <=> %s::vector) AS cosine_similarity "
            "FROM procedural_memory "
            "ORDER BY trigger_embedding <=> %s::vector ASC LIMIT %s"
        )
        
        try:
            rows = await db.backend.execute(query, (query_embedding, query_embedding, top_k))
            results: List[Tuple[str, Dict[str, Any], float]] = []
            
            for r in rows:
                wid = r.get("workflow_id")
                wf = r.get("successful_workflow") or {}
                score = float(r.get("cosine_similarity") or 0.0)
                results.append((wid, wf, score))
            
            logger.info(f"query_procedural_similar: found {len(results)} workflows")
            return results
            
        except Exception as e:
            logger.error(f"query_procedural_similar failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []

    async def bump_procedural_usage(self, workflow_id: str, by: int = 1) -> int:
        """Increment usage count for a workflow."""
        await self._ensure_tables()
        db = self._db or await DatabaseService.get_instance()
        
        try:
            q = "UPDATE procedural_memory SET usage_count = usage_count + %s WHERE workflow_id = %s"
            count = await db.backend.execute(q, (by, workflow_id))
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
        """Store a lesson learned from workflow execution."""
        await self._ensure_tables()
        db = self._db or await DatabaseService.get_instance()
        lid = str(uuid.uuid4())
        
        try:
            q = (
                "INSERT INTO procedural_lessons "
                "(lesson_id, trigger_embedding, goal_text, agent, status, error, fix_summary, working_params) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)"
            )
            await db.backend.execute(q, (
                lid, trigger_embedding, goal_text, agent, status, error, fix_summary, 
                json.dumps(working_params or {})
            ))
        except Exception as e:
            logger.warning(f"insert_lesson failed: {e}")
        
        return lid

    async def query_lessons_similar(
        self, 
        trigger_embedding: List[float], 
        agent: Optional[str], 
        top_k: int = 5
    ) -> List[Tuple[str, str, str, Dict[str, Any], float]]:
        """Query similar lessons for guidance."""
        await self._ensure_tables()
        db = self._db or await DatabaseService.get_instance()
        
        where = " WHERE 1=1"
        params: List[Any] = [trigger_embedding]
        
        if agent:
            where += " AND agent = %s"
            params.append(agent)
        
        params.append(top_k)
        
        q = (
            "WITH q AS (SELECT %s::vector AS v) "
            f"SELECT lesson_id, status, fix_summary, working_params, 1 - (trigger_embedding <=> q.v) AS cosine_similarity "
            f"FROM procedural_lessons, q {where} ORDER BY trigger_embedding <=> q.v ASC LIMIT %s"
        )
        
        try:
            rows = await db.backend.execute(q, tuple(params))
            out: List[Tuple[str, str, str, Dict[str, Any], float]] = []
            
            for r in rows:
                out.append((
                    r.get("lesson_id"),
                    r.get("status"),
                    r.get("fix_summary") or "",
                    r.get("working_params") or {},
                    float(r.get("cosine_similarity") or 0.0),
                ))
            
            return out
            
        except Exception as e:
            logger.warning(f"query_lessons_similar failed: {e}")
            return []

    async def query_message_workflows_for_session(
        self, 
        session_id: str, 
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Return message_workflows rows for messages belonging to a session.

        This performs a subquery join:
          message_workflows.message_id IN (SELECT m.id FROM messages m JOIN rounds r ON m.round_id=r.id WHERE r.session_id=%s)
        """
        await self._ensure_tables()
        db = self._db or await DatabaseService.get_instance()
        
        q = (
            "SELECT id, message_id, workflow_id, step_index, tags, metadata, created_at, updated_at "
            "FROM message_workflows WHERE message_id IN ("
            "  SELECT m.id FROM messages m JOIN rounds r ON m.round_id = r.id WHERE r.session_id = %s"
            ") ORDER BY created_at ASC LIMIT %s"
        )
        
        try:
            rows = await db.backend.execute(q, (session_id, limit))
            # rows is a list of dicts via PostgresDB
            return rows
        except Exception as e:
            logger.warning(f"query_message_workflows_for_session failed: {e}")
            return []

    async def query_procedural_similar(
        self, 
        task_name: str,
        query_text: str, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Query similar procedural memories by task name and query text."""
        await self._ensure_tables()
        db = self._db or await DatabaseService.get_instance()
        
        try:
            # For now, use simple text matching on trigger_pattern
            # In production, this would use embeddings
            query = (
                "SELECT workflow_id, trigger_pattern, successful_workflow, usage_count, created_at "
                "FROM procedural_memory "
                "WHERE trigger_pattern ILIKE %s OR trigger_pattern ILIKE %s "
                "ORDER BY usage_count DESC, created_at DESC LIMIT %s"
            )
            
            task_pattern = f"%{task_name}%"
            query_pattern = f"%{query_text}%"
            
            rows = await db.backend.execute(query, (task_pattern, query_pattern, limit))
            return rows
            
        except Exception as e:
            logger.warning(f"query_procedural_similar failed: {e}")
            return []

    async def get_lessons_by_task(
        self, 
        task_name: str, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get lessons for a specific task."""
        await self._ensure_tables()
        db = self._db or await DatabaseService.get_instance()
        
        try:
            query = (
                "SELECT lesson_id, goal_text, agent, status, error, fix_summary, working_params, created_at "
                "FROM procedural_lessons "
                "WHERE goal_text ILIKE %s "
                "ORDER BY created_at DESC LIMIT %s"
            )
            
            task_pattern = f"%{task_name}%"
            rows = await db.backend.execute(query, (task_pattern, limit))
            return rows
            
        except Exception as e:
            logger.warning(f"get_lessons_by_task failed: {e}")
            return []

    async def query_similar_workflows(
        self, 
        task_name: str,
        description: str, 
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Query similar workflows for reuse."""
        await self._ensure_tables()
        db = self._db or await DatabaseService.get_instance()
        
        try:
            query = (
                "SELECT workflow_id, trigger_pattern, successful_workflow, usage_count, created_at "
                "FROM procedural_memory "
                "WHERE trigger_pattern ILIKE %s OR trigger_pattern ILIKE %s "
                "ORDER BY usage_count DESC, created_at DESC LIMIT %s"
            )
            
            task_pattern = f"%{task_name}%"
            desc_pattern = f"%{description}%"
            
            rows = await db.backend.execute(query, (task_pattern, desc_pattern, limit))
            return rows
            
        except Exception as e:
            logger.warning(f"query_similar_workflows failed: {e}")
            return []

    async def get_task_statistics(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get statistics for all tasks with available experiences."""
        await self._ensure_tables()
        db = self._db or await DatabaseService.get_instance()
        
        try:
            # Get task statistics from procedural_memory
            workflow_query = (
                "SELECT "
                "  COALESCE(trigger_pattern, 'unknown') as task_name, "
                "  COUNT(*) as workflow_count, "
                "  SUM(usage_count) as total_usage, "
                "  MAX(created_at) as last_used "
                "FROM procedural_memory "
                "GROUP BY trigger_pattern "
                "ORDER BY total_usage DESC, last_used DESC "
                "LIMIT %s"
            )
            
            workflow_rows = await db.backend.execute(workflow_query, (limit,))
            
            # Get lesson statistics
            lesson_query = (
                "SELECT "
                "  goal_text as task_name, "
                "  COUNT(*) as lesson_count, "
                "  COUNT(CASE WHEN status = 'success' THEN 1 END) as success_count, "
                "  COUNT(CASE WHEN status = 'fail' THEN 1 END) as fail_count "
                "FROM procedural_lessons "
                "GROUP BY goal_text "
                "ORDER BY lesson_count DESC "
                "LIMIT %s"
            )
            
            lesson_rows = await db.backend.execute(lesson_query, (limit,))
            
            # Combine results
            task_stats = {}
            
            for row in workflow_rows:
                task_name = row.get("task_name", "unknown")
                task_stats[task_name] = {
                    "task_name": task_name,
                    "workflow_count": row.get("workflow_count", 0),
                    "total_usage": row.get("total_usage", 0),
                    "last_used": row.get("last_used"),
                    "lesson_count": 0,
                    "success_count": 0,
                    "fail_count": 0
                }
            
            for row in lesson_rows:
                task_name = row.get("task_name", "unknown")
                if task_name not in task_stats:
                    task_stats[task_name] = {
                        "task_name": task_name,
                        "workflow_count": 0,
                        "total_usage": 0,
                        "last_used": None,
                        "lesson_count": 0,
                        "success_count": 0,
                        "fail_count": 0
                    }
                
                task_stats[task_name]["lesson_count"] = row.get("lesson_count", 0)
                task_stats[task_name]["success_count"] = row.get("success_count", 0)
                task_stats[task_name]["fail_count"] = row.get("fail_count", 0)
            
            return list(task_stats.values())
            
        except Exception as e:
            logger.warning(f"get_task_statistics failed: {e}")
            return []