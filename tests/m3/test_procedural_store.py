"""Tests for ProceduralStore."""

import pytest
import uuid
from typing import Dict, Any, List
import json

from src.memfuse_core.procedural.store import ProceduralStore
from src.memfuse_core.services.database_service import DatabaseService


@pytest.fixture
async def store():
    """Create a ProceduralStore instance for testing."""
    db = await DatabaseService.get_instance()
    store = ProceduralStore(db)
    await store._ensure_tables()
    return store


@pytest.fixture
def sample_workflow():
    """Sample workflow data for testing."""
    return {
        "goal": "test workflow",
        "plan": [
            {"agent": "RAGQueryAgent", "input": {"query": "test"}},
            {"agent": "ReportGenerationAgent", "input": {}}
        ],
        "result_keys": ["report"]
    }


@pytest.fixture
def sample_embedding():
    """Sample embedding vector for testing."""
    return [0.1] * 384  # 384-dimensional vector


class TestProceduralStore:
    """Test cases for ProceduralStore."""

    async def test_ensure_tables(self, store):
        """Test that tables are created successfully."""
        # Should not raise any exceptions
        await store._ensure_tables()
        assert store._initialized

    async def test_log_message_workflow(self, store):
        """Test logging a message workflow."""
        message_id = str(uuid.uuid4())
        workflow_id = str(uuid.uuid4())
        
        mw_id = await store.log_message_workflow(
            message_id=message_id,
            workflow_id=workflow_id,
            step_index=0,
            tags=["test"],
            metadata={"test": "data"}
        )
        
        assert isinstance(mw_id, str)
        assert len(mw_id) > 0

    async def test_upsert_procedural_workflow(self, store, sample_workflow, sample_embedding):
        """Test upserting a procedural workflow."""
        workflow_id = str(uuid.uuid4())
        
        # Should not raise exceptions
        await store.upsert_procedural_workflow(
            workflow_id=workflow_id,
            trigger_embedding=sample_embedding,
            successful_workflow=sample_workflow,
            trigger_pattern="test_pattern"
        )

    async def test_query_procedural_similar(self, store, sample_workflow, sample_embedding):
        """Test querying similar procedural workflows."""
        workflow_id = str(uuid.uuid4())
        
        # Insert a workflow first
        await store.upsert_procedural_workflow(
            workflow_id=workflow_id,
            trigger_embedding=sample_embedding,
            successful_workflow=sample_workflow,
            trigger_pattern="test_pattern"
        )
        
        # Query for similar workflows
        results = await store.query_procedural_similar(
            query_embedding=sample_embedding,  
            top_k=5
        )
        
        assert isinstance(results, list)
        if results:  # May be empty if vector search is not available
            assert len(results[0]) == 3  # (workflow_id, workflow, score)

    async def test_query_procedural_similar_by_task(self, store):
        """Test querying similar workflows by task name."""
        results = await store.query_procedural_similar(
            task_name="test_task",
            query_text="test query",
            limit=5
        )
        
        assert isinstance(results, list)

    async def test_bump_procedural_usage(self, store, sample_workflow, sample_embedding):
        """Test bumping procedural workflow usage count."""
        workflow_id = str(uuid.uuid4())
        
        # Insert a workflow first
        await store.upsert_procedural_workflow(
            workflow_id=workflow_id,
            trigger_embedding=sample_embedding,
            successful_workflow=sample_workflow
        )
        
        # Bump usage
        count = await store.bump_procedural_usage(workflow_id, 1)
        assert isinstance(count, int)

    async def test_insert_lesson(self, store, sample_embedding):
        """Test inserting a lesson."""
        lesson_id = await store.insert_lesson(
            trigger_embedding=sample_embedding,
            goal_text="test goal",
            agent="RAGQueryAgent",
            status="success",
            error=None,
            fix_summary="test fix",
            working_params={"param": "value"}
        )
        
        assert isinstance(lesson_id, str)
        assert len(lesson_id) > 0

    async def test_query_lessons_similar(self, store, sample_embedding):
        """Test querying similar lessons."""
        # Insert a lesson first
        await store.insert_lesson(
            trigger_embedding=sample_embedding,
            goal_text="test goal",
            agent="RAGQueryAgent",
            status="success",
            error=None,
            fix_summary="test fix",
            working_params={"param": "value"}
        )
        
        # Query similar lessons
        results = await store.query_lessons_similar(
            trigger_embedding=sample_embedding,
            agent="RAGQueryAgent",
            top_k=5
        )
        
        assert isinstance(results, list)

    async def test_get_lessons_by_task(self, store):
        """Test getting lessons by task name."""
        results = await store.get_lessons_by_task(
            task_name="test_task",
            limit=10
        )
        
        assert isinstance(results, list)

    async def test_query_similar_workflows(self, store):
        """Test querying similar workflows."""
        results = await store.query_similar_workflows(
            task_name="test_task",
            description="test description",
            limit=5
        )
        
        assert isinstance(results, list)

    async def test_get_task_statistics(self, store):
        """Test getting task statistics."""
        results = await store.get_task_statistics(limit=50)
        
        assert isinstance(results, list)

    async def test_query_message_workflows_for_session(self, store):
        """Test querying message workflows for a session."""
        session_id = str(uuid.uuid4())
        
        results = await store.query_message_workflows_for_session(
            session_id=session_id,
            limit=50
        )
        
        assert isinstance(results, list)