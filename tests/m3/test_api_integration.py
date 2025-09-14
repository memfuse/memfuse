"""Integration tests for M3 API endpoints."""

import pytest
import uuid
import json
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient

from src.memfuse_core.api.messages import router as messages_router
from src.memfuse_core.api.query import router as query_router
from src.memfuse_core.models import MessageAdd, Message


@pytest.fixture
def mock_db_service():
    """Mock DatabaseService for testing."""
    mock_db = Mock()
    mock_db.get_user.return_value = {"name": "test_user"}
    mock_db.get_agent.return_value = {"name": "test_agent"}
    mock_db.get_session.return_value = {
        "user_id": "test_user_id",
        "agent_id": "test_agent_id",
        "name": "test_session"
    }
    mock_db.get_messages_by_session.return_value = []
    return mock_db


@pytest.fixture
def mock_memory_service():
    """Mock memory service for testing."""
    mock_service = Mock()
    mock_service.add = AsyncMock(return_value={
        "status": "success",
        "data": {"message_ids": ["msg1", "msg2"]}
    })
    mock_service.get_messages_by_session = AsyncMock(return_value=[])
    return mock_service


class TestMessagesAPIWithM3:
    """Test M3 integration with Messages API."""

    @pytest.mark.asyncio
    async def test_add_messages_with_task_eos_triggering(self, mock_db_service, mock_memory_service):
        """Test that task_eos metadata triggers M3 workflow."""
        session_id = str(uuid.uuid4())
        
        # Message with task_eos metadata
        messages_data = [
            {
                "role": "user",
                "content": "Complete the research task",
                "metadata": {
                    "task": "research_task",
                    "task_eos": True
                }
            }
        ]
        
        request = MessageAdd(messages=[Message(**msg) for msg in messages_data])
        
        with patch('src.memfuse_core.api.messages.DatabaseService.get_instance') as mock_db_get, \
             patch('src.memfuse_core.api.messages.get_service_for_session') as mock_get_service, \
             patch('src.memfuse_core.api.messages.get_task_messages') as mock_get_task_msgs, \
             patch('src.memfuse_core.api.messages.Orchestrator') as mock_orchestrator_class:
            
            mock_db_get.return_value = mock_db_service
            mock_get_service.return_value = mock_memory_service
            mock_get_task_msgs.return_value = []
            
            # Mock orchestrator
            mock_orchestrator = Mock()
            mock_orchestrator.handle_request = AsyncMock(return_value={
                "workflow_id": "test_workflow",
                "result": "M3 workflow completed"
            })
            mock_orchestrator_class.return_value = mock_orchestrator
            
            # Mock API key validation
            with patch('src.memfuse_core.api.messages.api_key_dependency', return_value={}):
                from src.memfuse_core.api.messages import add_messages
                
                response = await add_messages(
                    session_id=session_id,
                    request=request,
                    _api_key_data={}
                )
            
            # Verify M3 orchestrator was called
            mock_orchestrator.handle_request.assert_called_once()
            call_args = mock_orchestrator.handle_request.call_args
            assert call_args[1]["task_name"] == "research_task"
            assert call_args[1]["session_id"] == session_id
            
            # Verify response includes M3 result
            assert response.status == "success"
            assert "m3_workflow" in response.data

    @pytest.mark.asyncio
    async def test_add_messages_without_task_eos(self, mock_db_service, mock_memory_service):
        """Test normal message addition without task_eos."""
        session_id = str(uuid.uuid4())
        
        # Regular message without task_eos
        messages_data = [
            {
                "role": "user",
                "content": "Regular message",
                "metadata": {}
            }
        ]
        
        request = MessageAdd(messages=[Message(**msg) for msg in messages_data])
        
        with patch('src.memfuse_core.api.messages.DatabaseService.get_instance') as mock_db_get, \
             patch('src.memfuse_core.api.messages.get_service_for_session') as mock_get_service, \
             patch('src.memfuse_core.api.messages.Orchestrator') as mock_orchestrator_class:
            
            mock_db_get.return_value = mock_db_service
            mock_get_service.return_value = mock_memory_service
            
            mock_orchestrator = Mock()
            mock_orchestrator.handle_request = AsyncMock()
            mock_orchestrator_class.return_value = mock_orchestrator
            
            with patch('src.memfuse_core.api.messages.api_key_dependency', return_value={}):
                from src.memfuse_core.api.messages import add_messages
                
                response = await add_messages(
                    session_id=session_id,
                    request=request,
                    _api_key_data={}
                )
            
            # Verify M3 orchestrator was NOT called
            mock_orchestrator.handle_request.assert_not_called()
            
            # Verify normal response
            assert response.status == "success"
            assert "m3_workflow" not in response.data

    @pytest.mark.asyncio
    async def test_get_task_messages(self):
        """Test getting task-scoped messages."""
        session_id = str(uuid.uuid4())
        task_name = "test_task"
        
        # Mock messages with task metadata
        mock_messages = [
            {
                "id": "msg1",
                "content": "First task message",
                "metadata": {"task": "test_task"}
            },
            {
                "id": "msg2",
                "content": "Other task message",
                "metadata": {"task": "other_task"}
            },
            {
                "id": "msg3",
                "content": "Second task message",
                "metadata": {"task": "test_task"}
            }
        ]
        
        mock_db = Mock()
        mock_db.get_messages_by_session.return_value = mock_messages
        
        from src.memfuse_core.api.messages import get_task_messages
        
        result = await get_task_messages(mock_db, session_id, task_name)
        
        # Should only return messages with matching task
        assert len(result) == 2
        assert all(msg["metadata"]["task"] == "test_task" for msg in result)


class TestQueryAPI:
    """Test M3 Query API endpoints."""

    @pytest.mark.asyncio
    async def test_query_task_experiences(self):
        """Test querying task experiences."""
        session_id = str(uuid.uuid4())
        
        mock_db = Mock()
        mock_db.get_session.return_value = {"id": session_id}
        
        mock_store = Mock()
        mock_store.query_procedural_similar = AsyncMock(return_value=[
            {
                "workflow_id": "wf1",
                "trigger_pattern": "test_task",
                "successful_workflow": {"goal": "test"},
                "usage_count": 5
            }
        ])
        mock_store.get_lessons_by_task = AsyncMock(return_value=[
            {
                "lesson_id": "lesson1",
                "goal_text": "test_task",
                "status": "success",
                "fix_summary": "Fixed the issue"
            }
        ])
        
        with patch('src.memfuse_core.api.query.DatabaseService.get_instance') as mock_db_get, \
             patch('src.memfuse_core.api.query.ProceduralStore') as mock_store_class, \
             patch('src.memfuse_core.api.query.ensure_session_exists') as mock_ensure_session:
            
            mock_db_get.return_value = mock_db
            mock_store_class.return_value = mock_store
            mock_ensure_session.return_value = {"id": session_id}
            
            from src.memfuse_core.api.query import TaskExperienceQuery, query_task_experiences
            
            request = TaskExperienceQuery(
                task_name="test_task",
                query_text="test query",
                limit=10
            )
            
            with patch('src.memfuse_core.api.query.api_key_dependency', return_value={}):
                response = await query_task_experiences(
                    session_id=session_id,
                    request=request,
                    _api_key_data={}
                )
            
            assert response.status == "success"
            assert response.data["task_name"] == "test_task"
            assert "experiences" in response.data
            assert "lessons" in response.data

    @pytest.mark.asyncio
    async def test_query_workflows(self):
        """Test querying similar workflows."""
        session_id = str(uuid.uuid4())
        
        mock_db = Mock()
        mock_db.get_session.return_value = {"id": session_id}
        
        mock_store = Mock()
        mock_store.query_similar_workflows = AsyncMock(return_value=[
            {
                "workflow_id": "wf1",
                "trigger_pattern": "similar_task",
                "successful_workflow": {"goal": "similar goal"},
                "usage_count": 3
            }
        ])
        
        with patch('src.memfuse_core.api.query.DatabaseService.get_instance') as mock_db_get, \
             patch('src.memfuse_core.api.query.ProceduralStore') as mock_store_class, \
             patch('src.memfuse_core.api.query.ensure_session_exists') as mock_ensure_session:
            
            mock_db_get.return_value = mock_db
            mock_store_class.return_value = mock_store
            mock_ensure_session.return_value = {"id": session_id}
            
            from src.memfuse_core.api.query import WorkflowQuery, query_workflows
            
            request = WorkflowQuery(
                task_name="test_task",
                query_description="find similar workflows",
                limit=5
            )
            
            with patch('src.memfuse_core.api.query.api_key_dependency', return_value={}):
                response = await query_workflows(
                    session_id=session_id,
                    request=request,
                    _api_key_data={}
                )
            
            assert response.status == "success"
            assert response.data["task_name"] == "test_task"
            assert "workflows" in response.data

    @pytest.mark.asyncio
    async def test_list_tasks(self):
        """Test listing tasks with statistics."""
        session_id = str(uuid.uuid4())
        
        mock_db = Mock()
        mock_db.get_session.return_value = {"id": session_id}
        
        mock_store = Mock()
        mock_store.get_task_statistics = AsyncMock(return_value=[
            {
                "task_name": "task1",
                "workflow_count": 3,
                "total_usage": 10,
                "lesson_count": 2,
                "success_count": 2,
                "fail_count": 0
            },
            {
                "task_name": "task2",
                "workflow_count": 1,
                "total_usage": 5,
                "lesson_count": 1,
                "success_count": 0,
                "fail_count": 1
            }
        ])
        
        with patch('src.memfuse_core.api.query.DatabaseService.get_instance') as mock_db_get, \
             patch('src.memfuse_core.api.query.ProceduralStore') as mock_store_class, \
             patch('src.memfuse_core.api.query.ensure_session_exists') as mock_ensure_session:
            
            mock_db_get.return_value = mock_db
            mock_store_class.return_value = mock_store
            mock_ensure_session.return_value = {"id": session_id}
            
            from src.memfuse_core.api.query import list_tasks
            
            with patch('src.memfuse_core.api.query.api_key_dependency', return_value={}):
                response = await list_tasks(
                    session_id=session_id,
                    limit=50,
                    _api_key_data={}
                )
            
            assert response.status == "success"
            assert "tasks" in response.data
            assert len(response.data["tasks"]) == 2
            assert response.data["tasks"][0]["task_name"] == "task1"