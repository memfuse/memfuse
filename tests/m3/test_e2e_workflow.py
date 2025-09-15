"""End-to-end tests for M3 workflow functionality."""

import pytest
import uuid
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List

from src.memfuse_core.m3.orchestrator import Orchestrator
from src.memfuse_core.procedural.store import ProceduralStore
from src.memfuse_core.services.database_service import DatabaseService


@pytest.fixture
async def e2e_store():
    """Create a test ProceduralStore for E2E testing."""
    # Use a mock database for E2E testing to avoid real DB dependencies
    mock_db = Mock()
    mock_db.backend = Mock()
    
    # Mock database operations
    mock_db.backend.execute = AsyncMock()
    
    store = ProceduralStore(mock_db)
    store._initialized = True  # Skip table creation for testing
    return store


@pytest.fixture
def e2e_orchestrator(e2e_store):
    """Create an orchestrator for E2E testing."""
    with patch('src.memfuse_core.m3.orchestrator.ChatLLM') as mock_llm_class, \
         patch('src.memfuse_core.m3.orchestrator.RAGService') as mock_rag_class:
        
        # Mock LLM responses
        mock_llm = Mock()
        mock_llm.completion_json.return_value = '''
        {
            "steps": [
                {"agent": "RAGQueryAgent", "input": {"query": "research topic"}},
                {"agent": "ReportGenerationAgent", "input": {"data": "research results"}}
            ]
        }
        '''
        mock_llm.chat.return_value = "Generated report based on research findings."
        mock_llm_class.return_value = mock_llm
        
        # Mock RAG service
        mock_rag = Mock()
        mock_rag.chat = AsyncMock(return_value="Research findings from knowledge base.")
        mock_rag_class.return_value = mock_rag
        
        orchestrator = Orchestrator(store=e2e_store)
        return orchestrator


class TestE2EWorkflow:
    """End-to-end workflow tests for M3."""

    @pytest.mark.asyncio
    async def test_complete_workflow_execution(self, e2e_orchestrator, e2e_store):
        """Test complete M3 workflow execution from start to finish."""
        session_id = str(uuid.uuid4())
        user_goal = "Research artificial intelligence trends and generate a summary report"
        
        # Mock embedding creation
        with patch('src.memfuse_core.utils.embeddings.create_embedding') as mock_embedding:
            mock_embedding.return_value = [0.1] * 384
            
            # Mock store methods
            e2e_store.query_procedural_similar = AsyncMock(return_value=[])
            e2e_store.query_lessons_similar = AsyncMock(return_value=[])
            e2e_store.upsert_procedural_workflow = AsyncMock()
            e2e_store.insert_lesson = AsyncMock(return_value=str(uuid.uuid4()))
            
            # Execute workflow
            result = await e2e_orchestrator.handle_request(
                session_id=session_id,
                user_goal=user_goal
            )
        
        # Verify workflow execution
        assert isinstance(result, str)
        assert len(result) > 0
        
        # Verify workflow was stored
        e2e_store.upsert_procedural_workflow.assert_called_once()
        
        # Verify lessons were stored
        assert e2e_store.insert_lesson.call_count >= 1
        
        # Verify orchestrator state
        assert e2e_orchestrator.last_workflow_id is not None
        assert len(e2e_orchestrator.last_plan_steps) > 0

    @pytest.mark.asyncio
    async def test_workflow_reuse_scenario(self, e2e_orchestrator, e2e_store):
        """Test workflow reuse when similar workflow exists."""
        session_id = str(uuid.uuid4())
        user_goal = "Generate AI research report"
        
        # Mock similar workflow with high similarity score
        similar_workflow = {
            "goal": "Research AI and create report",
            "plan": [
                {"agent": "RAGQueryAgent", "input": {"query": "AI research"}},
                {"agent": "ReportGenerationAgent", "input": {}}
            ],
            "result_keys": ["report"]
        }
        
        with patch('src.memfuse_core.utils.embeddings.create_embedding') as mock_embedding:
            mock_embedding.return_value = [0.1] * 384
            
            # Mock store to return similar workflow
            e2e_store.query_procedural_similar = AsyncMock(return_value=[
                ("existing_workflow_id", similar_workflow, 0.95)  # High similarity
            ])
            e2e_store.query_lessons_similar = AsyncMock(return_value=[])
            e2e_store.bump_procedural_usage = AsyncMock(return_value=2)
            e2e_store.insert_lesson = AsyncMock(return_value=str(uuid.uuid4()))
            
            # Execute workflow
            result = await e2e_orchestrator.handle_request(
                session_id=session_id,
                user_goal=user_goal
            )
        
        # Verify workflow reuse
        assert isinstance(result, str)
        assert e2e_orchestrator.last_reused is True
        assert e2e_orchestrator.last_workflow_id == "existing_workflow_id"
        
        # Verify usage was bumped instead of creating new workflow
        e2e_store.bump_procedural_usage.assert_called_once_with("existing_workflow_id", 1)
        e2e_store.upsert_procedural_workflow.assert_not_called()

    @pytest.mark.asyncio
    async def test_workflow_with_history_messages(self, e2e_orchestrator, e2e_store):
        """Test workflow execution with message history."""
        session_id = str(uuid.uuid4())
        user_goal = "Continue the previous research"
        history_messages = [
            {"role": "user", "content": "Start researching AI trends"},
            {"role": "assistant", "content": "I'll help you research AI trends."}
        ]
        
        with patch('src.memfuse_core.utils.embeddings.create_embedding') as mock_embedding:
            mock_embedding.return_value = [0.1] * 384
            
            # Mock store methods
            e2e_store.query_procedural_similar = AsyncMock(return_value=[])
            e2e_store.query_lessons_similar = AsyncMock(return_value=[])
            e2e_store.upsert_procedural_workflow = AsyncMock()
            e2e_store.insert_lesson = AsyncMock(return_value=str(uuid.uuid4()))
            
            # Execute workflow with history
            result = await e2e_orchestrator.handle_request(
                session_id=session_id,
                user_goal=user_goal,
                history_messages=history_messages
            )
        
        # Verify execution completed
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_workflow_with_lessons_guidance(self, e2e_orchestrator, e2e_store):
        """Test workflow execution with lessons providing guidance."""
        session_id = str(uuid.uuid4())
        user_goal = "Research machine learning algorithms"
        
        # Mock lessons from previous similar tasks
        mock_lessons = [
            ("lesson1", "success", "Use specific ML keywords for better results", {"query_params": "detailed"}, 0.9),
            ("lesson2", "fail", "Avoid generic queries", {"error": "too_broad"}, 0.8)
        ]
        
        with patch('src.memfuse_core.utils.embeddings.create_embedding') as mock_embedding:
            mock_embedding.return_value = [0.1] * 384
            
            # Mock store methods
            e2e_store.query_procedural_similar = AsyncMock(return_value=[])
            e2e_store.query_lessons_similar = AsyncMock(return_value=mock_lessons)
            e2e_store.upsert_procedural_workflow = AsyncMock()
            e2e_store.insert_lesson = AsyncMock(return_value=str(uuid.uuid4()))
            
            # Execute workflow
            result = await e2e_orchestrator.handle_request(
                session_id=session_id,
                user_goal=user_goal
            )
        
        # Verify execution completed with lessons guidance
        assert isinstance(result, str)
        assert len(result) > 0
        
        # Verify lessons were queried
        e2e_store.query_lessons_similar.assert_called_once()

    @pytest.mark.asyncio
    async def test_workflow_error_handling(self, e2e_orchestrator, e2e_store):
        """Test workflow execution with agent errors."""
        session_id = str(uuid.uuid4())
        user_goal = "Research unavailable topic"
        
        # Mock RAG service to raise an error
        e2e_orchestrator.agents["RAGQueryAgent"].execute = AsyncMock(
            return_value={"error": "Knowledge base unavailable"}
        )
        
        with patch('src.memfuse_core.utils.embeddings.create_embedding') as mock_embedding:
            mock_embedding.return_value = [0.1] * 384
            
            # Mock store methods
            e2e_store.query_procedural_similar = AsyncMock(return_value=[])
            e2e_store.query_lessons_similar = AsyncMock(return_value=[])
            e2e_store.upsert_procedural_workflow = AsyncMock()
            e2e_store.insert_lesson = AsyncMock(return_value=str(uuid.uuid4()))
            
            # Execute workflow
            result = await e2e_orchestrator.handle_request(
                session_id=session_id,
                user_goal=user_goal
            )
        
        # Verify execution completed despite errors
        assert isinstance(result, str)
        
        # Verify failure lessons were stored
        lesson_calls = e2e_store.insert_lesson.call_args_list
        assert any("fail" in str(call) for call in lesson_calls)

    @pytest.mark.asyncio
    async def test_multiple_concurrent_workflows(self, e2e_store):
        """Test handling multiple concurrent workflows."""
        # Create multiple orchestrators for concurrent testing
        orchestrators = []
        
        for i in range(3):
            with patch('src.memfuse_core.m3.orchestrator.ChatLLM') as mock_llm_class, \
                 patch('src.memfuse_core.m3.orchestrator.RAGService') as mock_rag_class:
                
                mock_llm = Mock()
                mock_llm.completion_json.return_value = f'''
                {{
                    "steps": [
                        {{"agent": "RAGQueryAgent", "input": {{"query": "concurrent query {i}"}}}},
                        {{"agent": "ReportGenerationAgent", "input": {{}}}}
                    ]
                }}
                '''
                mock_llm.chat.return_value = f"Concurrent report {i}"
                mock_llm_class.return_value = mock_llm
                
                mock_rag = Mock()
                mock_rag.chat = AsyncMock(return_value=f"Concurrent research {i}")
                mock_rag_class.return_value = mock_rag
                
                orchestrator = Orchestrator(store=e2e_store)
                orchestrators.append(orchestrator)
        
        # Mock store methods for concurrent access
        e2e_store.query_procedural_similar = AsyncMock(return_value=[])
        e2e_store.query_lessons_similar = AsyncMock(return_value=[])
        e2e_store.upsert_procedural_workflow = AsyncMock()
        e2e_store.insert_lesson = AsyncMock(return_value=str(uuid.uuid4()))
        
        with patch('src.memfuse_core.utils.embeddings.create_embedding') as mock_embedding:
            mock_embedding.return_value = [0.1] * 384
            
            # Execute workflows concurrently
            tasks = []
            for i, orchestrator in enumerate(orchestrators):
                task = orchestrator.handle_request(
                    session_id=str(uuid.uuid4()),
                    user_goal=f"Concurrent research task {i}"
                )
                tasks.append(task)
            
            # Wait for all workflows to complete
            results = await asyncio.gather(*tasks)
        
        # Verify all workflows completed
        assert len(results) == 3
        for result in results:
            assert isinstance(result, str)
            assert len(result) > 0
        
        # Verify store was called for each workflow
        assert e2e_store.upsert_procedural_workflow.call_count == 3

    @pytest.mark.asyncio
    async def test_workflow_with_named_task(self, e2e_orchestrator, e2e_store):
        """Test workflow execution with a named task."""
        session_id = str(uuid.uuid4())
        user_goal = "Research and summarize"
        workflow_name = "research_summary_workflow"
        
        with patch('src.memfuse_core.utils.embeddings.create_embedding') as mock_embedding:
            mock_embedding.return_value = [0.1] * 384
            
            # Mock store methods
            e2e_store.query_procedural_similar = AsyncMock(return_value=[])
            e2e_store.query_lessons_similar = AsyncMock(return_value=[])
            e2e_store.upsert_procedural_workflow = AsyncMock()
            e2e_store.insert_lesson = AsyncMock(return_value=str(uuid.uuid4()))
            
            # Execute workflow with name
            result = await e2e_orchestrator.handle_request(
                session_id=session_id,
                user_goal=user_goal,
                workflow_name=workflow_name
            )
        
        # Verify execution completed
        assert isinstance(result, str)
        
        # Verify workflow was stored with name
        e2e_store.upsert_procedural_workflow.assert_called_once()
        call_args = e2e_store.upsert_procedural_workflow.call_args
        stored_workflow = call_args[0][2]  # Third argument is the workflow dict
        assert stored_workflow["workflow_name"] == workflow_name