"""Tests for M3 Orchestrator."""

import pytest
import uuid
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List

from src.memfuse_core.m3.orchestrator import Orchestrator, Planner, RAGQueryAgent, ReportGenerationAgent
from src.memfuse_core.m3.types import PlanStep
from src.memfuse_core.procedural.store import ProceduralStore
from src.memfuse_core.llm.chat import ChatLLM
from src.memfuse_core.rag.rag_service import RAGService


@pytest.fixture
def mock_llm():
    """Mock ChatLLM for testing."""
    llm = Mock(spec=ChatLLM)
    llm.completion_json.return_value = '{"steps": [{"agent": "RAGQueryAgent", "input": {"query": "test"}}]}'
    llm.chat.return_value = "Test response"
    return llm


@pytest.fixture
def mock_rag():
    """Mock RAGService for testing."""
    rag = Mock(spec=RAGService)
    rag.chat = AsyncMock(return_value="Test RAG response")
    return rag


@pytest.fixture
def mock_store():
    """Mock ProceduralStore for testing."""
    store = Mock(spec=ProceduralStore)
    store.query_procedural_similar = AsyncMock(return_value=[])
    store.query_lessons_similar = AsyncMock(return_value=[])
    store.upsert_procedural_workflow = AsyncMock()
    store.insert_lesson = AsyncMock(return_value=str(uuid.uuid4()))
    store.bump_procedural_usage = AsyncMock(return_value=1)
    return store


@pytest.fixture
def orchestrator(mock_store):
    """Create an Orchestrator with mocked dependencies."""
    with patch('src.memfuse_core.m3.orchestrator.ChatLLM') as mock_llm_class, \
         patch('src.memfuse_core.m3.orchestrator.RAGService') as mock_rag_class:
        
        mock_llm = Mock(spec=ChatLLM)
        mock_llm.completion_json.return_value = '{"steps": [{"agent": "RAGQueryAgent", "input": {"query": "test"}}]}'
        mock_llm.chat.return_value = "Test response"
        mock_llm_class.return_value = mock_llm
        
        mock_rag = Mock(spec=RAGService)
        mock_rag.chat = AsyncMock(return_value="Test RAG response")
        mock_rag_class.return_value = mock_rag
        
        orchestrator = Orchestrator(store=mock_store)
        return orchestrator


class TestPlanner:
    """Test cases for Planner."""

    def test_plan_with_valid_response(self, mock_llm):
        """Test planning with valid LLM response."""
        planner = Planner(mock_llm)
        mock_llm.completion_json.return_value = '''
        {
            "steps": [
                {"agent": "RAGQueryAgent", "input": {"query": "test query"}},
                {"agent": "ReportGenerationAgent", "input": {}}
            ]
        }
        '''
        
        steps = planner.plan("Test goal")
        
        assert len(steps) == 2
        assert steps[0].agent == "RAGQueryAgent"
        assert steps[0].input == {"query": "test query"}
        assert steps[1].agent == "ReportGenerationAgent"
        assert steps[1].input == {}

    def test_plan_with_invalid_response(self, mock_llm):
        """Test planning with invalid LLM response falls back to default."""
        planner = Planner(mock_llm)
        mock_llm.completion_json.return_value = "invalid json"
        
        steps = planner.plan("Test goal")
        
        # Should fall back to default plan
        assert len(steps) == 2
        assert steps[0].agent == "RAGQueryAgent"
        assert steps[1].agent == "ReportGenerationAgent"

    def test_plan_with_empty_steps(self, mock_llm):
        """Test planning with empty steps falls back to default."""
        planner = Planner(mock_llm)
        mock_llm.completion_json.return_value = '{"steps": []}'
        
        steps = planner.plan("Test goal")
        
        # Should fall back to default plan
        assert len(steps) == 2
        assert steps[0].agent == "RAGQueryAgent"
        assert steps[1].agent == "ReportGenerationAgent"


class TestRAGQueryAgent:
    """Test cases for RAGQueryAgent."""

    @pytest.mark.asyncio
    async def test_execute_with_query(self, mock_rag):
        """Test executing RAG query agent with valid query."""
        agent = RAGQueryAgent(mock_rag)
        session_id = str(uuid.uuid4())
        payload = {"query": "test query"}
        
        result = await agent.execute(session_id, payload)
        
        assert "answer" in result
        assert result["answer"] == "Test RAG response"
        mock_rag.chat.assert_called_once_with(session_id, "test query", history_messages=[])

    @pytest.mark.asyncio
    async def test_execute_without_query(self, mock_rag):
        """Test executing RAG query agent without query returns error."""
        agent = RAGQueryAgent(mock_rag)
        session_id = str(uuid.uuid4())
        payload = {}
        
        result = await agent.execute(session_id, payload)
        
        assert "error" in result
        assert result["error"] == "query required"

    @pytest.mark.asyncio
    async def test_execute_with_context_history(self, mock_rag):
        """Test executing RAG query agent with context history."""
        agent = RAGQueryAgent(mock_rag)
        session_id = str(uuid.uuid4())
        history = [{"role": "user", "content": "previous message"}]
        payload = {
            "query": "test query",
            "context": {"_history_messages": history}
        }
        
        result = await agent.execute(session_id, payload)
        
        assert "answer" in result
        mock_rag.chat.assert_called_once_with(session_id, "test query", history_messages=history)


class TestReportGenerationAgent:
    """Test cases for ReportGenerationAgent."""

    @pytest.mark.asyncio
    async def test_execute_with_data(self, mock_llm):
        """Test executing report generation agent."""
        agent = ReportGenerationAgent(mock_llm)
        session_id = str(uuid.uuid4())
        payload = {"points": ["point 1", "point 2"]}
        
        result = await agent.execute(session_id, payload)
        
        assert "report" in result
        assert result["report"] == "Test response"

    @pytest.mark.asyncio
    async def test_execute_with_llm_error(self, mock_llm):
        """Test executing report generation agent with LLM error."""
        agent = ReportGenerationAgent(mock_llm)
        mock_llm.chat.side_effect = Exception("LLM error")
        session_id = str(uuid.uuid4())
        payload = {"data": "test data"}
        
        result = await agent.execute(session_id, payload)
        
        assert "report" in result
        assert "note" in result
        assert "[offline]" in result["report"]
        assert result["note"] == "LLM error"


class TestOrchestrator:
    """Test cases for Orchestrator."""

    @pytest.mark.asyncio
    async def test_handle_request_basic(self, orchestrator):
        """Test basic request handling."""
        session_id = str(uuid.uuid4())
        user_goal = "Test goal"
        
        with patch('src.memfuse_core.utils.embeddings.create_embedding') as mock_embedding:
            mock_embedding.return_value = [0.1] * 384
            
            result = await orchestrator.handle_request(
                session_id=session_id,
                user_goal=user_goal
            )
        
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_handle_request_with_workflow_reuse(self, orchestrator, mock_store):
        """Test request handling with workflow reuse."""
        session_id = str(uuid.uuid4())
        user_goal = "Test goal"
        
        # Mock similar workflow with high score
        mock_workflow = {
            "plan": [
                {"agent": "RAGQueryAgent", "input": {"query": "reused query"}},
                {"agent": "ReportGenerationAgent", "input": {}}
            ]
        }
        mock_store.query_procedural_similar.return_value = [
            ("workflow_id", mock_workflow, 0.95)
        ]
        
        with patch('src.memfuse_core.utils.embeddings.create_embedding') as mock_embedding:
            mock_embedding.return_value = [0.1] * 384
            
            result = await orchestrator.handle_request(
                session_id=session_id,
                user_goal=user_goal
            )
        
        assert isinstance(result, str)
        assert orchestrator.last_reused is True
        mock_store.bump_procedural_usage.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_request_with_low_similarity(self, orchestrator, mock_store):
        """Test request handling with low similarity workflow (no reuse)."""
        session_id = str(uuid.uuid4())
        user_goal = "Test goal"
        
        # Mock similar workflow with low score
        mock_workflow = {"plan": []}
        mock_store.query_procedural_similar.return_value = [
            ("workflow_id", mock_workflow, 0.5)  # Below threshold
        ]
        
        with patch('src.memfuse_core.utils.embeddings.create_embedding') as mock_embedding:
            mock_embedding.return_value = [0.1] * 384
            
            result = await orchestrator.handle_request(
                session_id=session_id,
                user_goal=user_goal
            )
        
        assert isinstance(result, str)
        assert orchestrator.last_reused is False
        mock_store.upsert_procedural_workflow.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_request_with_history(self, orchestrator):
        """Test request handling with message history."""
        session_id = str(uuid.uuid4())
        user_goal = "Test goal"
        history = [{"role": "user", "content": "previous message"}]
        
        with patch('src.memfuse_core.utils.embeddings.create_embedding') as mock_embedding:
            mock_embedding.return_value = [0.1] * 384
            
            result = await orchestrator.handle_request(
                session_id=session_id,
                user_goal=user_goal,
                history_messages=history
            )
        
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_handle_request_with_workflow_name(self, orchestrator):
        """Test request handling with workflow name."""
        session_id = str(uuid.uuid4())
        user_goal = "Test goal"
        workflow_name = "test_workflow"
        
        with patch('src.memfuse_core.utils.embeddings.create_embedding') as mock_embedding:
            mock_embedding.return_value = [0.1] * 384
            
            result = await orchestrator.handle_request(
                session_id=session_id,
                user_goal=user_goal,
                workflow_name=workflow_name
            )
        
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_handle_request_embedding_failure(self, orchestrator):
        """Test request handling when embedding creation fails."""
        session_id = str(uuid.uuid4())
        user_goal = "Test goal"
        
        with patch('src.memfuse_core.utils.embeddings.create_embedding') as mock_embedding:
            mock_embedding.side_effect = Exception("Embedding error")
            
            result = await orchestrator.handle_request(
                session_id=session_id,
                user_goal=user_goal
            )
        
        # Should still work without embeddings
        assert isinstance(result, str)

    def test_orchestrator_initialization(self):
        """Test orchestrator initialization with default values."""
        with patch('src.memfuse_core.m3.orchestrator.ChatLLM'), \
             patch('src.memfuse_core.m3.orchestrator.RAGService'), \
             patch('src.memfuse_core.m3.orchestrator.ProceduralStore'):
            
            orchestrator = Orchestrator()
            
            assert orchestrator.procedural_top_k > 0
            assert 0 < orchestrator.procedural_reuse_threshold <= 1
            assert orchestrator.planner_max_attempts > 0
            assert "RAGQueryAgent" in orchestrator.agents
            assert "ReportGenerationAgent" in orchestrator.agents