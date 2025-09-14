"""Test cases for enhanced AgentExecutor."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path

from src.memfuse_core.m3.executor import AgentExecutor
from src.memfuse_core.m3.types import PlanStep
from src.memfuse_core.procedural.store import ProceduralStore


class TestEnhancedAgentExecutor:
    """Test enhanced AgentExecutor functionality."""
    
    @pytest.fixture
    def mock_store(self):
        store = AsyncMock(spec=ProceduralStore)
        store.query_lessons_similar.return_value = []
        store.insert_lesson.return_value = None
        return store
    
    @pytest.fixture
    def mock_agents(self):
        """Create mock agents for testing."""
        agents = {}
        
        # RAGQueryAgent mock
        rag_agent = AsyncMock()
        rag_agent.execute.return_value = {"answer": "Mock RAG response"}
        agents["RAGQueryAgent"] = rag_agent
        
        # WebSearchAgent mock
        web_agent = AsyncMock()
        web_agent.execute.return_value = {
            "results": [{"title": "Test result", "url": "http://example.com", "snippet": "test"}]
        }
        agents["WebSearchAgent"] = web_agent
        
        # ReportGenerationAgent mock
        report_agent = AsyncMock()
        report_agent.execute.return_value = {"report": "Mock report"}
        agents["ReportGenerationAgent"] = report_agent
        
        return agents
    
    @pytest.fixture
    def executor(self, mock_agents, mock_store):
        return AgentExecutor(mock_agents, mock_store, planner_max_attempts=2)
    
    def test_success_heuristic(self, executor):
        """Test success heuristic for different agents."""
        # RAGQueryAgent
        assert executor._success_heuristic("RAGQueryAgent", {"answer": "test"})
        assert not executor._success_heuristic("RAGQueryAgent", {"error": "failed"})
        assert not executor._success_heuristic("RAGQueryAgent", {})
        
        # WebSearchAgent
        assert executor._success_heuristic("WebSearchAgent", {"results": [{"title": "test"}]})
        assert not executor._success_heuristic("WebSearchAgent", {"results": []})
        
        # ReportGenerationAgent
        assert executor._success_heuristic("ReportGenerationAgent", {"report": "test"})
        assert not executor._success_heuristic("ReportGenerationAgent", {})
        
        # DatabaseQueryAgent
        assert executor._success_heuristic("DatabaseQueryAgent", {"rows": []})
        assert executor._success_heuristic("DatabaseQueryAgent", {"columns": []})
        assert not executor._success_heuristic("DatabaseQueryAgent", {})
    
    @pytest.mark.asyncio
    async def test_seed_params_from_lessons(self, executor):
        """Test parameter seeding from lessons."""
        # Mock lessons with successful parameters
        executor.store.query_lessons_similar.return_value = [
            ("lesson_1", "success", "", {"query": "seeded_query", "max_results": 10}, 0.9)
        ]
        
        payload = {"other_param": "value"}
        user_goal_vec = [0.1, 0.2, 0.3]
        
        seeded_payload = await executor._seed_params_from_lessons(
            "WebSearchAgent", "test goal", user_goal_vec, payload
        )
        
        assert seeded_payload["query"] == "seeded_query"
        assert seeded_payload["max_results"] == 10
        assert seeded_payload["other_param"] == "value"
    
    @pytest.mark.asyncio
    async def test_propose_input(self, executor):
        """Test LLM-based parameter proposal."""
        # Mock LLM response
        executor.llm.completion_json = Mock(return_value='{"query": "proposed_query", "max_results": 5}')
        
        context = {"step_0_answer": "previous result"}
        proposed = await executor._propose_input("WebSearchAgent", "test goal", context)
        
        assert proposed["query"] == "proposed_query"
        assert proposed["max_results"] == 5
    
    @pytest.mark.asyncio
    async def test_propose_input_fallback(self, executor):
        """Test fallback parameter proposals when LLM fails."""
        # Mock LLM to fail
        executor.llm.completion_json = Mock(side_effect=Exception("LLM failed"))
        
        context = {}
        
        # Test RAGQueryAgent fallback
        proposed = await executor._propose_input("RAGQueryAgent", "test goal", context)
        assert proposed["query"] == "test goal"
        
        # Test WebSearchAgent fallback
        proposed = await executor._propose_input("WebSearchAgent", "test goal", context)
        assert proposed["query"] == "test goal"
        assert proposed["max_results"] == 10
    
    @pytest.mark.asyncio
    async def test_execute_step_with_retries_success(self, executor, mock_agents):
        """Test successful step execution with retries."""
        step = PlanStep(agent="RAGQueryAgent", input={"query": "test"})\n        
        outcome, attempts_log = await executor._execute_step_with_retries(
            step, "session_1", "test goal", None, {}, 2
        )
        
        assert outcome["answer"] == "Mock RAG response"
        assert len(attempts_log) == 1
        assert attempts_log[0]["success"] is True
    
    @pytest.mark.asyncio
    async def test_execute_step_with_retries_failure(self, executor, mock_agents):
        """Test step execution with retries after failure."""
        # Make agent fail first, then succeed
        mock_agents["RAGQueryAgent"].execute.side_effect = [
            {"error": "First attempt failed"},
            {"answer": "Second attempt succeeded"}
        ]
        
        step = PlanStep(agent="RAGQueryAgent", input={"query": "test"})
        
        outcome, attempts_log = await executor._execute_step_with_retries(
            step, "session_1", "test goal", None, {}, 2
        )
        
        assert outcome["answer"] == "Second attempt succeeded"
        assert len(attempts_log) == 2
        assert attempts_log[0]["success"] is False
        assert attempts_log[1]["success"] is True
    
    @pytest.mark.asyncio
    async def test_execute_step_with_parameter_proposal(self, executor, mock_agents):
        """Test step execution with automatic parameter proposal."""
        # Mock LLM to propose parameters
        executor.llm.completion_json = Mock(return_value='{"query": "proposed_query"}')
        
        # Step with missing required parameter
        step = PlanStep(agent="RAGQueryAgent", input={})
        
        outcome, attempts_log = await executor._execute_step_with_retries(
            step, "session_1", "test goal", None, {}, 1
        )
        
        # Verify agent was called with proposed parameter
        mock_agents["RAGQueryAgent"].execute.assert_called_once()
        call_args = mock_agents["RAGQueryAgent"].execute.call_args[0][1]
        assert call_args["query"] == "proposed_query"
    
    @pytest.mark.asyncio
    async def test_execute_steps_integration(self, executor, mock_agents, tmp_path):
        """Test full step execution integration."""
        steps = [
            PlanStep(agent="RAGQueryAgent", input={"query": "test query"}),
            PlanStep(agent="ReportGenerationAgent", input={})
        ]
        
        context = {}
        run_dir = tmp_path / "test_run"
        run_dir.mkdir()
        
        executed, outcomes = await executor.execute_steps(
            session_id="session_1",
            steps=steps,
            context=context,
            run_dir=run_dir,
            user_goal_vec=None,
            user_goal="test goal"
        )
        
        # Verify execution
        assert len(executed) == 2
        assert len(outcomes) == 2
        
        # Verify context was updated
        assert "step_0_answer" in context
        assert "step_1_report" in context
        
        # Verify step files were created
        step_files = list(run_dir.glob("step_*.json"))
        assert len(step_files) == 2
    
    @pytest.mark.asyncio
    async def test_lesson_storage_on_success(self, executor, mock_agents):
        """Test that successful lessons are stored."""
        user_goal_vec = [0.1, 0.2, 0.3]
        step = PlanStep(agent="RAGQueryAgent", input={"query": "test"})
        
        await executor._execute_step_with_retries(
            step, "session_1", "test goal", user_goal_vec, {}, 1
        )
        
        # Verify lesson was stored
        executor.store.insert_lesson.assert_called_once()
        call_args = executor.store.insert_lesson.call_args[0]
        assert call_args[1] == "test goal"  # user_goal
        assert call_args[2] == "RAGQueryAgent"  # agent
        assert call_args[3] == "success"  # status
    
    @pytest.mark.asyncio
    async def test_lesson_storage_on_failure(self, executor, mock_agents):
        """Test that failure lessons are stored."""
        # Make agent always fail
        mock_agents["RAGQueryAgent"].execute.return_value = {"error": "Always fails"}
        
        user_goal_vec = [0.1, 0.2, 0.3]
        step = PlanStep(agent="RAGQueryAgent", input={"query": "test"})
        
        await executor._execute_step_with_retries(
            step, "session_1", "test goal", user_goal_vec, {}, 2
        )
        
        # Verify failure lesson was stored
        executor.store.insert_lesson.assert_called_once()
        call_args = executor.store.insert_lesson.call_args[0]
        assert call_args[3] == "fail"  # status
        assert "Always fails" in call_args[4]  # error message