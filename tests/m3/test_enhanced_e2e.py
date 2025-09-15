"""Enhanced end-to-end tests for M3 system."""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path

from src.memfuse_core.m3.orchestrator import Orchestrator
from src.memfuse_core.m3.types import PlanStep
from src.memfuse_core.procedural.store import ProceduralStore


class TestEnhancedM3E2E:
    """End-to-end tests for enhanced M3 system."""
    
    @pytest.fixture
    def mock_store(self):
        """Mock procedural store."""
        store = AsyncMock(spec=ProceduralStore)
        store.query_procedural_similar.return_value = []
        store.query_lessons_similar.return_value = []
        store.upsert_procedural_workflow.return_value = None
        store.insert_lesson.return_value = None
        store.bump_procedural_usage.return_value = None
        return store
    
    @pytest.fixture
    def orchestrator(self, mock_store):
        """Create orchestrator with mocked dependencies."""
        with patch('src.memfuse_core.m3.orchestrator.ProceduralStore', return_value=mock_store):
            orchestrator = Orchestrator(store=mock_store)
            
            # Mock LLM responses
            orchestrator.llm.completion_json = Mock(return_value=json.dumps({
                "steps": [
                    {"agent": "RAGQueryAgent", "input": {"query": "research machine learning"}},
                    {"agent": "WebSearchAgent", "input": {"query": "latest ML research", "sources": ["arxiv"]}},
                    {"agent": "ReportGenerationAgent", "input": {}}
                ]
            }))
            
            # Mock agent responses
            orchestrator.agents["RAGQueryAgent"].execute = AsyncMock(
                return_value={"answer": "Machine learning is a subset of AI..."}
            )
            orchestrator.agents["WebSearchAgent"].execute = AsyncMock(
                return_value={
                    "results": [
                        {"title": "New ML Paper", "url": "http://arxiv.org/abs/2024.001", "snippet": "Latest research..."}
                    ],
                    "total_found": 1
                }
            )
            orchestrator.agents["DatabaseQueryAgent"].execute = AsyncMock(
                return_value={"rows": [{"id": 1, "content": "Sample data"}], "row_count": 1}
            )
            orchestrator.agents["ShellCommandAgent"].execute = AsyncMock(
                return_value={"exit_code": 0, "output": "search results"}
            )
            orchestrator.agents["ReportGenerationAgent"].execute = AsyncMock(
                return_value={"report": "## Research Summary\n\nBased on the research..."}
            )
            
            return orchestrator
    
    @pytest.mark.asyncio
    async def test_complete_workflow_execution(self, orchestrator, tmp_path):
        """Test complete workflow execution with all enhancements."""
        # Mock embedding creation
        with patch('src.memfuse_core.utils.embeddings.create_embedding') as mock_embed:
            mock_embed.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
            
            # Mock learning system reflection
            orchestrator.learning_system.reflection_engine.llm.completion_json = Mock(
                return_value=json.dumps({
                    "overall_success": True,
                    "success_patterns": [
                        {"agent": "RAGQueryAgent", "pattern": "Clear queries work well", "working_params": {"query": "research machine learning"}}
                    ],
                    "failure_patterns": [],
                    "workflow_improvements": ["Consider using more specific search terms"],
                    "agent_recommendations": {"WebSearchAgent": "Include more sources"}
                })
            )
            
            # Set custom runs directory
            orchestrator.runs_base_dir = str(tmp_path)
            
            result = await orchestrator.handle_request(
                session_id="test_session_123",
                user_goal="Research recent developments in machine learning",
                workflow_name="ml_research_workflow"
            )
            
            # Verify result
            assert isinstance(result, str)
            assert "Research Summary" in result
            
            # Verify workflow was stored
            orchestrator.store.upsert_procedural_workflow.assert_called_once()
            
            # Verify lessons were stored (from reflection)
            assert orchestrator.store.insert_lesson.call_count > 0
            
            # Verify orchestrator state
            assert orchestrator.last_workflow_id is not None
            assert orchestrator.last_reused is False
            assert len(orchestrator.last_plan_steps) == 3
            assert orchestrator.last_reflection is not None
            
            # Verify run directory was created with artifacts
            run_dirs = list(tmp_path.glob("*/test_session_123"))
            assert len(run_dirs) == 1
            
            run_dir = run_dirs[0]
            assert (run_dir / "input.json").exists()
            assert (run_dir / "plan.json").exists()
            assert (run_dir / "report.txt").exists()
            assert (run_dir / "learning_reflection.json").exists()
            
            # Verify step artifacts
            step_files = list(run_dir.glob("step_*.json"))
            assert len(step_files) == 3
    
    @pytest.mark.asyncio
    async def test_workflow_reuse(self, orchestrator):
        """Test workflow reuse functionality."""
        # Mock existing workflow for reuse
        similar_workflow = {
            "goal": "Research machine learning",
            "plan": [
                {"agent": "RAGQueryAgent", "input": {"query": "machine learning"}},
                {"agent": "ReportGenerationAgent", "input": {}}
            ]
        }
        
        orchestrator.store.query_procedural_similar.return_value = [
            ("workflow_123", similar_workflow, 0.95)  # High similarity score
        ]
        
        with patch('src.memfuse_core.utils.embeddings.create_embedding') as mock_embed:
            mock_embed.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
            
            result = await orchestrator.handle_request(
                session_id="test_session_456",
                user_goal="Research machine learning applications"
            )
            
            # Verify workflow was reused
            assert orchestrator.last_reused is True
            assert orchestrator.last_workflow_id == "workflow_123"
            
            # Verify usage was bumped
            orchestrator.store.bump_procedural_usage.assert_called_once_with("workflow_123", 1)
    
    @pytest.mark.asyncio
    async def test_agent_retry_and_parameter_seeding(self, orchestrator):
        """Test agent retry mechanism and parameter seeding."""
        # Mock lessons for parameter seeding
        orchestrator.store.query_lessons_similar.return_value = [
            ("lesson_1", "success", "", {"query": "seeded query", "max_results": 10}, 0.9)
        ]
        
        # Make WebSearchAgent fail first, then succeed
        orchestrator.agents["WebSearchAgent"].execute = AsyncMock()
        orchestrator.agents["WebSearchAgent"].execute.side_effect = [
            {"error": "Network timeout"},  # First attempt fails
            {"results": [{"title": "Success", "url": "http://example.com"}], "total_found": 1}  # Second succeeds
        ]
        
        with patch('src.memfuse_core.utils.embeddings.create_embedding') as mock_embed:
            mock_embed.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
            
            result = await orchestrator.handle_request(
                session_id="test_session_789",
                user_goal="Search for recent AI papers"
            )
            
            # Verify agent was called multiple times (retry)
            assert orchestrator.agents["WebSearchAgent"].execute.call_count == 2
            
            # Verify final result is successful
            assert "Research Summary" in result
    
    @pytest.mark.asyncio
    async def test_multi_agent_workflow(self, orchestrator):
        """Test workflow with multiple different agents."""
        # Override planner to create a more complex workflow
        orchestrator.llm.completion_json = Mock(return_value=json.dumps({
            "steps": [
                {"agent": "DatabaseQueryAgent", "input": {"request": "Get recent user queries"}},
                {"agent": "RAGQueryAgent", "input": {"query": "analyze user behavior"}},
                {"agent": "WebSearchAgent", "input": {"query": "user behavior trends", "sources": ["duckduckgo"]}},
                {"agent": "ShellCommandAgent", "input": {"cmd": "rg", "pattern": "behavior", "path": "/tmp"}},
                {"agent": "ReportGenerationAgent", "input": {}}
            ]
        }))
        
        with patch('src.memfuse_core.utils.embeddings.create_embedding') as mock_embed:
            mock_embed.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
            
            result = await orchestrator.handle_request(
                session_id="test_session_multi",
                user_goal="Analyze user behavior patterns"
            )
            
            # Verify all agents were called
            orchestrator.agents["DatabaseQueryAgent"].execute.assert_called_once()
            orchestrator.agents["RAGQueryAgent"].execute.assert_called_once()
            orchestrator.agents["WebSearchAgent"].execute.assert_called_once()
            orchestrator.agents["ShellCommandAgent"].execute.assert_called_once()
            orchestrator.agents["ReportGenerationAgent"].execute.assert_called_once()
            
            # Verify workflow completed successfully
            assert "Research Summary" in result
            assert len(orchestrator.last_plan_steps) == 5
    
    @pytest.mark.asyncio
    async def test_learning_system_integration(self, orchestrator):
        """Test integration with learning system."""
        with patch('src.memfuse_core.utils.embeddings.create_embedding') as mock_embed:
            mock_embed.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
            
            # Execute workflow
            result = await orchestrator.handle_request(
                session_id="test_learning",
                user_goal="Test learning system"
            )
            
            # Verify learning system was called
            assert orchestrator.last_reflection is not None
            
            # Verify reflection contains expected structure
            reflection = orchestrator.last_reflection
            assert "overall_success" in reflection
            assert "success_patterns" in reflection
            assert "failure_patterns" in reflection
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, orchestrator):
        """Test error handling and recovery mechanisms."""
        # Make RAGQueryAgent fail completely
        orchestrator.agents["RAGQueryAgent"].execute = AsyncMock(
            return_value={"error": "Service unavailable"}
        )
        
        with patch('src.memfuse_core.utils.embeddings.create_embedding') as mock_embed:
            mock_embed.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
            
            result = await orchestrator.handle_request(
                session_id="test_error_handling",
                user_goal="Test error recovery"
            )
            
            # Workflow should still complete (other agents succeed)
            assert isinstance(result, str)
            
            # Verify failure lessons were stored
            failure_lessons = [
                call for call in orchestrator.store.insert_lesson.call_args_list
                if call[0][3] == "fail"  # status parameter
            ]
            assert len(failure_lessons) > 0
    
    @pytest.mark.asyncio 
    async def test_context_propagation(self, orchestrator):
        """Test that context is properly propagated between agents."""
        # Track context propagation
        call_contexts = []
        
        def track_context(session_id, payload):
            call_contexts.append(payload.get("context", {}))
            return {"answer": f"Response with context keys: {list(payload.get('context', {}).keys())}"}
        
        orchestrator.agents["RAGQueryAgent"].execute = AsyncMock(side_effect=track_context)
        
        with patch('src.memfuse_core.utils.embeddings.create_embedding') as mock_embed:
            mock_embed.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
            
            result = await orchestrator.handle_request(
                session_id="test_context",
                user_goal="Test context propagation",
                history_messages=[{"role": "user", "content": "Previous message"}]
            )
            
            # Verify context was propagated
            assert len(call_contexts) > 0
            
            # First agent should have history messages
            first_context = call_contexts[0]
            assert "_history_messages" in first_context
            
            # Later agents should have results from previous steps
            if len(call_contexts) > 1:
                later_context = call_contexts[-1]
                step_results = [key for key in later_context.keys() if key.startswith("step_")]
                assert len(step_results) > 0


@pytest.mark.integration
class TestM3GatewayIntegration:
    """Test M3 integration with gateway components."""
    
    @pytest.mark.asyncio
    async def test_m3_processor_integration(self):
        """Test M3 processor integration."""
        from src.memfuse_core.gateway.m3_processor import M3Processor
        from src.memfuse_core.interfaces.gateway_interface import RequestContext
        
        processor = M3Processor()
        
        # Mock orchestrator
        processor.orchestrator.handle_request = AsyncMock(
            return_value="M3 workflow result"
        )
        processor.orchestrator.last_workflow_id = "workflow_123"
        processor.orchestrator.last_reused = True
        processor.orchestrator.last_plan_steps = [
            PlanStep(agent="RAGQueryAgent", input={"query": "test"})
        ]
        
        # Test request data
        request_data = {
            "query": "Test M3 integration",
            "metadata": {
                "task_eos": True,
                "workflow_name": "test_workflow"
            }
        }
        
        context = RequestContext(
            session_id="test_session",
            user_id="test_user",
            agent_id="test_agent"
        )
        
        # Test M3 triggering
        should_trigger = processor.should_trigger_m3(request_data, context)
        assert should_trigger is True
        
        # Test M3 processing
        response = await processor.process_m3_request(request_data, context)
        
        assert response["status"] == "success"
        assert response["data"]["m3_result"] == "M3 workflow result"
        assert response["data"]["workflow_id"] == "workflow_123"
        assert response["data"]["workflow_reused"] is True