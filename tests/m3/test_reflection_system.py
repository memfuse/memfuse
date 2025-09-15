"""Test cases for M3 reflection and learning system."""

import pytest
import json
from unittest.mock import Mock, AsyncMock, patch

from src.memfuse_core.m3.reflection import ReflectionEngine, LearningSystem
from src.memfuse_core.m3.types import PlanStep
from src.memfuse_core.procedural.store import ProceduralStore


class TestReflectionEngine:
    """Test ReflectionEngine functionality."""
    
    @pytest.fixture
    def mock_store(self):
        store = AsyncMock(spec=ProceduralStore)
        store.insert_lesson.return_value = None
        return store
    
    @pytest.fixture
    def reflection_engine(self, mock_store):
        return ReflectionEngine(store=mock_store)
    
    @pytest.fixture
    def sample_execution_data(self):
        """Sample execution data for testing."""
        executed_steps = [
            (PlanStep(agent="RAGQueryAgent", input={"query": "test"}), {"answer": "test answer"}),
            (PlanStep(agent="WebSearchAgent", input={"query": "test"}), {"results": [{"title": "test"}]}),
            (PlanStep(agent="ReportGenerationAgent", input={}), {"report": "final report"})
        ]
        
        outcomes = [
            {"agent": "RAGQueryAgent", "success": True, "attempts": 1, "total_time": 1.5},
            {"agent": "WebSearchAgent", "success": True, "attempts": 1, "total_time": 2.0},
            {"agent": "ReportGenerationAgent", "success": True, "attempts": 1, "total_time": 0.8}
        ]
        
        return executed_steps, outcomes
    
    def test_build_execution_summary(self, reflection_engine, sample_execution_data):
        """Test building execution summary."""
        executed_steps, outcomes = sample_execution_data
        
        summary = reflection_engine._build_execution_summary(executed_steps, outcomes)
        
        assert summary["total_steps"] == 3
        assert summary["successful_steps"] == 3
        assert summary["failed_steps"] == 0
        assert len(summary["step_details"]) == 3
        
        # Check first step detail
        step_detail = summary["step_details"][0]
        assert step_detail["agent"] == "RAGQueryAgent"
        assert step_detail["success"] is True
        assert step_detail["attempts"] == 1
    
    @pytest.mark.asyncio
    async def test_generate_reflection(self, reflection_engine):
        """Test LLM-based reflection generation."""
        # Mock LLM response
        mock_reflection = {
            "overall_success": True,
            "success_patterns": [
                {"agent": "RAGQueryAgent", "pattern": "Clear query structure", "working_params": {"query": "specific question"}}
            ],
            "failure_patterns": [],
            "workflow_improvements": ["Use more specific queries"],
            "agent_recommendations": {"RAGQueryAgent": "Provide context"}
        }
        
        reflection_engine.llm.completion_json = Mock(return_value=json.dumps(mock_reflection))
        
        execution_summary = {"total_steps": 3, "successful_steps": 3, "failed_steps": 0}
        reflection = await reflection_engine._generate_reflection("test goal", execution_summary)
        
        assert reflection["overall_success"] is True
        assert len(reflection["success_patterns"]) == 1
        assert reflection["success_patterns"][0]["agent"] == "RAGQueryAgent"
    
    @pytest.mark.asyncio
    async def test_generate_reflection_llm_failure(self, reflection_engine):
        """Test reflection generation when LLM fails."""
        # Mock LLM to fail
        reflection_engine.llm.completion_json = Mock(side_effect=Exception("LLM failed"))
        
        execution_summary = {"total_steps": 2, "successful_steps": 1, "failed_steps": 1}
        reflection = await reflection_engine._generate_reflection("test goal", execution_summary)
        
        # Should return default structure
        assert "overall_success" in reflection
        assert "success_patterns" in reflection
        assert "failure_patterns" in reflection
        assert "error" in reflection
    
    @pytest.mark.asyncio
    async def test_store_lessons_from_reflection(self, reflection_engine):
        """Test storing lessons from reflection."""
        reflection = {
            "success_patterns": [
                {"agent": "RAGQueryAgent", "pattern": "Good pattern", "working_params": {"query": "test"}}
            ],
            "failure_patterns": [
                {"agent": "WebSearchAgent", "pattern": "Bad pattern", "recommended_fix": "Use better query", "example_params": {"query": "improved"}}
            ]
        }
        
        user_goal_vec = [0.1, 0.2, 0.3]
        await reflection_engine._store_lessons_from_reflection("test goal", user_goal_vec, reflection)
        
        # Verify lessons were stored
        assert reflection_engine.store.insert_lesson.call_count == 2
        
        # Check success lesson
        success_call = reflection_engine.store.insert_lesson.call_args_list[0][0]
        assert success_call[2] == "RAGQueryAgent"  # agent
        assert success_call[3] == "success"  # status
        
        # Check failure lesson  
        failure_call = reflection_engine.store.insert_lesson.call_args_list[1][0]
        assert failure_call[2] == "WebSearchAgent"  # agent
        assert failure_call[3] == "fail"  # status
    
    @pytest.mark.asyncio
    async def test_reflect_on_execution_integration(self, reflection_engine, sample_execution_data):
        """Test full reflection integration."""
        executed_steps, outcomes = sample_execution_data
        
        # Mock LLM response
        mock_reflection = {
            "overall_success": True,
            "success_patterns": [{"agent": "RAGQueryAgent", "working_params": {"query": "test"}}],
            "failure_patterns": [],
            "workflow_improvements": [],
            "agent_recommendations": {}
        }
        reflection_engine.llm.completion_json = Mock(return_value=json.dumps(mock_reflection))
        
        with patch('src.memfuse_core.utils.embeddings.create_embedding') as mock_embed:
            mock_embed.return_value = [0.1, 0.2, 0.3]
            
            reflection = await reflection_engine.reflect_on_execution(
                "test goal", executed_steps, outcomes
            )
            
            assert reflection["overall_success"] is True
            # Verify embedding was created and lessons stored
            mock_embed.assert_called_once_with("test goal")


class TestLearningSystem:
    """Test LearningSystem functionality."""
    
    @pytest.fixture
    def mock_store(self):
        store = AsyncMock(spec=ProceduralStore)
        store.query_lessons_similar.return_value = []
        return store
    
    @pytest.fixture
    def learning_system(self, mock_store):
        return LearningSystem(mock_store)
    
    @pytest.fixture
    def sample_execution_data(self):
        """Sample execution data for testing."""
        executed_steps = [
            (PlanStep(agent="RAGQueryAgent", input={"query": "test"}), {"answer": "test answer"})
        ]
        
        outcomes = [
            {"agent": "RAGQueryAgent", "success": True, "attempts": 1, "total_time": 1.5}
        ]
        
        return executed_steps, outcomes
    
    @pytest.mark.asyncio
    async def test_learn_from_workflow(self, learning_system, sample_execution_data):
        """Test learning from workflow execution."""
        executed_steps, outcomes = sample_execution_data
        
        # Mock reflection engine
        mock_reflection = {
            "overall_success": True,
            "success_patterns": [{"agent": "RAGQueryAgent", "working_params": {"query": "test"}}],
            "failure_patterns": []
        }
        
        with patch.object(learning_system.reflection_engine, 'reflect_on_execution') as mock_reflect:
            mock_reflect.return_value = mock_reflection
            
            with patch('src.memfuse_core.utils.embeddings.create_embedding') as mock_embed:
                mock_embed.return_value = [0.1, 0.2, 0.3]
                
                learning_summary = await learning_system.learn_from_workflow(
                    "test goal", executed_steps, outcomes, "workflow_123"
                )
                
                assert learning_summary["workflow_id"] == "workflow_123"
                assert learning_summary["goal"] == "test goal"
                assert learning_summary["overall_success"] is True
                assert learning_summary["lessons_stored"] == 1
    
    @pytest.mark.asyncio
    async def test_get_learning_insights(self, learning_system):
        """Test getting learning insights."""
        # Mock lessons data
        mock_lessons = [
            ("lesson_1", "success", "", {"query": "good_query"}, 0.9),
            ("lesson_2", "fail", "Error occurred", {"query": "bad_query"}, 0.8)
        ]
        learning_system.store.query_lessons_similar.return_value = mock_lessons
        
        with patch('src.memfuse_core.utils.embeddings.create_embedding') as mock_embed:
            mock_embed.return_value = [0.1, 0.2, 0.3]
            
            insights = await learning_system.get_learning_insights("test goal", "RAGQueryAgent")
            
            assert insights["user_goal"] == "test goal"
            assert insights["agent_filter"] == "RAGQueryAgent"
            assert insights["total_lessons"] == 2
            assert len(insights["success_lessons"]) == 1
            assert len(insights["failure_lessons"]) == 1
            assert insights["top_success_params"] == {"query": "good_query"}
    
    @pytest.mark.asyncio
    async def test_learning_system_error_handling(self, learning_system, sample_execution_data):
        """Test error handling in learning system."""
        executed_steps, outcomes = sample_execution_data
        
        # Mock reflection engine to fail
        with patch.object(learning_system.reflection_engine, 'reflect_on_execution') as mock_reflect:
            mock_reflect.side_effect = Exception("Reflection failed")
            
            learning_summary = await learning_system.learn_from_workflow(
                "test goal", executed_steps, outcomes
            )
            
            assert "error" in learning_summary
            assert "Reflection failed" in learning_summary["error"]


class TestIntegration:
    """Integration tests for reflection and learning."""
    
    @pytest.mark.asyncio
    async def test_full_learning_pipeline(self):
        """Test the complete learning pipeline."""
        # Create real components (with mocked dependencies)
        mock_store = AsyncMock(spec=ProceduralStore)
        mock_store.insert_lesson.return_value = None
        mock_store.query_lessons_similar.return_value = []
        
        learning_system = LearningSystem(mock_store)
        
        # Mock LLM in reflection engine
        learning_system.reflection_engine.llm.completion_json = Mock(
            return_value=json.dumps({
                "overall_success": True,
                "success_patterns": [{"agent": "RAGQueryAgent", "working_params": {"query": "test"}}],
                "failure_patterns": [],
                "workflow_improvements": ["Use more context"],
                "agent_recommendations": {"RAGQueryAgent": "Be more specific"}
            })
        )
        
        # Sample execution data
        executed_steps = [
            (PlanStep(agent="RAGQueryAgent", input={"query": "test"}), {"answer": "response"})
        ]
        outcomes = [{"agent": "RAGQueryAgent", "success": True, "attempts": 1, "total_time": 1.0}]
        
        with patch('src.memfuse_core.utils.embeddings.create_embedding') as mock_embed:
            mock_embed.return_value = [0.1, 0.2, 0.3]
            
            # Run learning
            learning_summary = await learning_system.learn_from_workflow(
                "test goal", executed_steps, outcomes, "workflow_123"
            )
            
            # Verify results
            assert learning_summary["overall_success"] is True
            assert learning_summary["lessons_stored"] == 1
            
            # Verify lesson was stored
            mock_store.insert_lesson.assert_called_once()
            
            # Get insights
            insights = await learning_system.get_learning_insights("test goal")
            assert insights["user_goal"] == "test goal"