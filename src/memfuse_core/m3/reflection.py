"""Reflection and learning system for M3."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from loguru import logger

from ..procedural.store import ProceduralStore
from ..utils.embeddings import create_embedding


class ReflectionEngine:
    """Analyzes workflow execution and extracts lessons for future improvement."""
    
    def __init__(self, llm=None, store: Optional[ProceduralStore] = None):
        self.llm = llm or self._create_default_llm()
        self.store = store or ProceduralStore()
    
    def _create_default_llm(self):
        """Create default LLM instance."""
        try:
            from ..llm.chat import ChatLLM
            return ChatLLM()
        except Exception as e:
            logger.warning(f"Failed to create ChatLLM for ReflectionEngine: {e}")
            return MockLLM()
    
    async def reflect_on_execution(
        self,
        user_goal: str,
        executed_steps: List[tuple],
        outcomes: List[Dict[str, Any]],
        user_goal_vec: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """Analyze workflow execution and extract lessons."""
        try:
            # Build execution summary for reflection
            execution_summary = self._build_execution_summary(executed_steps, outcomes)
            
            # Generate reflection using LLM
            reflection = await self._generate_reflection(user_goal, execution_summary)
            
            # Store lessons from reflection
            if user_goal_vec is not None:
                await self._store_lessons_from_reflection(user_goal, user_goal_vec, reflection)
            
            return reflection
            
        except Exception as e:
            logger.error(f"Reflection failed: {e}")
            return {"error": str(e)}
    
    def _build_execution_summary(
        self, 
        executed_steps: List[tuple], 
        outcomes: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build a summary of execution for reflection."""
        summary = {
            "total_steps": len(executed_steps),
            "successful_steps": 0,
            "failed_steps": 0,
            "step_details": []
        }
        
        for i, ((step, outcome), outcome_summary) in enumerate(zip(executed_steps, outcomes)):
            success = outcome_summary.get("success", False)
            
            if success:
                summary["successful_steps"] += 1
            else:
                summary["failed_steps"] += 1
            
            step_detail = {
                "step_index": i,
                "agent": step.agent,
                "success": success,
                "attempts": outcome_summary.get("attempts", 1),
                "total_time": outcome_summary.get("total_time", 0),
                "input_keys": list(step.input.keys()),
                "output_keys": list(outcome.keys()) if isinstance(outcome, dict) else [],
                "error": outcome_summary.get("final_error") or outcome.get("error") if isinstance(outcome, dict) else None
            }
            summary["step_details"].append(step_detail)
        
        return summary
    
    async def _generate_reflection(self, user_goal: str, execution_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Generate reflection using LLM."""
        system = (
            "You are a workflow reflection engine. Analyze the execution summary and extract lessons.\n"
            "Return strict JSON with the following structure:\n"
            "{\n"
            "  \"overall_success\": boolean,\n"
            "  \"success_patterns\": [{\"agent\": \"...\", \"pattern\": \"...\", \"working_params\": {...}}],\n"
            "  \"failure_patterns\": [{\"agent\": \"...\", \"pattern\": \"...\", \"recommended_fix\": \"...\", \"example_params\": {...}}],\n"
            "  \"workflow_improvements\": [\"...\"],\n"
            "  \"agent_recommendations\": {\"agent_name\": \"recommendation\"}\n"
            "}\n"
        )
        
        user_prompt = json.dumps({
            "user_goal": user_goal,
            "execution_summary": execution_summary
        }, ensure_ascii=False)
        
        try:
            raw_reflection = self.llm.completion_json(system, user_prompt)
            reflection = json.loads(raw_reflection or '{}')
            
            # Validate and clean reflection
            if not isinstance(reflection, dict):
                reflection = {}
            
            reflection.setdefault("overall_success", execution_summary["failed_steps"] == 0)
            reflection.setdefault("success_patterns", [])
            reflection.setdefault("failure_patterns", [])
            reflection.setdefault("workflow_improvements", [])
            reflection.setdefault("agent_recommendations", {})
            
            return reflection
            
        except Exception as e:
            logger.error(f"Failed to generate reflection: {e}")
            return {
                "overall_success": execution_summary["failed_steps"] == 0,
                "success_patterns": [],
                "failure_patterns": [],
                "workflow_improvements": [],
                "agent_recommendations": {},
                "error": str(e)
            }
    
    async def _store_lessons_from_reflection(
        self,
        user_goal: str,
        user_goal_vec: List[float],
        reflection: Dict[str, Any]
    ):
        """Store lessons extracted from reflection."""
        try:
            # Store success patterns as lessons
            for pattern in reflection.get("success_patterns", []):
                if not isinstance(pattern, dict):
                    continue
                
                agent = pattern.get("agent", "")
                working_params = pattern.get("working_params", {})
                
                if agent and isinstance(working_params, dict):
                    await self.store.insert_lesson(
                        user_goal_vec, user_goal, agent, "success",
                        None, pattern.get("pattern", ""), working_params
                    )
            
            # Store failure patterns as lessons
            for pattern in reflection.get("failure_patterns", []):
                if not isinstance(pattern, dict):
                    continue
                
                agent = pattern.get("agent", "")
                error_pattern = pattern.get("pattern", "")
                fix = pattern.get("recommended_fix", "")
                example_params = pattern.get("example_params", {})
                
                if agent and error_pattern:
                    await self.store.insert_lesson(
                        user_goal_vec, user_goal, agent, "fail",
                        error_pattern[:500], fix, example_params
                    )
            
        except Exception as e:
            logger.error(f"Failed to store lessons from reflection: {e}")


class LearningSystem:
    """Manages continuous learning from workflow executions."""
    
    def __init__(self, store: Optional[ProceduralStore] = None):
        self.store = store or ProceduralStore()
        self.reflection_engine = ReflectionEngine(store=self.store)
    
    async def learn_from_workflow(
        self,
        user_goal: str,
        executed_steps: List[tuple],
        outcomes: List[Dict[str, Any]],
        workflow_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Learn from a completed workflow execution."""
        try:
            # Create goal embedding for similarity matching
            user_goal_vec = None
            try:
                user_goal_vec = await create_embedding(user_goal)
            except Exception as e:
                logger.warning(f"Failed to create embedding for learning: {e}")
            
            # Generate reflection
            reflection = await self.reflection_engine.reflect_on_execution(
                user_goal, executed_steps, outcomes, user_goal_vec
            )
            
            # Update workflow success metrics
            if workflow_id and user_goal_vec is not None:
                await self._update_workflow_metrics(workflow_id, reflection)
            
            # Generate learning summary
            learning_summary = {
                "workflow_id": workflow_id,
                "goal": user_goal,
                "reflection": reflection,
                "lessons_stored": len(reflection.get("success_patterns", [])) + len(reflection.get("failure_patterns", [])),
                "overall_success": reflection.get("overall_success", False)
            }
            
            logger.info(f"Learning completed: {learning_summary['lessons_stored']} lessons stored")
            return learning_summary
            
        except Exception as e:
            logger.error(f"Learning from workflow failed: {e}")
            return {"error": str(e)}
    
    async def _update_workflow_metrics(self, workflow_id: str, reflection: Dict[str, Any]):
        """Update workflow success metrics based on reflection."""
        try:
            success_score = 1.0 if reflection.get("overall_success", False) else 0.0
            
            # Could extend this to update workflow metadata with success scores
            # For now, just log the metrics
            logger.info(f"Workflow {workflow_id} success score: {success_score}")
            
        except Exception as e:
            logger.debug(f"Failed to update workflow metrics: {e}")
    
    async def get_learning_insights(
        self, 
        user_goal: str, 
        agent_name: Optional[str] = None,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """Get learning insights for a specific goal and/or agent."""
        try:
            user_goal_vec = await create_embedding(user_goal)
            lessons = await self.store.query_lessons_similar(
                user_goal_vec, agent=agent_name, top_k=top_k
            )
            
            # Categorize lessons
            success_lessons = []
            failure_lessons = []
            
            for lesson_id, status, fix_summary, working_params, score in lessons:
                lesson_data = {
                    "lesson_id": lesson_id,
                    "score": score,
                    "fix_summary": fix_summary,
                    "working_params": working_params
                }
                
                if status == "success":
                    success_lessons.append(lesson_data)
                else:
                    failure_lessons.append(lesson_data)
            
            return {
                "user_goal": user_goal,
                "agent_filter": agent_name,
                "total_lessons": len(lessons),
                "success_lessons": success_lessons,
                "failure_lessons": failure_lessons,
                "top_success_params": success_lessons[0].get("working_params", {}) if success_lessons else {},
                "common_failures": [l["fix_summary"] for l in failure_lessons[:3]]
            }
            
        except Exception as e:
            logger.error(f"Failed to get learning insights: {e}")
            return {"error": str(e)}


class MockLLM:
    """Mock LLM for testing/fallback scenarios."""
    
    def completion_json(self, system: str, user: str) -> str:
        """Mock completion for reflection."""
        return json.dumps({
            "overall_success": True,
            "success_patterns": [],
            "failure_patterns": [],
            "workflow_improvements": ["Mock improvement"],
            "agent_recommendations": {}
        })