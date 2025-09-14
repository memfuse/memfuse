"""Enhanced Agent executor for M3 workflows with intelligent retry and parameter seeding."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from ..procedural.store import ProceduralStore
from .types import PlanStep


class AgentExecutor:
    """Enhanced executor that executes workflow steps with intelligent retry and parameter seeding."""
    
    def __init__(
        self, 
        agents: Dict[str, Any], 
        store: ProceduralStore, 
        planner_max_attempts: int = 3
    ) -> None:
        self.agents = agents
        self.store = store
        self.planner_max_attempts = max(1, planner_max_attempts)
        
        # Create LLM for parameter proposal
        self.llm = self._create_default_llm()
    
    def _create_default_llm(self):
        """Create default LLM instance for parameter proposal."""
        try:
            from ..llm.chat import ChatLLM
            return ChatLLM()
        except Exception as e:
            logger.warning(f"Failed to create ChatLLM for AgentExecutor: {e}")
            return MockLLM()

    @staticmethod
    def _success_heuristic(agent_name: str, output: Dict[str, Any]) -> bool:
        """Determine if agent execution was successful using heuristics."""
        if not isinstance(output, dict) or output.get("error"):
            return False
        
        # Agent-specific success criteria
        if agent_name == "RAGQueryAgent":
            return bool(output.get("answer"))
        elif agent_name == "ReportGenerationAgent":
            return bool(output.get("report"))
        elif agent_name == "WebSearchAgent":
            results = output.get("results", [])
            return isinstance(results, list) and len(results) > 0
        elif agent_name == "DatabaseQueryAgent":
            return "rows" in output or "columns" in output
        elif agent_name == "ShellCommandAgent":
            return output.get("exit_code") == 0 or bool(output.get("output"))
        
        # Default: success if no error
        return True
    
    async def _seed_params_from_lessons(
        self,
        agent_name: str, 
        user_goal: str,
        user_goal_vec: Optional[List[float]],
        payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Seed parameters from successful lessons."""
        if user_goal_vec is None:
            return payload
        
        try:
            # Query lessons for this agent and goal
            lessons = await self.store.query_lessons_similar(
                user_goal_vec, agent=agent_name, top_k=5
            )
            
            # Merge successful parameters
            for lesson_id, status, fix_summary, working_params, score in lessons:
                if status == 'success' and isinstance(working_params, dict):
                    # Add missing parameters from successful attempts
                    for key, value in working_params.items():
                        if key not in payload and key != "context":
                            payload[key] = value
                    break  # Use first successful lesson
                    
        except Exception as e:
            logger.debug(f"Failed to seed parameters from lessons: {e}")
        
        return payload
    
    async def _propose_input(
        self, 
        agent_name: str, 
        user_goal: str, 
        context: Dict[str, Any], 
        prior_attempt: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Use LLM to propose input parameters for an agent."""
        schema_hints = {
            "RAGQueryAgent": {"query": "string (derived from goal if missing)"},
            "DatabaseQueryAgent": {"request": "string (NL to SQL)", "schema_hint": "string?"},
            "WebSearchAgent": {"query": "string", "last_days": "int?", "max_results": "int?"},
            "ReportGenerationAgent": {"points": "object?", "data": "object?"},
            "ShellCommandAgent": {"cmd": "rg|echo", "pattern": "string", "path": "string?"},
        }
        
        schema_hint = schema_hints.get(agent_name, {})
        
        system = (
            "You are an autonomous agent parameter generator.\n"
            "Given a high-level goal and context, propose specific input parameters.\n"
            "Return ONLY a JSON object with the required parameters.\n"
            "Do NOT include explanations or additional text.\n"
        )
        
        user_prompt = json.dumps({
            "agent": agent_name,
            "goal": user_goal,
            "schema_hint": schema_hint,
            "context_keys": list(context.keys())[-5:],  # Last 5 context keys
            "prior_attempt": prior_attempt or {}
        }, ensure_ascii=False)
        
        try:
            raw = self.llm.completion_json(system, user_prompt)
            data = json.loads(raw or '{}')
            
            if isinstance(data, dict):
                return data
                
        except Exception as e:
            logger.debug(f"Failed to propose input via LLM: {e}")
        
        # Fallback parameter proposals
        if agent_name == "RAGQueryAgent":
            return {"query": user_goal}
        elif agent_name == "WebSearchAgent":
            return {"query": user_goal, "max_results": 10}
        elif agent_name == "ReportGenerationAgent":
            return {"points": {"title": user_goal, "context": list(context.keys())[-3:]}}
        
        return {}
    
    async def _execute_step_with_retries(
        self,
        step: PlanStep,
        session_id: str,
        user_goal: str,
        user_goal_vec: Optional[List[float]],
        context: Dict[str, Any],
        max_attempts: int
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Execute a single step with intelligent retries."""
        agent = self.agents.get(step.agent)
        if not agent:
            return {"error": f"Agent {step.agent} not found"}, []
        
        attempts_log = []
        payload = step.input.copy()
        
        # Seed parameters from lessons
        payload = await self._seed_params_from_lessons(
            step.agent, user_goal, user_goal_vec, payload
        )
        
        for attempt in range(1, max_attempts + 1):
            start_time = time.time()
            
            # Auto-propose parameters if missing critical fields
            needs_proposal = (
                (step.agent == "RAGQueryAgent" and not payload.get("query")) or
                (step.agent == "DatabaseQueryAgent" and not (payload.get("request") or payload.get("query"))) or
                (step.agent == "WebSearchAgent" and not payload.get("query")) or
                (step.agent == "ReportGenerationAgent" and not any(k in payload for k in ("points", "data")))
            )
            
            if needs_proposal:
                prior_attempt = attempts_log[-1] if attempts_log else None
                proposed = await self._propose_input(
                    step.agent, user_goal, context, prior_attempt
                )
                # Merge proposed parameters
                for k, v in proposed.items():
                    if k not in payload and k != "context":
                        payload[k] = v
            
            # Add context to payload
            exec_payload = payload.copy()
            exec_payload["context"] = context
            
            try:
                # Execute agent
                outcome = await agent.execute(session_id, exec_payload)
                elapsed = time.time() - start_time
                
                # Check success
                success = self._success_heuristic(step.agent, outcome)
                
                attempt_record = {
                    "attempt": attempt,
                    "input": {k: v for k, v in exec_payload.items() if k != "context"},
                    "success": success,
                    "elapsed_sec": elapsed,
                    "output_keys": list(outcome.keys()) if isinstance(outcome, dict) else [],
                    "error": outcome.get("error") if isinstance(outcome, dict) else None
                }
                attempts_log.append(attempt_record)
                
                if success:
                    # Store successful lesson
                    try:
                        if user_goal_vec is not None:
                            await self.store.insert_lesson(
                                user_goal_vec, user_goal, step.agent, "success", 
                                None, "", attempt_record["input"]
                            )
                    except Exception as e:
                        logger.debug(f"Failed to store success lesson: {e}")
                    
                    return outcome, attempts_log
                
                # If not successful and not last attempt, wait briefly
                if attempt < max_attempts:
                    await asyncio.sleep(min(1.0, 0.2 * attempt))
                    
            except Exception as e:
                elapsed = time.time() - start_time
                outcome = {"error": str(e)}
                
                attempt_record = {
                    "attempt": attempt,
                    "input": {k: v for k, v in exec_payload.items() if k != "context"},
                    "success": False,
                    "elapsed_sec": elapsed,
                    "error": str(e)
                }
                attempts_log.append(attempt_record)
        
        # All attempts failed - store failure lesson
        try:
            if user_goal_vec is not None:
                last_error = attempts_log[-1].get("error", "Unknown error") if attempts_log else "No attempts"
                await self.store.insert_lesson(
                    user_goal_vec, user_goal, step.agent, "fail",
                    str(last_error)[:500], "", 
                    attempts_log[-1].get("input", {}) if attempts_log else {}
                )
        except Exception as e:
            logger.debug(f"Failed to store failure lesson: {e}")
        
        return outcome, attempts_log

    async def execute_steps(
        self,
        session_id: str,
        steps: List[PlanStep],
        context: Dict[str, Any],
        run_dir: Optional[Path] = None,
        user_goal_vec: Optional[List[float]] = None,
        user_goal: str = ""
    ) -> Tuple[List[Tuple[PlanStep, Dict[str, Any]]], List[Dict[str, Any]]]:
        """Execute a list of workflow steps with intelligent retry and parameter seeding."""
        executed: List[Tuple[PlanStep, Dict[str, Any]]] = []
        outcomes: List[Dict[str, Any]] = []
        
        for i, step in enumerate(steps):
            logger.info(f"Executing step {i+1}/{len(steps)}: {step.agent}")
            
            try:
                # Execute step with retries
                outcome, attempts_log = await self._execute_step_with_retries(
                    step, session_id, user_goal, user_goal_vec, context, self.planner_max_attempts
                )
                
                executed.append((step, outcome))
                
                # Create outcome summary
                outcome_summary = {
                    "agent": step.agent,
                    "success": self._success_heuristic(step.agent, outcome),
                    "attempts": len(attempts_log),
                    "total_time": sum(att.get("elapsed_sec", 0) for att in attempts_log),
                    "final_error": outcome.get("error") if isinstance(outcome, dict) else None
                }
                outcomes.append(outcome_summary)
                
                # Update context with results for next steps
                if isinstance(outcome, dict):
                    for key, value in outcome.items():
                        if key not in ["error"]:
                            context[f"step_{i}_{key}"] = value
                
                # Log detailed step result
                if run_dir:
                    try:
                        step_file = run_dir / f"step_{i:02d}_{step.agent}.json"
                        detailed_log = {
                            "step": {"agent": step.agent, "input": step.input},
                            "outcome": outcome,
                            "attempts_log": attempts_log,
                            "outcome_summary": outcome_summary
                        }
                        step_file.write_text(json.dumps(detailed_log, ensure_ascii=False, indent=2))
                    except Exception as e:
                        logger.debug(f"Failed to write step log: {e}")
                
            except Exception as e:
                logger.error(f"Error executing step {step.agent}: {e}")
                outcome = {"error": str(e)}
                executed.append((step, outcome))
                outcomes.append({
                    "agent": step.agent,
                    "success": False,
                    "attempts": 0,
                    "error": str(e)
                })
        
        return executed, outcomes


class MockLLM:
    """Mock LLM for testing/fallback scenarios."""
    
    def completion_json(self, system: str, user: str) -> str:
        """Mock completion for parameter proposal."""
        return '{"query": "mock query"}'