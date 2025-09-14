"""Phase A Orchestrator for M3.

Implements a minimal Planner and basic agents to support M3 workflow processing.
Handles workflow reuse, planning, execution, and lesson storage.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

# Import these modules lazily to avoid circular dependencies
# from ..llm.chat import ChatLLM  
# from ..rag.rag_service import RAGService
from ..procedural.store import ProceduralStore
from ..utils.embeddings import create_embedding
from ..utils.global_config_manager import get_global_config_manager
from .config import get_m3_config
from .executor import AgentExecutor
from .reflection import LearningSystem
from .types import PlanStep


class Planner:
    """Plans complex tasks into ordered steps."""
    
    def __init__(self, llm=None) -> None:
        self.llm = llm or self._create_default_llm()
    
    def _create_default_llm(self):
        """Create default LLM instance."""
        try:
            from ..llm.chat import ChatLLM
            return ChatLLM()
        except Exception as e:
            logger.warning(f"Failed to create ChatLLM: {e}")
            return MockLLM()

    def plan(self, user_goal: str) -> List[PlanStep]:
        """Decompose a high-level goal into ordered steps."""
        system = (
            "You are a task planner. Decompose the high-level goal into ordered steps.\n"
            "Available agents: RAGQueryAgent, WebSearchAgent, DatabaseQueryAgent, ShellCommandAgent, ReportGenerationAgent.\n"
            "Agent descriptions:\n"
            "- RAGQueryAgent: Query internal knowledge base and documents\n"
            "- WebSearchAgent: Search the web (DuckDuckGo, arXiv) for current information\n"
            "- DatabaseQueryAgent: Query database using natural language to SQL conversion\n"
            "- ShellCommandAgent: Execute safe shell commands like ripgrep (rg)\n"
            "- ReportGenerationAgent: Generate final reports and summaries\n"
            "Return strict JSON: {\"steps\":[{\"agent\":<name>,\"input\":{...}}]}\n"
            "Keep 3-6 steps. Use RAG for internal data, WebSearch for current info, Database for structured queries.\n"
        )
        user = f"Goal: {user_goal}\nProduce steps now."
        
        raw = self.llm.completion_json(system, user)
        
        try:
            data = json.loads(raw or "{}")
        except Exception:
            data = {}
        
        steps = data.get("steps", []) if isinstance(data, dict) else []
        plan: List[PlanStep] = []
        
        for st in steps:
            if not isinstance(st, dict):
                continue
            agent = str(st.get("agent", "")).strip()
            if not agent:
                continue
            payload = st.get("input") or {}
            if not isinstance(payload, dict):
                payload = {}
            plan.append(PlanStep(agent=agent, input=payload))
        
        if plan:
            return plan
        
        # Fallback default plan
        return [
            PlanStep(agent="RAGQueryAgent", input={"query": user_goal}),
            PlanStep(agent="ReportGenerationAgent", input={}),
        ]


class RAGQueryAgent:
    """Agent that uses RAG service to answer queries."""
    
    def __init__(self, rag=None) -> None:
        self.rag = rag or self._create_default_rag()
    
    def _create_default_rag(self):
        """Create default RAG service instance."""
        try:
            from ..rag.rag_service import RAGService
            return RAGService()
        except Exception as e:
            logger.warning(f"Failed to create RAGService: {e}")
            return MockRAGService()

    async def execute(self, session_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a RAG query."""
        query = str(payload.get("query") or payload.get("question") or "").strip()
        if not query:
            return {"error": "query required"}
        
        # Try to extract lightweight history from payload.context if present
        hist = None
        ctx = payload.get("context") if isinstance(payload, dict) else None
        if isinstance(ctx, dict):
            hist = ctx.get("_history_messages")
            if not isinstance(hist, list):
                hist = None
        
        ans = await self.rag.chat(session_id, query, history_messages=hist or [])
        return {"answer": ans}


class ReportGenerationAgent:
    """Agent that generates reports from data."""
    
    def __init__(self, llm=None) -> None:
        self.llm = llm or self._create_default_llm()
    
    def _create_default_llm(self):
        """Create default LLM instance."""
        try:
            from ..llm.chat import ChatLLM
            return ChatLLM()
        except Exception as e:
            logger.warning(f"Failed to create ChatLLM: {e}")
            return MockLLM()

    async def execute(self, session_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a report from the provided data."""
        points = payload.get("points") or payload.get("data") or payload
        text = json.dumps(points, ensure_ascii=False)
        system = "You are a precise report writer. Summarize inputs into a concise brief."
        
        try:
            res = self.llm.chat(system, [{"role": "user", "content": text}])
            return {"report": res}
        except Exception as e:
            return {"report": f"[offline] {text[:500]}", "note": str(e)}


class Orchestrator:
    """Main M3 orchestrator that handles workflow reuse, planning, and execution."""
    
    def __init__(self, store: Optional[ProceduralStore] = None) -> None:
        # Create components with lazy initialization
        self.llm = self._create_default_llm()
        self.rag = self._create_default_rag()
        self.planner = Planner(self.llm)
        # Import agents
        from .agents.websearch import WebSearchAgent
        from .agents.database import DatabaseQueryAgent
        from .agents.shell import ShellCommandAgent
        
        self.agents = {
            "RAGQueryAgent": RAGQueryAgent(self.rag),
            "ReportGenerationAgent": ReportGenerationAgent(self.llm),
            "WebSearchAgent": WebSearchAgent(),
            "DatabaseQueryAgent": DatabaseQueryAgent(self.llm),
            "ShellCommandAgent": ShellCommandAgent(),
        }
        self.store = store or ProceduralStore()
        
        # Initialize learning system
        self.learning_system = LearningSystem(self.store)
        
        # Debug / last-run info
        self.last_workflow_id: Optional[str] = None
        self.last_reused: bool = False
        self.last_plan_steps: List[PlanStep] = []
        self.last_reflection: Optional[Dict[str, Any]] = None
        
        # Load M3 configuration
        self.config = get_m3_config()
        self.procedural_top_k = self.config.max_workflow_reuse_candidates
        self.procedural_reuse_threshold = self.config.workflow_reuse_threshold
        self.planner_max_attempts = self.config.max_agent_retries
        self.runs_base_dir = os.getenv("RUNS_BASE_DIR", "runs")
    
    def _create_default_llm(self):
        """Create default LLM instance."""
        try:
            from ..llm.chat import ChatLLM
            return ChatLLM()
        except Exception as e:
            logger.warning(f"Failed to create ChatLLM: {e}")
            return MockLLM()
    
    def _create_default_rag(self):
        """Create default RAG service instance."""
        try:
            from ..rag.rag_service import RAGService
            return RAGService()
        except Exception as e:
            logger.warning(f"Failed to create RAGService: {e}")
            return MockRAGService()

    async def handle_request(
        self,
        session_id: str,
        user_goal: str,
        workflow_name: Optional[str] = None,
        history_messages: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Handle an M3 request with workflow reuse, planning, and execution."""
        # Prepare run directory
        base_dir = self.runs_base_dir or os.getenv("RUNS_BASE_DIR", "runs")
        run_dir = Path(base_dir) / time.strftime('%Y%m%d_%H%M%S') / session_id
        
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "input.json").write_text(json.dumps({
                "session_id": session_id, 
                "goal": user_goal
            }, ensure_ascii=False, indent=2))
        except Exception:
            pass
        
        # Try reuse first (soft-fail if any issue)
        steps: List[PlanStep]
        wid_reused: Optional[str] = None
        self.last_workflow_id = None
        self.last_reused = False
        self.last_plan_steps = []
        goal_vec = None
        
        try:
            goal_vec = await create_embedding(user_goal)
        except Exception:
            goal_vec = None
        
        try:
            vec = goal_vec
            recs = await self.store.query_procedural_similar(
                vec, max(1, self.procedural_top_k)
            ) if vec is not None else []
            
            if recs:
                wid, wf, score = recs[0]
                if score >= self.procedural_reuse_threshold:
                    plan_list = wf.get("plan", []) if isinstance(wf, dict) else []
                    cand = [
                        PlanStep(agent=str(s.get("agent", "")), input=s.get("input") or {})
                        for s in plan_list
                        if isinstance(s, dict) and str(s.get("agent", "")).strip()
                    ]
                    if cand:
                        steps = cand
                        wid_reused = wid
                        self.last_reused = True
                    else:
                        steps = self.planner.plan(user_goal)
                else:
                    steps = self.planner.plan(user_goal)
            else:
                steps = self.planner.plan(user_goal)
        except Exception:
            steps = self.planner.plan(user_goal)

        # Expose plan steps for external logging use
        try:
            self.last_plan_steps = steps[:]
        except Exception:
            self.last_plan_steps = []

        # Persist plan
        try:
            (run_dir / "plan.json").write_text(json.dumps({
                "steps": [{"agent": s.agent, "input": s.input} for s in steps]
            }, ensure_ascii=False, indent=2))
        except Exception:
            pass

        context: Dict[str, Any] = {}
        if history_messages is not None:
            try:
                context["_history_messages"] = history_messages
            except Exception:
                pass
        if workflow_name:
            try:
                context["_workflow_name"] = workflow_name
            except Exception:
                pass
        
        last_output: Dict[str, Any] = {}
        self.last_step_outcomes: List[Dict[str, Any]] = []

        # Pre-lessons (global) to help parameterization and observability
        try:
            pre_lessons: Dict[str, Any] = {}
            if goal_vec is not None:
                lessons_list = await self.store.query_lessons_similar(goal_vec, agent=None, top_k=5)
                pre_lessons = {
                    "total": len(lessons_list),
                    "success": [
                        {"lesson_id": lid, "fix_summary": fx, "working_params": wp, "score": sc}
                        for (lid, st, fx, wp, sc) in lessons_list if st == 'success'
                    ],
                    "fail": [
                        {"lesson_id": lid, "fix_summary": fx, "working_params": wp, "score": sc}
                        for (lid, st, fx, wp, sc) in lessons_list if st == 'fail'
                    ],
                }
            context["_pre_lessons"] = pre_lessons
            
            try:
                (run_dir / "pre_lessons.json").write_text(json.dumps(pre_lessons, ensure_ascii=False, indent=2))
            except Exception:
                pass
        except Exception:
            pass
        
        # Use AgentExecutor to run steps
        executor = AgentExecutor(self.agents, self.store, planner_max_attempts=self.planner_max_attempts)
        executed, outcomes = await executor.execute_steps(
            session_id=session_id,
            steps=steps,
            context=context,
            run_dir=run_dir,
            user_goal_vec=goal_vec,
            user_goal=user_goal,
        )
        self.last_step_outcomes = outcomes
        
        if executed:
            last_output = executed[-1][1]
        
        # Final text
        if "report" in last_output:
            final_text = str(last_output.get("report") or "")
        elif "answer" in last_output:
            final_text = str(last_output.get("answer") or "")
        else:
            final_text = json.dumps({"result": context}, ensure_ascii=False)

        # Persist usage/workflow/lessons (soft-fail)
        try:
            vec = await create_embedding(user_goal)
            if self.last_reused and wid_reused:
                await self.store.bump_procedural_usage(wid_reused, 1)
                self.last_workflow_id = wid_reused
                try:
                    (run_dir / "reused.json").write_text(json.dumps({"workflow_id": wid_reused}, ensure_ascii=False, indent=2))
                except Exception:
                    pass
            else:
                workflow = {
                    "goal": user_goal,
                    "workflow_name": workflow_name,
                    "plan": [{"agent": s.agent, "input": s.input} for (s, _o) in executed],
                    "result_keys": list(last_output.keys()),
                }
                wid_new = str(uuid.uuid4())
                await self.store.upsert_procedural_workflow(wid_new, vec, workflow)
                self.last_workflow_id = wid_new
                try:
                    (run_dir / "workflow.json").write_text(json.dumps({
                        "workflow_id": wid_new, 
                        "workflow": workflow
                    }, ensure_ascii=False, indent=2))
                except Exception:
                    pass
            
            # lessons
            for s, o in executed:
                if isinstance(o, dict) and (o.get("report") or o.get("answer")):
                    await self.store.insert_lesson(vec, user_goal, s.agent, "success", None, "", s.input)
                elif isinstance(o, dict) and o.get("error"):
                    await self.store.insert_lesson(vec, user_goal, s.agent, "fail", str(o.get("error"))[:500], "", s.input)
        except Exception:
            pass

        # reflection summary: write regardless of DB status
        try:
            def _ok(out: Dict[str, Any]) -> bool:
                return bool(isinstance(out, dict) and (out.get("report") or out.get("answer")) and not out.get("error"))

            summary = {
                "total_steps": len(executed),
                "success": sum(1 for _s, o in executed if _ok(o)),
                "fail": sum(1 for _s, o in executed if not _ok(o)),
            }
            steps_ref = [
                {
                    "agent": s.agent,
                    "success": _ok(o),
                    "keys": list(o.keys()) if isinstance(o, dict) else [],
                }
                for s, o in executed
            ]
            (run_dir / "reflection.json").write_text(json.dumps({
                "summary": summary, 
                "steps": steps_ref
            }, ensure_ascii=False, indent=2))
        except Exception:
            pass

        # Post-execution reflection and learning
        try:
            learning_summary = await self.learning_system.learn_from_workflow(
                user_goal, executed, outcomes, self.last_workflow_id
            )
            self.last_reflection = learning_summary.get("reflection", {})
            
            # Write reflection to run directory
            try:
                (run_dir / "learning_reflection.json").write_text(
                    json.dumps(learning_summary, ensure_ascii=False, indent=2)
                )
            except Exception:
                pass
                
        except Exception as e:
            logger.debug(f"Post-execution learning failed: {e}")

        # Write final report
        try:
            (run_dir / "report.txt").write_text(final_text)
        except Exception:
            pass

        return final_text


class MockLLM:
    """Mock LLM for testing/fallback scenarios."""
    
    def completion_json(self, system: str, user: str) -> str:
        """Mock JSON completion."""
        return '{"steps": [{"agent": "RAGQueryAgent", "input": {"query": "mock query"}}, {"agent": "ReportGenerationAgent", "input": {}}]}'
    
    async def chat(self, system: str, messages: List[Dict[str, str]]) -> str:
        """Mock chat completion."""
        return "This is a mock response for M3 testing."


class MockRAGService:
    """Mock RAG service for testing/fallback scenarios."""
    
    async def chat(self, session_id: str, query: str, history_messages=None) -> str:
        """Mock RAG chat."""
        return f"Mock RAG response for query: {query}"