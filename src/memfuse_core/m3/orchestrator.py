from __future__ import annotations

"""Phase A Orchestrator (skeleton) for M3.

Implements a minimal Planner and two agents (RAGQueryAgent, ReportGenerationAgent)
to support unit tests and incremental integration. Procedural memory reuse and
lessons storage will be wired in later steps against ProceduralStore.
"""

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

from ..llm.chat import ChatLLM
from ..rag.rag_service import RAGService
from ..procedural.store import ProceduralStore
from ..utils.embeddings import create_embedding
from ..utils.global_config_manager import get_global_config_manager
import os
import time
from pathlib import Path


@dataclass
class PlanStep:
    agent: str
    input: Dict[str, Any]


class Planner:
    def __init__(self, llm: ChatLLM) -> None:
        self.llm = llm

    def plan(self, user_goal: str) -> List[PlanStep]:
        system = (
            "You are a task planner. Decompose the high-level goal into ordered steps.\n"
            "Available agents: RAGQueryAgent, ReportGenerationAgent.\n"
            "Return strict JSON: {\"steps\":[{\"agent\":<name>,\"input\":{...}}]}\n"
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
    def __init__(self, rag: RAGService) -> None:
        self.rag = rag

    async def execute(self, session_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        query = str(payload.get("query") or payload.get("question") or "").strip()
        if not query:
            return {"error": "query required"}
        # try to extract lightweight history from payload.context if present
        hist = None
        ctx = payload.get("context") if isinstance(payload, dict) else None
        if isinstance(ctx, dict):
            hist = ctx.get("_history_messages")
            if not isinstance(hist, list):
                hist = None
        ans = await self.rag.chat(session_id, query, history_messages=hist or [])
        return {"answer": ans}


class ReportGenerationAgent:
    def __init__(self, llm: ChatLLM) -> None:
        self.llm = llm

    async def execute(self, session_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        points = payload.get("points") or payload.get("data") or payload
        text = json.dumps(points, ensure_ascii=False)
        system = "You are a precise report writer. Summarize inputs into a concise brief."
        try:
            res = self.llm.chat(system, [{"role": "user", "content": text}])
            return {"report": res}
        except Exception as e:
            return {"report": f"[offline] {text[:500]}", "note": str(e)}


class WebSearchAgent:
    def __init__(self) -> None:
        pass

    async def execute(self, session_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        # Placeholder agent: disabled by default to avoid network dependency
        return {"error": "web search disabled in Phase A"}


class ShellCommandAgent:
    def __init__(self) -> None:
        self.allowed = str(os.getenv("ALLOW_SHELL_AGENT", "false")).lower() in ("1", "true", "yes")

    async def execute(self, session_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.allowed:
            return {"error": "shell agent disabled"}
        cmd = payload.get("cmd") or payload.get("command")
        if not cmd or not isinstance(cmd, str):
            return {"error": "cmd required"}
        # Extremely restricted: only allow safe commands like 'echo'
        parts = cmd.strip().split()
        if not parts or parts[0] not in ("echo",):
            return {"error": "command not allowed"}
        try:
            import subprocess
            proc = subprocess.run(parts, capture_output=True, text=True, timeout=5)
            return {"exit": proc.returncode, "output": (proc.stdout or proc.stderr)[:1000]}
        except Exception as e:
            return {"error": f"shell exec failed: {e}"}


class DatabaseQueryAgent:
    def __init__(self) -> None:
        pass

    async def execute(self, session_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        # Minimal placeholder using DatabaseService if available; safe select-only
        try:
            from ..services.database_service import DatabaseService
        except Exception:
            return {"error": "database service unavailable"}
        query = str(payload.get("query") or "").strip()
        if not query or not query.lower().startswith("select"):
            return {"error": "only SELECT allowed"}
        try:
            db = await DatabaseService.get_instance()
            rows = await db.backend.execute(query, tuple())  # type: ignore[attr-defined]
            # Limit results for safety
            if isinstance(rows, list) and rows and isinstance(rows[0], dict):
                headers = list(rows[0].keys())
            else:
                headers = []
            return {"headers": headers[:50], "rows": rows[:20] if isinstance(rows, list) else []}
        except Exception as e:
            return {"error": f"db query failed: {e}"}


class Orchestrator:
    def __init__(self, store: Optional[ProceduralStore] = None) -> None:
        self.llm = ChatLLM()
        self.rag = RAGService()
        self.planner = Planner(self.llm)
        self.agents = {
            "RAGQueryAgent": RAGQueryAgent(self.rag),
            "ReportGenerationAgent": ReportGenerationAgent(self.llm),
            "WebSearchAgent": WebSearchAgent(),
            "ShellCommandAgent": ShellCommandAgent(),
            "DatabaseQueryAgent": DatabaseQueryAgent(),
        }
        self.store = store or ProceduralStore()
        # Debug / last-run info
        self.last_workflow_id: Optional[str] = None
        self.last_reused: bool = False
        self.last_plan_steps: List[PlanStep] = []
        # Controls via env (fallback defaults)
        self.procedural_top_k = 5
        self.procedural_reuse_threshold = 0.9
        self.planner_max_attempts = 3
        self.runs_base_dir = os.getenv("RUNS_BASE_DIR", "runs")
        try:
            cfg = get_global_config_manager()
            # Read from global config if available
            self.procedural_top_k = int(cfg.get("memory.layers.m3.procedural_top_k", self.procedural_top_k))
            self.procedural_reuse_threshold = float(cfg.get("memory.layers.m3.procedural_reuse_threshold", self.procedural_reuse_threshold))
            self.planner_max_attempts = int(cfg.get("memory.layers.m3.planner_max_attempts", self.planner_max_attempts))
            self.runs_base_dir = str(cfg.get("memory.layers.m3.runs_base_dir", self.runs_base_dir))
        except Exception:
            # Fallback to environment variables
            try:
                self.procedural_top_k = int(os.getenv("PROCEDURAL_TOP_K", str(self.procedural_top_k)))
            except Exception:
                pass
            try:
                self.procedural_reuse_threshold = float(os.getenv("PROCEDURAL_REUSE_THRESHOLD", str(self.procedural_reuse_threshold)))
            except Exception:
                pass

    async def handle_request(self, session_id: str, user_goal: str) -> str:
        # Prepare run directory
        base_dir = self.runs_base_dir or os.getenv("RUNS_BASE_DIR", "runs")
        run_dir = Path(base_dir) / time.strftime('%Y%m%d_%H%M%S') / session_id
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "input.json").write_text(json.dumps({"session_id": session_id, "goal": user_goal}, ensure_ascii=False, indent=2))
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
            recs = await self.store.query_procedural_similar(vec, max(1, self.procedural_top_k)) if vec is not None else []
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
            (run_dir / "plan.json").write_text(
                json.dumps({"steps": [{"agent": s.agent, "input": s.input} for s in steps]}, ensure_ascii=False, indent=2)
            )
        except Exception:
            pass

        context: Dict[str, Any] = {}
        last_output: Dict[str, Any] = {}
        executed: List[Tuple[PlanStep, Dict[str, Any]]] = []
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
        for idx, step in enumerate(steps):
            agent = self.agents.get(step.agent)
            if not agent:
                continue
            try:
                t0 = time.perf_counter()
                payload = dict(step.input)
                payload.setdefault("context", context)
                # Lessons-informed parameterization: try to seed payload from prior successful params for this agent
                success_params: List[Dict[str, Any]] = []
                avoid_patterns: List[str] = []
                if goal_vec is not None:
                    try:
                        agent_lessons = await self.store.query_lessons_similar(goal_vec, agent=step.agent, top_k=5)
                        for _lid, st, fx, wp, _sc in agent_lessons:
                            if st == 'success' and isinstance(wp, dict):
                                success_params.append(wp)
                            elif st == 'fail' and fx:
                                avoid_patterns.append(str(fx))
                        if success_params:
                            # Merge first successful params into payload (do not override provided values)
                            for k, v in success_params[0].items():
                                if k not in payload:
                                    payload[k] = v
                    except Exception:
                        pass
                # Attempt execution with simple success heuristic
                attempts = 0
                success = False
                out: Dict[str, Any] = {}
                for attempt in range(max(1, int(self.planner_max_attempts or 1))):
                    attempts = attempt + 1
                    try:
                        out = await agent.execute(session_id, payload)
                    except Exception:
                        out = {"error": "agent execution failed"}
                    # Heuristic success
                    def _ok(a: str, o: Dict[str, Any]) -> bool:
                        if not isinstance(o, dict):
                            return False
                        if o.get("error"):
                            return False
                        if a == "RAGQueryAgent":
                            return bool(o.get("answer"))
                        if a == "ReportGenerationAgent":
                            return bool(o.get("report"))
                        return True
                    success = _ok(step.agent, out)
                    if success:
                        break
                last_output = out
                context[step.agent] = out
                duration_ms = int((time.perf_counter() - t0) * 1000)
                executed.append((step, out))
                self.last_step_outcomes.append({
                    "agent": step.agent,
                    "success": bool(success),
                    "attempts": attempts,
                    "duration_ms": duration_ms,
                    "error": (str(out.get("error"))[:200] if isinstance(out, dict) and out.get("error") else None),
                })
                # Write per-step trace
                try:
                    (run_dir / f"step_{idx}_{step.agent}.json").write_text(
                        json.dumps({
                            "input": {k: v for k, v in payload.items() if k != "context"},
                            "output": out,
                            "attempts": attempts,
                            "success": success,
                            "lessons": {"success_examples": len(success_params), "avoid_patterns": len(avoid_patterns)},
                            "duration_ms": duration_ms,
                        }, ensure_ascii=False, indent=2)
                    )
                except Exception:
                    pass
            except Exception:
                continue
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
                    "plan": [{"agent": s.agent, "input": s.input} for (s, _o) in executed],
                    "result_keys": list(last_output.keys()),
                }
                import uuid as _uuid
                wid_new = str(_uuid.uuid4())
                await self.store.upsert_procedural_workflow(wid_new, vec, workflow)
                self.last_workflow_id = wid_new
                try:
                    (run_dir / "workflow.json").write_text(json.dumps({"workflow_id": wid_new, "workflow": workflow}, ensure_ascii=False, indent=2))
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
            (run_dir / "reflection.json").write_text(
                json.dumps({"summary": summary, "steps": steps_ref}, ensure_ascii=False, indent=2)
            )
        except Exception:
            pass

        # Write final report
        try:
            (run_dir / "report.txt").write_text(final_text)
        except Exception:
            pass

        return final_text
