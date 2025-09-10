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


class Orchestrator:
    def __init__(self, store: Optional[ProceduralStore] = None) -> None:
        self.llm = ChatLLM()
        self.rag = RAGService()
        self.planner = Planner(self.llm)
        self.agents = {
            "RAGQueryAgent": RAGQueryAgent(self.rag),
            "ReportGenerationAgent": ReportGenerationAgent(self.llm),
        }
        self.store = store or ProceduralStore()
        # Debug / last-run info
        self.last_workflow_id: Optional[str] = None
        self.last_reused: bool = False
        # Controls via env (fallback defaults)
        try:
            self.procedural_top_k = int(os.getenv("PROCEDURAL_TOP_K", "5"))
        except Exception:
            self.procedural_top_k = 5
        try:
            self.procedural_reuse_threshold = float(os.getenv("PROCEDURAL_REUSE_THRESHOLD", "0.9"))
        except Exception:
            self.procedural_reuse_threshold = 0.9

    async def handle_request(self, session_id: str, user_goal: str) -> str:
        # Prepare run directory
        base_dir = os.getenv("RUNS_BASE_DIR", "runs")
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
        try:
            vec = await create_embedding(user_goal)
            recs = await self.store.query_procedural_similar(vec, max(1, self.procedural_top_k))
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
        for idx, step in enumerate(steps):
            agent = self.agents.get(step.agent)
            if not agent:
                continue
            try:
                payload = dict(step.input)
                payload.setdefault("context", context)
                # Await agent execution (agents are async)
                out = await agent.execute(session_id, payload)
                last_output = out
                context[step.agent] = out
                executed.append((step, out))
                # Write per-step trace
                try:
                    (run_dir / f"step_{idx}_{step.agent}.json").write_text(
                        json.dumps({"input": {k: v for k, v in payload.items() if k != "context"}, "output": out}, ensure_ascii=False, indent=2)
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
            # reflection summary
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
        except Exception:
            pass

        # Write final report
        try:
            (run_dir / "report.txt").write_text(final_text)
        except Exception:
            pass

        return final_text
