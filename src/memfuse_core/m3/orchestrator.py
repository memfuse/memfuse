from __future__ import annotations

"""Phase A Orchestrator (skeleton) for M3.

Implements a minimal Planner and two agents (RAGQueryAgent, ReportGenerationAgent)
to support unit tests and incremental integration. Procedural memory reuse and
lessons storage will be wired in later steps against ProceduralStore.
"""

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from ..llm.chat import ChatLLM
from ..rag.rag_service import RAGService


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

    def execute(self, session_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        query = str(payload.get("query") or payload.get("question") or "").strip()
        if not query:
            return {"error": "query required"}
        ans = self.rag.chat(session_id, query)
        return {"answer": ans}


class ReportGenerationAgent:
    def __init__(self, llm: ChatLLM) -> None:
        self.llm = llm

    def execute(self, session_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        points = payload.get("points") or payload.get("data") or payload
        text = json.dumps(points, ensure_ascii=False)
        system = "You are a precise report writer. Summarize inputs into a concise brief."
        try:
            res = self.llm.chat(system, [{"role": "user", "content": text}])
            return {"report": res}
        except Exception as e:
            return {"report": f"[offline] {text[:500]}", "note": str(e)}


class Orchestrator:
    def __init__(self) -> None:
        self.llm = ChatLLM()
        self.rag = RAGService()
        self.planner = Planner(self.llm)
        self.agents = {
            "RAGQueryAgent": RAGQueryAgent(self.rag),
            "ReportGenerationAgent": ReportGenerationAgent(self.llm),
        }

    def handle_request(self, session_id: str, user_goal: str) -> str:
        steps = self.planner.plan(user_goal)
        context: Dict[str, Any] = {}
        last_output: Dict[str, Any] = {}
        for step in steps:
            agent = self.agents.get(step.agent)
            if not agent:
                continue
            try:
                payload = dict(step.input)
                payload.setdefault("context", context)
                out = agent.execute(session_id, payload)
                last_output = out
                context[step.agent] = out
            except Exception:
                continue
        # Final text response
        if "report" in last_output:
            return str(last_output.get("report") or "")
        if "answer" in last_output:
            return str(last_output.get("answer") or "")
        # fallback to simple stitched summary
        return json.dumps({"result": context}, ensure_ascii=False)

