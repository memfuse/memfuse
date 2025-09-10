from __future__ import annotations

"""AgentExecutor abstraction for M3 Orchestrator.

Provides a reusable component to execute planned steps with:
- lessons-informed parameter seeding
- retry with simple success heuristics
- per-step artifact writing
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Callable

from loguru import logger

from .types import PlanStep


class AgentExecutor:
    def __init__(
        self,
        agents: Dict[str, Any],
        store: Any,  # ProceduralStore-like
        planner_max_attempts: int = 3,
    ) -> None:
        self.agents = agents
        self.store = store
        self.planner_max_attempts = max(1, int(planner_max_attempts or 1))

    @staticmethod
    def _success_heuristic(agent_name: str, output: Dict[str, Any]) -> bool:
        if not isinstance(output, dict) or output.get("error"):
            return False
        if agent_name == "RAGQueryAgent":
            return bool(output.get("answer"))
        if agent_name == "ReportGenerationAgent":
            return bool(output.get("report"))
        if agent_name == "WebSearchAgent":
            res = output.get("results")
            return isinstance(res, list) and len(res) > 0
        if agent_name == "DatabaseQueryAgent":
            return ("rows" in output) or ("headers" in output)
        if agent_name == "ShellCommandAgent":
            return output.get("exit") == 0 or bool(output.get("output"))
        return True

    async def _seed_params_from_lessons(
        self,
        agent_name: str,
        user_goal_vec: Optional[List[float]],
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        if user_goal_vec is None:
            return payload
        try:
            lessons = await self.store.query_lessons_similar(user_goal_vec, agent=agent_name, top_k=5)
        except Exception:
            return payload
        # Merge first success params into payload if missing
        try:
            for _lid, status, _fix, wparams, _sc in lessons:
                if status == 'success' and isinstance(wparams, dict):
                    for k, v in wparams.items():
                        if k not in payload:
                            payload[k] = v
                    break
        except Exception:
            pass
        return payload

    async def execute_steps(
        self,
        session_id: str,
        steps: List[PlanStep],
        context: Dict[str, Any],
        run_dir: Path,
        user_goal_vec: Optional[List[float]] = None,
    ) -> Tuple[List[Tuple[PlanStep, Dict[str, Any]]], List[Dict[str, Any]]]:
        executed: List[Tuple[PlanStep, Dict[str, Any]]] = []
        outcomes: List[Dict[str, Any]] = []

        for idx, step in enumerate(steps):
            agent = self.agents.get(step.agent)
            if not agent:
                continue

            try:
                t0 = time.perf_counter()
                payload = dict(step.input) if isinstance(step.input, dict) else {}
                payload.setdefault("context", context)
                payload = await self._seed_params_from_lessons(step.agent, user_goal_vec, payload)

                attempts = 0
                success = False
                out: Dict[str, Any] = {}
                for attempt in range(self.planner_max_attempts):
                    attempts = attempt + 1
                    try:
                        out = await agent.execute(session_id, payload)
                    except Exception as e:
                        out = {"error": f"agent execution failed: {e}"}
                    success = self._success_heuristic(step.agent, out)
                    if success:
                        break

                duration_ms = int((time.perf_counter() - t0) * 1000)
                executed.append((step, out))
                context[step.agent] = out
                outcomes.append({
                    "agent": step.agent,
                    "success": bool(success),
                    "attempts": attempts,
                    "duration_ms": duration_ms,
                    "error": (str(out.get("error"))[:200] if isinstance(out, dict) and out.get("error") else None),
                })
                # per-step artifact
                try:
                    (run_dir / f"step_{idx}_{step.agent}.json").write_text(
                        json.dumps({
                            "input": {k: v for k, v in payload.items() if k != "context"},
                            "output": out,
                            "attempts": attempts,
                            "success": success,
                            "duration_ms": duration_ms,
                        }, ensure_ascii=False, indent=2)
                    )
                except Exception:
                    pass
            except Exception as e:
                logger.debug(f"AgentExecutor: step {idx} failed: {e}")
                continue

        return executed, outcomes
