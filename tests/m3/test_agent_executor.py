import pytest


class DummyAgent:
    def __init__(self):
        self.calls = 0

    async def execute(self, session_id, payload):
        self.calls += 1
        # Fail once, then succeed
        if self.calls == 1:
            return {"error": "transient"}
        return {"report": "ok"}


class DummyStore:
    async def query_lessons_similar(self, *a, **kw):
        return []


@pytest.mark.asyncio
async def test_agent_executor_retries_and_records_outcomes(tmp_path):
    from memfuse_core.m3.executor import AgentExecutor
    from memfuse_core.m3.types import PlanStep

    agent = DummyAgent()
    agents = {"ReportGenerationAgent": agent}
    store = DummyStore()

    executor = AgentExecutor(agents, store, planner_max_attempts=2)
    steps = [PlanStep(agent="ReportGenerationAgent", input={})]
    executed, outcomes = await executor.execute_steps(
        session_id="sess-t",
        steps=steps,
        context={},
        run_dir=tmp_path,
        user_goal_vec=None,
    )

    assert len(executed) == 1
    assert len(outcomes) == 1
    assert outcomes[0]["attempts"] == 2
    assert outcomes[0]["success"] is True
    # step artifact exists
    files = list(tmp_path.glob("step_0_ReportGenerationAgent.json"))
    assert files, "step artifact not created"
