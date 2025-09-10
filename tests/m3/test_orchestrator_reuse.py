import pytest


class FakeStore:
    def __init__(self):
        self.bumped = 0

    async def query_procedural_similar(self, emb, top_k):
        # Return a single high-score workflow with one report step
        wf = {"plan": [{"agent": "ReportGenerationAgent", "input": {"points": {"title": "reused"}}}]}
        return [("wid-1", wf, 0.99)]

    async def bump_procedural_usage(self, wid, by):
        self.bumped += by
        return by

    async def upsert_procedural_workflow(self, *a, **kw):
        return None

    async def insert_lesson(self, *a, **kw):
        return "lid"


@pytest.mark.asyncio
async def test_orchestrator_reuse_path_sets_flag(monkeypatch):
    from memfuse_core.m3.orchestrator import Orchestrator
    from memfuse_core.llm import chat as chat_mod

    # Force offline LLM to be fast and predictable
    def fake_chat(self, system_prompt, messages):
        return "ok"

    monkeypatch.setattr(chat_mod.ChatLLM, "chat", fake_chat, raising=False)

    store = FakeStore()
    orch = Orchestrator(store=store)
    out = await orch.handle_request("sess-x", "Some goal")
    assert isinstance(out, str)
    assert orch.last_reused is True
    assert orch.last_workflow_id == "wid-1"
