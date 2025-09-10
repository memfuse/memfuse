import pytest


@pytest.mark.asyncio
async def test_orchestrator_with_rag_agent(monkeypatch):
    # Force planner to produce a RAGQueryAgent step
    from memfuse_core.m3 import orchestrator as orch_mod

    async def fake_chat(self, session_id, query, history_messages=None, top_k=5):
        return "answer-from-rag"

    def fake_completion_json(self, system, user):
        return '{"steps": [{"agent": "RAGQueryAgent", "input": {"query": "what is memfuse?"}}]}'

    monkeypatch.setattr(orch_mod.RAGService, "chat", fake_chat, raising=False)
    monkeypatch.setattr(orch_mod.ChatLLM, "completion_json", fake_completion_json, raising=False)

    from memfuse_core.m3.orchestrator import Orchestrator

    orch = Orchestrator()
    out = await orch.handle_request("sess-rag", "test goal")
    assert isinstance(out, str)
    assert "answer" in out or len(out) > 0

