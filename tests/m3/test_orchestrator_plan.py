from memfuse_core.m3.orchestrator import Orchestrator
import pytest


@pytest.mark.asyncio
async def test_orchestrator_minimal_flow(monkeypatch):
    # speed up by avoiding heavy embedding/model loads
    from memfuse_core.utils import embeddings as emb_mod
    monkeypatch.setattr(emb_mod, "create_embedding", lambda text: [0.0] * 16, raising=False)

    from memfuse_core.llm import chat as chat_mod
    monkeypatch.setattr(chat_mod.ChatLLM, "chat", lambda self, sys, msgs: "ok", raising=False)
    monkeypatch.setattr(
        chat_mod.ChatLLM,
        "completion_json",
        lambda self, sys, u: '{"steps": [{"agent": "ReportGenerationAgent", "input": {"points": {"x":1}}}]}',
        raising=False,
    )

    orch = Orchestrator()
    result = await orch.handle_request("session-test", "Explain MemFuse in one sentence")
    assert isinstance(result, str)
    assert len(result) > 0
