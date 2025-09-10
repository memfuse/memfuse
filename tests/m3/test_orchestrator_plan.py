from memfuse_core.m3.orchestrator import Orchestrator


def test_orchestrator_minimal_flow():
    orch = Orchestrator()
    # Should not raise and should return a non-empty string
    result = orch.handle_request("session-test", "Explain MemFuse in one sentence")
    assert isinstance(result, str)
    assert len(result) > 0

