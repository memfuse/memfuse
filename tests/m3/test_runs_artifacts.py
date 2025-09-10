import os
import json
from pathlib import Path
import pytest


@pytest.mark.asyncio
async def test_orchestrator_writes_run_artifacts(tmp_path, monkeypatch):
    # Force runs base dir to tmp
    monkeypatch.setenv("RUNS_BASE_DIR", str(tmp_path))

    # Make ChatLLM deterministic and offline
    from memfuse_core.llm import chat as chat_mod

    def fake_chat(self, system_prompt, messages):
        return "ok"

    def fake_completion_json(self, system_prompt, user_prompt):
        return json.dumps({"steps": [
            {"agent": "ReportGenerationAgent", "input": {"points": {"title": "x"}}}
        ]})

    monkeypatch.setattr(chat_mod.ChatLLM, "chat", fake_chat, raising=False)
    monkeypatch.setattr(chat_mod.ChatLLM, "completion_json", fake_completion_json, raising=False)

    from memfuse_core.m3.orchestrator import Orchestrator

    orch = Orchestrator()
    out = await orch.handle_request("sess-artifacts", "gen")
    assert isinstance(out, str)

    # Find the run directory under tmp_path / */sess-artifacts
    session_dirs = list(tmp_path.glob("*/sess-artifacts"))
    assert session_dirs, f"No run dir created under {tmp_path}"
    run_dir = session_dirs[0]

    # Check files
    assert (run_dir / "input.json").exists()
    assert (run_dir / "plan.json").exists()
    assert (run_dir / "report.txt").exists()

