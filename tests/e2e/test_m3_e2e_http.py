import os
import time
import uuid
import json
import pytest
import httpx


def _api_base() -> str:
    return os.getenv("MEMFUSE_API_BASE", "http://localhost:8000/api/v1").rstrip("/")


@pytest.mark.e2e
def test_m3_e2e_http_flow():
    # Skip unless explicitly enabled
    if os.getenv("MEMFUSE_E2E", "0") not in ("1", "true", "TRUE", "yes", "YES"):
        pytest.skip("Set MEMFUSE_E2E=1 to run live HTTP E2E test")

    base = _api_base()
    client = httpx.Client(timeout=30.0)

    # 1) Create user
    user_name = f"user_e2e_{uuid.uuid4().hex[:8]}"
    r = client.post(f"{base}/users", json={"name": user_name})
    assert r.status_code in (200, 201), r.text
    data = r.json().get("data", {})
    user_id = (data.get("user") or {}).get("id") or data.get("user_id")
    assert user_id, r.text

    # 2) Create agent
    agent_name = f"agent_e2e_{uuid.uuid4().hex[:8]}"
    r = client.post(f"{base}/agents", json={"name": agent_name})
    assert r.status_code in (200, 201), r.text
    data = r.json().get("data", {})
    agent_id = (data.get("agent") or {}).get("id") or data.get("agent_id")
    assert agent_id, r.text

    # 3) Create session
    r = client.post(f"{base}/sessions", json={"user_id": user_id, "agent_id": agent_id, "name": f"sess-{user_name}-{agent_name}"})
    assert r.status_code in (200, 201), r.text
    data = r.json().get("data", {})
    session_id = (data.get("session") or {}).get("id") or data.get("session_id")
    assert session_id, r.text

    # 4) Send M3-triggered messages (task + task_eos)
    payload = {
        "messages": [
            {"role": "user", "content": "Search articles about agent memory.", "metadata": {"task": "op_websearch_memory"}},
            {"role": "user", "content": "Summarize findings about agent memory.", "metadata": {"task": "op_websearch_memory", "task_eos": True}},
        ]
    }
    r = client.post(f"{base}/sessions/{session_id}/messages", json=payload)
    assert r.status_code in (200, 201), r.text
    data = r.json().get("data", {})
    # Assistant reply may be present if orchestrator ran successfully
    # We assert success status and leave deeper checks to gateway/unit tests
    assert isinstance(data, dict)

    # 5) Query M3 by task
    query_body = {
        "query": "memory patterns",
        "top_k": 5,
        "metadata": {"task": "op_websearch_memory"},
        "session_id": session_id,
        "include_workflows": True,
    }
    r = client.post(f"{base}/users/{user_id}/query", json=query_body)
    assert r.status_code == 200, r.text
    qd = r.json().get("data", {})
    assert isinstance(qd, dict), r.text

