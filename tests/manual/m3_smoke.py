#!/usr/bin/env python3
"""
M3 Smoke Test (Manual)

This script is intended for manual verification while a MemFuse server is running.

It will:
1) Create a user and agent
2) Create a session
3) Submit a user message with metadata.tag = "m3" to trigger orchestration
4) Call the M3 query endpoint

Environment:
- MEMFUSE_API_BASE (default: http://localhost:8000/api/v1)

Note: Not collected by pytest. Run with: python tests/manual/m3_smoke.py
"""

import os
import uuid
import json
import requests


def api_base() -> str:
    return os.getenv("MEMFUSE_API_BASE", "http://localhost:8000/api/v1").rstrip("/")


def main() -> None:
    base = api_base()
    print(f"Using API base: {base}")

    # 1) Create user
    user_name = f"user_{str(uuid.uuid4())[:8]}"
    r = requests.post(f"{base}/users", json={"name": user_name})
    r.raise_for_status()
    user_id = r.json()["data"]["user"]["id"] if "user" in r.json().get("data", {}) else r.json()["data"].get("user_id")
    print(f"User created: {user_name} ({user_id})")

    # 2) Create agent
    agent_name = f"agent_{str(uuid.uuid4())[:8]}"
    r = requests.post(f"{base}/agents", json={"name": agent_name})
    r.raise_for_status()
    agent_id = r.json()["data"].get("agent")["id"] if "agent" in r.json().get("data", {}) else r.json()["data"].get("agent_id")
    print(f"Agent created: {agent_name} ({agent_id})")

    # 3) Create session
    r = requests.post(
        f"{base}/sessions",
        json={"user_id": user_id, "agent_id": agent_id, "name": f"sess-{user_name}-{agent_name}"},
    )
    r.raise_for_status()
    session_id = r.json()["data"].get("session")["id"] if "session" in r.json().get("data", {}) else r.json()["data"].get("session_id")
    print(f"Session created: {session_id}")

    # 4) Submit M3 message
    m3_msg = {
        "messages": [
            {"role": "user", "content": "Research latest LLM memory trends and summarize.", "metadata": {"tag": "m3"}}
        ]
    }
    r = requests.post(f"{base}/sessions/{session_id}/messages", json=m3_msg)
    r.raise_for_status()
    data = r.json().get("data", {})
    print(f"M3 add_messages response: {json.dumps(data, ensure_ascii=False)}")

    # 5) Query M3
    payload = {"query": "memory patterns", "top_k": 5, "metadata": {"tag": "m3"}}
    r = requests.post(f"{base}/users/{user_id}/query", json=payload)
    if r.status_code == 200:
        print("M3 query results:", json.dumps(r.json().get("data", {}), ensure_ascii=False)[:800])
    else:
        print("M3 query failed:", r.text)


if __name__ == "__main__":
    main()

