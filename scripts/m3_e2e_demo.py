#!/usr/bin/env python3
"""
M3 End-to-End Demo (HTTP)

Run after starting the server and DB:
  poetry run python scripts/memfuse_launcher.py --start-db --optimize-db
  poetry run memfuse-core

Then run this script:
  MEMFUSE_API_BASE=http://localhost:8000/api/v1 python scripts/m3_e2e_demo.py

This demo:
1) Creates a user and agent
2) Creates a session
3) Sends messages with metadata.task and final metadata.task_eos=true to trigger M3
4) Prints assistant reply message id and workflow id
5) Repeats the workflow to demonstrate reuse (same workflow id)
6) Queries M3 by task to inspect workflows/lessons/session_workflows
"""

import os
import sys
import uuid
import json
import time
import httpx


def api_base() -> str:
    return os.getenv("MEMFUSE_API_BASE", "http://localhost:8000/api/v1").rstrip("/")


def pretty(obj):
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        return str(obj)


def main() -> int:
    base = api_base()
    print(f"Using API base: {base}")
    client = httpx.Client(timeout=30.0)

    try:
        # 1) Create user
        user_name = f"user_e2e_{uuid.uuid4().hex[:8]}"
        r = client.post(f"{base}/users", json={"name": user_name})
        r.raise_for_status()
        data = r.json().get("data", {})
        user_id = (data.get("user") or {}).get("id") or data.get("user_id")
        print(f"User created: {user_name} ({user_id})")

        # 2) Create agent
        agent_name = f"agent_e2e_{uuid.uuid4().hex[:8]}"
        r = client.post(f"{base}/agents", json={"name": agent_name})
        r.raise_for_status()
        data = r.json().get("data", {})
        agent_id = (data.get("agent") or {}).get("id") or data.get("agent_id")
        print(f"Agent created: {agent_name} ({agent_id})")

        # 3) Create session
        r = client.post(f"{base}/sessions", json={"user_id": user_id, "agent_id": agent_id, "name": f"sess-{user_name}-{agent_name}"})
        r.raise_for_status()
        data = r.json().get("data", {})
        session_id = (data.get("session") or {}).get("id") or data.get("session_id")
        print(f"Session created: {session_id}")

        task = "op_websearch_memory"
        # 4) First run: send task messages (EOS triggers M3)
        payload = {
            "messages": [
                {"role": "user", "content": "Search articles about agent memory.", "metadata": {"task": task}},
                {"role": "user", "content": "Summarize findings about agent memory.", "metadata": {"task": task, "task_eos": True}},
            ]
        }
        r = client.post(f"{base}/sessions/{session_id}/messages", json=payload)
        r.raise_for_status()
        d1 = r.json().get("data", {})
        wf1 = d1.get("workflow_id")
        print("First run result:")
        print(pretty(d1))

        # 5) Second run: same workflow to demonstrate reuse
        time.sleep(1.0)
        r = client.post(f"{base}/sessions/{session_id}/messages", json=payload)
        r.raise_for_status()
        d2 = r.json().get("data", {})
        wf2 = d2.get("workflow_id")
        print("Second run result:")
        print(pretty(d2))
        if wf1 and wf2 and wf1 == wf2:
            print(f"✅ Reuse confirmed: workflow_id {wf2}")
        else:
            print(f"ℹ️ Reuse not confirmed (wf1={wf1}, wf2={wf2}); still OK if changes are substantial.")

        # 6) Query M3 by task to inspect workflows/lessons/session_workflows
        query_body = {
            "query": "memory patterns",
            "top_k": 5,
            "metadata": {"task": task},
            "session_id": session_id,
            "include_workflows": True,
        }
        r = client.post(f"{base}/users/{user_id}/query", json=query_body)
        r.raise_for_status()
        qd = r.json().get("data", {})
        print("M3 query results (by task):")
        print(pretty(qd))

        print("\nDone.")
        return 0
    except httpx.HTTPError as e:
        print(f"HTTP error: {e}")
        try:
            print(f"Response: {e.response.text}")
        except Exception:
            pass
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

