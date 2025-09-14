#!/usr/bin/env python3
"""
M3 End-to-End Demo (HTTP) – Current Branch

Run after starting the server and DB:
  poetry run python scripts/memfuse_launcher.py --recreate-db
  poetry run memfuse-core

Then run this script:
  MEMFUSE_API_BASE=http://localhost:8000/api/v1 poetry run python scripts/m3_e2e_demo.py

This demo:
1) Creates a user and agent
2) Creates a session
3) Sends messages with metadata.task，最后一条 metadata.task_eos=true 以触发 M3（写路径）
4) 打印 add_messages 返回的 message_ids（严格 Schema，不含多余字段）
5) 再次发送相同消息（可用于触发编排复用）
6) 携带同名 task 做检索（读路径），校验 M1（episodic）Schema 与 metadata 范式

注意：本脚本对服务响应做了容错，兼容是否返回 workflow_id/assistant_message_id 等可选字段。
"""

import os
import sys
import uuid
import json
import time
import httpx


API_KEY = os.getenv("MEMFUSE_API_KEY", "test-api-key")


def api_base() -> str:
    return os.getenv("MEMFUSE_API_BASE", "http://localhost:8000/api/v1").rstrip("/")


def pretty(obj):
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        return str(obj)


def assert_episodic_schema(result: dict, has_session_id: bool):
    # Required fields
    for f in ("id", "relevance_score", "memory_type", "created_at", "updated_at", "metadata"):
        assert f in result, f"Missing field: {f}"

    # Episodic vs semantic shape
    if result["memory_type"] == "episodic":
        assert "content" in result and "fact" not in result
    elif result["memory_type"] == "semantic":
        assert "content" not in result and "fact" in result
        assert isinstance(result["fact"], dict)
        assert "text" in result["fact"] and "triples" in result["fact"]
    else:
        raise AssertionError(f"Unknown memory_type: {result['memory_type']}")

    # Metadata constraints
    md = result["metadata"]
    for f in ("user_id", "agent_id", "session_id", "session_name", "scope"):
        assert f in md, f"Missing metadata field: {f}"

    # Scope rules
    if has_session_id:
        assert md["scope"] in ("in_session", "cross_session")
    else:
        assert md["scope"] is None

    # Forbidden fields removed
    for bad in ("score", "type", "role", "level", "retrieval", "source"):
        assert bad not in result and bad not in md


def main() -> int:
    base = api_base()
    print(f"Using API base: {base}")
    headers = {"X-API-Key": API_KEY}
    client = httpx.Client(timeout=60.0, headers=headers)

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
        r = client.post(
            f"{base}/sessions",
            json={"user_id": user_id, "agent_id": agent_id, "name": f"sess-{user_name}-{agent_name}"},
        )
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
        d1 = r.json()
        print("First add_messages response:")
        print(pretty(d1))
        wf1 = (d1.get("data") or {}).get("workflow_id")
        reused1 = (d1.get("data") or {}).get("workflow_reused")
        assert d1.get("status") == "success" and d1.get("code") in (200, 201)
        assert "message_ids" in (d1.get("data") or {})

        # 5) Second run: same workflow to potentially reuse
        time.sleep(0.5)
        r = client.post(f"{base}/sessions/{session_id}/messages", json=payload)
        r.raise_for_status()
        d2 = r.json()
        print("Second add_messages response:")
        print(pretty(d2))
        wf2 = (d2.get("data") or {}).get("workflow_id")
        reused2 = (d2.get("data") or {}).get("workflow_reused")

        if wf1 and wf2:
            if wf1 == wf2:
                print(f"✅ Workflow reuse detected: workflow_id={wf2}, reused flags: first={reused1}, second={reused2}")
            else:
                print(f"ℹ️ Different workflow ids (wf1={wf1}, wf2={wf2}); reuse may not have been applied.")
        else:
            print("ℹ️ No workflow_id provided by server; reuse cannot be inferred from add_messages. (This is optional.)")

        # 6) Query by task (read path) and validate M1 schema/metadata rules
        # Prefer episodic routing, and use a query text that matches the messages content.
        # Our messages include phrases like "Search articles about agent memory" and
        # "Summarize findings about agent memory.", so we use a matching query.
        query_body_episodic = {
            "query": "agent memory",
            "top_k": 10,
            "session_id": session_id,
            "agent_id": agent_id,
            "metadata": {"task": "recent", "mode": "episodic"}
        }
        r = client.post(f"{base}/users/{user_id}/query", json=query_body_episodic)
        r.raise_for_status()
        q = r.json()
        print("Query (episodic) response:")
        print(pretty(q))

        assert q.get("status") == "success" and q.get("code") == 200
        qd = q.get("data") or {}
        assert isinstance(qd.get("results"), list)
        results = qd.get("results")
        if len(results) == 0:
            # as a fallback, show messages list to aid debugging
            print("No retrieval results; listing messages for the session for diagnostics...")
            gr = client.get(f"{base}/sessions/{session_id}/messages")
            print(pretty(gr.json()))
            raise AssertionError("retrieval returned 0 results; expected episodic results from buffer")

        # Validate episodic/semantic shape and metadata contract on each result
        for res in results:
            assert_episodic_schema(res, has_session_id=True)

        print("\n✅ M3 + M1 Schema E2E validation completed.")
        return 0
    except httpx.HTTPError as e:
        print(f"HTTP error: {e}")
        try:
            print(f"Response: {e.response.text}")
        except Exception:
            pass
        return 1
    except AssertionError as e:
        print(f"Assertion failed: {e}")
        return 2
    except Exception as e:
        print(f"Error: {e}")
        return 3


if __name__ == "__main__":
    sys.exit(main())
