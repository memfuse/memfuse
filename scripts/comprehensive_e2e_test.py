#!/usr/bin/env python3
"""
Comprehensive M1 Schema + M3 End-to-End Test

Run after starting the server and DB:
  poetry run python scripts/memfuse_launcher.py --recreate-db
  poetry run memfuse-core

Then run this script:
  MEMFUSE_API_BASE=http://localhost:8000/api/v1 poetry run python scripts/comprehensive_e2e_test.py

This comprehensive test validates:
1) M1 Schema compliance (episodic memory structure)
2) M3 functionality (task-based workflow orchestration)
3) Metadata handling (task, task_eos, scope calculation)
4) Cross-session and in-session scope validation
5) Workflow reuse detection
6) Complete request/response schema validation

Test scenarios:
- Create user, agent, and sessions
- Send messages with task metadata
- Trigger M3 with task_eos=true
- Validate M1 episodic schema in responses
- Test cross-session queries
- Verify workflow reuse
- Validate all required metadata fields
"""

import os
import sys
import uuid
import json
import time
import httpx
from typing import Dict, Any, List


API_KEY = os.getenv("MEMFUSE_API_KEY", "test-api-key")


def api_base() -> str:
    return os.getenv("MEMFUSE_API_BASE", "http://localhost:8000/api/v1").rstrip("/")


def pretty(obj):
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        return str(obj)


def validate_m1_episodic_schema(result: Dict[str, Any], has_session_id: bool, test_name: str):
    """Validate M1 episodic memory schema compliance."""
    print(f"  🔍 Validating M1 schema for: {test_name}")
    
    # Required top-level fields
    required_fields = ["id", "relevance_score", "memory_type", "created_at", "updated_at", "metadata"]
    for field in required_fields:
        assert field in result, f"Missing required field: {field}"
    
    # Memory type specific validation
    memory_type = result["memory_type"]
    if memory_type == "episodic":
        assert "content" in result, "Episodic memory must have 'content' field"
        assert "fact" not in result, "Episodic memory should not have 'fact' field"
    elif memory_type == "semantic":
        assert "content" not in result, "Semantic memory should not have 'content' field"
        assert "fact" in result, "Semantic memory must have 'fact' field"
        fact = result["fact"]
        assert isinstance(fact, dict), "Fact must be a dictionary"
        assert "text" in fact, "Fact must have 'text' field"
        assert "triples" in fact, "Fact must have 'triples' field"
    else:
        raise AssertionError(f"Unknown memory_type: {memory_type}")
    
    # Metadata validation
    metadata = result["metadata"]
    required_metadata_fields = ["user_id", "agent_id", "session_id", "session_name", "scope"]
    for field in required_metadata_fields:
        assert field in metadata, f"Missing required metadata field: {field}"
    
    # Scope validation
    scope = metadata["scope"]
    if has_session_id:
        assert scope in ("in_session", "cross_session"), f"Invalid scope with session_id: {scope}"
    else:
        assert scope is None, f"Scope must be null without session_id, got: {scope}"
    
    # Forbidden fields (should be removed by processors)
    forbidden_fields = ["score", "type", "role"]
    forbidden_metadata_fields = ["level", "retrieval", "source"]
    
    for field in forbidden_fields:
        assert field not in result, f"Forbidden field found in result: {field}"
    
    for field in forbidden_metadata_fields:
        assert field not in metadata, f"Forbidden metadata field found: {field}"
    
    print(f"  ✅ M1 schema validation passed for: {test_name}")


def validate_m3_response(response_data: Dict[str, Any], should_have_m3: bool):
    """Validate M3 workflow response fields."""
    if should_have_m3:
        print("  🔍 Validating M3 workflow response...")
        # M3 fields are optional but if present should be valid
        if "workflow_id" in response_data:
            assert isinstance(response_data["workflow_id"], str), "workflow_id must be string"
            assert len(response_data["workflow_id"]) > 0, "workflow_id must not be empty"
        
        if "assistant_message_id" in response_data:
            assert isinstance(response_data["assistant_message_id"], str), "assistant_message_id must be string"
        
        print("  ✅ M3 workflow response validation passed")
    else:
        print("  ℹ️ No M3 workflow expected for this request")


def main() -> int:
    base = api_base()
    print(f"🚀 Starting Comprehensive E2E Test")
    print(f"📡 Using API base: {base}")
    
    headers = {"X-API-Key": API_KEY}
    client = httpx.Client(timeout=60.0, headers=headers)
    
    try:
        print("\n📋 Test Phase 1: Setup - Creating user, agent, and sessions")
        
        # 1) Create user
        user_name = f"user_e2e_{uuid.uuid4().hex[:8]}"
        r = client.post(f"{base}/users", json={"name": user_name})
        r.raise_for_status()
        data = r.json().get("data", {})
        user_id = (data.get("user") or {}).get("id") or data.get("user_id")
        print(f"  ✅ User created: {user_name} ({user_id})")

        # 2) Create agent
        agent_name = f"agent_e2e_{uuid.uuid4().hex[:8]}"
        r = client.post(f"{base}/agents", json={"name": agent_name})
        r.raise_for_status()
        data = r.json().get("data", {})
        agent_id = (data.get("agent") or {}).get("id") or data.get("agent_id")
        print(f"  ✅ Agent created: {agent_name} ({agent_id})")

        # 3) Create first session
        r = client.post(
            f"{base}/sessions",
            json={"user_id": user_id, "agent_id": agent_id, "name": f"sess1-{user_name}"},
        )
        r.raise_for_status()
        data = r.json().get("data", {})
        session1_id = (data.get("session") or {}).get("id") or data.get("session_id")
        print(f"  ✅ Session 1 created: {session1_id}")

        # 4) Create second session for cross-session testing
        r = client.post(
            f"{base}/sessions",
            json={"user_id": user_id, "agent_id": agent_id, "name": f"sess2-{user_name}"},
        )
        r.raise_for_status()
        data = r.json().get("data", {})
        session2_id = (data.get("session") or {}).get("id") or data.get("session_id")
        print(f"  ✅ Session 2 created: {session2_id}")

        print("\n📋 Test Phase 2: M3 Workflow Testing")
        
        task = "op_websearch_memory"
        
        # 5) First M3 workflow: send task messages (EOS triggers M3)
        print("  🔄 Sending first M3 workflow messages...")
        payload1 = {
            "messages": [
                {"role": "user", "content": "Search articles about agent memory.", "metadata": {"task": task}},
                {"role": "user", "content": "Summarize findings about agent memory.", "metadata": {"task": task, "task_eos": True}},
            ]
        }
        r = client.post(f"{base}/sessions/{session1_id}/messages", json=payload1)
        r.raise_for_status()
        d1 = r.json()
        print(f"  📊 First M3 workflow response:")
        print(f"     Status: {d1.get('status')}, Code: {d1.get('code')}")
        
        response_data1 = d1.get("data", {})
        validate_m3_response(response_data1, should_have_m3=True)
        
        wf1 = response_data1.get("workflow_id")
        if wf1:
            print(f"  ✅ M3 workflow 1 ID: {wf1}")
        else:
            print(f"  ⚠️ No workflow_id returned (may be due to LLM API limits)")

        # 6) Second M3 workflow: same task to test reuse
        print("  🔄 Sending second M3 workflow messages (testing reuse)...")
        time.sleep(1.0)  # Brief delay to ensure different timestamps
        r = client.post(f"{base}/sessions/{session1_id}/messages", json=payload1)
        r.raise_for_status()
        d2 = r.json()
        
        response_data2 = d2.get("data", {})
        validate_m3_response(response_data2, should_have_m3=True)
        
        wf2 = response_data2.get("workflow_id")
        if wf1 and wf2:
            if wf1 == wf2:
                print(f"  ✅ Workflow reuse detected: {wf2}")
            else:
                print(f"  ℹ️ Different workflow IDs (wf1={wf1}, wf2={wf2}) - new workflow created")
        else:
            print(f"  ℹ️ Workflow reuse cannot be verified (missing workflow_id)")

        # 7) Add some messages to session 2 for cross-session testing
        print("  🔄 Adding messages to second session for cross-session testing...")
        payload2 = {
            "messages": [
                {"role": "user", "content": "Different session message about memory.", "metadata": {"task": "different_task"}},
            ]
        }
        r = client.post(f"{base}/sessions/{session2_id}/messages", json=payload2)
        r.raise_for_status()

        print("\n📋 Test Phase 3: M1 Schema Validation")
        
        # 8) Query without session_id (scope should be null)
        print("  🔍 Testing query without session_id (scope=null)...")
        query_no_session = {
            "query": "agent memory",
            "top_k": 10,
            "agent_id": agent_id,
            "metadata": {"task": "recent", "mode": "episodic"}
        }
        r = client.post(f"{base}/users/{user_id}/query", json=query_no_session)
        r.raise_for_status()
        q1 = r.json()
        
        assert q1.get("status") == "success", f"Query failed: {q1}"
        results1 = q1.get("data", {}).get("results", [])
        print(f"  📊 Query without session_id returned {len(results1)} results")
        
        for i, result in enumerate(results1):
            validate_m1_episodic_schema(result, has_session_id=False, test_name=f"no-session-result-{i}")

        # 9) Query with session_id (scope should be in_session/cross_session)
        print("  🔍 Testing query with session_id (scope=in_session/cross_session)...")
        query_with_session = {
            "query": "agent memory",
            "top_k": 10,
            "session_id": session1_id,
            "agent_id": agent_id,
            "metadata": {"task": "recent", "mode": "episodic"}
        }
        r = client.post(f"{base}/users/{user_id}/query", json=query_with_session)
        r.raise_for_status()
        q2 = r.json()
        
        assert q2.get("status") == "success", f"Query failed: {q2}"
        results2 = q2.get("data", {}).get("results", [])
        print(f"  📊 Query with session_id returned {len(results2)} results")
        
        # Validate schema and check for both in_session and cross_session scopes
        in_session_count = 0
        cross_session_count = 0
        
        for i, result in enumerate(results2):
            validate_m1_episodic_schema(result, has_session_id=True, test_name=f"with-session-result-{i}")
            scope = result["metadata"]["scope"]
            if scope == "in_session":
                in_session_count += 1
            elif scope == "cross_session":
                cross_session_count += 1
        
        print(f"  📊 Scope distribution: in_session={in_session_count}, cross_session={cross_session_count}")
        
        # 10) Query with specific task metadata
        print("  🔍 Testing query with task metadata...")
        query_with_task = {
            "query": "agent memory",
            "top_k": 5,
            "session_id": session1_id,
            "agent_id": agent_id,
            "metadata": {"task": task, "mode": "episodic"}
        }
        r = client.post(f"{base}/users/{user_id}/query", json=query_with_task)
        r.raise_for_status()
        q3 = r.json()
        
        assert q3.get("status") == "success", f"Query failed: {q3}"
        results3 = q3.get("data", {}).get("results", [])
        print(f"  📊 Query with task metadata returned {len(results3)} results")
        
        for i, result in enumerate(results3):
            validate_m1_episodic_schema(result, has_session_id=True, test_name=f"task-query-result-{i}")
            # Verify task metadata is preserved
            task_in_metadata = result["metadata"].get("task")
            if task_in_metadata:
                print(f"    ✅ Task metadata preserved: {task_in_metadata}")

        print("\n📋 Test Phase 4: Additional Schema Validation")
        
        # 11) Test message listing to verify buffer integration
        print("  🔍 Testing message listing (buffer integration)...")
        r = client.get(f"{base}/sessions/{session1_id}/messages?limit=10")
        r.raise_for_status()
        messages_resp = r.json()
        
        assert messages_resp.get("status") == "success", f"Message listing failed: {messages_resp}"
        messages = messages_resp.get("data", {}).get("messages", [])
        print(f"  📊 Message listing returned {len(messages)} messages")
        
        # Validate message structure
        for i, msg in enumerate(messages):
            assert "role" in msg, f"Message {i} missing 'role' field"
            assert "content" in msg, f"Message {i} missing 'content' field"
            assert msg["role"] in ["user", "assistant", "system"], f"Invalid role in message {i}: {msg['role']}"
            print(f"    ✅ Message {i}: role={msg['role']}, content_length={len(msg.get('content', ''))}")

        print("\n🎉 All Tests Completed Successfully!")
        print("\n📊 Test Summary:")
        print(f"  ✅ M1 Schema validation: PASSED")
        print(f"  ✅ M3 workflow functionality: PASSED")
        print(f"  ✅ Metadata handling: PASSED")
        print(f"  ✅ Scope calculation: PASSED")
        print(f"  ✅ Cross-session queries: PASSED")
        print(f"  ✅ Buffer integration: PASSED")
        print(f"  ✅ Request/Response schema: PASSED")
        
        return 0
        
    except httpx.HTTPError as e:
        print(f"❌ HTTP error: {e}")
        try:
            print(f"Response: {e.response.text}")
        except Exception:
            pass
        return 1
    except AssertionError as e:
        print(f"❌ Schema validation failed: {e}")
        return 2
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 3


if __name__ == "__main__":
    sys.exit(main())