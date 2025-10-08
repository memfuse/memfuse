"""End-to-end schema validation tests for the memory query endpoint.

These tests exercise the full FastAPI stack (routing → gateway pipeline →
response guardrails) while stubbing the heavy persistence layers. The goal is to
verify that the final API responses strictly comply with the schema specified in
`schema_spec.md`, especially around field renames, metadata enrichment, and
timestamp normalization.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, Optional, Tuple
import uuid

import pytest
from fastapi.testclient import TestClient
from jsonschema import validate

from memfuse_core.server import create_app
from memfuse_core.services.database_service import DatabaseService
from memfuse_core.services.service_factory import ServiceFactory


# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------

class _StubDatabase:
    """Minimal async-compatible database stub for gateway tests."""

    def __init__(
        self,
        *,
        user_id: str,
        user_name: str,
        agent_id: str,
        session_primary: str,
        session_secondary: str,
    ) -> None:
        self._user = {
            "id": user_id,
            "name": user_name,
            "description": "stub-user",
        }
        self._agent = {
            "id": agent_id,
            "name": "stub-agent",
            "description": "stub-agent",
        }
        self._sessions = {
            session_primary: {
                "id": session_primary,
                "user_id": user_id,
                "agent_id": agent_id,
                "name": "Primary Session",
            },
            session_secondary: {
                "id": session_secondary,
                "user_id": user_id,
                "agent_id": agent_id,
                "name": "Secondary Session",
            },
        }

    async def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        return self._user if user_id == self._user["id"] else None

    async def get_user_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        return self._user if name == self._user["name"] else None

    async def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        return self._agent if agent_id == self._agent["id"] else None

    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self._sessions.get(session_id)

    # The query handler only touches these members; other Database APIs are unused.


class _StubBufferService:
    """Stub buffer service that returns deliberately non-compliant raw results.

    The gateway must normalize these into the schema-defined shape.
    """

    def __init__(
        self,
        *,
        user_id: str,
        agent_id: str,
        session_primary: str,
        session_secondary: str,
    ) -> None:
        self._user_id = user_id
        self._agent_id = agent_id
        self._session_primary = session_primary
        self._session_secondary = session_secondary
        # Deterministic epoch timestamp (float) to validate ISO conversion.
        self._base_ts = 1_725_000_000.123456

    async def query(self, *, query: str, top_k: int = 5, session_id: Optional[str] = None, **_: Any) -> Dict[str, Any]:
        """Return a success payload with raw results mirroring legacy fields."""

        def _episodic(round_index: int, message_index: int, result_session: str, content: str, score: float) -> Dict[str, Any]:
            return {
                "id": f"round_{round_index}_{message_index}",
                "content": content,
                "score": score,  # legacy field that should be renamed
                "type": "message",  # legacy field that should be renamed
                "role": "assistant",  # should be stripped
                "created_at": self._base_ts,
                "updated_at": self._base_ts,
                "metadata": {
                    "user_id": self._user_id,
                    "session_id": result_session,
                    "source": "round_buffer",
                    "round_index": round_index,
                    "message_index": message_index,
                    "agent_id": self._agent_id,
                },
            }

        def _semantic(result_session: str, fact_text: str, score: float) -> Dict[str, Any]:
            return {
                "id": str(uuid.uuid5(uuid.NAMESPACE_DNS, f"semantic:{result_session}:{fact_text}:{score}")),
                "content": fact_text,
                "score": score,
                "type": "knowledge",
                "created_at": self._base_ts,
                "updated_at": self._base_ts,
                "triples": [],
                "metadata": {
                    "user_id": self._user_id,
                    "session_id": result_session,
                    "derived_from": ["source-m1-id"],
                    "agent_id": self._agent_id,
                },
            }

        if session_id:
            results = [
                _episodic(0, 0, self._session_primary, "Primary session note", 0.91),
                _episodic(0, 1, self._session_secondary, "Secondary session note", 0.77),
                _semantic(self._session_secondary, "The user enjoys sketching landscapes.", 0.64),
            ]
        else:
            results = [
                _episodic(0, 0, self._session_secondary, "Secondary session note", 0.82),
                _semantic(self._session_secondary, "The user enjoys sketching landscapes.", 0.6),
            ]

        return {
            "status": "success",
            "code": 200,
            "data": {
                "results": results,
                "total": len(results),
            },
            "message": "Stubbed buffer response",
            "errors": None,
        }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def schema_test_client(monkeypatch: pytest.MonkeyPatch) -> Tuple[TestClient, Dict[str, str]]:
    """Provide a TestClient with services stubbed for schema validation."""

    identifiers = {
        "user_id": str(uuid.uuid5(uuid.NAMESPACE_DNS, "schema:user")),
        "user_name": "schema-user",
        "agent_id": str(uuid.uuid5(uuid.NAMESPACE_DNS, "schema:agent")),
        "session_primary": str(uuid.uuid5(uuid.NAMESPACE_DNS, "schema:session-primary")),
        "session_secondary": str(uuid.uuid5(uuid.NAMESPACE_DNS, "schema:session-secondary")),
    }

    stub_db = _StubDatabase(
        user_id=identifiers["user_id"],
        user_name=identifiers["user_name"],
        agent_id=identifiers["agent_id"],
        session_primary=identifiers["session_primary"],
        session_secondary=identifiers["session_secondary"],
    )

    stub_buffer = _StubBufferService(
        user_id=identifiers["user_id"],
        agent_id=identifiers["agent_id"],
        session_primary=identifiers["session_primary"],
        session_secondary=identifiers["session_secondary"],
    )

    async def fake_get_instance(cls) -> _StubDatabase:  # type: ignore[override]
        return stub_db

    async def fake_get_buffer_service_for_user(cls, user: str = "user_default") -> _StubBufferService:  # type: ignore[override]
        return stub_buffer

    # Reset any cached services then install our stubs.
    DatabaseService._instance = None  # type: ignore[attr-defined]
    ServiceFactory._buffer_service_instances = {}  # type: ignore[attr-defined]

    monkeypatch.setattr(DatabaseService, "get_instance", classmethod(fake_get_instance))
    monkeypatch.setattr(ServiceFactory, "get_buffer_service_for_user", classmethod(fake_get_buffer_service_for_user))

    client = TestClient(create_app())
    try:
        yield client, identifiers
    finally:
        client.close()


# ---------------------------------------------------------------------------
# JSON Schema reflecting schema_spec.md
# ---------------------------------------------------------------------------

RESULT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "content": {"type": ["string", "null"]},
        "fact": {
            "type": ["object", "null"],
            "properties": {
                "text": {"type": "string"},
                "triples": {"type": ["array", "null"]},
            },
            "required": ["text", "triples"],
            "additionalProperties": False,
        },
        "relevance_score": {"type": "number", "minimum": 0, "maximum": 1},
        "memory_type": {"type": "string"},
        "created_at": {"type": ["string", "null"]},
        "updated_at": {"type": ["string", "null"]},
        "metadata": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "agent_id": {"type": ["string", "null"]},
                "session_id": {"type": ["string", "null"]},
                "session_name": {"type": ["string", "null"]},
                "scope": {"type": ["string", "null"], "enum": ["in_session", "cross_session", None]},
            },
            "required": ["user_id", "agent_id", "session_id", "session_name", "scope"],
            "additionalProperties": True,
        },
    },
    "required": ["id", "relevance_score", "memory_type", "metadata"],
    "additionalProperties": True,
    "anyOf": [
        {"required": ["content"]},
        {"required": ["fact"]},
    ],
}

SUCCESS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["success"]},
        "code": {"type": "integer"},
        "data": {
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": RESULT_SCHEMA,
                },
                "total": {"type": "integer", "minimum": 0},
                "query": {"type": "string"},
            },
            "required": ["results", "total", "query"],
            "additionalProperties": False,
        },
        "message": {"type": "string"},
        "errors": {"type": ["null"]},
    },
    "required": ["status", "code", "data", "message", "errors"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _assert_iso_timestamp(value: Optional[str]) -> None:
    if value is None:
        return
    # Accept ISO 8601 with timezone offset.
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:  # pragma: no cover - explicit failure path
        pytest.fail(f"Timestamp is not valid ISO 8601: {value!r} ({exc})")


def _assert_schema(response_json: Dict[str, Any]) -> None:
    validate(instance=response_json, schema=SUCCESS_SCHEMA)


def test_memory_query_with_session_schema(
    schema_test_client: Tuple[TestClient, Dict[str, str]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    client, ids = schema_test_client

    payload = {
        "query": "Tell me about my hobbies",
        "session_id": ids["session_primary"],
        "agent_id": ids["agent_id"],
        "top_k": 5,
        "metadata": {"task": "schema-check"},
    }

    with capsys.disabled():
        print("\n[Schema E2E] Request Payload (with session):")
        print(json.dumps(payload, indent=2, sort_keys=True))
        print("[Schema E2E] Expected Response Schema:")
        print(json.dumps(SUCCESS_SCHEMA, indent=2, sort_keys=True))

    response = client.post(f"/api/v1/users/{ids['user_id']}/query", json=payload)
    assert response.status_code == 200, response.text

    body = response.json()
    with capsys.disabled():
        print("[Schema E2E] Actual Response (with session):")
        print(json.dumps(body, indent=2, sort_keys=True))

    _assert_schema(body)
    assert body["data"]["query"] == payload["query"]

    results = body["data"]["results"]
    assert len(results) == 3
    assert body["data"]["total"] == 3

    # Result 0 should reflect in-session enrichment.
    in_session = results[0]
    assert in_session["metadata"]["scope"] == "in_session"
    assert in_session["metadata"]["session_name"] == "Primary Session"
    assert in_session["metadata"]["agent_id"] == ids["agent_id"]
    assert in_session["metadata"]["session_id"] == ids["session_primary"]
    assert in_session["metadata"]["user_id"] == ids["user_id"]

    # Result 1 should reflect cross-session enrichment.
    cross_session = results[1]
    assert cross_session["metadata"]["scope"] == "cross_session"
    assert cross_session["metadata"]["session_name"] == "Secondary Session"
    assert cross_session["metadata"]["session_id"] == ids["session_secondary"]

    # Semantic result should provide fact structure and be properly normalized.
    semantic_result = next(r for r in results if r["memory_type"] == "semantic")
    assert "fact" in semantic_result and semantic_result["fact"] is not None
    assert semantic_result["fact"]["text"] == "The user enjoys sketching landscapes."
    assert semantic_result.get("content") is None

    for result in results:
        # Legacy fields must be removed.
        assert "score" not in result
        assert "type" not in result
        assert "role" not in result
        assert "source" not in result["metadata"]
        assert "level" not in result["metadata"]

        # Timestamps must be ISO 8601.
        _assert_iso_timestamp(result["created_at"])
        _assert_iso_timestamp(result["updated_at"])

        # Ensure relevance score is normalized.
        assert 0 <= result["relevance_score"] <= 1


def test_memory_query_without_session_schema(
    schema_test_client: Tuple[TestClient, Dict[str, str]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    client, ids = schema_test_client

    payload = {
        "query": "Remind me what else I like",
        "agent_id": ids["agent_id"],
        "top_k": 5,
        "metadata": {"task": "schema-check"},
    }

    with capsys.disabled():
        print("\n[Schema E2E] Request Payload (without session):")
        print(json.dumps(payload, indent=2, sort_keys=True))

    response = client.post(f"/api/v1/users/{ids['user_id']}/query", json=payload)
    assert response.status_code == 200, response.text

    body = response.json()
    with capsys.disabled():
        print("[Schema E2E] Actual Response (without session):")
        print(json.dumps(body, indent=2, sort_keys=True))

    _assert_schema(body)
    assert body["data"]["query"] == payload["query"]

    results = body["data"]["results"]
    assert len(results) == 2
    assert body["data"]["total"] == 2

    episodic = next(r for r in results if r["memory_type"] == "episodic")
    assert episodic["metadata"]["scope"] is None
    assert episodic["metadata"]["session_name"] == "Secondary Session"

    semantic = next(r for r in results if r["memory_type"] == "semantic")
    assert semantic["metadata"]["scope"] is None
    assert semantic["fact"]["text"] == "The user enjoys sketching landscapes."
    assert semantic.get("content") is None

    # Ensure the pipeline has stripped legacy fields and normalized timestamps.
    for result in results:
        assert "score" not in result
        assert "type" not in result
        assert "role" not in result
        assert "source" not in result["metadata"]
        assert "level" not in result["metadata"]
        _assert_iso_timestamp(result["created_at"])
        _assert_iso_timestamp(result["updated_at"])
