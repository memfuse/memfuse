import os
from types import SimpleNamespace

from tests.performance.api.soak_helpers import (
    get_sessions_range,
    select_session_count,
    get_turn_batch_bounds,
    build_messages_from_feeder,
    run_tick,
)
from scripts.perf.message_feeder import FeederIndex, DatasetSession, FeederCursor

pytestmark = pytest.mark.unit


def test_session_count_from_env(monkeypatch):
    monkeypatch.setenv("SESSIONS_PER_USER_MIN", "5")
    monkeypatch.setenv("SESSIONS_PER_USER_MAX", "7")
    mn, mx = get_sessions_range()
    assert (mn, mx) == (5, 7)
    c = select_session_count()
    assert 5 <= c <= 7


def test_turn_batch_bounds(monkeypatch):
    monkeypatch.setenv("TURN_BATCH_MIN", "2")
    monkeypatch.setenv("TURN_BATCH_MAX", "4")
    mn, mx = get_turn_batch_bounds()
    assert (mn, mx) == (2, 4)


def test_build_message_batch_from_feeder():
    idx = FeederIndex(sessions=[DatasetSession(uid="s1", turns=[{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}])])
    cur = FeederCursor(idx, 0)
    out = build_messages_from_feeder(cur, 3, max_chars=1)
    assert len(out) == 3
    # truncated to 1 char
    assert all(len(m["content"]) == 1 for m in out)


def test_tick_ordering(monkeypatch):
    calls = []
    def list_fn(client, sid, **kwargs):
        calls.append(("list", sid))
    def add_fn(client, sid, batch):
        calls.append(("add", sid, len(batch) if batch else 0))
    def query_fn(client, uid, session_id=None):
        calls.append(("query", uid, session_id))

    idx = FeederIndex(sessions=[DatasetSession(uid="s1", turns=[{"role": "user", "content": "x"}] )])
    cur = FeederCursor(idx, 0)
    client = SimpleNamespace()
    run_tick(client, user_id="u1", session_id="sess1", cursor=cur, list_fn=list_fn, add_fn=add_fn, query_fn=query_fn, batch_min=1, batch_max=1, query_with_session_prob=1.0)
    assert [c[0] for c in calls] == ["list", "add", "query"]
import pytest
