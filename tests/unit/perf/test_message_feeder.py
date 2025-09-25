import io
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

from scripts.perf.message_feeder import (
    FeederCursor,
    FeederIndex,
    load_dataset,
    map_to_session,
)


def _write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_load_dataset_parses_sessions(tmp_path: Path):
    # Two lines; first with 2 sessions, second with 1 session
    rows = [
        {
            "haystack_sessions": [
                [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ],
                [
                    {"speaker": "user", "text": "quest"},
                ],
            ]
        },
        {
            "haystack_sessions": [
                [
                    {"author": "user", "message": "ping"},
                    {"role": "assistant", "content": "pong"},
                ]
            ]
        },
    ]
    p = tmp_path / "sample.jsonl"
    _write_jsonl(p, rows)

    idx = load_dataset(str(p))
    assert isinstance(idx, FeederIndex)
    assert idx.size == 3
    assert all(len(sess.turns) >= 1 for sess in idx.sessions)
    # UIDs stable format
    assert idx.sessions[0].uid.startswith("line0-sess0")


def test_shard_mapping_deterministic(tmp_path: Path):
    # Minimal dataset with 5 sessions
    rows = [{"haystack_sessions": [[{"role": "user", "content": str(i)}]]} for i in range(5)]
    p = tmp_path / "d.jsonl"
    _write_jsonl(p, rows)
    idx = load_dataset(str(p))

    s1 = map_to_session(idx, user_id="u1", session_idx=0, strategy="hash")
    s2 = map_to_session(idx, user_id="u1", session_idx=0, strategy="hash")
    assert s1 == s2

    rr1 = map_to_session(idx, user_id="u1", session_idx=1, strategy="round_robin")
    rr2 = map_to_session(idx, user_id="u1", session_idx=1, strategy="round_robin")
    assert rr1 == rr2
    assert 0 <= rr1 < idx.size


def test_cursor_wraparound_and_batch(tmp_path: Path):
    rows = [
        {
            "haystack_sessions": [
                [
                    {"role": "user", "content": "a"},
                    {"role": "assistant", "content": "b"},
                ]
            ]
        }
    ]
    p = tmp_path / "d.jsonl"
    _write_jsonl(p, rows)
    idx = load_dataset(str(p))
    cur = FeederCursor(idx, 0)

    # Two turns available; request 3 to force wrap-around
    b1 = cur.next_batch(batch_size=3)
    assert len(b1) == 3
    assert [m["content"] for m in b1] == ["a", "b", "a"]

    # Subsequent call continues sequence
    b2 = cur.next_batch(batch_size=2)
    assert [m["content"] for m in b2] == ["b", "a"]


def test_batch_and_char_cap(tmp_path: Path):
    long_text = "x" * 100
    rows = [
        {
            "haystack_sessions": [
                [
                    {"role": "user", "content": long_text},
                    {"role": "assistant", "content": long_text},
                ]
            ]
        }
    ]
    p = tmp_path / "d.jsonl"
    _write_jsonl(p, rows)
    idx = load_dataset(str(p))
    cur = FeederCursor(idx, 0)
    out = cur.next_batch(batch_size=2, max_chars=10)
    assert all(len(m["content"]) == 10 for m in out)
