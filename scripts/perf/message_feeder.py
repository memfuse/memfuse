#!/usr/bin/env python3
"""
Dataset-backed message feeder for long-run performance simulations.

Reads JSONL dataset (e.g., datasets/lme_s_mc10.json), extracts
`haystack_sessions` per record, and exposes deterministic sharding and
per-session cursors that yield message batches suitable for POST
`/sessions/:id/messages` payloads.

Public API:
- load_dataset(path: str) -> FeederIndex
- map_to_session(index, user_id: str, session_idx: int, strategy: str = 'hash') -> int
- FeederCursor(index, dataset_session_idx: int)
  - next_batch(batch_size: int = 1, max_chars: int | None = None) -> list[dict]

Environment (optional):
- DATASET_PATH (default: datasets/lme_s_mc10.json)
- DATASET_SHARDING (hash|round_robin, default: hash)
- TURN_BATCH_SIZE (default: 1..3 when caller requests None)
- MAX_CHARS (default: None)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ----------------------------
# Data structures
# ----------------------------

@dataclass(frozen=True)
class DatasetSession:
    """A single conversation session consisting of ordered turns."""

    uid: str  # stable identifier (e.g., f"line{L}-sess{S}")
    turns: List[Dict[str, Any]]  # each: {role: str, content: str}


@dataclass
class FeederIndex:
    """Index of all sessions in the dataset."""

    sessions: List[DatasetSession]

    @property
    def size(self) -> int:
        return len(self.sessions)


def _normalize_turn(t: Any) -> Optional[Dict[str, Any]]:
    """Normalize a raw turn to {role, content} dict, or None if invalid."""
    if not isinstance(t, dict):
        return None
    # Common keys for role/content variants
    role = t.get("role") or t.get("speaker") or t.get("author")
    content = t.get("content") or t.get("text") or t.get("message")
    if role is None or content is None:
        return None
    role_s = str(role).strip().lower()
    if role_s not in ("user", "assistant", "system"):
        # Default anything else to 'user' for the API shape
        role_s = "user"
    content_s = str(content)
    return {"role": role_s, "content": content_s}


def load_dataset(path: str) -> FeederIndex:
    """Load JSONL dataset and extract haystack_sessions into an index.

    Each JSONL line is expected to be a JSON object that may contain a
    `haystack_sessions` field: a list of sessions, where each session is a list
    of turn dicts.
    """
    p = Path(path)
    sessions: List[DatasetSession] = []
    if not p.exists():
        return FeederIndex(sessions=sessions)

    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            raw_sessions = obj.get("haystack_sessions")
            if not isinstance(raw_sessions, list):
                continue
            for sess_idx, raw_turns in enumerate(raw_sessions):
                if not isinstance(raw_turns, list):
                    continue
                turns: List[Dict[str, Any]] = []
                for t in raw_turns:
                    norm = _normalize_turn(t)
                    if norm is not None:
                        turns.append(norm)
                if not turns:
                    continue
                uid = f"line{line_no}-sess{sess_idx}"
                sessions.append(DatasetSession(uid=uid, turns=turns))

    return FeederIndex(sessions=sessions)


def _hash_str(s: str) -> int:
    # Deterministic hash independent of Python's randomization
    h = 2166136261
    for ch in s.encode("utf-8"):
        h ^= ch
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def map_to_session(index: FeederIndex, user_id: str, session_idx: int, strategy: str = "hash") -> int:
    """Map a (user_id, session_idx) pair to a dataset session index.

    - hash: FNV-1a hash of f"{user_id}-{session_idx}" modulo index.size
    - round_robin: (hash(user_id) + session_idx) % index.size
    """
    n = max(1, index.size)
    if strategy == "round_robin":
        base = _hash_str(user_id)
        return (base + int(session_idx)) % n
    # default: hash
    key = f"{user_id}-{int(session_idx)}"
    return _hash_str(key) % n


class FeederCursor:
    """Cursor over a single dataset session, with wrap-around semantics."""

    def __init__(self, index: FeederIndex, dataset_session_idx: int):
        if index.size == 0:
            raise ValueError("Empty FeederIndex; load_dataset() returned no sessions")
        if dataset_session_idx < 0 or dataset_session_idx >= index.size:
            raise IndexError("dataset_session_idx out of range")
        self._index = index
        self._sess = index.sessions[dataset_session_idx]
        self._pos = 0

    @property
    def session_uid(self) -> str:
        return self._sess.uid

    def next_batch(self, batch_size: int = 1, max_chars: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return the next batch of messages and advance cursor.

        Args:
            batch_size: number of turns to emit (wraps around on end)
            max_chars: if provided, truncate `content` to at most this size
        Returns:
            List of message dicts: {role, content, metadata}
        """
        if batch_size <= 0:
            return []
        out: List[Dict[str, Any]] = []
        total = len(self._sess.turns)
        for _ in range(batch_size):
            t = self._sess.turns[self._pos]
            self._pos = (self._pos + 1) % total
            content = t.get("content", "")
            if isinstance(max_chars, int) and max_chars > 0:
                content = content[: max_chars]
            out.append({
                "role": t.get("role", "user"),
                "content": content,
                "metadata": {"source": "dataset", "session_uid": self._sess.uid},
            })
        return out


# Convenience: defaults from env (used by future callers)

def env_dataset_path() -> str:
    return os.getenv("DATASET_PATH", "datasets/lme_s_mc10.json")


def env_sharding() -> str:
    return os.getenv("DATASET_SHARDING", "hash").lower()


def env_turn_batch_size(default_min: int = 1, default_max: int = 3) -> Tuple[int, int]:
    try:
        return int(os.getenv("TURN_BATCH_MIN", str(default_min))), int(os.getenv("TURN_BATCH_MAX", str(default_max)))
    except Exception:
        return default_min, default_max


def env_max_chars() -> Optional[int]:
    v = os.getenv("MAX_CHARS", "").strip()
    try:
        return int(v) if v else None
    except Exception:
        return None

