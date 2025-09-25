import os
import random
from typing import Callable, List, Optional


def get_sessions_range(default_min: int = 5, default_max: int = 10) -> tuple[int, int]:
    def _int(name: str, dv: int) -> int:
        try:
            return int(os.getenv(name, str(dv)))
        except Exception:
            return dv
    mn = _int("SESSIONS_PER_USER_MIN", default_min)
    mx = _int("SESSIONS_PER_USER_MAX", default_max)
    if mx < mn:
        mn, mx = mx, mn
    mn = max(1, mn)
    mx = max(mn, mx)
    return mn, mx


def select_session_count() -> int:
    mn, mx = get_sessions_range()
    return random.randint(mn, mx)


def get_turn_batch_bounds(default_min: int = 1, default_max: int = 3) -> tuple[int, int]:
    def _int(name: str, dv: int) -> int:
        try:
            return int(os.getenv(name, str(dv)))
        except Exception:
            return dv
    mn = _int("TURN_BATCH_MIN", default_min)
    mx = _int("TURN_BATCH_MAX", default_max)
    if mx < mn:
        mn, mx = mx, mn
    mn = max(1, mn)
    mx = max(mn, mx)
    return mn, mx


def get_max_chars() -> Optional[int]:
    v = (os.getenv("MAX_CHARS") or "").strip()
    try:
        return int(v) if v else None
    except Exception:
        return None


def build_messages_from_feeder(cursor, n: int, max_chars: Optional[int]) -> List[dict]:
    n = max(1, int(n))
    return cursor.next_batch(batch_size=n, max_chars=max_chars)


def run_tick(
    client,
    user_id: str,
    session_id: str,
    cursor,
    list_fn: Callable[..., object],
    add_fn: Callable[..., object],
    query_fn: Callable[..., object],
    batch_min: int = 1,
    batch_max: int = 3,
    query_with_session_prob: float = 0.7,
) -> None:
    # 1) list
    list_fn(client, session_id, limit=50, order="desc")
    # 2) add
    n = random.randint(max(1, batch_min), max(batch_min, batch_max))
    max_chars = get_max_chars()
    msgs = build_messages_from_feeder(cursor, n, max_chars)
    add_fn(client, session_id, msgs)
    # 3) query
    sid = session_id if random.random() < query_with_session_prob else None
    query_fn(client, user_id, session_id=sid)

