import os
import random
from typing import List

from locust import HttpUser, between

from utils import (
    api_url,
    ensure_user,
    ensure_agent,
    create_session,
    add_messages,
    list_messages,
    update_messages,
    delete_messages,
    query_user_memory,
    get_session_chunks,
    make_message_batch,
)
from profile_loader import load_profile
from soak_helpers import (
    get_sessions_range,
    get_turn_batch_bounds,
    run_tick,
)
from scripts.perf.message_feeder import load_dataset, map_to_session, FeederCursor, env_dataset_path, env_sharding, env_turn_batch_size, env_max_chars


# Load scenario profile (env PROFILE, PROFILE_PATH)
_PROFILE = load_profile()
_DATASET = load_dataset(env_dataset_path())
_SHARD_STRATEGY = env_sharding()
_TURN_MIN, _TURN_MAX = env_turn_batch_size()
_MAX_CHARS = env_max_chars()

# Think time (allow env overrides)
WAIT_MIN_MS = int(os.getenv("WAIT_MIN_MS", str(_PROFILE.get("think_time_ms", {}).get("min", 100))))
WAIT_MAX_MS = int(os.getenv("WAIT_MAX_MS", str(_PROFILE.get("think_time_ms", {}).get("max", 500))))

# Message size profile (allow env override)
MSG_SIZE_PROFILE = os.getenv("MSG_SIZE_PROFILE", _PROFILE.get("message_size", "mixed"))


class ApiUser(HttpUser):
    wait_time = between(WAIT_MIN_MS / 1000.0, WAIT_MAX_MS / 1000.0)

    def on_start(self):
        # Create/ensure state for this VU
        user = ensure_user(self.client)
        agent = ensure_agent(self.client)
        self.user_id = user["id"]
        self.agent_id = agent["id"]
        self._message_ids: List[str] = []
        # Create 5–10 sessions per user (configurable)
        mn, mx = get_sessions_range()
        sess_count = random.randint(mn, mx)
        self._session_ids: List[str] = []
        self._feeder: dict[str, FeederCursor] = {}
        for sidx in range(sess_count):
            sess = create_session(self.client, user_id=self.user_id, agent_id=self.agent_id)
            sid = sess["id"]
            self._session_ids.append(sid)
            # Map to dataset session and create a cursor if dataset available
            if _DATASET.size > 0:
                ds_idx = map_to_session(_DATASET, self.user_id, sidx, strategy=_SHARD_STRATEGY)
                self._feeder[sid] = FeederCursor(_DATASET, ds_idx)
            # Optionally seed a small batch at start to warm paths
            try:
                if _DATASET.size > 0:
                    cur = self._feeder.get(sid)
                    if cur is not None:
                        msgs = cur.next_batch(batch_size=max(1, _TURN_MIN), max_chars=_MAX_CHARS)
                        mids = add_messages(self.client, sid, msgs)
                        if mids:
                            self._message_ids.extend(mids)
                else:
                    mids = add_messages(self.client, sid, make_message_batch(MSG_SIZE_PROFILE))
                    if mids:
                        self._message_ids.extend(mids)
            except Exception:
                pass

        # Back-compat: provide a default session_id for legacy tasks
        if getattr(self, "_session_ids", None):
            self.session_id = random.choice(self._session_ids)
        else:
            # Fallback: create one if none exists (shouldn't happen)
            session = create_session(self.client, user_id=self.user_id, agent_id=self.agent_id)
            self.session_id = session["id"]

    def t_messages_add_list(self):
        # Add
        mids = add_messages(self.client, self.session_id, make_message_batch(MSG_SIZE_PROFILE))
        if mids:
            self._message_ids.extend(mids)

        # List
        _ = list_messages(
            self.client,
            self.session_id,
            limit=random.choice([20, 50, 100]),
            order=random.choice(["asc", "desc"]),
        )

    def t_messages_update(self):
        if not self._message_ids:
            return
        sample = random.sample(self._message_ids, k=min(len(self._message_ids), random.choice([1, 2, 3])))
        new_msgs = make_message_batch(MSG_SIZE_PROFILE, n=len(sample))
        update_messages(self.client, self.session_id, sample, new_msgs)

    def t_messages_delete(self):
        if not self._message_ids:
            return
        # Delete a small subset to keep session active
        k = 1 if len(self._message_ids) == 1 else max(1, len(self._message_ids) // 10)
        sample = random.sample(self._message_ids, k=min(len(self._message_ids), k))
        delete_messages(self.client, self.session_id, sample)
        # Remove from local cache
        self._message_ids = [m for m in self._message_ids if m not in sample]

    def t_user_query(self):
        # With or without session scope
        sess_choice = None
        if getattr(self, "_session_ids", None):
            sess_choice = random.choice(self._session_ids)
        else:
            sess_choice = getattr(self, "session_id", None)
        sess = sess_choice if random.random() < 0.7 else None
        query_user_memory(self.client, self.user_id, session_id=sess)

    def t_session_chunks(self):
        sid = random.choice(self._session_ids) if getattr(self, "_session_ids", None) else self.session_id
        get_session_chunks(self.client, sid, limit=random.choice([20, 50]))

    def t_user_tick(self):
        # Composite tick: list -> add (dataset) -> query
        session_id = random.choice(self._session_ids) if getattr(self, "_session_ids", None) else self.session_id
        cursor = None
        if hasattr(self, "_feeder"):
            cursor = self._feeder.get(session_id)
        if cursor is None and _DATASET.size > 0 and getattr(self, "_session_ids", None):
            # Fallback: derive a cursor using first session index
            try:
                ds_idx = map_to_session(_DATASET, self.user_id, 0, strategy=_SHARD_STRATEGY)
                cursor = FeederCursor(_DATASET, ds_idx)
            except Exception:
                cursor = None
        # Bounds for messages per tick
        turn_min, turn_max = get_turn_batch_bounds(_TURN_MIN, _TURN_MAX)
        def _list_fn(client, sid, **kwargs):
            return list_messages(client, sid, **kwargs)
        def _add_fn(client, sid, batch):
            if cursor is not None:
                return add_messages(client, sid, batch)
            # fallback to synthetic batch
            return add_messages(client, sid, make_message_batch(MSG_SIZE_PROFILE, n=len(batch) if batch else None))
        def _query_fn(client, uid, session_id=None):
            return query_user_memory(client, uid, session_id=session_id)
        run_tick(
            self.client,
            self.user_id,
            session_id,
            cursor or (lambda *a, **k: None),
            _list_fn,
            _add_fn,
            _query_fn,
            batch_min=turn_min,
            batch_max=turn_max,
        )

# Apply dynamic task weights from profile
_W = _PROFILE.get("weights", {})
ApiUser.tasks = {
    ApiUser.t_messages_add_list: int(_W.get("messages_add_list", 6)),
    ApiUser.t_messages_update: int(_W.get("messages_update", 2)),
    ApiUser.t_messages_delete: int(_W.get("messages_delete", 1)),
    ApiUser.t_user_query: int(_W.get("user_query", 2)),
    ApiUser.t_session_chunks: int(_W.get("session_chunks", 1)),
    ApiUser.t_user_tick: int(_W.get("user_tick", 10)),
}
