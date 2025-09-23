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


# Load scenario profile (env PROFILE, PROFILE_PATH)
_PROFILE = load_profile()

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
        session = create_session(self.client, user_id=user["id"], agent_id=agent["id"])

        # Store for tasks
        self.user_id = user["id"]
        self.agent_id = agent["id"]
        self.session_id = session["id"]
        self._message_ids: List[str] = []

        # Seed a few messages
        mids = add_messages(self.client, self.session_id, make_message_batch(MSG_SIZE_PROFILE))
        if mids:
            self._message_ids.extend(mids)

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
        sess = self.session_id if random.random() < 0.7 else None
        query_user_memory(self.client, self.user_id, session_id=sess)

    def t_session_chunks(self):
        get_session_chunks(self.client, self.session_id, limit=random.choice([20, 50]))

# Apply dynamic task weights from profile
_W = _PROFILE.get("weights", {})
ApiUser.tasks = {
    ApiUser.t_messages_add_list: int(_W.get("messages_add_list", 6)),
    ApiUser.t_messages_update: int(_W.get("messages_update", 2)),
    ApiUser.t_messages_delete: int(_W.get("messages_delete", 1)),
    ApiUser.t_user_query: int(_W.get("user_query", 2)),
    ApiUser.t_session_chunks: int(_W.get("session_chunks", 1)),
}
