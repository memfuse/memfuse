import os
import random
import string
import time
import uuid
from typing import Dict, List, Optional, Tuple


API_PREFIX = os.getenv("API_PREFIX", "/api/v1").rstrip("/")
ENTITY_PREFIX = os.getenv("ENTITY_PREFIX", "perf")


def _bearer_headers() -> Dict[str, str]:
    token = os.getenv("API_KEY")
    header_name = os.getenv("API_KEY_HEADER", "Authorization")
    if token:
        return {header_name: f"Bearer {token}"}
    return {}


def api_url(path: str) -> str:
    path = path if path.startswith("/") else f"/{path}"
    return f"{API_PREFIX}{path}"


def rand_suffix(n: int = 8) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=n))


def random_text(size_profile: str = "mixed") -> str:
    if size_profile == "short":
        n = random.randint(50, 200)
    elif size_profile == "long":
        n = random.randint(2000, 8000)
    else:  # mixed
        bucket = random.random()
        if bucket < 0.6:
            n = random.randint(50, 200)
        elif bucket < 0.9:
            n = random.randint(500, 1000)
        else:
            n = random.randint(2000, 4000)
    return "lorem " * (n // 6)


def ensure_user(client, name: Optional[str] = None) -> Dict:
    name = name or f"{ENTITY_PREFIX}_user_{rand_suffix()}"
    headers = _bearer_headers()

    # Try fetch by name
    # Treat 404 (not found) as a successful check to avoid counting expected
    # idempotency lookups as Locust failures.
    with client.get(
        api_url(f"/users"),
        params={"name": name},
        headers=headers,
        name="GET /users?name=",
        catch_response=True,
    ) as r:
        if r.status_code == 200:
            js = r.json()
            users = (js.get("data") or {}).get("users") or []
            if users:
                r.success()
                return users[0]
        elif r.status_code == 404:
            # Expected when ensuring resources; mark as success to avoid skewing failure stats
            r.success()

    # Create
    r = client.post(api_url("/users"), json={"name": name, "description": "perf user"}, headers=headers, name="POST /users")
    r.raise_for_status()
    return (r.json().get("data") or {}).get("user")


def ensure_agent(client, name: Optional[str] = None) -> Dict:
    name = name or f"{ENTITY_PREFIX}_agent_{rand_suffix()}"
    headers = _bearer_headers()

    # Try fetch by name
    with client.get(
        api_url(f"/agents"),
        params={"name": name},
        headers=headers,
        name="GET /agents?name=",
        catch_response=True,
    ) as r:
        if r.status_code == 200:
            js = r.json()
            agents = (js.get("data") or {}).get("agents") or []
            if agents:
                r.success()
                return agents[0]
        elif r.status_code == 404:
            r.success()

    # Create
    r = client.post(api_url("/agents"), json={"name": name, "description": "perf agent"}, headers=headers, name="POST /agents")
    r.raise_for_status()
    return (r.json().get("data") or {}).get("agent")


def create_session(client, user_id: str, agent_id: str, name: Optional[str] = None) -> Dict:
    headers = _bearer_headers()
    if not name:
        name = f"{ENTITY_PREFIX}_sess_{rand_suffix()}"
    payload = {"user_id": user_id, "agent_id": agent_id, "name": name}
    r = client.post(api_url("/sessions"), json=payload, headers=headers, name="POST /sessions")
    r.raise_for_status()
    return (r.json().get("data") or {}).get("session")


def add_messages(client, session_id: str, messages: List[Dict]) -> List[str]:
    headers = _bearer_headers()
    r = client.post(api_url(f"/sessions/{session_id}/messages"), json={"messages": messages}, headers=headers, name="POST /sessions/:id/messages")
    r.raise_for_status()
    js = r.json()
    return ((js.get("data") or {}).get("message_ids")) or []


def list_messages(client, session_id: str, limit: int = 20, order: str = "desc") -> List[Dict]:
    headers = _bearer_headers()
    r = client.get(
        api_url(f"/sessions/{session_id}/messages"),
        params={"limit": str(limit), "sort_by": "timestamp", "order": order},
        headers=headers,
        name="GET /sessions/:id/messages",
    )
    r.raise_for_status()
    js = r.json()
    return ((js.get("data") or {}).get("messages")) or []


def update_messages(client, session_id: str, message_ids: List[str], new_messages: List[Dict]) -> None:
    headers = _bearer_headers()
    payload = {"message_ids": message_ids, "new_messages": new_messages}
    r = client.put(api_url(f"/sessions/{session_id}/messages"), json=payload, headers=headers, name="PUT /sessions/:id/messages")
    r.raise_for_status()


def delete_messages(client, session_id: str, message_ids: List[str]) -> None:
    headers = _bearer_headers()
    payload = {"message_ids": message_ids}
    r = client.delete(api_url(f"/sessions/{session_id}/messages"), json=payload, headers=headers, name="DELETE /sessions/:id/messages")
    r.raise_for_status()


def query_user_memory(client, user_id: str, session_id: Optional[str] = None) -> None:
    headers = _bearer_headers()
    q = {
        "query": f"test query {rand_suffix()}",
        "session_id": session_id,
        "top_k": random.choice([5, 10, 20]),
        "include_messages": True,
        "include_knowledge": False,
        "metadata": {"source": "locust"},
    }
    r = client.post(api_url(f"/users/{user_id}/query"), json=q, headers=headers, name="POST /users/:id/query")
    r.raise_for_status()


def get_session_chunks(client, session_id: str, limit: int = 20) -> None:
    headers = _bearer_headers()
    r = client.get(api_url(f"/sessions/{session_id}/chunks"), params={"limit": str(limit)}, headers=headers, name="GET /sessions/:id/chunks")
    r.raise_for_status()


def make_message_batch(profile: str = "mixed", n: Optional[int] = None) -> List[Dict]:
    count = n if n is not None else random.choice([1, 2, 3])
    batch: List[Dict] = []
    for i in range(count):
        role = random.choice(["user", "assistant"]) if i % 2 else "user"
        batch.append({
            "role": role,
            "content": random_text(profile)[:4000],  # safety cap
            "metadata": {"task": random.choice(["", "general", "support"]) or "general"},
        })
    return batch
