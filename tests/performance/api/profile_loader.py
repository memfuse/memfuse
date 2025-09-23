import json
import os
from typing import Any, Dict


DEFAULT_PROFILE: Dict[str, Any] = {
    "name": "default",
    "weights": {
        "messages_add_list": 6,
        "messages_update": 2,
        "messages_delete": 1,
        "user_query": 2,
        "session_chunks": 1,
    },
    "think_time_ms": {"min": 100, "max": 500},
    "message_size": "mixed",
}


def _default_profiles_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "profiles")


def _resolve_profile_path(profile_name: str) -> str:
    # Allow overriding with explicit path
    override_path = os.getenv("PROFILE_PATH")
    if override_path:
        return override_path
    return os.path.join(_default_profiles_dir(), f"{profile_name}.json")


def load_profile() -> Dict[str, Any]:
    name = os.getenv("PROFILE", "load")
    path = _resolve_profile_path(name)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            data.setdefault("name", name)
            return data
    except Exception:
        # Fallback to baked defaults
        p = DEFAULT_PROFILE.copy()
        p["name"] = name
        return p

