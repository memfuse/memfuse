#!/usr/bin/env python3
"""
Seed MemFuse with users, agents, sessions, and messages for performance testing.

The script is idempotent for names: it "ensures" users/agents/sessions exist,
creating them only if missing. Messages are always appended when requested.

Examples
  # Minimal seed with defaults
  poetry run python scripts/perf/seed_data.py \
    --base-url http://localhost:8000 --users 3 --agents 1 --sessions-per-user 2 --messages-per-session 10

  # Larger dataset with mixed message sizes and moderate concurrency
  poetry run python scripts/perf/seed_data.py \
    --base-url http://localhost:8000 --users 10 --agents 2 --sessions-per-user 3 \
    --messages-per-session 50 --msg-size-profile mixed --concurrency 8

Auth
  If API key validation is enabled on the server, pass --api-key YOUR_TOKEN
  and optionally --api-key-header (defaults to Authorization, sent as Bearer).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import string
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import aiohttp


def rand_suffix(n: int = 8) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=n))


def random_text(size_profile: str = "mixed") -> str:
    if size_profile == "short":
        n = random.randint(50, 200)
    elif size_profile == "long":
        n = random.randint(2000, 8000)
    else:
        bucket = random.random()
        if bucket < 0.6:
            n = random.randint(50, 200)
        elif bucket < 0.9:
            n = random.randint(500, 1000)
        else:
            n = random.randint(2000, 4000)
    return ("lorem ipsum dolor sit amet ") * max(1, n // 26)


@dataclass
class SeedConfig:
    base_url: str
    api_prefix: str
    api_key: Optional[str]
    api_key_header: str
    entity_prefix: str
    users: int
    agents: int
    sessions_per_user: int
    messages_per_session: int
    msg_size_profile: str
    per_agent_sessions: bool
    concurrency: int


def build_headers(cfg: SeedConfig) -> Dict[str, str]:
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if cfg.api_key:
        headers[cfg.api_key_header] = f"Bearer {cfg.api_key}"
    return headers


def u(cfg: SeedConfig, path: str) -> str:
    path = path if path.startswith("/") else f"/{path}"
    return f"{cfg.base_url}{cfg.api_prefix}{path}"


async def ensure_user(session: aiohttp.ClientSession, cfg: SeedConfig, name: str) -> Dict[str, Any]:
    # Try GET by name (returns 404 if not found)
    params = {"name": name}
    async with session.get(u(cfg, "/users"), params=params) as r:
        if r.status == 200:
            js = await r.json()
            users = (js.get("data") or {}).get("users") or []
            if users:
                return users[0]
        # else 4xx -> create

    payload = {"name": name, "description": "perf user"}
    async with session.post(u(cfg, "/users"), json=payload) as r:
        r.raise_for_status()
        js = await r.json()
        return (js.get("data") or {}).get("user")


async def ensure_agent(session: aiohttp.ClientSession, cfg: SeedConfig, name: str) -> Dict[str, Any]:
    params = {"name": name}
    async with session.get(u(cfg, "/agents"), params=params) as r:
        if r.status == 200:
            js = await r.json()
            agents = (js.get("data") or {}).get("agents") or []
            if agents:
                return agents[0]

    payload = {"name": name, "description": "perf agent"}
    async with session.post(u(cfg, "/agents"), json=payload) as r:
        r.raise_for_status()
        js = await r.json()
        return (js.get("data") or {}).get("agent")


async def ensure_session(session: aiohttp.ClientSession, cfg: SeedConfig, user: Dict[str, Any], agent: Dict[str, Any], name: str) -> Dict[str, Any]:
    params = {"name": name, "user_id": user["id"]}
    async with session.get(u(cfg, "/sessions"), params=params) as r:
        if r.status == 200:
            js = await r.json()
            sessions = (js.get("data") or {}).get("sessions") or []
            if sessions:
                return sessions[0]

    payload = {"user_id": user["id"], "agent_id": agent["id"], "name": name}
    async with session.post(u(cfg, "/sessions"), json=payload) as r:
        r.raise_for_status()
        js = await r.json()
        return (js.get("data") or {}).get("session")


async def add_messages(session: aiohttp.ClientSession, cfg: SeedConfig, session_id: str, count: int, size_profile: str) -> List[str]:
    messages = []
    for i in range(count):
        role = "user" if i % 2 == 0 else "assistant"
        messages.append({
            "role": role,
            "content": random_text(size_profile)[:4000],
            "metadata": {"task": random.choice(["general", "support", "qa"])},
        })
    async with session.post(u(cfg, f"/sessions/{session_id}/messages"), json={"messages": messages}) as r:
        r.raise_for_status()
        js = await r.json()
        return ((js.get("data") or {}).get("message_ids")) or []


async def seed_user(
    session: aiohttp.ClientSession,
    cfg: SeedConfig,
    user_idx: int,
    agents_cache: List[Dict[str, Any]],
) -> Dict[str, Any]:
    user_name = f"{cfg.entity_prefix}_user_{user_idx:03d}"
    user = await ensure_user(session, cfg, user_name)

    created_sessions: List[Dict[str, Any]] = []
    used_agents = agents_cache[: cfg.agents]

    for a_idx, agent in enumerate(used_agents):
        n_sessions = cfg.sessions_per_user if cfg.per_agent_sessions else (cfg.sessions_per_user if a_idx == 0 else 0)
        for s in range(n_sessions):
            sess_name = f"{cfg.entity_prefix}_sess_{user_idx:03d}_{a_idx:02d}_{s:02d}"
            sess = await ensure_session(session, cfg, user, agent, sess_name)
            await add_messages(session, cfg, sess["id"], cfg.messages_per_session, cfg.msg_size_profile)
            created_sessions.append(sess)

    return {"user": user, "sessions": created_sessions}


async def run_seed(cfg: SeedConfig) -> Dict[str, Any]:
    headers = build_headers(cfg)
    timeout = aiohttp.ClientTimeout(total=None, connect=30)
    connector = aiohttp.TCPConnector(limit=cfg.concurrency)
    async with aiohttp.ClientSession(headers=headers, timeout=timeout, connector=connector) as http:
        # Prepare agents (shared across users)
        agents: List[Dict[str, Any]] = []
        for i in range(cfg.agents):
            a_name = f"{cfg.entity_prefix}_agent_{i:03d}"
            agent = await ensure_agent(http, cfg, a_name)
            agents.append(agent)

        # Seed users concurrently with bounded tasks
        sem = asyncio.Semaphore(cfg.concurrency)
        results: List[Dict[str, Any]] = []

        async def worker(idx: int):
            async with sem:
                res = await seed_user(http, cfg, idx, agents)
                results.append(res)

        tasks = [asyncio.create_task(worker(i)) for i in range(cfg.users)]
        await asyncio.gather(*tasks)

        return {"agents": agents, "users": results}


def parse_args() -> SeedConfig:
    p = argparse.ArgumentParser(description="Seed MemFuse dataset for perf testing")
    p.add_argument("--base-url", default=os.getenv("BASE_URL", "http://localhost:8000"))
    p.add_argument("--api-prefix", default=os.getenv("API_PREFIX", "/api/v1"))
    p.add_argument("--api-key", default=os.getenv("API_KEY"))
    p.add_argument("--api-key-header", default=os.getenv("API_KEY_HEADER", "Authorization"))
    p.add_argument("--entity-prefix", default=os.getenv("ENTITY_PREFIX", "perf"))

    p.add_argument("--users", type=int, default=3)
    p.add_argument("--agents", type=int, default=1)
    p.add_argument("--sessions-per-user", type=int, default=2)
    p.add_argument("--messages-per-session", type=int, default=20)
    p.add_argument("--msg-size-profile", choices=["short", "mixed", "long"], default=os.getenv("MSG_SIZE_PROFILE", "mixed"))
    p.add_argument("--per-agent-sessions", action="store_true", help="Create sessions_per_user per agent for each user")
    p.add_argument("--concurrency", type=int, default=int(os.getenv("SEED_CONCURRENCY", "6")))

    args = p.parse_args()
    return SeedConfig(
        base_url=args.base_url.rstrip("/"),
        api_prefix=("/" + args.api_prefix.strip("/")).rstrip("/"),
        api_key=args.api_key,
        api_key_header=args.api_key_header,
        entity_prefix=args.entity_prefix,
        users=args.users,
        agents=args.agents,
        sessions_per_user=args.sessions_per_user,
        messages_per_session=args.messages_per_session,
        msg_size_profile=args.msg_size_profile,
        per_agent_sessions=bool(args.per_agent_sessions),
        concurrency=max(1, args.concurrency),
    )


async def amain() -> int:
    cfg = parse_args()
    result = await run_seed(cfg)
    print(json.dumps(result, indent=2))
    return 0


def main() -> int:
    return asyncio.run(amain())


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

