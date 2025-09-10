#!/usr/bin/env python3
"""
Quick inspector for a single M1 chunk and its session context using
SimplifiedMemoryService helper methods.

Usage:
  poetry run python scripts/m2_fetch_context.py \
    --chunk-id 60b36d37-6856-4fbe-aaf8-8a32a04a5d8d \
    --context-before 4

Environment:
  Uses the service's default DB config or POSTGRES_* env vars.

Notes:
  - Disables M2 background worker for this script run.
  - Prints chunk summary and a formatted list of prior context chunks.
"""

import argparse
import asyncio
import json
from typing import Optional

from datetime import datetime


async def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a single M1 chunk + context")
    parser.add_argument("--chunk-id", required=True, help="Target M1 chunk UUID")
    parser.add_argument(
        "--context-before",
        type=int,
        default=4,
        help="Number of context chunks before target to fetch",
    )
    parser.add_argument(
        "--user",
        default="user_default",
        help="User name for service context (default: user_default)",
    )
    args = parser.parse_args()

    # Lazy import within script path context
    from memfuse_core.services.simplified_memory_service import SimplifiedMemoryService

    # Disable M2 worker in this script to avoid background noise
    cfg = {"m2_enabled": False}
    svc = SimplifiedMemoryService(cfg=cfg, user=args.user)

    await svc.initialize()

    try:
        chunk = await svc._get_m1_chunk(args.chunk_id, user_id=None)
        if not chunk:
            print(f"❌ Chunk not found: {args.chunk_id}")
            return 1

        print("\n=== Target Chunk ===")
        print(f"chunk_id     : {chunk.get('chunk_id')}")
        print(f"user_id      : {chunk.get('user_id')}")
        print(f"session_id   : {chunk.get('session_id')}")
        print(f"token_count  : {chunk.get('token_count')}")
        created_at = chunk.get('created_at')
        if isinstance(created_at, datetime):
            created_at = created_at.isoformat()
        print(f"created_at   : {created_at}")
        print(f"m2_status    : {chunk.get('m2_status')}")
        print(f"m0_raw_ids   : {len(chunk.get('m0_raw_ids') or [])} items")

        # Fetch context
        context = await svc._get_session_context_for_chunk(
            args.chunk_id, context_chunks_before=args.context_before, user_id=None
        )
        if context is None:
            print("❌ Context result is None (target chunk may be missing session_id)")
            return 2

        print("\n=== Context (oldest → newest) ===")
        print(f"requested: {args.context_before}, fetched: {len(context)}\n")
        for i, c in enumerate(context, 1):
            c_created = getattr(c, "created_at", None)
            if isinstance(c_created, datetime):
                c_created = c_created.isoformat()
            print(
                f"[{i}] chunk_id={getattr(c, 'chunk_id', '')} "
                f"tokens={getattr(c, 'token_count', 0)} "
                f"created_at={c_created} "
                f"m2_status={getattr(c, 'm2_status', None)}"
            )

        # Optional: print a compact preview of the target content
        content_preview = (chunk.get('content') or '')[:200].replace('\n', ' ')
        print(f"\n=== Target Content Preview (200 chars) ===\n{content_preview}\n")

        return 0
    finally:
        await svc.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

