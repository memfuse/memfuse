#!/usr/bin/env python3
"""
End-to-end fact extraction for a single chunk using service helpers + LiteLLM.

- Fetches target chunk and prior context via SimplifiedMemoryService methods
- Builds the M2 extraction prompt with the existing templates
- Calls LiteLLMProvider (configured like scripts/smoke_litellm.py) to extract facts
  using the same _extract_facts_with_retry logic as the service

Usage:
  poetry run python scripts/m2_extract_facts.py \
    --chunk-id 60b36d37-6856-4fbe-aaf8-8a32a04a5d8d \
    --context-before 4 \
    --budget 2000 \
    [--model gpt-5-nano] \
    [--user user_default]

Environment:
  Preferred (Qwen-style): QWEN_API_KEY, QWEN_BASE_URL, QWEN_MODEL
  Alternative (OpenAI-compatible): OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_COMPATIBLE_MODEL
"""

import argparse
import asyncio
import os
from datetime import datetime


async def main() -> int:
    parser = argparse.ArgumentParser(description="Extract facts from a single M1 chunk")
    parser.add_argument("--chunk-id", required=True, help="Target M1 chunk UUID")
    parser.add_argument("--context-before", type=int, default=4, help="How many prior context chunks")
    parser.add_argument("--budget", type=int, default=2000, help="Token budget for context")
    parser.add_argument("--model", default=None, help="Override model name (default from env)")
    parser.add_argument(
        "--print-prompt",
        action="store_true",
        help="Print the constructed system/user prompts and exit",
    )
    parser.add_argument("--user", default="user_default", help="Service user (default: user_default)")
    args = parser.parse_args()

    # Lazy imports
    from memfuse_core.services.simplified_memory_service import SimplifiedMemoryService
    from memfuse_core.llm.base import LLMRequest
    from memfuse_core.llm.providers.litellm import LiteLLMProvider

    # Configure provider similar to scripts/smoke_litellm.py
    api_key = (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("QWEN_API_KEY")
    )
    base_url = (
        os.getenv("OPENAI_BASE_URL")
        or os.getenv("QWEN_BASE_URL")
    )
    model = (
        args.model
        or os.getenv("OPENAI_COMPATIBLE_MODEL")
        or os.getenv("QWEN_MODEL")
    )

    provider = LiteLLMProvider({
        "api_key": api_key,
        "base_url": base_url,
        "timeout": 30.0,
    })

    # Disable M2 worker for this script to avoid background interference
    cfg = {"m2_enabled": False}
    svc = SimplifiedMemoryService(cfg=cfg, user=args.user)
    await svc.initialize()

    try:
        # Fetch target chunk + context
        chunk = await svc._get_m1_chunk(args.chunk_id, user_id=None)
        if not chunk:
            print(f"❌ Chunk not found: {args.chunk_id}")
            return 1

        context = await svc._get_session_context_for_chunk(
            args.chunk_id, context_chunks_before=args.context_before, user_id=None
        )
        context = context or []

        # Apply token budget
        context_trunc = svc._apply_token_budget_limit(context, limit=args.budget)

        # Build LLM messages
        messages = svc._build_fact_extraction_prompt(chunk, context_trunc)

        # Optionally print the prompts and exit
        if args.print_prompt:
            print("\n=== System Prompt ===\n")
            print(messages[0]["content"]) if messages else None
            print("\n=== User Prompt ===\n")
            print(messages[1]["content"]) if len(messages) > 1 else None
            return 0

        # Build request
        req = LLMRequest(
            messages=messages,
            model=model,
            temperature=0.3,
            max_tokens=1500,
        )

        # Extract facts using service retry logic
        facts = await svc._extract_facts_with_retry(provider, req)

        print("\n=== Extraction Summary ===")
        print(f"model          : {model}")
        print(f"context_before : requested={args.context_before}, used={len(context_trunc)}")
        print(f"facts_count    : {len(facts)}")

        if facts:
            print("\n=== Facts ===")
            for i, f in enumerate(facts, 1):
                print(f"{i}. {f}")
        else:
            print("No facts extracted.")

        return 0
    finally:
        await svc.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
