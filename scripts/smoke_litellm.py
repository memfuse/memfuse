#!/usr/bin/env python3
"""LiteLLM smoke test for structured + streaming calls.

Uses env vars: OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_COMPATIBLE_MODEL.
Falls back to loading .env if present.
"""

import asyncio
import os
import sys
from typing import Optional


def load_env_file(path: str = ".env") -> None:
    """Simple .env loader (no external deps)."""
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    # strip optional quotes
                    value = value.strip().strip('"').strip("'")
                    os.environ.setdefault(key, value)
    except Exception:
        # Non-fatal
        pass


async def main() -> int:
    # Ensure repo src/ is importable
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    SRC = os.path.join(ROOT, "src")
    if SRC not in sys.path:
        sys.path.insert(0, SRC)

    load_env_file()

    from memfuse_core.llm.providers.litellm import LiteLLMProvider
    from memfuse_core.llm.base import LLMRequest
    from memfuse_core.models.m2_extraction import FactExtractionResponse

    api_key = os.getenv("QWEN_API_KEY")
    base_url = os.getenv("QWEN_BASE_URL")
    model = os.getenv("QWEN_MODEL") or "gpt-5-nano"
    # api_key = os.getenv("OPENAI_API_KEY")
    # base_url = os.getenv("OPENAI_BASE_URL")
    # model = os.getenv("OPENAI_COMPATIBLE_MODEL") or "gpt-5-nano"

    provider = LiteLLMProvider(
        {
            "api_key": api_key,
            "base_url": base_url,
            "timeout": 30.0,
        }
    )

    # Structured test
    messages = [
        {
            "role": "user",
            "content": (
                "Extract at most 2 simple facts with short content and a dummy"
                " source id list from: 'The Eiffel Tower is in Paris. It was built in 1889.'"
            ),
        }
    ]
    req = LLMRequest(messages=messages, model=model, temperature=0.1, max_tokens=200)
    print(f"Model: {model}")
    print("Calling generate_structured(...) via LiteLLMProvider ...")
    resp = await provider.generate_structured(req, FactExtractionResponse)
    print("structured.success:", resp.success)
    print("structured.model:", resp.model)
    print("structured.error:", resp.error)
    parsed_type: Optional[str] = type(resp.parsed_data).__name__ if resp.parsed_data else None
    print("structured.parsed_type:", parsed_type)
    try:
        facts_count = len(resp.parsed_data.facts) if resp.parsed_data else 0
    except Exception:
        facts_count = 0
    print("structured.parsed_data:", resp.parsed_data)
    print("structured.facts_count:", facts_count)

    # Streaming test
    print("\nCalling generate_stream(...) via LiteLLMProvider ...")
    req2 = LLMRequest(
        messages=[{"role": "user", "content": "Say 'hello' in one short line."}],
        model=model,
        temperature=0.1,
        max_tokens=20,
        stream=True,
    )
    chunks = []
    async for chunk in provider.generate_stream(req2):
        if chunk:
            chunks.append(chunk)
            if len("".join(chunks)) > 200:
                break
    out = "".join(chunks)
    print("stream.length:", len(out))
    print("stream.preview:", out[:120].replace("\n", "\\n"))

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

