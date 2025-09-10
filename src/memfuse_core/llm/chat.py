from __future__ import annotations

"""Unified ChatLLM abstraction compatible with MVP semantics.

Two primary entry points:
- completion_json(system_prompt: str, user_prompt: str) -> str
- chat(system_prompt: str, messages: list[dict]) -> str

Implementation strategy:
- Prefer OpenAI-compatible client if environment provides config.
- Provide offline fallbacks for tests and air-gapped environments.
"""

import json
import os
from typing import List, Dict, Any


class ChatLLM:
    def __init__(self) -> None:
        # OpenAI-compatible settings
        self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.base_url = os.getenv("OPENAI_BASE_URL", "").strip()
        self.model = os.getenv("OPENAI_COMPATIBLE_MODEL", "").strip() or "gpt-4o-mini"
        self._online = bool(self.api_key and self.model)

        # Lazy init client to avoid import cost when offline
        self._client = None

    # ---------------------------- Internals ----------------------------
    def _ensure_client(self) -> None:
        if not self._online:
            return
        if self._client is not None:
            return
        try:
            from openai import OpenAI  # type: ignore

            if self.base_url:
                self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            else:
                self._client = OpenAI(api_key=self.api_key)
        except Exception:
            # downgrade to offline mode if client creation fails
            self._online = False
            self._client = None

    # ---------------------------- Public API ----------------------------
    def completion_json(self, system_prompt: str, user_prompt: str) -> str:
        """Return a JSON string response for planner-like requests.

        If online, call OpenAI-compatible Chat Completions.
        Otherwise, return a minimal safe default JSON.
        """
        if self._online:
            try:
                self._ensure_client()
                if self._client is not None:
                    resp = self._client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=0,
                    )
                    content = resp.choices[0].message.content if resp.choices else "{}"
                    # Ensure it's JSON; if not, extract JSON substring fallback
                    try:
                        json.loads(content)
                        return content
                    except Exception:
                        # heuristic: find first/last brace
                        s = content.find("{")
                        e = content.rfind("}")
                        if s != -1 and e != -1 and e > s:
                            blob = content[s : e + 1]
                            json.loads(blob)  # validate
                            return blob
                        return "{}"
            except Exception:
                # fall through to offline fallback
                pass

        # Offline fallback: minimal plan structure
        return json.dumps({
            "steps": [
                {"agent": "RAGQueryAgent", "input": {}},
                {"agent": "ReportGenerationAgent", "input": {}},
            ]
        })

    def chat(self, system_prompt: str, messages: List[Dict[str, Any]]) -> str:
        """Return a plain text assistant response.

        messages is a list of {role, content}.
        """
        if self._online:
            try:
                self._ensure_client()
                if self._client is not None:
                    resp = self._client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            *messages,
                        ],
                        temperature=0.3,
                    )
                    return resp.choices[0].message.content or ""
            except Exception:
                pass

        # Offline fallback: echo-summarize last user content
        last_user = next((m for m in reversed(messages) if m.get("role") == "user"), None)
        content = last_user.get("content") if last_user else ""
        return f"[offline] Summary: {content[:500]}"

