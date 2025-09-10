from __future__ import annotations

"""Minimal RAG service (Phase A) for MemFuse Core.

This is a lightweight placeholder to provide a stable interface for the
Orchestrator's RAGQueryAgent during Phase A. It avoids DB dependencies
for unit tests and will be extended/realigned in later phases.
"""

from typing import List
from ..llm.chat import ChatLLM


class RAGService:
    def __init__(self) -> None:
        self._llm = ChatLLM()

    def ingest_document(self, source: str, content: str, chunk_size: int = 800) -> int:
        # Placeholder: real ingestion will split + embed + persist
        # Phase A keeps a no-op to avoid DB requirements in unit tests
        if not content:
            return 0
        # simulate n chunks
        words = content.split()
        return max(1, len(words) // max(1, chunk_size))

    def chat(self, session_id: str, user_query: str) -> str:
        # Minimal chat that leverages LLM to produce an answer
        system = "You are MemFuse RAG assistant. Provide concise answers."
        return self._llm.chat(system, [{"role": "user", "content": user_query}])

