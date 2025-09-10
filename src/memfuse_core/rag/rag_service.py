from __future__ import annotations

"""Minimal RAG service (Phase A) for MemFuse Core.

This is a lightweight placeholder to provide a stable interface for the
Orchestrator's RAGQueryAgent during Phase A. It avoids DB dependencies
for unit tests and will be extended/realigned in later phases.
"""

from typing import List, Tuple, Dict, Any
from ..llm.chat import ChatLLM
from ..utils.embeddings import create_embedding, cosine_similarity


class RAGService:
    """Pragmatic RAG service with in-memory per-session index.

    This Phase A.5 uplift provides:
    - Simple chunking
    - In-memory vector index per session
    - Retrieval to build minimal context for LLM
    """

    def __init__(self) -> None:
        self._llm = ChatLLM()
        # session_id -> list of (chunk_text, embedding)
        self._index: Dict[str, List[Tuple[str, List[float]]]] = {}

    # ----------------------- Index helpers -----------------------
    def _chunk_text(self, text: str, chunk_size: int = 200) -> List[str]:
        words = text.split()
        out: List[str] = []
        cur: List[str] = []
        for w in words:
            cur.append(w)
            if len(cur) >= chunk_size:
                out.append(" ".join(cur))
                cur = []
        if cur:
            out.append(" ".join(cur))
        return out if out else ([text] if text else [])

    async def _add_to_index(self, session_id: str, chunks: List[str]) -> int:
        if not chunks:
            return 0
        embs = [await create_embedding(c) for c in chunks]
        arr = self._index.setdefault(session_id, [])
        for c, e in zip(chunks, embs):
            arr.append((c, e))
        return len(chunks)

    async def ensure_session_index(self, session_id: str, history_messages: List[Dict[str, Any]], chunk_size: int = 200) -> int:
        """Build index from history messages if empty.

        history_messages: list of dicts with keys at least {role, content}
        """
        if self._index.get(session_id):
            return 0
        texts: List[str] = []
        for m in history_messages or []:
            try:
                role = str(m.get("role", ""))
                if role in ("user", "assistant"):
                    content = str(m.get("content", ""))
                    if content:
                        texts.extend(self._chunk_text(content, chunk_size=chunk_size))
            except Exception:
                continue
        return await self._add_to_index(session_id, texts)

    # ----------------------- Public API -----------------------
    async def ingest_document(self, session_id: str, content: str, chunk_size: int = 800, source: str | None = None) -> int:
        chunks = self._chunk_text(content, chunk_size=chunk_size)
        return await self._add_to_index(session_id, chunks)

    async def _retrieve(self, session_id: str, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        qemb = await create_embedding(query)
        arr = self._index.get(session_id, [])
        scored: List[Tuple[str, float]] = []
        for text, emb in arr:
            try:
                s = float(cosine_similarity(emb, qemb))
            except Exception:
                s = 0.0
            scored.append((text, s))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[: top_k]

    async def chat(self, session_id: str, user_query: str, history_messages: List[Dict[str, Any]] | None = None, top_k: int = 5) -> str:
        # Build index if needed
        try:
            await self.ensure_session_index(session_id, history_messages or [], chunk_size=200)
        except Exception:
            pass
        # Retrieve
        try:
            retrieved = await self._retrieve(session_id, user_query, top_k=max(1, top_k))
        except Exception:
            retrieved = []
        # Build context
        context_lines = ["[Retrieved Chunks]"]
        for i, (text, score) in enumerate(retrieved, start=1):
            context_lines.append(f"({i}) score={score:.2f}: {text[:300]}")
        context_lines.append("[End Retrieved]")
        context = "\n".join(context_lines)
        # Compose messages (system prompt passed separately; do not duplicate inside messages)
        system = "You are MemFuse RAG assistant. Use retrieved chunks when relevant."
        messages = [
            {"role": "user", "content": f"Context:\n{context}\n\nQuery: {user_query}"},
        ]
        return self._llm.chat(system, messages)
