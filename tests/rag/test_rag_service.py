import pytest


@pytest.mark.asyncio
async def test_rag_ingest_and_chat(monkeypatch):
    # speed up embeddings
    from memfuse_core.utils import embeddings as emb_mod
    monkeypatch.setattr(emb_mod, "create_embedding", lambda text: [0.1] * 8, raising=False)
    monkeypatch.setattr(emb_mod, "cosine_similarity", lambda a, b: 0.9, raising=False)

    from memfuse_core.llm import chat as chat_mod
    monkeypatch.setattr(chat_mod.ChatLLM, "chat", lambda self, sys, msgs: "ok", raising=False)

    from memfuse_core.rag.rag_service import RAGService

    rag = RAGService()
    # ingest a small doc into session index
    n = await rag.ingest_document("sess1", "A long document about memory systems and MemFuse design.", chunk_size=5)
    assert n > 0
    # chat should use retrieved context
    out = await rag.chat("sess1", "Explain memory", history_messages=[])
    assert isinstance(out, str) and len(out) > 0


@pytest.mark.asyncio
async def test_rag_build_index_from_history(monkeypatch):
    from memfuse_core.utils import embeddings as emb_mod
    monkeypatch.setattr(emb_mod, "create_embedding", lambda text: [0.2] * 8, raising=False)
    from memfuse_core.llm import chat as chat_mod
    monkeypatch.setattr(chat_mod.ChatLLM, "chat", lambda self, sys, msgs: "ok", raising=False)

    from memfuse_core.rag.rag_service import RAGService

    rag = RAGService()
    # ensure builds from history
    hist = [
        {"role": "user", "content": "user says about memory retrieval"},
        {"role": "assistant", "content": "assistant discusses embeddings"},
    ]
    added = await rag.ensure_session_index("sess2", hist, chunk_size=3)
    assert added > 0
    out = await rag.chat("sess2", "What about embeddings?", history_messages=hist)
    assert isinstance(out, str)

