# RAG Integration Plan (Phase A)

Place the MVP-like RAG service under `src/memfuse_core/rag/` with:

## Components
- `RAGService` with ingest/chat
- History-aware indexing and retrieval

## Embeddings
- Use `src/memfuse_core/utils/embeddings.py` (MiniLM) by default.

