# RAG Integration Plan (Phase A)

Place the MVP-like RAG service under `src/memfuse_core/rag/` with these components:

## Components
- `RAGService`
  - `ingest_document(source, content, chunk_size)`
  - `chat(session_id, user_query)`
- Helpers:
  - `ContextController` (history truncation + context window management)
  - `SessionIndexer` (ensure session chunks embedded and indexed)
  - `BasicRetrievalStrategy` (embed query → vector search, prefer session scope when configured)

## Embeddings
- Use `src/memfuse_core/utils/embeddings.py` for embeddings (default MiniLM 384d) to replace MVP’s Jina client.
- Honor `config/database/default.yaml` embedding dimension when we finalize the schema.

## Persistence
- Keep M0/M1 writes via existing Core layers when feasible.
- For Phase A, allow a minimal RAG service implementation that mirrors MVP behavior; in Phase B, align with Vector Store + HybridBuffer for performance.

## Testing
- Start with unit tests (no network) and later add integration tests with DB.
