# Buffer Plugin Architecture (Minimal)

This document describes the minimal plugin mechanism for the Buffer layer. It introduces a lightweight hook lifecycle for QueryBuffer so that cross-cutting augmentations can be added without modifying core logic. The default behavior remains unchanged unless plugins are enabled through configuration.

## Goals

- Keep QueryBuffer core simple, stable, and testable
- Enable small, composable behaviors via plugins
- No DB dependency for unit-level integration tests

## Hook lifecycle

Execution order inside QueryBuffer:

1. before_retrieve(ctx)
2. perform buffer retrieval
3. after_retrieve(results, ctx)
4. merge/sort/rerank (existing logic)
5. after_merge(results, ctx)

### Plugin order and rerank

Inside QueryBuffer, the effective execution order when rerank is enabled is:

- Sort (according to sort_by/order)
- Rerank (if rerank_handler provided and use_rerank=True)
- after_merge plugins, in the configured order

Notes:

- Plugins do not run before rerank except for optional after_retrieve hooks that only touch buffer-side results.
- Typical recommended order for after_merge plugins: deduplicate → score_clip → result_enricher → field_keep_or_remove.
- Rerank uses a small cache keyed by query + result ids; repeated queries avoid re-running rerank.

Where ctx is a small dict including: { query_text, top_k, sort_by, order }. Plugins can be extended to use more context in the future.

## Plugin Port & Examples

- BufferPlugin Protocol defines optional hooks: before_retrieve, after_retrieve, after_merge
- Built-in examples:
  - RagAnnotatorPlugin: ensure metadata.source exists (default: "buffer")
  - ScoreClipPlugin: clamp score into [min, max] to improve robustness
  - SessionAnnotatorPlugin: ensure metadata has session_id/agent_id (with optional defaults)
  - DeduplicatePlugin: remove duplicates by key (default: id) or fallback content hash

## Configuration

Hydra defaults include the buffer_plugins group:

- config/config.yaml adds: `- buffer_plugins: pipeline`
- config/buffer_plugins/pipeline.yaml defines a list of plugins with name/enabled/params

Example:

```yaml
plugins:
  - name: rag_annotator
    enabled: true
    params:
      source: buffer
  - name: score_clip
    enabled: true
    params:
      max: 0.9
  - name: session_annotator
    enabled: true
    params:
      default_session_id: s1
      default_agent_id: a1
  - name: deduplicate
    enabled: true
    params:
      key: id
  - name: result_enricher
    enabled: true
    params:
      stage: merge
      include_query_len: true
  - name: field_keep_or_remove
    enabled: true
    params:
      keep_fields: []
      remove_fields: ["metadata.source", "metadata.observability.query_len"]
```

## Timeout and fallback (storage retrieval)

You can configure a minimal timeout for storage retrieval and let QueryBuffer gracefully fall back when storage is slow/unavailable. When a timeout or exception occurs, QueryBuffer continues without storage results (and on session queries, returns Hybrid-only data if available).

Configuration example:

```yaml
buffer:
  retrieval_timeout_seconds: 0.2  # seconds
```

Behavior:

- Mixed path (buffer + storage): storage call wrapped by asyncio.wait_for; timeout → continue with buffer results only
- Session path: storage part times out → return Hybrid results only (still sorted/limited)
- No exception propagated; logs contain a warning

### Common plugin combinations

- Dedup + Enrich + Remove
  - Order: deduplicate → result_enricher → field_keep_or_remove
  - Effect: remove dupes, add observability metadata, then drop sensitive aux fields
- Session annotate + Rag annotate
  - Ensure session_id/agent_id present and metadata.source tagged for downstream filters

## Testing

- Unit-level integration tests mock QueryBuffer.buffer_retrieval.retrieve to avoid DB
- New tests:
  - tests/unit/buffer/test_buffer_plugins.py
  - tests/unit/buffer/test_buffer_plugins_extra.py

These validate that plugin order and effects are deterministic and reversible by configuration.

## Related

- See also: execution order across layers in docs/architecture/execution_order.md
