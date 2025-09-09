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
      include_stats: true
  - name: session_annotator
    enabled: true
    params:
      default_session_id: s1
      default_agent_id: a1
  - name: deduplicate
    enabled: true
    params:
      key: id
      include_stats: true
        # When include_stats: true, will emit in metadata.observability on first item:
        # - dedup_removed_count
        # - dedup_unique_count
        # - dedup_key_source (e.g., 'id' or 'id|fallback' when some items lacked the key)
  - name: result_enricher
    enabled: true
    params:
      stage: merge
      include_query_len: true
      include_rerank_cache: true
      include_plugin_order: true
  - name: field_keep_or_remove
    enabled: true
    params:
      keep_fields: []

      remove_fields: ["metadata.source", "metadata.observability.query_len"]
```

Note: When `score_clip.include_stats: true`, each batch will emit `metadata.observability.score_clip_stats` on the first item with summary fields: `count_clipped`, `min_before/max_before`, `min_after/max_after`, and thresholds `min_threshold/max_threshold`. Gateway can optionally surface this at the top level via `gateway.debug.include_score_clip_stats`.

## Timeout and fallback (storage retrieval)

You can configure a minimal timeout for storage retrieval and let QueryBuffer gracefully fall back when storage is slow/unavailable. When a timeout or exception occurs, QueryBuffer continues without storage results (and on session queries, returns Hybrid-only data if available).

Configuration example:

```yaml
buffer:
  retrieval_timeout_seconds: 0.2  # seconds (global default)
  retrieval_per_store:
    VectorStore:
      timeout_seconds: 1.0  # longer timeout for vector operations
    KeywordStore:
      timeout_seconds: 0.1  # faster timeout for keyword search
      retry:
        enabled: false      # disable retry for keyword (fast fail)
```

Behavior:

- Mixed path (buffer + storage): storage call wrapped by asyncio.wait_for; timeout → continue with buffer results only
- Session path: storage part times out → return Hybrid results only (still sorted/limited)
- No exception propagated; logs contain a warning
- Per-store overrides: Use `retrieval_per_store.<ClassName>.timeout_seconds` to override global timeout for specific store classes

### Optional retry (transient errors)

For transient failures you can enable a minimal retry with fixed backoff. Timeouts apply per-attempt.

```yaml
buffer:
  retrieval_retry:
    enabled: true
    max_attempts: 3        # total attempts
    backoff_ms: 5          # sleep between attempts
```

Notes:

- Retry only triggers on exceptions (including timeouts); success returns immediately.
- Safe default is disabled (no retries).

### Common plugin combinations

- Dedup + Enrich + Remove
  - Order: deduplicate → result_enricher → field_keep_or_remove
  - Effect: remove dupes, add observability metadata, then drop sensitive aux fields
- Session annotate + Rag annotate
  - Ensure session_id/agent_id present and metadata.source tagged for downstream filters

## Gateway debug aggregation (optional)

To aid troubleshooting, Gateway can aggregate rerank cache hits across results and expose a top-level flag in the API response when enabled.

Config:

```yaml
gateway:
  debug:
    enabled: true
    include_rerank_cache_hit: true
    include_plugin_order: true
```

Effect (response excerpt):

```json
{
  "status": "success",
  "data": {
    "results": [...],
    "total": 10,
    "metadata": {
      "observability": {
        "rerank_cache_hit": true,
        "plugin_order": ["DeduplicatePlugin", "ResultEnricherPlugin", "FieldKeepOrRemovePlugin"],
        "durations": {
          "response_processor": 0.125,
          "metadata_enricher": 0.089,
          "scope_calculator": 0.034,
          "field_remover": 0.012
        }
      }
    }
  }
}
```

### End-to-end example: Rerank + Multi-plugins

Input (simplified):

```json
{
  "query": "how to deploy",
  "top_k": 5
}
```

Config (excerpt):

```yaml
buffer_plugins:
  plugins:
    - name: deduplicate
      enabled: true
      params: { key: id }
    - name: score_clip
      enabled: true
      params: { min: 0.0, max: 0.9 }
    - name: result_enricher
      enabled: true
      params: { include_rerank_cache: true, include_plugin_order: true }
    - name: field_keep_or_remove
      enabled: true
      params:
        keep_fields: []
        remove_fields: ["metadata.source"]

gateway:
  debug:
    enabled: true
    include_rerank_cache_hit: true
    include_plugin_order: true
    include_score_range: true
    include_dedup_removed_count: true
    include_score_clip_stats: true
    include_dedup_unique_count: true
    include_dedup_key_source: true
    include_durations: true  # optional: transformation stage timings (ms)

### Cache-hit example

When rerank cache is enabled and a subsequent query hits the cache, each item may carry `metadata.observability.rerank_cache_hit: true` (depending on your enricher settings), and Gateway can aggregate this as a top-level flag when `gateway.debug.include_rerank_cache_hit: true`.

Response excerpt:

```json
{
  "status": "success",
  "data": {
    "results": [
      {"id": "...", "relevance_score": 0.83, "metadata": {"observability": {"rerank_cache_hit": true}}},
      {"id": "...", "relevance_score": 0.77, "metadata": {"observability": {"rerank_cache_hit": true}}}
    ],
    "total": 2,
    "metadata": {"observability": {"rerank_cache_hit": true}}
  }
}
```

Expected response (excerpt):

```json
{
  "status": "success",
  "data": {
    "results": [
      {"id": "...", "score": 0.82, "metadata": {"observability": {"stage": "after_merge", "query_len": 12, "rerank_cache_hit": false, "plugin_order": ["DeduplicatePlugin", "ScoreClipPlugin", "ResultEnricherPlugin", "FieldKeepOrRemovePlugin"]}}},
      {"id": "...", "score": 0.75, "metadata": {"observability": {"stage": "after_merge", "query_len": 12, "rerank_cache_hit": false, "plugin_order": ["DeduplicatePlugin", "ScoreClipPlugin", "ResultEnricherPlugin", "FieldKeepOrRemovePlugin"]}}}
    ],
    "total": 2,
    "metadata": {"observability": {"rerank_cache_hit": false, "plugin_order": ["DeduplicatePlugin", "ScoreClipPlugin", "ResultEnricherPlugin", "FieldKeepOrRemovePlugin"], "score_range": {"min": 0.75, "max": 0.82}}}
  }
}
```

### End-to-end (Cache-hit, repeated query)

This demonstrates a cache hit on the second request when rerank is enabled and the query + candidate ids are identical.

Input #1:

```json
{"query": "how to deploy", "top_k": 5}
```

Input #2 (repeat the same query shortly after):

```json
{"query": "how to deploy", "top_k": 5}
```

Config (excerpt):

```yaml
buffer_plugins:
  plugins:
    - name: result_enricher
      enabled: true
      params:
        include_rerank_cache: true
        include_plugin_order: true

gateway:
  debug:
    enabled: true
    include_rerank_cache_hit: true
```

Output (second request excerpt, showing cache hit):

```json
{
  "status": "success",
  "data": {
    "results": [
      {"id": "A1", "relevance_score": 0.84, "metadata": {"observability": {"rerank_cache_hit": true}}},
      {"id": "B2", "relevance_score": 0.79, "metadata": {"observability": {"rerank_cache_hit": true}}}
    ],
    "total": 2,
    "metadata": {"observability": {"rerank_cache_hit": true}}
  }
}
```

## Testing

- Unit-level integration tests mock QueryBuffer.buffer_retrieval.retrieve to avoid DB
- New tests:
  - tests/unit/buffer/test_buffer_plugins.py
  - tests/unit/buffer/test_buffer_plugins_extra.py

These validate that plugin order and effects are deterministic and reversible by configuration.

## Related

- See also: execution order across layers in docs/architecture/execution_order.md
