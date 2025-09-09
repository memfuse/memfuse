# Outbound Filters (Gateway)

This page documents outbound filters that run after response transformation and before final guardrail validation. They are intended for masking, truncation, or light annotations without modifying core service logic.

## Execution order

Outbound filters execute in the configured order after the Gateway transforms the response and before Guardrail.validate_response/audit.

```text
services → gateway transform → outbound filters → guardrail.validate_response → client
```

See execution_order.md for the full end-to-end picture.

## Built-in filters

- max_length
  - Truncates `result.content` to a configured maximum length and appends a suffix (default `...`).
  - Sets `metadata.length_truncated = true` on affected items.
- sensitive_word / sensitive_words
  - Processes configured words in `result.content` and optionally in metadata strings.
  - Case-insensitive by default; set `case_insensitive: false` for exact case matching.
  - Action modes: `mask` (replace with token), `flag` (mark only), `drop` (remove result).
  - Optional recursive metadata processing with `recurse_metadata: true`.
  - Sets `metadata.sensitive_hit = true` when any sensitive content is detected.
- composite_content
  - Advanced multi-dimensional validation combining length, semantic, and structural checks.
  - Configurable strategies: `lenient` (flag only), `strict` (drop on any violation), `custom` (per-violation actions).
  - Length validation with min/max constraints and configurable actions.
  - Semantic validation with required keywords and forbidden patterns (regex support).
  - Structural validation for required fields and metadata depth limits.
  - Sets `metadata.composite_violations[]` with detailed violation information.
- content_quality
  - Content quality assessment with multi-dimensional scoring (0.0-1.0).
  - Dimensions: completeness, relevance, clarity, accuracy with configurable weights.
  - Actions: `flag` (mark low quality), `drop` (remove), `annotate` (prefix score).
  - Sets `metadata.quality_score` and optionally `metadata.low_quality = true`.

## Configuration

Example combined configuration:

```yaml
gateway:
  pipeline:
    inbound: []
    outbound:
      - name: max_length
        enabled: true
      - name: sensitive_word   # alias: sensitive_words
        enabled: true
      - name: composite_content
        enabled: false  # optional advanced validation
      - name: content_quality
        enabled: false  # optional quality scoring

guardrail:
  length:
    enabled: true
    max_content_length: 20
    suffix: "..."
  sensitive:
    enabled: true
    words: ["forbidden", "secret"]
    mask_token: "[SENSITIVE]"
    case_insensitive: true
    recurse_metadata: false  # default: only process content
    action: "mask"           # default: mask | flag | drop
  composite:
    enabled: false
    strategy: lenient  # lenient|strict|custom
    length:
      min_length: 10
      max_length: 1000
      action: flag  # flag|truncate|drop
    semantic:
      required_keywords: []  # e.g., ["important", "data"]
      forbidden_patterns: []  # regex patterns, e.g., ["forbidden", "bad.*word"]
      action: flag
    structural:
      required_fields: []  # e.g., ["content", "score"]
      max_metadata_depth: 5
      action: flag
    policy:
      fail_fast: false  # stop on first violation
      aggregate_violations: true
  quality:
    enabled: false
    min_score: 0.6  # threshold for quality actions
    action: flag  # flag|drop|annotate
    weights:
      completeness: 0.3
      relevance: 0.3
      clarity: 0.2
      accuracy: 0.2
```

Notes:

- Filters are no-ops unless the corresponding `guardrail.*.enabled` flag is true.
- Order matters: `max_length` running before `sensitive_word` truncates first, potentially reducing mask hits.

## End-to-end examples

### Example A: Max length truncation

Input results (excerpt):

```json
{
  "data": {
    "results": [
      {"id": "1", "content": "this is a very long content beyond limit", "metadata": {} }
    ]
  }
}
```

Config:

```yaml
guardrail:
  length: { enabled: true, max_content_length: 10, suffix: "..." }
```

Output (excerpt):

```json
{
  "data": {
    "results": [
      {"id": "1", "content": "this is a ...", "metadata": {"length_truncated": true}}
    ]
  }
}
```

### Example B: Sensitive word masking

Input results (excerpt):

```json
{
  "data": {"results": [{"id": "1", "content": "a Forbidden and SECRET thing"}]}
}
```

Config:

```yaml
guardrail:
  sensitive:
    enabled: true
    words: ["forbidden", "secret"]
    mask_token: "[BLOCKED]"
    case_insensitive: true
```

Output (excerpt):

```json
{
  "data": {
    "results": [
      {"id": "1", "content": "a [BLOCKED] and [BLOCKED] thing", "metadata": {"sensitive_hit": true}}
    ]
  }
}
```

### Example C: Combined filters (order matters)

Input results (excerpt):

```json
{
  "data": {
    "results": [
      {"id": "1", "content": "This contains forbidden information that exceeds the limit"}
    ]
  }
}
```

Config (max_length runs first, then sensitive_word):

```yaml
gateway:
  pipeline:
    outbound:
      - name: max_length
        enabled: true
      - name: sensitive_word
        enabled: true

guardrail:
  length: { enabled: true, max_content_length: 25, suffix: "..." }
  sensitive: { enabled: true, words: ["forbidden"], mask_token: "[MASKED]" }
```

Output (excerpt):

```json
{
  "data": {
    "results": [
      {
        "id": "1",
        "content": "This contains [MASKED]...",
        "metadata": {"length_truncated": true, "sensitive_hit": true}
      }
    ]
  }
}
```

Note: The content was first truncated to "This contains forbidden..." then masked to "This contains [MASKED]...".

### Example D: Action modes (flag vs mask vs drop)

Input results (excerpt):

```json
{
  "data": {
    "results": [
      {"id": "1", "content": "Contains secret data"},
      {"id": "2", "content": "Normal content"}
    ]
  }
}
```

Config with `action: "flag"`:

```yaml
guardrail:
  sensitive: { enabled: true, words: ["secret"], action: "flag" }
```

Output (flag mode - content unchanged, metadata marked):

```json
{
  "data": {
    "results": [
      {"id": "1", "content": "Contains secret data", "metadata": {"sensitive_hit": true}},
      {"id": "2", "content": "Normal content"}
    ]
  }
}
```

Config with `action: "drop"`:

```yaml
guardrail:
  sensitive: { enabled: true, words: ["secret"], action: "drop" }
```

Output (drop mode - sensitive results removed):

```json
{
  "data": {
    "results": [
      {"id": "2", "content": "Normal content"}
    ]
  }
}
```

## Best practices

- Keep outbound filters lightweight; avoid heavy parsing of large payloads.
- Prefer explicit configuration flags and safe defaults (disabled by default).
- Add new filters as separate classes and register them in `gateway.filters` to keep responsibilities isolated.
