# Outbound Filters (Gateway)

This page documents outbound filters that run after response transformation and before final guardrail validation. They are intended for masking, truncation, or light annotations without modifying core service logic.

## Execution order

Outbound filters execute in the configured order after the Gateway transforms the response and before Guardrail.validate_response/audit.

```
services → gateway transform → outbound filters → guardrail.validate_response → client
```

See execution_order.md for the full end-to-end picture.

## Built-in filters

- max_length
  - Truncates `result.content` to a configured maximum length and appends a suffix (default `...`).
  - Sets `metadata.length_truncated = true` on affected items.
- sensitive_word / sensitive_words
  - Masks configured words in `result.content` with a token (default `[SENSITIVE]`).
  - Case-insensitive by default; set `case_insensitive: false` for exact case matching.
  - Sets `metadata.sensitive_hit = true` when any masking occurs.

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

## Best practices

- Keep outbound filters lightweight; avoid heavy parsing of large payloads.
- Prefer explicit configuration flags and safe defaults (disabled by default).
- Add new filters as separate classes and register them in `gateway.filters` to keep responsibilities isolated.

