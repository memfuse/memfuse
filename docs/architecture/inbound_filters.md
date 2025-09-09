# Inbound Filters (Gateway)

This page documents inbound filters that run on requests before they are routed into Buffer/Guardrail/Services. Inbound filters are lightweight, side‑effect free transformations or validations.

## Execution order

Inbound filters execute in the configured order, then Guardrail.validate_request() is applied, then the request is routed to services.

```
client → inbound filters → guardrail.validate_request → gateway routing → services
```

See execution_order.md for the full end‑to‑end picture.

## Available filters

- request_normalizer
  - Coerces query to string (None/number → string, None → "")
  - Clamps top_k into configured bounds

## Configuration

Example minimal configuration enabling the normalizer:

```yaml
gateway:
  request_normalizer:
    enabled: true
    default_top_k: 5
    min_top_k: 1
    max_top_k: 50
  pipeline:
    inbound:
      - name: request_normalizer
        enabled: true
    outbound: []
```

Notes:
- If pipeline.inbound is empty, no inbound filters run.
- Guardrail request validation runs after inbound filters, so normalization can make otherwise invalid inputs acceptable (e.g., top_k coercion).

## Validation interplay

- Guardrail.validate_request requires:
  - query is string
  - top_k is coercible to int
- If request_normalizer runs first, it can help satisfy the validation.

## Future filters (placeholders)

- rate_limit: Check caller‑level quotas and short‑circuit if exceeded
- authn_authz: Verify identity and authorize operation
- schema_validate: Strict schema check with structured errors

You can add placeholders in configuration today to plan for future filters, e.g.:

```yaml
gateway:
  pipeline:
    inbound:
      - name: rate_limit
        enabled: false  # placeholder, not implemented
```

## Best practices

- Keep inbound filters fast and deterministic.
- Prefer normalization before validation so Guardrail can reason over a consistent shape.
- Keep configuration close to the filter with sensible defaults (see request_normalizer).

