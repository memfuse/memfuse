# M3 Testing Guide (Poetry)

This guide shows how to validate the M3 + Schema pipeline locally using Poetry.

## 1) Recreate DB with new schema
Direct schema changes are applied (no backward compatibility). Recreate the DB:

- Recreate container and auto-init schema:
  - `poetry run python scripts/memfuse_launcher.py --recreate-db`
- Or use the manager:
  - Validate: `poetry run python scripts/database_manager.py validate`
  - Recreate: `poetry run python scripts/database_manager.py recreate`

## 2) Manual gateway-level test (fast, no HTTP)
Runs a mock gateway flow to verify Schema normalization and scope logic:

- `poetry run python scripts/test_schema_m3.py`

It prints three cases:
- Without session_id → scope is null; no forbidden fields; renames applied.
- With session_id → in_session/cross_session tagging; required metadata present.
- With task metadata → task flows into result metadata (guidance optional).

## 3) Contract tests (API response schema)
Strict schema validation at API layer:

- `poetry run pytest -q tests/contract/test_memory_api_contract.py`

## 4) End-to-end schema checks (run individually)
To avoid intermittent asyncio loop teardown issues, run these one by one:

- `poetry run pytest -q tests/integration/test_end_to_end_schema_compliance.py::TestEndToEndSchemaCompliance::test_complete_schema_without_session_id`
- `poetry run pytest -q tests/integration/test_end_to_end_schema_compliance.py::TestEndToEndSchemaCompliance::test_complete_schema_with_session_id`
- `poetry run pytest -q tests/integration/test_end_to_end_schema_compliance.py::TestEndToEndSchemaCompliance::test_semantic_memory_schema_compliance`
- `poetry run pytest -q tests/integration/test_end_to_end_schema_compliance.py::TestEndToEndSchemaCompliance::test_task_eos_handling_schema_compliance`

## 5) Optional: enable query-time guidance
If you want guidance strings under `result.metadata.m3_guidance` during queries,
set in `config/m3/default.yaml`:

```yaml
m3:
  enable_query_guidance: true
```

This does not change the top-level response shape and remains schema-safe.

## Notes
- Do not use files in `memfuse_mvp/` for tests; that directory is deprecated and will be removed.
- All commands must be executed with `poetry run ...`.
