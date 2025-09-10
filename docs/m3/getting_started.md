# Getting Started (M3 Migration)

This guide shows how to prepare and validate the M3 migration scaffolding.

## Prerequisites
- Poetry
- `.env` populated (you copied MVP env vars already)

## Validate Docs and Schemas
```bash
poetry run pytest -q tests/m3
```

Expected: tests pass, confirming docs and schema plans are present.

## Next (once implementation lands)
- Start services
```bash
poetry run python scripts/memfuse_launcher.py
```
- Use the API with `metadata.tag = "m3"` to trigger orchestration.

Refer to `docs/m3/api.md` for intended endpoints and usage.
