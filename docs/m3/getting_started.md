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

## Run services
Start the DB and server:
```bash
poetry run python scripts/memfuse_launcher.py --start-db --optimize-db
poetry run memfuse-core
```

## End-to-end M3 demo (HTTP)
Ensure your LLM environment is configured (for example):
- OPENAI_API_KEY
- OPENAI_BASE_URL
- OPENAI_COMPATIBLE_MODEL

Open a new terminal and run:
```bash
MEMFUSE_API_BASE=http://localhost:8000/api/v1 \
  python scripts/m3_e2e_demo.py
```

What the demo does:
- Creates a user, agent, and session
- Sends a workflow with metadata.task and final metadata.task_eos=true to trigger M3
- Prints assistant message id and a workflow_id
- Runs the same workflow again to demonstrate reuse (the workflow_id should remain the same)
- Queries M3 results for that task to display procedural_memory, lessons, and session_workflows

Notes:
- First execution may encounter errors depending on environment; subsequent runs will record lessons and can reuse the learned workflow for similar goals.

For endpoint details, see docs/m3/api.md.
