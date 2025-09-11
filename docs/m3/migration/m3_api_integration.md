# M3 API Integration (Phase A)

This note records how we will integrate M3 into existing APIs without modifying the `messages` table.

## Principles
- Use `metadata` to carry M3 intent signal (e.g., `{"tag": "m3"}`) instead of query string `tag=m3`.
- Persist workflow-related metadata via an auxiliary table referencing `messages` (`message_workflows`).
- Keep current APIs backward compatible.

## Message Creation / Chat
- When creating a user message or calling chat under a session, if `metadata.tag == 'm3'` and M3 is enabled:
  - Route the request to Orchestrator.
  - Orchestrator writes assistant reply and logs per-step entries into `message_workflows`.

## Query API
- `POST /api/v1/users/{user_id}/query` accepts `metadata.tag == 'm3'` to query:
  - `procedural_memory`, `procedural_lessons`, and session `message_workflows`.

## Observability
- Execution artifacts are saved under `runs/{timestamp}/{session_id}/`.

