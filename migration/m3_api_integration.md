# M3 API Integration (Phase A)

This note records how we will integrate M3 into existing APIs without modifying the `messages` table.

## Principles
- Use `metadata` to carry M3 intent signal (e.g., `{"tag": "m3"}`) instead of query string `tag=m3`.
- Persist workflow-related metadata via an auxiliary table referencing `messages` (`message_workflows`).
- Keep current APIs backward compatible.

## Message Creation / Chat
- When creating a user message or calling chat under a session, if `metadata.tag == 'm3'` and M3 is enabled:
  - Route the request to Orchestrator (Phase A implementation to follow in the coding phase).
  - The Orchestrator will produce:
    - An assistant reply message (standard `messages` insert).
    - One or more `message_workflows` rows referencing the written messages with:
      - `workflow_id`
      - `step_index`
      - `tags` (e.g., `["m3", "workflow"]`)
      - `metadata` (e.g., `{ "m3_enabled": true }`)

## Query API
- Introduce a unified query endpoint `POST /api/v1/users/{user_id}/query` that accepts a body like:
```json
{
  "query": "...",
  "top_k": 10,
  "metadata": {"tag": "m3"}
}
```
- If `metadata.tag == 'm3'`: search across:
  - `procedural_memory` for similar workflows (retrieved with vector similarity on `trigger_embedding`).
  - Recent session workflow logs from `message_workflows` for this user (optional scoping).
  - `procedural_lessons` for similar lessons.

## Filtering / Listing
- `GET /api/v1/sessions/{session_id}/messages` remains unchanged. Clients can correlate `messages.id` with `message_workflows.message_id` to filter/aggregate by M3-specific attributes (tags, workflow_id, step_index).

## Backward Compatibility
- Legacy callers that used `tag=m3` on query string can transition to using `metadata.tag`. We can optionally accept both in the API request layer, but persistence only uses the `message_workflows` table.

## Observability
- Execution artifacts (plan/trace/reflection/report) are saved under `runs/{timestamp}/{session_id}/...`.
