# M3 API Usage

Phase A implementation provides minimal orchestration, RAG, and query integration. M3 is gated by `memory.layers.m3.enabled`.
Trigger migration: use `metadata.task` and `metadata.task_eos` instead of legacy `tag=m3`. A short deprecation window exists behind `memory.layers.m3.legacy_tag_trigger` (default: false).

## Message Creation / Chat
- Use existing sessions/messages endpoints.
- To invoke M3 behavior, send a final user message for a workflow with:
  - `metadata.task`: The workflow name (e.g., `op_websearch_memory`)
  - `metadata.task_eos: true`: Marks end-of-sequence; M3 triggers only on EOS.
- M3 builds context by filtering session history to messages where `metadata.task` matches the workflow name.
- The orchestrator reply is added as an assistant message; per-step logs are written to `message_workflows` including `task` and `task_eos` in metadata.

Example (progress messages, not triggering):

```json
{
  "messages": [
    {"role":"user", "content":"Search articles about agent memory.", "metadata": {"task": "op_websearch_memory"}},
    {"role":"assistant", "content":"Found 3 sources.", "metadata": {"task": "op_websearch_memory"}}
  ]
}
```

Example (final EOS message that triggers M3):

```bash
BASE="http://localhost:8000/api/v1"
SESSION_ID="<your-session-id>"

curl -s -X POST "$BASE/sessions/$SESSION_ID/messages" \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [
      {"role":"user", "content":"Summarize findings about agent memory.", "metadata": {"task": "op_websearch_memory", "task_eos": true}}
    ]
  }' | jq .
```

## Query (M3 focus)
- `POST /api/v1/users/{user_id}/query`
  - Use the existing users query endpoint; when `metadata.task` is present, the server routes to M3-specific logic to search:
    - `procedural_memory` for similar workflows
    - `message_workflows` for session-scoped workflow logs (when `session_id` provided)
    - `procedural_lessons` for related execution lessons

Optional filters:
- `session_id`: include session-specific `message_workflows` (default true via `include_workflows`)
- `include_workflows`: whether to include `message_workflows`
- `task`: filter `session_workflows` by task/workflow name
- `filter_workflow_id`: filter by a known workflow id
- `filter_tags`: (array) any-match tags for `message_workflows`
- `filter_agent`: filter lessons by agent (server-side when possible)
- `filter_status`: `success` or `fail` for lessons
- `min_score`: minimum similarity score (0–1) for workflows/lessons

Example (body metadata):

```bash
BASE="http://localhost:8000/api/v1"
USER_ID="<your-user-id>"

curl -s -X POST "$BASE/users/$USER_ID/query" \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "memory patterns",
    "top_k": 5,
    "metadata": {"task": "op_websearch_memory"},
    "session_id": "<optional-session>",
    "include_workflows": true,
    "filter_status": "success",
    "min_score": 0.8
  }' | jq .
```

Compatibility: a short deprecation window for `tag=m3` can be enabled via `memory.layers.m3.legacy_tag_trigger`.

## Filtering
- Clients can join `messages` with `message_workflows` on `messages.id = message_workflows.message_id` to filter on tags/workflow_id/step_index.
