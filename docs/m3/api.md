# M3 API Usage

Phase A implementation provides minimal orchestration, RAG, and query integration. M3 is gated by `memory.layers.m3.enabled`.

## Message Creation / Chat
- Use existing sessions/messages endpoints.
- To invoke M3 behavior, include `{"tag": "m3"}` in the message `metadata`.
- M3 orchestration will route the request and log results using `message_workflows`.

Example:

```bash
BASE="http://localhost:8000/api/v1"
SESSION_ID="<your-session-id>"

curl -s -X POST "$BASE/sessions/$SESSION_ID/messages" \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [
      {"role":"user", "content":"Research latest LLM memory trends and summarize.", "metadata": {"tag": "m3"}}
    ]
  }' | jq .
```

## Query (M3 focus)
- `POST /api/v1/users/{user_id}/query`
  - Use the existing users query endpoint; when `metadata.tag == 'm3'`, the server routes to M3-specific logic to search:
    - `procedural_memory` for similar workflows
    - `message_workflows` for session-scoped workflow logs (when `session_id` provided)
    - `procedural_lessons` for related execution lessons

Optional filters:
- `session_id`: include session-specific `message_workflows` (default true via `include_workflows`)
- `include_workflows`: whether to include `message_workflows`
- `filter_workflow_id`: filter by a known workflow id
- `filter_tags`: (array) any-match tags for `message_workflows`
- `filter_agent`: filter lessons by agent (server-side when possible)
- `filter_status`: `success` or `fail` for lessons
- `min_score`: minimum similarity score (0–1) for workflows/lessons

Example:

```bash
BASE="http://localhost:8000/api/v1"
USER_ID="<your-user-id>"

curl -s -X POST "$BASE/users/$USER_ID/query" \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "memory patterns",
    "top_k": 5,
    "metadata": {"tag": "m3"},
    "session_id": "<optional-session>",
    "include_workflows": true,
    "filter_status": "success",
    "min_score": 0.8
  }' | jq .
```

## Filtering
- Clients can join `messages` with `message_workflows` on `messages.id = message_workflows.message_id` to filter on tags/workflow_id/step_index.
