# M3 API Usage (Planned)

This is the intended API surface for Phase A. No behavior change is implemented in this patch.

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
  - Request body includes `{"metadata": {"tag": "m3"}}` to focus on workflows and lessons.
  - Returns:
    - Similar workflows from `procedural_memory`
    - Session workflow logs from `message_workflows`
    - Lessons from `procedural_lessons`

Example:

```bash
BASE="http://localhost:8000/api/v1"
USER_ID="<your-user-id>"

curl -s -X POST "$BASE/users/$USER_ID/query" \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "memory patterns",
    "top_k": 5,
    "metadata": {"tag": "m3"}
  }' | jq .
```

## Filtering
- Clients can join `messages` with `message_workflows` on `messages.id = message_workflows.message_id` to filter on tags/workflow_id/step_index.
