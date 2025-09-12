# M3 End-to-End Test (HTTP)

This guide shows how to run a minimal end‑to‑end test of the M3 pipeline via HTTP. It assumes the API server is running and the database is available.

## Prerequisites
- Start the core services (DB + server):

```
poetry run python scripts/memfuse_launcher.py --start-db --optimize-db
poetry run memfuse-core  # or poetry run python -m memfuse_core
```

- Confirm the API is reachable at `http://localhost:8000` (or your configured host/port).

## Steps

1) Create a user

```
BASE="http://localhost:8000/api/v1"
USER_NAME="user_e2e_$RANDOM"

curl -s -X POST "$BASE/users" \
  -H 'Content-Type: application/json' \
  -d "{\"name\": \"$USER_NAME\"}" | jq .
```

Save `user.id`.

2) Create an agent

```
AGENT_NAME="agent_e2e_$RANDOM"
curl -s -X POST "$BASE/agents" \
  -H 'Content-Type: application/json' \
  -d "{\"name\": \"$AGENT_NAME\"}" | jq .
```

Save `agent.id`.

3) Create a session

```
SESSION_ID=$(curl -s -X POST "$BASE/sessions" \
  -H 'Content-Type: application/json' \
  -d "{\"user_id\": \"<user-id>\", \"agent_id\": \"<agent-id>\", \"name\": \"sess-e2e\"}" \
  | jq -r .data.session_id)
```

4) Send task messages (final message has `task_eos: true`)

```
curl -s -X POST "$BASE/sessions/$SESSION_ID/messages" \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [
      {"role":"user", "content":"Search articles about agent memory.", "metadata": {"task": "op_websearch_memory"}},
      {"role":"user", "content":"Summarize findings about agent memory.", "metadata": {"task": "op_websearch_memory", "task_eos": true}}
    ]
  }' | jq .
```

The response includes `assistant_message_id` (if M3 orchestrator replied) and may include `workflow_id`.

5) Query M3 results (by task)

```
curl -s -X POST "$BASE/users/<user-id>/query" \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "memory patterns",
    "top_k": 5,
    "metadata": {"task": "op_websearch_memory"},
    "session_id": "'$SESSION_ID'",
    "include_workflows": true
  }' | jq .
```

This returns:
- `procedural_memory`: similar workflows
- `lessons`: related execution lessons
- `session_workflows`: step‑by‑step logs associated with this session (filtered by `task`).

## Notes
- The API accepts metadata on messages as a first‑class field. All parsing and routing of metadata happen in the Gateway.
- M3 triggers only when the final user message includes `metadata.task_eos == true`. The `metadata.task` value is used as the workflow name and to scope history.
- To allow legacy `tag=m3` triggers temporarily, set `memory.layers.m3.legacy_tag_trigger: true` in the config.
