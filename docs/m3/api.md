# M3 API Usage (Planned)

This is the intended API surface for Phase A. No behavior change is implemented in this patch.

## Message Creation / Chat
- Use existing sessions/messages endpoints.
- To invoke M3 behavior, include `{"tag": "m3"}` in the message `metadata`.
- M3 orchestration will route the request and log results using `message_workflows`.

## Query (M3 focus)
- `POST /api/v1/users/{user_id}/query`
  - Request body includes `{"metadata": {"tag": "m3"}}` to focus on workflows and lessons.
  - Returns:
    - Similar workflows from `procedural_memory`
    - Session workflow logs from `message_workflows`
    - Lessons from `procedural_lessons`

## Filtering
- Clients can join `messages` with `message_workflows` on `messages.id = message_workflows.message_id` to filter on tags/workflow_id/step_index.
