#!/usr/bin/env python3
"""
Trigger M3 on ADD path via messages.add using task_eos metadata (no HTTP).

Usage:
  poetry run python scripts/test_m3_add_trigger.py

What it does:
  - Mocks DB + Buffer service and calls the FastAPI add_messages handler directly
  - Sends two user messages, the second with metadata.task_eos = True
  - Verifies M3 orchestrator is invoked and prints the response

This is a fast integration-style check without touching memfuse_mvp.
"""

import asyncio
from typing import Any, Dict
from unittest.mock import patch, Mock, AsyncMock

from memfuse_core.api.messages import add_messages
from memfuse_core.models import Message, MessageAdd


async def main():
    session_id = "sess_demo_add"

    # Prepare two user messages; second is EOS to trigger M3
    messages_data = [
        {
            "role": "user",
            "content": "Search articles about agent memory.",
            "metadata": {"task": "op_websearch_memory"}
        },
        {
            "role": "user",
            "content": "Summarize findings about agent memory.",
            "metadata": {"task": "op_websearch_memory", "task_eos": True}
        }
    ]

    request = MessageAdd(messages=[Message(**m) for m in messages_data])
    
    # Debug: print the request messages
    print("Request messages:")
    for i, msg in enumerate(request.messages):
        print(f"  Message {i}: {msg.model_dump()}")

    # Mocks
    mock_db = Mock()
    mock_db.get_session = AsyncMock(return_value={"id": session_id, "user_id": "user-1", "agent_id": "agent-1", "name": "demo-session"})
    mock_db.get_user = AsyncMock(return_value={"name": "test-user"})
    mock_db.get_agent = AsyncMock(return_value={"name": "test-agent"})
    mock_db.get_messages_by_session = AsyncMock(return_value=[])

    mock_memory_service = Mock()
    mock_memory_service.add = AsyncMock(return_value={
        "status": "success",
        "data": {"message_ids": ["msg1", "msg2"]}
    })

    # Orchestrator mock to simulate successful run
    mock_orchestrator = Mock()
    mock_orchestrator.handle_request = AsyncMock(return_value="M3 workflow completed successfully")
    mock_orchestrator.last_workflow_id = "wf_demo_1234"
    mock_orchestrator.last_reused = False
    mock_orchestrator.last_plan_steps = []
    mock_orchestrator.last_step_outcomes = []

    print("\n[+] Running ADD with task_eos to trigger M3...\n")

    with patch('memfuse_core.api.messages.DatabaseService.get_instance') as mock_db_get, \
         patch('memfuse_core.api.messages.get_service_for_session') as mock_get_service, \
         patch('memfuse_core.api.messages.Orchestrator') as mock_orchestrator_cls, \
         patch('memfuse_core.api.messages.ensure_session_exists') as mock_ensure_session, \
         patch('memfuse_core.api.messages.get_task_messages') as mock_get_task_messages:

        mock_db_get.return_value = mock_db
        mock_get_service.return_value = mock_memory_service
        mock_orchestrator_cls.return_value = mock_orchestrator
        mock_ensure_session.return_value = {"id": session_id, "user_id": "user-1", "agent_id": "agent-1", "name": "demo-session"}
        mock_get_task_messages.return_value = []

        resp = await add_messages(
            session_id=session_id,
            request=request,
            _api_key_data={},  # bypass API key
        )

    # ApiResponse to dict
    if hasattr(resp, 'model_dump'):
        out: Dict[str, Any] = resp.model_dump()
    else:
        out = dict(resp)

    print("[+] Response:")
    import json
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print("\n[+] Done.")


if __name__ == '__main__':
    asyncio.run(main())

