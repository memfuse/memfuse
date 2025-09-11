import pytest


class FakeDB:
    async def get_session(self, session_id):
        return {"id": session_id, "user_id": "u1", "agent_id": "a1", "name": "s"}

    async def get_user(self, user_id):
        return {"id": user_id, "name": "user_default"}

    async def get_agent(self, agent_id):
        return {"id": agent_id, "name": "agent_default"}


class FakeService:
    def __init__(self):
        self.add_calls = []
        self.history = []

    async def initialize(self):
        return True

    async def add(self, messages, session_id=None):
        self.add_calls.append((session_id, messages))
        return {"status": "success", "data": {"message_ids": ["mid-assistant"]}}

    async def get_messages_by_session(self, session_id, limit, sort_by, order, buffer_only=None):
        # return mixed history; only two messages match the task
        return [
            {"role": "user", "content": "x", "metadata": {"task": "op_websearch_memory"}},
            {"role": "assistant", "content": "y", "metadata": {"task": "op_websearch_memory"}},
            {"role": "user", "content": "z", "metadata": {"task": "other"}},
        ]


@pytest.mark.asyncio
async def test_task_eos_triggers_and_filters_history(monkeypatch):
    # Patch config: enable M3, disable legacy
    from memfuse_core.api import messages as msg_mod
    from memfuse_core.utils import global_config_manager as gcm
    class GC:
        def get(self, key, default=None):
            if key == "memory.layers.m3.enabled":
                return True
            if key == "memory.layers.m3.legacy_tag_trigger":
                return False
            if key == "memory.layers.m3.max_history_scan":
                return 50
            return default
    monkeypatch.setattr(gcm, "get_global_config_manager", lambda: GC(), raising=False)

    # Patch DB and service
    async def _gi(cls=None):
        return FakeDB()
    monkeypatch.setattr(msg_mod.DatabaseService, "get_instance", classmethod(_gi), raising=False)

    fake_service = FakeService()
    async def fake_get_service_for_session(session, session_id):
        return fake_service
    monkeypatch.setattr(msg_mod, "get_service_for_session", fake_get_service_for_session, raising=False)
    # Ensure metadata is visible by overriding conversion util
    monkeypatch.setattr(msg_mod, "convert_pydantic_to_dict", lambda x: [
        {"role": "user", "content": "intermediate", "metadata": {"task": "op_websearch_memory"}},
        {"role": "user", "content": "final summary", "metadata": {"task": "op_websearch_memory", "task_eos": True}},
    ], raising=False)

    # Patch orchestrator to check inputs and return a stub text
    calls = {}
    async def fake_handle(self, session_id, user_goal, workflow_name=None, history_messages=None):
        calls["session_id"] = session_id
        calls["user_goal"] = user_goal
        calls["workflow_name"] = workflow_name
        calls["history_messages"] = history_messages
        return "ok"

    from memfuse_core.m3 import orchestrator as orch_mod
    monkeypatch.setattr(orch_mod.Orchestrator, "handle_request", fake_handle, raising=False)

    # Build request
    from memfuse_core.models import MessageAdd
    req = MessageAdd(messages=[
        {"role": "user", "content": "intermediate", "metadata": {"task": "op_websearch_memory"}},
        {"role": "user", "content": "final summary", "metadata": {"task": "op_websearch_memory", "task_eos": True}},
    ])

    resp = await msg_mod.add_messages("s1", req, tag=None, _api_key_data={})
    assert resp.status == "success"
    assert calls.get("workflow_name") == "op_websearch_memory"
    assert calls.get("user_goal") == "final summary"
    # history should be filtered to two items with the task
    hist = calls.get("history_messages")
    assert isinstance(hist, list)
    assert len(hist) == 2


@pytest.mark.asyncio
async def test_task_without_eos_does_not_trigger(monkeypatch):
    # Patch config: enable M3
    from memfuse_core.api import messages as msg_mod
    from memfuse_core.utils import global_config_manager as gcm
    class GC:
        def get(self, key, default=None):
            if key == "memory.layers.m3.enabled":
                return True
            if key == "memory.layers.m3.legacy_tag_trigger":
                return False
            if key == "memory.layers.m3.max_history_scan":
                return 50
            return default
    monkeypatch.setattr(gcm, "get_global_config_manager", lambda: GC(), raising=False)

    async def _gi2(cls=None):
        return FakeDB()
    monkeypatch.setattr(msg_mod.DatabaseService, "get_instance", classmethod(_gi2), raising=False)

    fake_service = FakeService()
    async def fake_get_service_for_session(session, session_id):
        return fake_service
    monkeypatch.setattr(msg_mod, "get_service_for_session", fake_get_service_for_session, raising=False)
    monkeypatch.setattr(msg_mod, "convert_pydantic_to_dict", lambda x: [
        {"role": "user", "content": "progress", "metadata": {"task": "op_websearch_memory"}},
    ], raising=False)

    triggered = {"called": False}
    async def fake_handle(self, session_id, user_goal, workflow_name=None, history_messages=None):
        triggered["called"] = True
        return "ok"
    from memfuse_core.m3 import orchestrator as orch_mod
    monkeypatch.setattr(orch_mod.Orchestrator, "handle_request", fake_handle, raising=False)

    from memfuse_core.models import MessageAdd
    req = MessageAdd(messages=[
        {"role": "user", "content": "progress", "metadata": {"task": "op_websearch_memory"}},
    ])

    resp = await msg_mod.add_messages("s1", req, tag=None, _api_key_data={})
    assert resp.status == "success"
    assert triggered["called"] is False
