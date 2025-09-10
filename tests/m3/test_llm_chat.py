from memfuse_core.llm.chat import ChatLLM


def test_chatllm_offline_chat_returns_text():
    llm = ChatLLM()
    out = llm.chat("You are helpful.", [{"role": "user", "content": "Summarize this text."}])
    assert isinstance(out, str) and len(out) > 0


def test_completion_json_minimal_structure():
    llm = ChatLLM()
    raw = llm.completion_json("Planner", "Goal: test")
    assert isinstance(raw, str)
    # Must be JSON or empty JSON fallback
    import json

    data = json.loads(raw)
    assert isinstance(data, dict)
    # allow missing steps, planner will fallback, but our default includes steps
    assert "steps" in data or data == {}

