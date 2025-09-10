import pytest


@pytest.mark.asyncio
async def test_websearch_agent_parses_results(monkeypatch):
    from memfuse_core.m3.agents.websearch import WebSearchAgent

    agent = WebSearchAgent()

    async def fake_fetch(url, params):
        return {
            "RelatedTopics": [
                {"Text": "MemFuse - memory system", "FirstURL": "https://example.com/memfuse"},
                {"Topics": [
                    {"Text": "Procedural Memory", "FirstURL": "https://example.com/procedural"}
                ]}
            ],
            "AbstractText": "",
        }

    monkeypatch.setattr(agent, "_fetch_json", fake_fetch, raising=True)
    out = await agent.execute("sess-web", {"query": "memfuse", "max_results": 2})
    assert isinstance(out, dict)
    assert out.get("results") and len(out["results"]) == 2
    assert out["results"][0]["title"].lower().startswith("memfuse")

