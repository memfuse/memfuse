import pytest
from typing import Any, Dict

from src.memfuse_core.gateway.api_gateway import MemoryApiGateway
from src.memfuse_core.utils.global_config_manager import get_global_config_manager


class FakeBufferService:
    def __init__(self, results):
        self.results = results

    async def query(self, query: str, top_k: int = 5, **kwargs) -> Dict[str, Any]:
        return {
            "status": "success",
            "code": 200,
            "data": {"results": self.results, "total": len(self.results)},
            "message": "ok",
            "errors": None,
        }


@pytest.mark.asyncio
async def test_output_remove_complex_combination_paths():
    gcm = get_global_config_manager()
    await gcm.hot_reload(
        {
            "gateway": {
                "pipeline": {"inbound": [], "outbound": [{"name": "output_remove", "enabled": True}]}
            },
            "guardrail": {
                "output": {
                    "enabled": True,
                    "remove_fields": [
                        # tags
                        "metadata.tags.0.k",            # index remove
                        "metadata.tags.*.x",            # wildcard remove all x
                        "metadata.tags.*{k=B}.v",       # conditional remove v where k==B
                        # attribs
                        "metadata.attribs.1.inner.secret",   # deep remove on specific index
                        "metadata.attribs.*{type=aux}.note", # conditional on type
                    ],
                }
            },
        }
    )

    results = [
        {
            "id": "1",
            "content": "hello",
            "score": 0.9,
            "metadata": {
                "tags": [
                    {"k": "A", "v": 1, "x": 10},
                    {"k": "B", "v": 2, "x": 20},
                    {"k": "C", "v": 3, "x": 30},
                ],
                "attribs": [
                    {"type": "main", "note": "keep-main"},
                    {"type": "aux",  "note": "drop-note", "inner": {"secret": "S", "keep": True}},
                    {"type": "main", "note": "keep-main-2"},
                ],
            },
        }
    ]

    gw = MemoryApiGateway(buffer_service=FakeBufferService(results), db_service=None)
    resp = await gw.process_request({"user_id": "u", "agent_id": "a", "session_id": "s", "query": "q", "top_k": 5})

    out = resp["data"]["results"][0]
    tags = out["metadata"]["tags"]
    attribs = out["metadata"]["attribs"]

    # tags checks
    assert "k" not in tags[0]  # index removal
    assert all("x" not in t for t in tags)  # wildcard removal
    tB = next(t for t in tags if t.get("k") == "B") if any("k" in t and t.get("k") == "B" for t in tags) else None
    # after index removal, k left only on non-indexed elements; for k==B, v removed
    if tB:
        assert "v" not in tB

    # attribs checks
    assert "secret" not in attribs[1].get("inner", {})  # deep index removal
    aux = next(a for a in attribs if a.get("type") == "aux")
    assert "note" not in aux  # conditional remove on type
    # non-aux keep note
    for a in attribs:
        if a.get("type") != "aux":
            assert "note" in a

