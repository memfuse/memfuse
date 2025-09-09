import pytest
from typing import Any, Dict, List

from src.memfuse_core.buffer.query_buffer import QueryBuffer
from src.memfuse_core.utils.global_config_manager import get_global_config_manager


@pytest.mark.asyncio
async def test_multi_plugins_apply_after_rerank_and_effects_visible():
    # Configure multiple plugins
    gcm = get_global_config_manager()
    await gcm.hot_reload(
        {
            "buffer_plugins": {
                "plugins": [
                    {"name": "deduplicate", "enabled": True, "params": {"key": "id"}},
                    {"name": "score_clip", "enabled": True, "params": {"min": 0.0, "max": 0.8}},
                    {"name": "result_enricher", "enabled": True, "params": {"stage": "after_merge", "include_query_len": True}},
                    {"name": "field_keep_or_remove", "enabled": True, "params": {"remove_fields": ["metadata.observability.query_len"]}},
                ]
            }
        }
    )

    # Retrieval handler returns duplicates and out-of-order scores
    async def retrieval_handler(query_text: str, max_results: int) -> List[Dict[str, Any]]:
        return [
            {"id": "a", "content": "alpha-high", "score": 0.95, "created_at": "2024-01-02T00:00:00", "metadata": {}},
            {"id": "b", "content": "bravo", "score": 0.50, "created_at": "2024-01-01T00:00:00", "metadata": {}},
            {"id": "a", "content": "alpha-low",  "score": 0.10, "created_at": "2024-01-03T00:00:00", "metadata": {}},
        ][:max_results]

    qb = QueryBuffer(retrieval_handler=retrieval_handler)

    async def rerank_handler(query_text: str, results: List[Dict[str, Any]]):
        # Reverse to force 'a'(high) before 'b' and 'a'(low) at end
        return list(reversed(results))

    qb.rerank_handler = rerank_handler

    out = await qb.query("q", top_k=3, sort_by="score", order="asc", use_rerank=True)

    # After rerank (reversed) then deduplicate, we should keep first 'a' (high) and drop later 'a' (low)
    assert [r["id"] for r in out] == ["a", "b"]

    # Score clipped to <= 0.8
    a = out[0]
    assert a["score"] == 0.8

    # Enricher added observability.stage but query_len removed by field_keep_or_remove
    md = a.get("metadata", {})
    obs = md.get("observability", {})
    assert obs.get("stage") == "after_merge"
    assert "query_len" not in obs

