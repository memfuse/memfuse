import pytest
from typing import Any, Dict, List

from src.memfuse_core.buffer.plugins import DeduplicatePlugin, ScoreClipPlugin


@pytest.mark.asyncio
async def test_deduplicate_plugin_emits_removed_count_when_enabled():
    plugin = DeduplicatePlugin(key="id", include_stats=True)
    ctx: Dict[str, Any] = {}
    results: List[Dict[str, Any]] = [
        {"id": "1", "content": "a", "score": 0.3},
        {"id": "1", "content": "a", "score": 0.3},
        {"id": "2", "content": "b", "score": 0.2},
    ]

    out = plugin.after_merge(results, ctx)

    # One duplicate should be removed
    assert len(out) == 2
    # Stats attached to first item and ctx
    first = out[0]
    obs = first.get("metadata", {}).get("observability", {})
    assert obs.get("dedup_removed_count") == 1
    assert obs.get("dedup_unique_count") == 2
    assert obs.get("dedup_key_source") == "id"
    assert ctx.get("dedup_removed_count") == 1
    assert ctx.get("dedup_unique_count") == 2
    assert ctx.get("dedup_key_source") == "id"


@pytest.mark.asyncio
async def test_score_clip_plugin_emits_stats_when_enabled():
    plugin = ScoreClipPlugin(min=0.1, max=0.9, include_stats=True)
    ctx: Dict[str, Any] = {}
    results: List[Dict[str, Any]] = [
        {"id": "1", "content": "a", "score": 0.05},
        {"id": "2", "content": "b", "score": 0.95},
        {"id": "3", "content": "c", "score": 0.5},
    ]

    out = plugin.after_merge(results, ctx)

    # Scores should be clipped into [0.1, 0.9]
    vals = [r.get("score") for r in out]
    assert vals == [0.1, 0.9, 0.5]

    # Stats attached to first item and ctx
    first = out[0]
    obs = first.get("metadata", {}).get("observability", {})
    scs = obs.get("score_clip_stats")
    assert isinstance(scs, dict)
    assert scs.get("count_clipped") == 2
    assert scs.get("min_before") == 0.05 and scs.get("max_before") == 0.95
    assert scs.get("min_after") == 0.1 and scs.get("max_after") == 0.9
    assert scs.get("min_threshold") == 0.1 and scs.get("max_threshold") == 0.9
    # ctx mirror
    assert ctx.get("score_clip_stats", {}) == scs

