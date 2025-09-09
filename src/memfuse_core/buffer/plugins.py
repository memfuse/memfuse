"""Buffer plugin port, registry and loader.

Plugins can hook into QueryBuffer lifecycle to augment results without
changing core logic. Hooks are lightweight and synchronous.
"""
from __future__ import annotations
from typing import Any, Dict, List, Protocol, Type


class BufferPlugin(Protocol):
    def before_retrieve(self, ctx: Dict[str, Any]) -> None:  # optional
        ...

    def after_retrieve(self, results: List[Dict[str, Any]], ctx: Dict[str, Any]) -> List[Dict[str, Any]]:  # optional
        ...

    def after_merge(self, results: List[Dict[str, Any]], ctx: Dict[str, Any]) -> List[Dict[str, Any]]:  # optional
        ...


# Example plugins
class RagAnnotatorPlugin:
    """Annotate results with a simple source tag if missing."""

    def __init__(self, **params: Any) -> None:
        self.source = params.get("source", "buffer")

    def after_retrieve(self, results: List[Dict[str, Any]], ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
        for r in results:
            md = r.setdefault("metadata", {}) if isinstance(r, dict) else None
            if isinstance(md, dict) and not md.get("source"):
                md["source"] = self.source
        return results


class ScoreClipPlugin:
    """Clip scores into a configurable range to stabilize downstream logic.

    Params:
      min: float (default 0.0)
      max: float (default 1.0)
      include_stats: bool (default False) — if True, write score_clip_stats to observability
    """

    def __init__(self, **params: Any) -> None:
        self.min_v = float(params.get("min", 0.0))
        self.max_v = float(params.get("max", 1.0))
        self.include_stats = bool(params.get("include_stats", False))

    def after_merge(self, results: List[Dict[str, Any]], ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
        min_before = None
        max_before = None
        min_after = None
        max_after = None
        count_clipped = 0
        for r in results:
            if not isinstance(r, dict):
                continue
            s = r.get("score")
            if isinstance(s, (int, float)):
                s_float = float(s)
                min_before = s_float if min_before is None else min(min_before, s_float)
                max_before = s_float if max_before is None else max(max_before, s_float)
                clipped = max(self.min_v, min(self.max_v, s_float))
                if clipped != s_float:
                    count_clipped += 1
                r["score"] = clipped
                min_after = clipped if min_after is None else min(min_after, clipped)
                max_after = clipped if max_after is None else max(max_after, clipped)
        if self.include_stats and results:
            stats = {
                "count_clipped": int(count_clipped),
                "min_before": float(min_before) if min_before is not None else None,
                "max_before": float(max_before) if max_before is not None else None,
                "min_after": float(min_after) if min_after is not None else None,
                "max_after": float(max_after) if max_after is not None else None,
                "min_threshold": float(self.min_v),
                "max_threshold": float(self.max_v),
            }
            # Attach to first result's observability to avoid duplicating across all items
            first = results[0]
            if isinstance(first, dict):
                md = first.setdefault("metadata", {})
                obs = md.setdefault("observability", {})
                obs["score_clip_stats"] = stats
            # Also expose in ctx for potential upstream aggregation
            ctx["score_clip_stats"] = stats
        return results


class SessionAnnotatorPlugin:
    """Ensure session/agent fields exist in metadata; can use defaults via params.

    Params:
      default_session_id: optional default value when missing
      default_agent_id: optional default value when missing
    """

    def __init__(self, **params: Any) -> None:
        self.default_session_id = params.get("default_session_id")
        self.default_agent_id = params.get("default_agent_id")

    def after_merge(self, results: List[Dict[str, Any]], ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
        for r in results:
            if not isinstance(r, dict):
                continue
            md = r.setdefault("metadata", {})
            if isinstance(md, dict):
                md.setdefault("session_id", ctx.get("session_id", self.default_session_id))
                md.setdefault("agent_id", ctx.get("agent_id", self.default_agent_id))
        return results


class DeduplicatePlugin:
    """Remove duplicate results by key or content hash in after_merge phase.

    Params:
      key: field name to use for deduplication (default: 'id')
      include_stats: bool (default False)   if True, write dedup_removed_count to observability
    """

    def __init__(self, **params: Any) -> None:
        self.key = params.get("key", "id")
        self.include_stats = bool(params.get("include_stats", False))

    def after_merge(self, results: List[Dict[str, Any]], ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
        seen = set()
        unique: List[Dict[str, Any]] = []
        for r in results:
            if not isinstance(r, dict):
                continue
            k = r.get(self.key)
            if k is None:
                # fallback to hash of content
                k = (r.get("content"), r.get("score"))
            if k in seen:
                continue
            seen.add(k)
            unique.append(r)
        removed = max(0, len(results) - len(unique))
        if self.include_stats and unique:
            first = unique[0]
            md = first.setdefault("metadata", {})
            obs = md.setdefault("observability", {})
            obs["dedup_removed_count"] = int(removed)
            ctx["dedup_removed_count"] = int(removed)
        return unique


class ResultEnricherPlugin:
    """Add lightweight observability fields into result metadata.

    Params:
      stage: optional stage label (default: 'after_merge')
      include_query_len: bool, whether to include query length in metadata
      include_rerank_cache: bool, whether to include last rerank cache hit flag
      include_plugin_order: bool, whether to include executed plugin order
    """

    def __init__(self, **params: Any) -> None:
        self.stage = params.get("stage", "after_merge")
        self.include_query_len = bool(params.get("include_query_len", True))
        self.include_rerank_cache = bool(params.get("include_rerank_cache", False))
        self.include_plugin_order = bool(params.get("include_plugin_order", False))

    def after_merge(self, results: List[Dict[str, Any]], ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
        q = ctx.get("query_text", "") or ""
        rk = bool(ctx.get("rerank_cache_hit", False))
        order = ctx.get("plugin_order")
        for r in results:
            if not isinstance(r, dict):
                continue
            md = r.setdefault("metadata", {})
            obs = md.setdefault("observability", {})
            if isinstance(obs, dict):
                obs.setdefault("stage", self.stage)
                if self.include_query_len:
                    obs.setdefault("query_len", len(q))
                if self.include_rerank_cache:
                    # Always reflect latest cache status
                    obs["rerank_cache_hit"] = rk
                if self.include_plugin_order and isinstance(order, list):
                    # Keep plugin order up-to-date
                    obs["plugin_order"] = order
        return results


class FieldKeepOrRemovePlugin:
    """Keep or remove configured dotted fields on each result dict.

    Params:
      keep_fields: list[str] (if provided, keep only these fields)
      remove_fields: list[str] (fields to remove)
    """

    def __init__(self, **params: Any) -> None:
        self.keep_fields = list(params.get("keep_fields", []) or [])
        self.remove_fields = list(params.get("remove_fields", []) or [])

    def _get_root_keys(self, dotted: str) -> str:
        return dotted.split(".")[0] if dotted else ""

    def _remove_path(self, obj: Dict[str, Any], dotted: str) -> None:
        parts = dotted.split('.') if dotted else []
        if not parts:
            return
        cur = obj
        for p in parts[:-1]:
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                return
        last = parts[-1]
        if isinstance(cur, dict) and last in cur:
            try:
                del cur[last]
            except Exception:
                pass

    def after_merge(self, results: List[Dict[str, Any]], ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
        processed: List[Dict[str, Any]] = []
        for r in results:
            if not isinstance(r, dict):
                continue
            item = r
            if self.keep_fields:
                # Keep only specified root fields
                keep_roots = {self._get_root_keys(f) for f in self.keep_fields}
                item = {k: v for k, v in r.items() if k in keep_roots}
            # Apply removals (dotted)
            for f in self.remove_fields:
                self._remove_path(item, f)
            processed.append(item)
        return processed


def build_plugins_from_config(cfg: Dict[str, Any] | None) -> List[BufferPlugin]:
    """Instantiate plugins from buffer_plugins config.

    Expected cfg structure:
    {
      "plugins": [
        {"name": "rag_annotator", "enabled": true, "params": {...}},
        {"name": "score_clip", "enabled": true, "params": {"max": 0.9}}
      ]
    }
    """
    if not isinstance(cfg, dict):
        return []

    registry: Dict[str, Type] = {
        "rag_annotator": RagAnnotatorPlugin,
        "score_clip": ScoreClipPlugin,
        "session_annotator": SessionAnnotatorPlugin,
        "deduplicate": DeduplicatePlugin,
        "result_enricher": ResultEnricherPlugin,
        "field_keep_or_remove": FieldKeepOrRemovePlugin,
    }

    created: List[BufferPlugin] = []
    for item in cfg.get("plugins", []) or []:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not item.get("enabled", True) or not name:
            continue
        cls = registry.get(name)
        if cls:
            params = item.get("params", {}) or {}
            created.append(cls(**params))
    return created

