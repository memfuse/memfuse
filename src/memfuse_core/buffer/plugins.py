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
    """Clip scores into a configurable range to stabilize downstream logic."""

    def __init__(self, **params: Any) -> None:
        self.min_v = float(params.get("min", 0.0))
        self.max_v = float(params.get("max", 1.0))

    def after_merge(self, results: List[Dict[str, Any]], ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
        for r in results:
            if isinstance(r, dict):
                s = r.get("score")
                if isinstance(s, (int, float)):
                    r["score"] = max(self.min_v, min(self.max_v, float(s)))
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
    """

    def __init__(self, **params: Any) -> None:
        self.key = params.get("key", "id")

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
        return unique


class ResultEnricherPlugin:
    """Add lightweight observability fields into result metadata.

    Params:
      stage: optional stage label (default: 'after_merge')
      include_query_len: bool, whether to include query length in metadata
    """

    def __init__(self, **params: Any) -> None:
        self.stage = params.get("stage", "after_merge")
        self.include_query_len = bool(params.get("include_query_len", True))

    def after_merge(self, results: List[Dict[str, Any]], ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
        q = ctx.get("query_text", "") or ""
        for r in results:
            if not isinstance(r, dict):
                continue
            md = r.setdefault("metadata", {})
            obs = md.setdefault("observability", {})
            if isinstance(obs, dict):
                obs.setdefault("stage", self.stage)
                if self.include_query_len:
                    obs.setdefault("query_len", len(q))
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

