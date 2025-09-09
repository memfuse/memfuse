"""Buffer plugin port, registry and loader.

Plugins can hook into QueryBuffer lifecycle to augment results without
changing core logic. Hooks are lightweight and synchronous.
"""
from __future__ import annotations
from typing import Any, Dict, List, Protocol, Tuple, Type


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

