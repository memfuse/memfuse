"""Inbound and outbound filters for Gateway.

These are lightweight, synchronous filters to normalize/validate requests and
responses around the gateway pipeline. They are intentionally minimal to avoid
blocking async flows; heavy logic belongs to Guardrail or processors.
"""
from typing import Any, Dict, Protocol, Tuple, List
import re
from ..interfaces.gateway_interface import RequestContext
from ..utils.global_config_manager import get_global_config_manager


class InboundFilter(Protocol):
    def apply(self, request_data: Dict[str, Any], context: RequestContext) -> Dict[str, Any]:
        ...


class OutboundFilter(Protocol):
    def apply(self, response: Dict[str, Any], context: RequestContext) -> Dict[str, Any]:
        ...


class NoOpInboundFilter:
    """No-op inbound filter (placeholder)."""
    def apply(self, request_data: Dict[str, Any], context: RequestContext) -> Dict[str, Any]:
        return request_data


class NoOpOutboundFilter:
    """No-op outbound filter (placeholder)."""
    def apply(self, response: Dict[str, Any], context: RequestContext) -> Dict[str, Any]:
        return response


class ConfigOutputRemovalFilter:
    """Remove configured fields from results (uses guardrail.output config)."""

    def __init__(self):
        gcm = get_global_config_manager()
        cfg = gcm.get_section("guardrail") if gcm.is_initialized() else {}
        output_cfg = cfg.get("output", cfg.get("output_filter", {})) or {}
        self.enabled = bool(output_cfg.get("enabled", False))
        self.fields = list(output_cfg.get("remove_fields", []) or [])

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

    def apply(self, response: Dict[str, Any], context: RequestContext) -> Dict[str, Any]:
        if not self.enabled or not self.fields:
            return response
        data = response.get("data", {})
        results = data.get("results", [])
        for r in results:
            for f in self.fields:
                self._remove_path(r, f)
        return response


class ConfigPIIRedactorFilter:
    """Redact basic PII like emails/phones when guardrail.pii is enabled."""

    EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    PHONE_RE = re.compile(r"\+?(?:\d[\s-]?){10,}")

    def __init__(self):
        gcm = get_global_config_manager()
        cfg = gcm.get_section("guardrail") if gcm.is_initialized() else {}
        pii_cfg = cfg.get("pii", {}) or {}
        self.enabled = bool(pii_cfg.get("enabled", False))
        self.redact = bool(pii_cfg.get("redact", True))

    def _redact_text(self, text: str) -> str:
        if not isinstance(text, str):
            return text
        text = self.EMAIL_RE.sub("[REDACTED:EMAIL]", text)
        text = self.PHONE_RE.sub("[REDACTED:PHONE]", text)
        return text

    def _redact_dict(self, d: Dict[str, Any]) -> None:
        for k, v in list(d.items()):
            if isinstance(v, str):
                d[k] = self._redact_text(v)
            elif isinstance(v, dict):
                self._redact_dict(v)

    def apply(self, response: Dict[str, Any], context: RequestContext) -> Dict[str, Any]:
        if not self.enabled or not self.redact:
            return response
        data = response.get("data", {})
        results = data.get("results", [])
        for r in results:
            if "content" in r and isinstance(r["content"], str):
                r["content"] = self._redact_text(r["content"])  # redact in content
            if "metadata" in r and isinstance(r["metadata"], dict):
                self._redact_dict(r["metadata"])  # redact in metadata strings
        return response


class ConfigToxicityAnnotatorFilter:
    """Annotate responses with a toxicity flag using simple keyword heuristic.

    Note: minimal heuristic for unit tests; not a production classifier.
    """

    KEYWORDS = {"toxic", "abuse", "insult"}

    def __init__(self):
        gcm = get_global_config_manager()
        cfg = gcm.get_section("guardrail") if gcm.is_initialized() else {}
        tox_cfg = cfg.get("toxicity", {}) or {}
        self.enabled = bool(tox_cfg.get("enabled", False))
        self.threshold = float(tox_cfg.get("threshold", 0.9))

    def _score_text(self, text: str) -> float:
        if not isinstance(text, str) or not text:
            return 0.0
        lower = text.lower()
        hits = sum(1 for k in self.KEYWORDS if k in lower)
        return min(1.0, hits / 1.0)  # any hit -> 1.0, else 0.0

    def apply(self, response: Dict[str, Any], context: RequestContext) -> Dict[str, Any]:
        if not self.enabled:
            return response
        results = response.get("data", {}).get("results", [])
        for r in results:
            score = 0.0
            if "content" in r:
                score = max(score, self._score_text(r.get("content")))
            # mark if score >= (1 - (1-threshold)) => score >= threshold (simple)
            if score >= self.threshold:
                md = r.setdefault("metadata", {})
                md["toxicity_flag"] = True
                md["toxicity_score"] = score
        return response


def build_filters_from_config(cfg: Dict[str, Any] | None) -> Tuple[List[InboundFilter], List[OutboundFilter]]:
    """Build inbound/outbound filter instances from gateway pipeline config.

    Expected cfg structure:
    {
      "pipeline": {
        "inbound": [{"name": "noop_inbound", "enabled": true}],
        "outbound": [{"name": "noop_outbound", "enabled": false}]
      }
    }
    """
    inbound: List[InboundFilter] = []
    outbound: List[OutboundFilter] = []
    if not cfg:
        return inbound, outbound

    pipeline = cfg.get("pipeline") if isinstance(cfg, dict) else None
    if not isinstance(pipeline, dict):
        return inbound, outbound

    # Registry of available filters
    inbound_registry: Dict[str, type] = {
        "noop_inbound": NoOpInboundFilter,
    }
    outbound_registry: Dict[str, type] = {
        "noop_outbound": NoOpOutboundFilter,
        "output_remove": ConfigOutputRemovalFilter,
        "pii_redact": ConfigPIIRedactorFilter,
        "toxicity_mark": ConfigToxicityAnnotatorFilter,
    }

    # Inbound
    for item in pipeline.get("inbound", []) or []:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        enabled = item.get("enabled", True)
        if not enabled or not name:
            continue
        cls = inbound_registry.get(name)
        if cls:
            inbound.append(cls())

    # Outbound
    for item in pipeline.get("outbound", []) or []:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        enabled = item.get("enabled", True)
        if not enabled or not name:
            continue
        cls = outbound_registry.get(name)
        if cls:
            outbound.append(cls())

    return inbound, outbound

