"""Inbound and outbound filters for Gateway.

These are lightweight, synchronous filters to normalize/validate requests and
responses around the gateway pipeline. They are intentionally minimal to avoid
blocking async flows; heavy logic belongs to Guardrail or processors.
"""
from typing import Any, Dict, Protocol, Tuple, List
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

