"""Inbound and outbound filters for Gateway.

These are lightweight, synchronous filters to normalize/validate requests and
responses around the gateway pipeline. They are intentionally minimal to avoid
blocking async flows; heavy logic belongs to Guardrail or processors.
"""
from typing import Any, Dict, Protocol, Tuple, List
from ..interfaces.gateway_interface import RequestContext


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

