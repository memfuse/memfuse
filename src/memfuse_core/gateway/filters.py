"""Inbound and outbound filters for Gateway.

These are lightweight, synchronous filters to normalize/validate requests and
responses around the gateway pipeline. They are intentionally minimal to avoid
blocking async flows; heavy logic belongs to Guardrail or processors.
"""
from typing import Any, Dict, Protocol
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

