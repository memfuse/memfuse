"""
MemFuse Observability Package

This package provides comprehensive observability capabilities for MemFuse:
- Distributed tracing with OpenTelemetry integration
- Request correlation and context propagation
- Performance monitoring integration
- Custom metrics and attributes for MemFuse operations
"""

from .tracing import (
    MemFuseTracer,
    TraceConfig,
    get_tracer,
    initialize_tracing,
    trace_operation
)

__all__ = [
    "MemFuseTracer",
    "TraceConfig", 
    "get_tracer",
    "initialize_tracing",
    "trace_operation"
]
