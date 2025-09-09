"""
Distributed tracing integration for MemFuse using OpenTelemetry.

This module provides comprehensive request-level tracing capabilities:
- Automatic span creation for gateway operations
- Context propagation across service boundaries
- Custom attributes for MemFuse-specific metadata
- Integration with existing performance monitoring
"""

import time
import uuid
from typing import Any, Dict, List, Optional, Union
from contextlib import contextmanager, asynccontextmanager
from dataclasses import dataclass
import threading
from loguru import logger

try:
    from opentelemetry import trace
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    from opentelemetry.instrumentation.asyncio import AsyncioInstrumentor
    from opentelemetry.propagate import inject, extract
    from opentelemetry.trace.status import Status, StatusCode
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    logger.warning("OpenTelemetry not available. Tracing will be disabled.")

from ..utils.global_config_manager import get_global_config_manager
from ..interfaces.gateway_interface import RequestContext


@dataclass
class TraceConfig:
    """Configuration for distributed tracing."""
    enabled: bool = False
    service_name: str = "memfuse-core"
    service_version: str = "1.0.0"
    exporter_type: str = "console"  # console, jaeger, otlp
    jaeger_endpoint: Optional[str] = None
    otlp_endpoint: Optional[str] = None
    sample_rate: float = 1.0
    include_request_body: bool = False
    include_response_body: bool = False
    max_attribute_length: int = 1000


class MemFuseTracer:
    """
    MemFuse-specific distributed tracing implementation.
    
    Provides request-level tracing with automatic span creation,
    context propagation, and MemFuse-specific metadata collection.
    """
    
    def __init__(self):
        self._initialized = False
        self._tracer = None
        self._config = TraceConfig()
        self._local = threading.local()
        
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the tracer with configuration."""
        # Load configuration first (even if OpenTelemetry is not available)
        gcm = get_global_config_manager()
        if gcm.is_initialized():
            trace_cfg = gcm.get_section("tracing") or {}
        else:
            trace_cfg = config or {}

        self._config = TraceConfig(
            enabled=bool(trace_cfg.get("enabled", False)),
            service_name=str(trace_cfg.get("service_name", "memfuse-core")),
            service_version=str(trace_cfg.get("service_version", "1.0.0")),
            exporter_type=str(trace_cfg.get("exporter_type", "console")),
            jaeger_endpoint=trace_cfg.get("jaeger_endpoint"),
            otlp_endpoint=trace_cfg.get("otlp_endpoint"),
            sample_rate=float(trace_cfg.get("sample_rate", 1.0)),
            include_request_body=bool(trace_cfg.get("include_request_body", False)),
            include_response_body=bool(trace_cfg.get("include_response_body", False)),
            max_attribute_length=int(trace_cfg.get("max_attribute_length", 1000))
        )

        if not OPENTELEMETRY_AVAILABLE:
            logger.warning("OpenTelemetry not available. Tracing disabled.")
            return

        if not self._config.enabled:
            logger.info("Distributed tracing is disabled")
            return
            
        # Set up resource
        resource = Resource.create({
            "service.name": self._config.service_name,
            "service.version": self._config.service_version,
            "service.instance.id": str(uuid.uuid4())
        })
        
        # Set up tracer provider
        tracer_provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(tracer_provider)
        
        # Set up exporter
        exporter = self._create_exporter()
        if exporter:
            span_processor = BatchSpanProcessor(exporter)
            tracer_provider.add_span_processor(span_processor)
        
        # Get tracer
        self._tracer = trace.get_tracer(__name__)
        
        # Instrument common libraries
        try:
            RequestsInstrumentor().instrument()
            AsyncioInstrumentor().instrument()
        except Exception as e:
            logger.warning(f"Failed to instrument libraries: {e}")
        
        self._initialized = True
        logger.info(f"Distributed tracing initialized with {self._config.exporter_type} exporter")
    
    def _create_exporter(self):
        """Create span exporter based on configuration."""
        if self._config.exporter_type == "console":
            return ConsoleSpanExporter()
        elif self._config.exporter_type == "jaeger":
            if not self._config.jaeger_endpoint:
                logger.error("Jaeger endpoint not configured")
                return None
            return JaegerExporter(
                agent_host_name=self._config.jaeger_endpoint.split(":")[0],
                agent_port=int(self._config.jaeger_endpoint.split(":")[1]) if ":" in self._config.jaeger_endpoint else 14268
            )
        elif self._config.exporter_type == "otlp":
            if not self._config.otlp_endpoint:
                logger.error("OTLP endpoint not configured")
                return None
            return OTLPSpanExporter(endpoint=self._config.otlp_endpoint)
        else:
            logger.error(f"Unknown exporter type: {self._config.exporter_type}")
            return None
    
    def is_enabled(self) -> bool:
        """Check if tracing is enabled and initialized."""
        return self._initialized and self._config.enabled and self._tracer is not None
    
    def get_current_trace_id(self) -> Optional[str]:
        """Get current trace ID if available."""
        if not self.is_enabled():
            return None
        
        current_span = trace.get_current_span()
        if current_span and current_span.is_recording():
            trace_id = current_span.get_span_context().trace_id
            return f"{trace_id:032x}"
        return None
    
    def get_current_span_id(self) -> Optional[str]:
        """Get current span ID if available."""
        if not self.is_enabled():
            return None
        
        current_span = trace.get_current_span()
        if current_span and current_span.is_recording():
            span_id = current_span.get_span_context().span_id
            return f"{span_id:016x}"
        return None
    
    def inject_context(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Inject tracing context into headers."""
        if not self.is_enabled():
            return headers
        
        inject(headers)
        return headers
    
    def extract_context(self, headers: Dict[str, str]) -> None:
        """Extract tracing context from headers."""
        if not self.is_enabled():
            return
        
        context = extract(headers)
        if context:
            # Set the extracted context as current
            token = trace.set_span_in_context(trace.get_current_span(), context)
            # Store token for cleanup (if needed)
            if not hasattr(self._local, 'context_tokens'):
                self._local.context_tokens = []
            self._local.context_tokens.append(token)
    
    @contextmanager
    def trace_operation(self, operation_name: str, attributes: Optional[Dict[str, Any]] = None):
        """Context manager for tracing synchronous operations."""
        if not self.is_enabled():
            yield
            return
        
        with self._tracer.start_as_current_span(operation_name) as span:
            try:
                # Add attributes
                if attributes:
                    self._add_attributes(span, attributes)
                
                yield span
                
                # Mark as successful
                span.set_status(Status(StatusCode.OK))
                
            except Exception as e:
                # Mark as error
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise
    
    @asynccontextmanager
    async def trace_async_operation(self, operation_name: str, attributes: Optional[Dict[str, Any]] = None):
        """Context manager for tracing asynchronous operations."""
        if not self.is_enabled():
            yield
            return
        
        with self._tracer.start_as_current_span(operation_name) as span:
            try:
                # Add attributes
                if attributes:
                    self._add_attributes(span, attributes)
                
                yield span
                
                # Mark as successful
                span.set_status(Status(StatusCode.OK))
                
            except Exception as e:
                # Mark as error
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise
    
    def _add_attributes(self, span, attributes: Dict[str, Any]) -> None:
        """Add attributes to span with proper formatting."""
        for key, value in attributes.items():
            if value is None:
                continue
            
            # Convert value to string and truncate if needed
            str_value = str(value)
            if len(str_value) > self._config.max_attribute_length:
                str_value = str_value[:self._config.max_attribute_length] + "..."
            
            span.set_attribute(key, str_value)
    
    def add_request_attributes(self, span, context, request_data: Optional[Dict[str, Any]] = None) -> None:
        """Add MemFuse-specific request attributes to span."""
        if not span or not span.is_recording():
            return

        # Handle both RequestContext objects and dictionaries
        if isinstance(context, dict):
            # Dictionary input
            user_id = context.get("user_id", "")
            user_name = context.get("user_name")
            agent_id = context.get("agent_id")
            agent_name = context.get("agent_name")
            session_id = context.get("session_id")
            session_name = context.get("session_name")
        else:
            # RequestContext object
            user_id = getattr(context, 'user_id', "") or ""
            user_name = getattr(context, 'user_name', None)
            agent_id = getattr(context, 'agent_id', None)
            agent_name = getattr(context, 'agent_name', None)
            session_id = getattr(context, 'session_id', None)
            session_name = getattr(context, 'session_name', None)

        # Request context attributes
        span.set_attribute("memfuse.user_id", user_id)
        if user_name:
            span.set_attribute("memfuse.user_name", user_name)
        if agent_id:
            span.set_attribute("memfuse.agent_id", agent_id)
        if agent_name:
            span.set_attribute("memfuse.agent_name", agent_name)
        if session_id:
            span.set_attribute("memfuse.session_id", session_id)
        if session_name:
            span.set_attribute("memfuse.session_name", session_name)

        # Handle operation_type
        if isinstance(context, dict):
            operation_type = context.get("operation_type")
        else:
            operation_type = getattr(context, 'operation_type', None)

        if operation_type:
            span.set_attribute("memfuse.operation_type", str(operation_type))

        # Request data attributes (if enabled)
        if self._config.include_request_body and request_data:
            if "query" in request_data:
                query = str(request_data["query"])
                if len(query) > self._config.max_attribute_length:
                    query = query[:self._config.max_attribute_length] + "..."
                span.set_attribute("memfuse.query", query)

            if "top_k" in request_data:
                span.set_attribute("memfuse.top_k", int(request_data["top_k"]))

        # Request metadata
        if isinstance(context, dict):
            request_metadata = context.get("request_metadata")
        else:
            request_metadata = getattr(context, 'request_metadata', None)

        if request_metadata:
            for key, value in request_metadata.items():
                attr_key = f"memfuse.metadata.{key}"
                str_value = str(value)
                if len(str_value) > self._config.max_attribute_length:
                    str_value = str_value[:self._config.max_attribute_length] + "..."
                span.set_attribute(attr_key, str_value)
    
    def add_response_attributes(self, span, response: Dict[str, Any]) -> None:
        """Add response attributes to span."""
        if not span or not span.is_recording():
            return
        
        # Response status
        if "status" in response:
            span.set_attribute("memfuse.response.status", str(response["status"]))
        if "code" in response:
            span.set_attribute("memfuse.response.code", int(response["code"]))
        
        # Response data summary
        if "data" in response and isinstance(response["data"], dict):
            data = response["data"]
            if "results" in data and isinstance(data["results"], list):
                span.set_attribute("memfuse.response.result_count", len(data["results"]))
            if "total" in data:
                span.set_attribute("memfuse.response.total", int(data["total"]))
        
        # Response body (if enabled)
        if self._config.include_response_body:
            response_str = str(response)
            if len(response_str) > self._config.max_attribute_length:
                response_str = response_str[:self._config.max_attribute_length] + "..."
            span.set_attribute("memfuse.response.body", response_str)


# Global tracer instance
_tracer: Optional[MemFuseTracer] = None


def get_tracer() -> MemFuseTracer:
    """Get global tracer instance."""
    global _tracer
    if _tracer is None:
        _tracer = MemFuseTracer()
    return _tracer


def initialize_tracing(config: Optional[Dict[str, Any]] = None) -> MemFuseTracer:
    """Initialize distributed tracing."""
    tracer = get_tracer()
    tracer.initialize(config)
    return tracer


def trace_operation(operation_name: str, attributes: Optional[Dict[str, Any]] = None):
    """Decorator for tracing operations."""
    def decorator(func):
        if hasattr(func, '__call__'):
            if hasattr(func, '__code__') and func.__code__.co_flags & 0x80:  # CO_COROUTINE
                # Async function
                async def async_wrapper(*args, **kwargs):
                    tracer = get_tracer()
                    async with tracer.trace_async_operation(operation_name, attributes):
                        return await func(*args, **kwargs)
                return async_wrapper
            else:
                # Sync function
                def sync_wrapper(*args, **kwargs):
                    tracer = get_tracer()
                    with tracer.trace_operation(operation_name, attributes):
                        return func(*args, **kwargs)
                return sync_wrapper
        return func
    return decorator
