import pytest
import asyncio
from unittest.mock import Mock, patch
from typing import Dict, Any

from memfuse_core.observability.tracing import (
    MemFuseTracer,
    TraceConfig,
    get_tracer,
    initialize_tracing,
    trace_operation
)
from memfuse_core.interfaces.gateway_interface import RequestContext, OperationType


@pytest.fixture
def mock_opentelemetry():
    """Mock OpenTelemetry components."""
    with patch('memfuse_core.observability.tracing.OPENTELEMETRY_AVAILABLE', True):
        # Mock all OpenTelemetry imports at module level
        with patch.dict('sys.modules', {
            'opentelemetry': Mock(),
            'opentelemetry.trace': Mock(),
            'opentelemetry.exporter.jaeger.thrift': Mock(),
            'opentelemetry.exporter.otlp.proto.grpc.trace_exporter': Mock(),
            'opentelemetry.sdk.trace': Mock(),
            'opentelemetry.sdk.trace.export': Mock(),
            'opentelemetry.sdk.resources': Mock(),
            'opentelemetry.instrumentation.requests': Mock(),
            'opentelemetry.instrumentation.asyncio': Mock(),
            'opentelemetry.propagate': Mock(),
            'opentelemetry.trace.status': Mock()
        }):
            # Import the module after mocking
            import importlib
            import memfuse_core.observability.tracing as tracing_module
            importlib.reload(tracing_module)

            # Set up mocks
            mock_tracer = Mock()
            mock_span = Mock()
            mock_span.is_recording.return_value = True
            mock_span.get_span_context.return_value = Mock(
                trace_id=0x12345678901234567890123456789012,
                span_id=0x1234567890123456
            )
            mock_tracer.start_as_current_span.return_value.__enter__ = Mock(return_value=mock_span)
            mock_tracer.start_as_current_span.return_value.__exit__ = Mock(return_value=None)

            # Mock the trace module functions
            tracing_module.trace = Mock()
            tracing_module.trace.get_tracer.return_value = mock_tracer
            tracing_module.trace.get_current_span.return_value = mock_span
            tracing_module.trace.set_tracer_provider = Mock()

            # Mock other components
            tracing_module.Resource = Mock()
            tracing_module.Resource.create = Mock(return_value=Mock())
            tracing_module.TracerProvider = Mock()
            tracing_module.BatchSpanProcessor = Mock()
            tracing_module.ConsoleSpanExporter = Mock()
            tracing_module.inject = Mock()
            tracing_module.extract = Mock(return_value=Mock())

            yield {
                'trace': tracing_module.trace,
                'tracer': mock_tracer,
                'span': mock_span
            }


def test_trace_config_defaults():
    """Test TraceConfig default values."""
    config = TraceConfig()
    assert config.enabled is False
    assert config.service_name == "memfuse-core"
    assert config.service_version == "1.0.0"
    assert config.exporter_type == "console"
    assert config.sample_rate == 1.0
    assert config.include_request_body is False
    assert config.include_response_body is False
    assert config.max_attribute_length == 1000


def test_tracer_initialization_disabled():
    """Test tracer initialization when tracing is disabled."""
    tracer = MemFuseTracer()
    tracer.initialize({"enabled": False})
    
    assert not tracer.is_enabled()
    assert tracer.get_current_trace_id() is None
    assert tracer.get_current_span_id() is None


def test_tracer_initialization_no_opentelemetry():
    """Test tracer initialization when OpenTelemetry is not available."""
    with patch('memfuse_core.observability.tracing.OPENTELEMETRY_AVAILABLE', False):
        tracer = MemFuseTracer()
        tracer.initialize({"enabled": True})
        
        assert not tracer.is_enabled()


def test_tracer_initialization_enabled(mock_opentelemetry):
    """Test tracer initialization when enabled."""
    # Create a fresh tracer instance to avoid singleton issues
    from memfuse_core.observability.tracing import MemFuseTracer
    tracer = MemFuseTracer()
    tracer.initialize({
        "enabled": True,
        "service_name": "test-service",
        "exporter_type": "console"
    })

    assert tracer.is_enabled()
    assert tracer._config.service_name == "test-service"
    assert tracer._config.exporter_type == "console"


def test_trace_id_extraction(mock_opentelemetry):
    """Test trace ID extraction."""
    tracer = MemFuseTracer()
    tracer.initialize({"enabled": True})
    
    trace_id = tracer.get_current_trace_id()
    assert trace_id == "12345678901234567890123456789012"


def test_span_id_extraction(mock_opentelemetry):
    """Test span ID extraction."""
    tracer = MemFuseTracer()
    tracer.initialize({"enabled": True})
    
    span_id = tracer.get_current_span_id()
    assert span_id == "1234567890123456"


def test_context_injection(mock_opentelemetry):
    """Test context injection into headers."""
    tracer = MemFuseTracer()
    tracer.initialize({"enabled": True})
    
    headers = {"existing": "header"}
    result_headers = tracer.inject_context(headers)
    
    # Should return the same headers object (modified in place)
    assert result_headers is headers
    mock_opentelemetry['trace'].inject.assert_called_once()


def test_context_extraction(mock_opentelemetry):
    """Test context extraction from headers."""
    tracer = MemFuseTracer()
    tracer.initialize({"enabled": True})
    
    headers = {"traceparent": "00-12345678901234567890123456789012-1234567890123456-01"}
    tracer.extract_context(headers)
    
    mock_opentelemetry['trace'].extract.assert_called_once_with(headers)


def test_trace_operation_sync(mock_opentelemetry):
    """Test synchronous operation tracing."""
    tracer = MemFuseTracer()
    tracer.initialize({"enabled": True})
    
    with tracer.trace_operation("test_operation", {"key": "value"}) as span:
        assert span is not None
    
    # Verify span was started and attributes were set
    mock_tracer = mock_opentelemetry['tracer']
    mock_tracer.start_as_current_span.assert_called_with("test_operation")


@pytest.mark.asyncio
async def test_trace_operation_async(mock_opentelemetry):
    """Test asynchronous operation tracing."""
    tracer = MemFuseTracer()
    tracer.initialize({"enabled": True})
    
    async with tracer.trace_async_operation("test_async_operation", {"key": "value"}) as span:
        assert span is not None
        await asyncio.sleep(0.001)  # Simulate async work
    
    # Verify span was started
    mock_tracer = mock_opentelemetry['tracer']
    mock_tracer.start_as_current_span.assert_called_with("test_async_operation")


def test_add_request_attributes(mock_opentelemetry):
    """Test adding request attributes to span."""
    tracer = MemFuseTracer()
    tracer.initialize({"enabled": True, "include_request_body": True})
    
    context = RequestContext(
        user_id="user123",
        user_name="Test User",
        agent_id="agent456",
        session_id="session789",
        operation_type=OperationType.QUERY,
        query="test query"
    )
    
    request_data = {
        "query": "test query",
        "top_k": 5
    }
    
    mock_span = mock_opentelemetry['span']
    tracer.add_request_attributes(mock_span, context, request_data)
    
    # Verify attributes were set
    expected_calls = [
        ("memfuse.user_id", "user123"),
        ("memfuse.user_name", "Test User"),
        ("memfuse.agent_id", "agent456"),
        ("memfuse.session_id", "session789"),
        ("memfuse.operation_type", "OperationType.QUERY"),
        ("memfuse.query", "test query"),
        ("memfuse.top_k", 5)
    ]
    
    for attr_name, attr_value in expected_calls:
        mock_span.set_attribute.assert_any_call(attr_name, attr_value)


def test_add_response_attributes(mock_opentelemetry):
    """Test adding response attributes to span."""
    tracer = MemFuseTracer()
    tracer.initialize({"enabled": True})
    
    response = {
        "status": "success",
        "code": 200,
        "data": {
            "results": [{"id": "1"}, {"id": "2"}],
            "total": 2
        }
    }
    
    mock_span = mock_opentelemetry['span']
    tracer.add_response_attributes(mock_span, response)
    
    # Verify response attributes were set
    expected_calls = [
        ("memfuse.response.status", "success"),
        ("memfuse.response.code", 200),
        ("memfuse.response.result_count", 2),
        ("memfuse.response.total", 2)
    ]
    
    for attr_name, attr_value in expected_calls:
        mock_span.set_attribute.assert_any_call(attr_name, attr_value)


def test_attribute_truncation(mock_opentelemetry):
    """Test attribute value truncation."""
    tracer = MemFuseTracer()
    tracer.initialize({"enabled": True, "max_attribute_length": 10})
    
    mock_span = mock_opentelemetry['span']
    
    # Test with long attribute value
    long_value = "a" * 20
    tracer._add_attributes(mock_span, {"long_attr": long_value})
    
    # Should be truncated to 10 chars + "..."
    mock_span.set_attribute.assert_called_with("long_attr", "aaaaaaaaaa...")


def test_trace_operation_decorator_sync(mock_opentelemetry):
    """Test trace operation decorator for sync functions."""
    @trace_operation("decorated_sync_op")
    def sync_function(x, y):
        return x + y
    
    # Initialize tracer
    tracer = get_tracer()
    tracer.initialize({"enabled": True})
    
    result = sync_function(1, 2)
    assert result == 3


@pytest.mark.asyncio
async def test_trace_operation_decorator_async(mock_opentelemetry):
    """Test trace operation decorator for async functions."""
    @trace_operation("decorated_async_op")
    async def async_function(x, y):
        await asyncio.sleep(0.001)
        return x + y
    
    # Initialize tracer
    tracer = get_tracer()
    tracer.initialize({"enabled": True})
    
    result = await async_function(1, 2)
    assert result == 3


def test_global_tracer_singleton():
    """Test that get_tracer returns singleton instance."""
    tracer1 = get_tracer()
    tracer2 = get_tracer()
    assert tracer1 is tracer2


def test_initialize_tracing_function(mock_opentelemetry):
    """Test initialize_tracing function."""
    config = {"enabled": True, "service_name": "test-service"}
    initialize_tracing(config)
    
    tracer = get_tracer()
    assert tracer.is_enabled()
    assert tracer._config.service_name == "test-service"


def test_disabled_tracer_operations():
    """Test that disabled tracer operations are no-ops."""
    tracer = MemFuseTracer()
    tracer.initialize({"enabled": False})
    
    # All operations should be no-ops
    assert tracer.get_current_trace_id() is None
    assert tracer.get_current_span_id() is None
    
    headers = {"test": "header"}
    result = tracer.inject_context(headers)
    assert result is headers  # Should return unchanged
    
    tracer.extract_context(headers)  # Should not raise
    
    # Context managers should yield without doing anything
    with tracer.trace_operation("test") as span:
        assert span is None


@pytest.mark.asyncio
async def test_disabled_tracer_async_operations():
    """Test that disabled tracer async operations are no-ops."""
    tracer = MemFuseTracer()
    tracer.initialize({"enabled": False})
    
    async with tracer.trace_async_operation("test") as span:
        assert span is None
