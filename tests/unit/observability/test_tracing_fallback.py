"""
Tests for tracing functionality when OpenTelemetry is not available.

These tests verify that the system gracefully handles the absence of OpenTelemetry
and provides appropriate fallback behavior.
"""

import pytest
from unittest.mock import patch, MagicMock

# Import the tracing module directly to test fallback behavior
from src.memfuse_core.observability.tracing import (
    MemFuseTracer,
    TraceConfig,
    get_tracer,
    initialize_tracing,
    OPENTELEMETRY_AVAILABLE
)


class TestTracingFallback:
    """Test tracing functionality when OpenTelemetry is not available."""
    
    def test_opentelemetry_availability_detection(self):
        """Test that OpenTelemetry availability is correctly detected."""
        # This test verifies the import detection works
        # In our current environment, OpenTelemetry is not available
        assert OPENTELEMETRY_AVAILABLE is False
    
    def test_tracer_initialization_without_opentelemetry(self):
        """Test tracer initialization when OpenTelemetry is not available."""
        tracer = MemFuseTracer()
        tracer.initialize({
            "enabled": True,
            "service_name": "test-service",
            "exporter_type": "console"
        })
        
        # When OpenTelemetry is not available, tracer should be disabled
        assert not tracer.is_enabled()
        assert tracer._config.service_name == "test-service"
        assert tracer._config.exporter_type == "console"
    
    def test_trace_operations_without_opentelemetry(self):
        """Test that trace operations work without OpenTelemetry."""
        tracer = MemFuseTracer()
        tracer.initialize({"enabled": True})
        
        # All operations should work but return None/empty values
        assert tracer.get_current_trace_id() is None
        assert tracer.get_current_span_id() is None
        
        # Context operations should not fail
        headers = {}
        tracer.inject_context(headers)
        assert headers == {}  # Should remain empty
        
        context = tracer.extract_context({"traceparent": "test"})
        assert context is None
    
    def test_sync_trace_operation_without_opentelemetry(self):
        """Test synchronous trace operation without OpenTelemetry."""
        tracer = MemFuseTracer()
        tracer.initialize({"enabled": True})
        
        # Should not fail and return None
        with tracer.trace_operation("test_operation") as span:
            assert span is None
    
    @pytest.mark.asyncio
    async def test_async_trace_operation_without_opentelemetry(self):
        """Test asynchronous trace operation without OpenTelemetry."""
        tracer = MemFuseTracer()
        tracer.initialize({"enabled": True})
        
        # Should not fail and return None
        async with tracer.trace_async_operation("test_operation") as span:
            assert span is None
    
    def test_attribute_handling_without_opentelemetry(self):
        """Test attribute handling when OpenTelemetry is not available."""
        tracer = MemFuseTracer()
        tracer.initialize({"enabled": True})
        
        # Mock span to test attribute handling
        mock_span = MagicMock()
        
        # Should not fail when adding attributes
        request_data = {
            "user_id": "test_user",
            "query": "test query",
            "top_k": 5
        }
        
        # This should not raise an exception
        tracer.add_request_attributes(mock_span, request_data)
        
        response_data = {
            "status": "success",
            "results_count": 3
        }
        
        # This should not raise an exception
        tracer.add_response_attributes(mock_span, response_data)
    
    def test_global_tracer_singleton_without_opentelemetry(self):
        """Test global tracer singleton when OpenTelemetry is not available."""
        tracer1 = get_tracer()
        tracer2 = get_tracer()
        
        # Should return the same instance
        assert tracer1 is tracer2
        
        # Should be disabled when OpenTelemetry is not available
        assert not tracer1.is_enabled()
    
    def test_initialize_tracing_function_without_opentelemetry(self):
        """Test initialize_tracing function when OpenTelemetry is not available."""
        config = {
            "enabled": True,
            "service_name": "test-service",
            "exporter_type": "console"
        }
        
        # Should not raise an exception
        tracer = initialize_tracing(config)
        
        # Should return a tracer instance
        assert isinstance(tracer, MemFuseTracer)
        
        # Should be disabled when OpenTelemetry is not available
        assert not tracer.is_enabled()
    
    def test_trace_config_defaults(self):
        """Test TraceConfig default values."""
        config = TraceConfig()
        
        assert config.enabled is False
        assert config.service_name == "memfuse-core"
        assert config.service_version == "1.0.0"
        assert config.exporter_type == "console"
        assert config.sample_rate == 1.0
        assert config.include_request_body is False
        assert config.include_response_body is False
    
    def test_disabled_tracer_operations(self):
        """Test that disabled tracer operations work correctly."""
        tracer = MemFuseTracer()
        tracer.initialize({"enabled": False})
        
        # Should be disabled
        assert not tracer.is_enabled()
        
        # All operations should work but be no-ops
        assert tracer.get_current_trace_id() is None
        assert tracer.get_current_span_id() is None
        
        headers = {}
        tracer.inject_context(headers)
        assert headers == {}
        
        context = tracer.extract_context({"traceparent": "test"})
        assert context is None
        
        # Trace operations should work
        with tracer.trace_operation("test") as span:
            assert span is None
    
    @pytest.mark.asyncio
    async def test_disabled_tracer_async_operations(self):
        """Test that disabled tracer async operations work correctly."""
        tracer = MemFuseTracer()
        tracer.initialize({"enabled": False})
        
        # Async trace operations should work
        async with tracer.trace_async_operation("test") as span:
            assert span is None


class TestTracingConfiguration:
    """Test tracing configuration handling."""
    
    def test_config_from_dict(self):
        """Test creating TraceConfig from dictionary."""
        config_dict = {
            "enabled": True,
            "service_name": "custom-service",
            "exporter_type": "jaeger",
            "sample_rate": 0.5
        }
        
        config = TraceConfig(**config_dict)
        
        assert config.enabled is True
        assert config.service_name == "custom-service"
        assert config.exporter_type == "jaeger"
        assert config.sample_rate == 0.5
    
    def test_config_partial_override(self):
        """Test partial configuration override."""
        config = TraceConfig(enabled=True, service_name="test-service")
        
        assert config.enabled is True
        assert config.service_name == "test-service"
        # Other values should be defaults
        assert config.exporter_type == "console"
        assert config.sample_rate == 1.0
    
    def test_tracer_config_update(self):
        """Test updating tracer configuration."""
        tracer = MemFuseTracer()
        
        # Initial config
        tracer.initialize({"enabled": False, "service_name": "initial"})
        assert not tracer.is_enabled()
        assert tracer._config.service_name == "initial"
        
        # Update config
        tracer.initialize({"enabled": True, "service_name": "updated"})
        # Should still be disabled due to OpenTelemetry not being available
        assert not tracer.is_enabled()
        assert tracer._config.service_name == "updated"


class TestTracingErrorHandling:
    """Test error handling in tracing functionality."""
    
    def test_tracer_initialization_with_invalid_config(self):
        """Test tracer initialization with invalid configuration."""
        tracer = MemFuseTracer()
        
        # Should handle invalid config gracefully
        tracer.initialize(None)
        assert not tracer.is_enabled()
        
        # Should handle empty config
        tracer.initialize({})
        assert not tracer.is_enabled()
    
    def test_attribute_truncation_without_opentelemetry(self):
        """Test attribute truncation when OpenTelemetry is not available."""
        tracer = MemFuseTracer()
        tracer.initialize({"enabled": True})
        
        # Mock span
        mock_span = MagicMock()
        
        # Long attribute value
        long_value = "a" * 1000
        
        # Should not fail
        tracer._add_attributes(mock_span, {"long_attr": long_value})
        
        # Since OpenTelemetry is not available, this is essentially a no-op
        # but it should not raise an exception
    
    def test_context_operations_with_invalid_data(self):
        """Test context operations with invalid data."""
        tracer = MemFuseTracer()
        tracer.initialize({"enabled": True})
        
        # Should handle None gracefully
        tracer.inject_context(None)
        context = tracer.extract_context(None)
        assert context is None
        
        # Should handle invalid headers
        headers = {"invalid": "data"}
        tracer.inject_context(headers)
        context = tracer.extract_context(headers)
        assert context is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
