"""
Unit tests for Prometheus metrics integration.
"""

import pytest
from unittest.mock import MagicMock, patch
import time

from src.memfuse_core.observability.metrics import (
    MemFuseMetrics,
    MetricLabels,
    get_metrics,
    initialize_metrics,
    PROMETHEUS_AVAILABLE
)


class TestMetricLabels:
    """Test MetricLabels dataclass."""
    
    def test_labels_creation(self):
        """Test metric labels creation."""
        labels = MetricLabels(
            user_id="user123",
            session_id="session456",
            layer="m1",
            operation="query",
            status="success"
        )
        
        assert labels.user_id == "user123"
        assert labels.session_id == "session456"
        assert labels.layer == "m1"
        assert labels.operation == "query"
        assert labels.status == "success"
    
    def test_labels_defaults(self):
        """Test metric labels with defaults."""
        labels = MetricLabels()
        
        assert labels.user_id is None
        assert labels.session_id is None
        assert labels.layer is None
        assert labels.operation is None
        assert labels.status is None


@pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="Prometheus client not available")
class TestMemFuseMetrics:
    """Test MemFuseMetrics class."""
    
    @pytest.fixture
    def metrics(self):
        """Create metrics instance for testing."""
        with patch('src.memfuse_core.observability.metrics.get_global_config_manager'):
            metrics = MemFuseMetrics()
            metrics.enabled = True  # Force enable for testing
            return metrics
    
    def test_metrics_initialization(self, metrics):
        """Test metrics initialization."""
        assert metrics.enabled is True
        assert metrics.registry is not None
        assert len(metrics._metrics) > 0
        
        # Check that key metrics are initialized
        assert 'requests_total' in metrics._metrics
        assert 'request_duration' in metrics._metrics
        assert 'memory_operations' in metrics._metrics
        assert 'filter_executions' in metrics._metrics
        assert 'cache_operations' in metrics._metrics
        assert 'errors_total' in metrics._metrics
    
    def test_record_request(self, metrics):
        """Test request recording."""
        # Should not raise exception
        metrics.record_request("query", "success", "user123")
        metrics.record_request("store", "error", None)
        
        # Verify metric was incremented
        counter = metrics._metrics['requests_total']
        assert counter._value._value > 0
    
    def test_time_request_context_manager(self, metrics):
        """Test request timing context manager."""
        with metrics.time_request("test_operation", "m1"):
            time.sleep(0.01)  # Small delay
        
        # Verify histogram was updated
        histogram = metrics._metrics['request_duration']
        assert histogram._count._value > 0
    
    def test_record_memory_operation(self, metrics):
        """Test memory operation recording."""
        metrics.record_memory_operation("m1", "add", "success")
        metrics.record_memory_operation("m2", "query", "error")
        
        counter = metrics._metrics['memory_operations']
        assert counter._value._value > 0
    
    def test_update_memory_records(self, metrics):
        """Test memory records count update."""
        metrics.update_memory_records("m1", 100, "user123")
        metrics.update_memory_records("m2", 50)
        
        gauge = metrics._metrics['memory_records']
        # Gauge should have been set
        assert len(gauge._metrics) > 0
    
    def test_record_filter_execution(self, metrics):
        """Test filter execution recording."""
        metrics.record_filter_execution("sensitive_word", "outbound", "success", 0.05)
        
        counter = metrics._metrics['filter_executions']
        histogram = metrics._metrics['filter_duration']
        
        assert counter._value._value > 0
        assert histogram._count._value > 0
    
    def test_record_filter_violation(self, metrics):
        """Test filter violation recording."""
        metrics.record_filter_violation("content_filter", "pii", "high")
        
        counter = metrics._metrics['filter_violations']
        assert counter._value._value > 0
    
    def test_record_cache_operation(self, metrics):
        """Test cache operation recording."""
        metrics.record_cache_operation("regex", "get", "hit")
        metrics.record_cache_operation("content", "set", "success")
        
        counter = metrics._metrics['cache_operations']
        assert counter._value._value > 0
    
    def test_update_cache_hit_rate(self, metrics):
        """Test cache hit rate update."""
        metrics.update_cache_hit_rate("regex", 0.85)
        
        gauge = metrics._metrics['cache_hit_rate']
        assert len(gauge._metrics) > 0
    
    def test_record_error(self, metrics):
        """Test error recording."""
        metrics.record_error("validation_error", "gateway", "error")
        metrics.record_error("timeout", "rag", "warning")
        
        counter = metrics._metrics['errors_total']
        assert counter._value._value > 0
    
    def test_record_semantic_validation(self, metrics):
        """Test semantic validation recording."""
        metrics.record_semantic_validation("similarity", "violation", 0.9)
        metrics.record_semantic_validation("coherence", "passed")
        
        counter = metrics._metrics['semantic_validations']
        histogram = metrics._metrics['semantic_similarity_scores']
        
        assert counter._value._value > 0
        assert histogram._count._value > 0
    
    def test_get_metrics_output(self, metrics):
        """Test metrics output generation."""
        # Record some metrics first
        metrics.record_request("test", "success")
        metrics.record_error("test_error", "test_component")
        
        output = metrics.get_metrics()
        
        assert isinstance(output, str)
        assert len(output) > 0
        assert "memfuse_requests_total" in output
        assert "memfuse_errors_total" in output
    
    def test_get_content_type(self, metrics):
        """Test content type for metrics."""
        content_type = metrics.get_content_type()
        assert "text/plain" in content_type


class TestMemFuseMetricsDisabled:
    """Test MemFuseMetrics when disabled."""
    
    @pytest.fixture
    def disabled_metrics(self):
        """Create disabled metrics instance."""
        with patch('src.memfuse_core.observability.metrics.get_global_config_manager'):
            metrics = MemFuseMetrics()
            metrics.enabled = False
            return metrics
    
    def test_disabled_metrics_operations(self, disabled_metrics):
        """Test that disabled metrics don't raise exceptions."""
        # All operations should be no-ops
        disabled_metrics.record_request("test", "success")
        disabled_metrics.record_memory_operation("m1", "add", "success")
        disabled_metrics.record_filter_execution("test", "inbound", "success", 0.1)
        disabled_metrics.record_error("test", "component")
        
        # Should return empty string
        output = disabled_metrics.get_metrics()
        assert output == ""


@pytest.mark.skipif(PROMETHEUS_AVAILABLE, reason="Testing without Prometheus")
class TestMetricsWithoutPrometheus:
    """Test metrics behavior when Prometheus is not available."""
    
    def test_metrics_without_prometheus(self):
        """Test metrics creation without Prometheus."""
        metrics = MemFuseMetrics()
        
        # Should not crash
        assert metrics.enabled is False
        assert len(metrics._metrics) == 0
        
        # Operations should be no-ops
        metrics.record_request("test", "success")
        output = metrics.get_metrics()
        assert output == ""


class TestMetricsGlobals:
    """Test global metrics functions."""
    
    def test_get_metrics_singleton(self):
        """Test that get_metrics returns singleton."""
        metrics1 = get_metrics()
        metrics2 = get_metrics()
        
        assert metrics1 is metrics2
    
    def test_initialize_metrics(self):
        """Test metrics initialization."""
        # Should not raise exception
        initialize_metrics()
        
        # Should create global instance
        metrics = get_metrics()
        assert metrics is not None


class TestMetricsErrorHandling:
    """Test error handling in metrics."""
    
    @pytest.fixture
    def metrics_with_mock_error(self):
        """Create metrics with mocked error conditions."""
        with patch('src.memfuse_core.observability.metrics.get_global_config_manager'):
            metrics = MemFuseMetrics()
            metrics.enabled = True
            return metrics
    
    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="Prometheus client not available")
    def test_error_handling_in_record_request(self, metrics_with_mock_error):
        """Test error handling in record_request."""
        # Mock the counter to raise an exception
        with patch.object(metrics_with_mock_error._metrics['requests_total'], 'labels', side_effect=Exception("Test error")):
            # Should not raise exception
            metrics_with_mock_error.record_request("test", "success")
    
    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="Prometheus client not available")
    def test_error_handling_in_get_metrics(self, metrics_with_mock_error):
        """Test error handling in get_metrics."""
        with patch('src.memfuse_core.observability.metrics.generate_latest', side_effect=Exception("Test error")):
            output = metrics_with_mock_error.get_metrics()
            assert output == ""


class TestMetricsConfiguration:
    """Test metrics configuration loading."""
    
    def test_config_loading_enabled(self):
        """Test config loading when metrics are enabled."""
        mock_config = {
            "metrics": {
                "enabled": True
            }
        }
        
        with patch('src.memfuse_core.observability.metrics.get_global_config_manager') as mock_gcm:
            mock_gcm.return_value.is_initialized.return_value = True
            mock_gcm.return_value.get_section.return_value = mock_config["metrics"]
            
            metrics = MemFuseMetrics()
            assert metrics.enabled is True
    
    def test_config_loading_disabled(self):
        """Test config loading when metrics are disabled."""
        mock_config = {
            "metrics": {
                "enabled": False
            }
        }
        
        with patch('src.memfuse_core.observability.metrics.get_global_config_manager') as mock_gcm:
            mock_gcm.return_value.is_initialized.return_value = True
            mock_gcm.return_value.get_section.return_value = mock_config["metrics"]
            
            metrics = MemFuseMetrics()
            assert metrics.enabled is False
    
    def test_config_loading_error(self):
        """Test config loading with error."""
        with patch('src.memfuse_core.observability.metrics.get_global_config_manager', side_effect=Exception("Config error")):
            metrics = MemFuseMetrics()
            assert metrics.enabled is False
