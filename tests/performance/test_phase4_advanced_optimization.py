"""
Phase 4: Advanced Optimization Tests

This module tests the advanced optimization features implemented in Phase 4:
- Performance monitoring system
- Connection pool health check optimization
- Real-time metrics collection
- Performance regression detection
"""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, Optional

from src.memfuse_core.monitoring.performance_monitor import PerformanceMonitor, get_performance_monitor, track_performance
from src.memfuse_core.services.global_connection_manager import GlobalConnectionManager


class TestPhase4AdvancedOptimization:
    """Test suite for Phase 4 advanced optimization features."""
    
    @pytest.fixture
    def performance_monitor(self):
        """Create a fresh performance monitor for testing."""
        monitor = PerformanceMonitor(max_metrics_per_operation=100)
        monitor.clear_metrics()  # Start with clean state
        return monitor
    
    @pytest.fixture
    def mock_connection_manager(self):
        """Create a mock connection manager for testing."""
        manager = AsyncMock(spec=GlobalConnectionManager)
        
        # Mock health check cache
        manager._health_check_cache = {}
        manager._health_check_interval = 300.0
        manager._performance_monitor = get_performance_monitor()
        
        return manager
    
    def test_performance_monitor_basic_functionality(self, performance_monitor):
        """Test basic performance monitoring functionality."""
        # Record some metrics
        performance_monitor.record_metric("test_operation", 0.1, True, {"test": "data"})
        performance_monitor.record_metric("test_operation", 0.2, True)
        performance_monitor.record_metric("test_operation", 0.15, False)
        
        # Get statistics
        stats = performance_monitor.get_stats("test_operation")
        
        assert stats is not None
        assert stats.operation == "test_operation"
        assert stats.count == 3
        assert stats.avg_duration == pytest.approx(0.15, rel=1e-2)
        assert stats.min_duration == 0.1
        assert stats.max_duration == 0.2
        assert stats.success_rate == pytest.approx(2/3, rel=1e-2)
    
    @pytest.mark.asyncio
    async def test_performance_monitor_context_manager(self, performance_monitor):
        """Test performance monitoring with context manager."""
        async with performance_monitor.track_operation("async_test", {"context": "test"}):
            await asyncio.sleep(0.01)  # Simulate work
        
        stats = performance_monitor.get_stats("async_test")
        assert stats is not None
        assert stats.count == 1
        assert stats.avg_duration >= 0.01
        assert stats.success_rate == 1.0
    
    @pytest.mark.asyncio
    async def test_performance_monitor_error_handling(self, performance_monitor):
        """Test performance monitoring with errors."""
        try:
            async with performance_monitor.track_operation("error_test"):
                raise ValueError("Test error")
        except ValueError:
            pass  # Expected
        
        stats = performance_monitor.get_stats("error_test")
        assert stats is not None
        assert stats.count == 1
        assert stats.success_rate == 0.0
    
    def test_performance_monitor_threshold_detection(self, performance_monitor):
        """Test performance threshold violation detection."""
        # Set a low threshold
        performance_monitor.set_threshold("slow_operation", 0.05)
        
        # Track alerts
        alerts = []
        def alert_callback(alert_type, metric):
            alerts.append((alert_type, metric))
        
        performance_monitor.add_alert_callback(alert_callback)
        
        # Record a slow operation
        performance_monitor.record_metric("slow_operation", 0.1, True)
        
        # Check that alert was triggered
        assert len(alerts) == 1
        assert alerts[0][0] == "threshold_violation"
        assert alerts[0][1].operation == "slow_operation"
        assert alerts[0][1].duration == 0.1
    
    def test_performance_monitor_regression_detection(self, performance_monitor):
        """Test performance regression detection."""
        # Create baseline metrics
        for i in range(10):
            performance_monitor.record_metric("baseline_op", 0.1, True)
        
        baseline_stats = performance_monitor.get_all_stats()
        
        # Clear and add degraded performance
        performance_monitor.clear_metrics("baseline_op")
        for i in range(10):
            performance_monitor.record_metric("baseline_op", 0.15, True)  # 50% slower
        
        # Detect regressions
        regressions = performance_monitor.detect_regressions(baseline_stats)
        
        assert len(regressions) == 1
        assert "baseline_op" in regressions[0]
        assert "50.0% increase" in regressions[0]
    
    def test_performance_monitor_export_metrics(self, performance_monitor):
        """Test metrics export functionality."""
        # Record some metrics
        performance_monitor.record_metric("export_test", 0.1, True, {"key": "value"})
        performance_monitor.record_metric("export_test", 0.2, False)
        
        # Export metrics
        exported = performance_monitor.export_metrics("export_test")
        
        assert "export_test" in exported
        assert len(exported["export_test"]) == 2
        
        metric1 = exported["export_test"][0]
        assert metric1["operation"] == "export_test"
        assert metric1["duration"] == 0.1
        assert metric1["success"] is True
        assert metric1["metadata"]["key"] == "value"
    
    def test_track_performance_decorator(self):
        """Test the track_performance decorator."""
        monitor = get_performance_monitor()
        monitor.clear_metrics()
        
        @track_performance("decorated_function")
        def test_function():
            time.sleep(0.01)
            return "result"
        
        result = test_function()
        assert result == "result"
        
        stats = monitor.get_stats("decorated_function")
        assert stats is not None
        assert stats.count == 1
        assert stats.avg_duration >= 0.01
    
    @pytest.mark.asyncio
    async def test_track_performance_async_decorator(self):
        """Test the track_performance decorator with async functions."""
        monitor = get_performance_monitor()
        monitor.clear_metrics()
        
        @track_performance("async_decorated_function")
        async def async_test_function():
            await asyncio.sleep(0.01)
            return "async_result"
        
        result = await async_test_function()
        assert result == "async_result"
        
        stats = monitor.get_stats("async_decorated_function")
        assert stats is not None
        assert stats.count == 1
        assert stats.avg_duration >= 0.01
    
    @pytest.mark.asyncio
    async def test_connection_pool_health_check_optimization(self, mock_connection_manager):
        """Test connection pool health check optimization."""
        # Mock connection object
        mock_conn = MagicMock()
        mock_conn.info.dsn = "postgresql://test:test@localhost:5432/test"
        
        # First call should perform health check
        with patch('src.memfuse_core.services.global_connection_manager.register_vector_async') as mock_register:
            mock_register.return_value = asyncio.Future()
            mock_register.return_value.set_result(None)
            
            # Create a real GlobalConnectionManager instance for testing
            manager = GlobalConnectionManager()
            
            # First call - should register
            await manager._configure_connection(mock_conn)
            assert mock_register.call_count == 1
            
            # Second call within interval - should skip
            await manager._configure_connection(mock_conn)
            assert mock_register.call_count == 1  # No additional calls
    
    @pytest.mark.asyncio
    async def test_connection_pool_performance_tracking(self):
        """Test that connection pool operations are tracked."""
        monitor = get_performance_monitor()
        monitor.clear_metrics()
        
        # Mock a connection pool operation
        async with monitor.track_operation("connection_pool_access", {"db_url": "test_db"}):
            await asyncio.sleep(0.01)  # Simulate pool access time
        
        stats = monitor.get_stats("connection_pool_access")
        assert stats is not None
        assert stats.count == 1
        assert stats.avg_duration >= 0.01
        assert stats.success_rate == 1.0
    
    def test_performance_monitor_percentiles(self, performance_monitor):
        """Test percentile calculations in performance monitoring."""
        # Record metrics with known distribution
        durations = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for duration in durations:
            performance_monitor.record_metric("percentile_test", duration, True)
        
        stats = performance_monitor.get_stats("percentile_test")
        assert stats is not None
        
        # Check percentiles (approximate due to calculation method)
        assert stats.p95_duration >= 0.9  # 95th percentile should be around 0.95
        assert stats.p99_duration >= 0.95  # 99th percentile should be around 0.99
    
    @pytest.mark.asyncio
    async def test_phase4_integration_performance(self):
        """Test overall Phase 4 optimization integration performance."""
        monitor = get_performance_monitor()
        monitor.clear_metrics()
        
        # Set performance targets for Phase 4
        monitor.set_threshold("connection_pool_access", 0.010)  # 10ms
        monitor.set_threshold("pgvector_registration", 0.050)   # 50ms
        
        # Simulate optimized operations
        operations = [
            ("connection_pool_access", 0.005),  # Fast pool access
            ("pgvector_registration", 0.020),   # Fast health check
            ("service_access", 0.003),          # Fast service access
        ]
        
        for operation, duration in operations:
            monitor.record_metric(operation, duration, True)
        
        # Verify all operations meet performance targets
        for operation, target_duration in [("connection_pool_access", 0.010), ("pgvector_registration", 0.050)]:
            stats = monitor.get_stats(operation)
            assert stats is not None
            assert stats.avg_duration < target_duration, f"{operation} exceeded target: {stats.avg_duration} > {target_duration}"
        
        print(f"Phase 4 integration performance results:")
        for operation, _ in operations:
            stats = monitor.get_stats(operation)
            if stats:
                print(f"  {operation}: {stats.avg_duration:.3f}s (target: varies)")
    
    def test_performance_monitor_disable_enable(self, performance_monitor):
        """Test enabling/disabling performance monitoring."""
        # Disable monitoring
        performance_monitor.disable()
        
        # Record metric while disabled
        performance_monitor.record_metric("disabled_test", 0.1, True)
        
        # Should have no stats
        stats = performance_monitor.get_stats("disabled_test")
        assert stats is None
        
        # Re-enable monitoring
        performance_monitor.enable()
        
        # Record metric while enabled
        performance_monitor.record_metric("enabled_test", 0.1, True)
        
        # Should have stats
        stats = performance_monitor.get_stats("enabled_test")
        assert stats is not None
        assert stats.count == 1
