"""
Performance monitoring system for MemFuse optimization tracking.

This module provides comprehensive performance monitoring capabilities including:
- Operation timing and latency tracking
- Resource usage monitoring
- Performance regression detection
- Real-time metrics collection
"""

import asyncio
import time
import threading
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from contextlib import asynccontextmanager
import statistics
import logging

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Individual performance metric data point."""
    operation: str
    duration: float
    timestamp: float
    success: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceStats:
    """Aggregated performance statistics."""
    operation: str
    count: int
    avg_duration: float
    min_duration: float
    max_duration: float
    p95_duration: float
    p99_duration: float
    success_rate: float
    last_updated: float


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system.
    
    P4 OPTIMIZATION: Real-time performance tracking and regression detection.
    """
    
    def __init__(self, max_metrics_per_operation: int = 1000):
        self.max_metrics_per_operation = max_metrics_per_operation
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_metrics_per_operation))
        self._lock = threading.RLock()
        self._enabled = True
        
        # Performance thresholds for regression detection
        self._thresholds = {
            "connection_pool_access": 0.010,  # 10ms
            "m0_operation": 0.100,            # 100ms
            "m1_operation": 0.150,            # 150ms
            "m2_operation": 0.150,            # 150ms
            "buffer_flush": 0.050,            # 50ms
            "service_access": 0.010,          # 10ms
        }
        
        # Alert callbacks
        self._alert_callbacks: List[Callable[[str, PerformanceMetric], None]] = []
    
    def enable(self):
        """Enable performance monitoring."""
        self._enabled = True
        logger.info("PerformanceMonitor: Monitoring enabled")
    
    def disable(self):
        """Disable performance monitoring."""
        self._enabled = False
        logger.info("PerformanceMonitor: Monitoring disabled")
    
    def set_threshold(self, operation: str, threshold_seconds: float):
        """Set performance threshold for an operation."""
        with self._lock:
            self._thresholds[operation] = threshold_seconds
            logger.debug(f"PerformanceMonitor: Set threshold for {operation}: {threshold_seconds}s")
    
    def add_alert_callback(self, callback: Callable[[str, PerformanceMetric], None]):
        """Add callback for performance alerts."""
        self._alert_callbacks.append(callback)
    
    def record_metric(self, operation: str, duration: float, success: bool = True, metadata: Optional[Dict[str, Any]] = None):
        """Record a performance metric."""
        if not self._enabled:
            return
        
        metric = PerformanceMetric(
            operation=operation,
            duration=duration,
            timestamp=time.time(),
            success=success,
            metadata=metadata or {}
        )
        
        with self._lock:
            self._metrics[operation].append(metric)
            
            # Check for threshold violations
            threshold = self._thresholds.get(operation)
            if threshold and duration > threshold:
                self._trigger_alert(f"threshold_violation", metric)
    
    @asynccontextmanager
    async def track_operation(self, operation: str, metadata: Optional[Dict[str, Any]] = None):
        """Context manager for tracking operation performance."""
        if not self._enabled:
            yield
            return
        
        start_time = time.time()
        success = True
        
        try:
            yield
        except Exception as e:
            success = False
            if metadata is None:
                metadata = {}
            metadata["error"] = str(e)
            raise
        finally:
            duration = time.time() - start_time
            self.record_metric(operation, duration, success, metadata)
    
    def get_stats(self, operation: str) -> Optional[PerformanceStats]:
        """Get aggregated statistics for an operation."""
        with self._lock:
            metrics = list(self._metrics.get(operation, []))
        
        if not metrics:
            return None
        
        durations = [m.duration for m in metrics]
        successful_metrics = [m for m in metrics if m.success]
        
        return PerformanceStats(
            operation=operation,
            count=len(metrics),
            avg_duration=statistics.mean(durations),
            min_duration=min(durations),
            max_duration=max(durations),
            p95_duration=self._percentile(durations, 95),
            p99_duration=self._percentile(durations, 99),
            success_rate=len(successful_metrics) / len(metrics) if metrics else 0.0,
            last_updated=time.time()
        )
    
    def get_all_stats(self) -> Dict[str, PerformanceStats]:
        """Get statistics for all tracked operations."""
        with self._lock:
            operations = list(self._metrics.keys())
        
        return {op: self.get_stats(op) for op in operations if self.get_stats(op)}
    
    def detect_regressions(self, baseline_stats: Dict[str, PerformanceStats]) -> List[str]:
        """Detect performance regressions compared to baseline."""
        regressions = []
        current_stats = self.get_all_stats()
        
        for operation, baseline in baseline_stats.items():
            current = current_stats.get(operation)
            if not current:
                continue
            
            # Check for significant performance degradation (>20% increase in avg duration)
            if current.avg_duration > baseline.avg_duration * 1.2:
                regression_msg = (
                    f"Performance regression in {operation}: "
                    f"{baseline.avg_duration:.3f}s -> {current.avg_duration:.3f}s "
                    f"({((current.avg_duration / baseline.avg_duration - 1) * 100):.1f}% increase)"
                )
                regressions.append(regression_msg)
                logger.warning(f"PerformanceMonitor: {regression_msg}")
        
        return regressions
    
    def clear_metrics(self, operation: Optional[str] = None):
        """Clear metrics for specific operation or all operations."""
        with self._lock:
            if operation:
                self._metrics[operation].clear()
                logger.debug(f"PerformanceMonitor: Cleared metrics for {operation}")
            else:
                self._metrics.clear()
                logger.debug("PerformanceMonitor: Cleared all metrics")
    
    def export_metrics(self, operation: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Export metrics for analysis."""
        with self._lock:
            if operation:
                operations = [operation] if operation in self._metrics else []
            else:
                operations = list(self._metrics.keys())
            
            return {
                op: [
                    {
                        "operation": m.operation,
                        "duration": m.duration,
                        "timestamp": m.timestamp,
                        "success": m.success,
                        "metadata": m.metadata
                    }
                    for m in self._metrics[op]
                ]
                for op in operations
            }
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))
    
    def _trigger_alert(self, alert_type: str, metric: PerformanceMetric):
        """Trigger performance alert."""
        for callback in self._alert_callbacks:
            try:
                callback(alert_type, metric)
            except Exception as e:
                logger.error(f"PerformanceMonitor: Alert callback failed: {e}")


# Global performance monitor instance
_global_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor


def track_performance(operation: str, metadata: Optional[Dict[str, Any]] = None):
    """Decorator for tracking function performance."""
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                monitor = get_performance_monitor()
                async with monitor.track_operation(operation, metadata):
                    return await func(*args, **kwargs)
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                monitor = get_performance_monitor()
                start_time = time.time()
                success = True
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    success = False
                    raise
                finally:
                    duration = time.time() - start_time
                    monitor.record_metric(operation, duration, success, metadata)
            return sync_wrapper
    return decorator
