"""
Prometheus metrics integration for MemFuse.

This module provides comprehensive metrics collection for monitoring
MemFuse performance, usage patterns, and system health.
"""

import time
from typing import Dict, Any, Optional, List
from contextlib import contextmanager
from dataclasses import dataclass
from loguru import logger

try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Summary, Info,
        CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("Prometheus client not available. Metrics collection disabled.")

    # Define dummy classes when Prometheus is not available
    class CollectorRegistry:
        pass

    class Counter:
        def __init__(self, *args, **kwargs):
            pass

    class Histogram:
        def __init__(self, *args, **kwargs):
            pass

    class Gauge:
        def __init__(self, *args, **kwargs):
            pass

    class Summary:
        def __init__(self, *args, **kwargs):
            pass

    class Info:
        def __init__(self, *args, **kwargs):
            pass

    def generate_latest(registry):
        return b""

    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"

from ..utils.global_config_manager import get_global_config_manager


@dataclass
class MetricLabels:
    """Standard metric labels for MemFuse."""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    layer: Optional[str] = None
    operation: Optional[str] = None
    status: Optional[str] = None
    error_type: Optional[str] = None


class MemFuseMetrics:
    """
    Comprehensive metrics collection for MemFuse.
    
    Provides Prometheus-compatible metrics for:
    - Request/response patterns
    - Memory layer operations
    - Gateway filter performance
    - Cache hit rates
    - Error tracking
    - Resource utilization
    """
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.enabled = False
        self.registry = registry or CollectorRegistry()
        self._metrics: Dict[str, Any] = {}
        
        if PROMETHEUS_AVAILABLE:
            self._initialize_metrics()
            self._load_config()
    
    def _load_config(self):
        """Load metrics configuration."""
        try:
            gcm = get_global_config_manager()
            if gcm.is_initialized():
                metrics_cfg = gcm.get_section("metrics") or {}
                self.enabled = bool(metrics_cfg.get("enabled", False))
                
                if self.enabled:
                    logger.info("MemFuseMetrics: Enabled with Prometheus integration")
                else:
                    logger.info("MemFuseMetrics: Disabled by configuration")
        except Exception as e:
            logger.warning(f"Failed to load metrics config: {e}")
            self.enabled = False
    
    def _initialize_metrics(self):
        """Initialize all Prometheus metrics."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        # Request metrics
        self._metrics['requests_total'] = Counter(
            'memfuse_requests_total',
            'Total number of requests processed',
            ['operation', 'status', 'user_id'],
            registry=self.registry
        )
        
        self._metrics['request_duration'] = Histogram(
            'memfuse_request_duration_seconds',
            'Request processing duration',
            ['operation', 'layer'],
            registry=self.registry,
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        )
        
        # Memory layer metrics
        self._metrics['memory_operations'] = Counter(
            'memfuse_memory_operations_total',
            'Memory layer operations',
            ['layer', 'operation', 'status'],
            registry=self.registry
        )
        
        self._metrics['memory_records'] = Gauge(
            'memfuse_memory_records_count',
            'Number of records in memory layers',
            ['layer', 'user_id'],
            registry=self.registry
        )
        
        # Gateway filter metrics
        self._metrics['filter_executions'] = Counter(
            'memfuse_filter_executions_total',
            'Filter execution count',
            ['filter_name', 'direction', 'status'],
            registry=self.registry
        )
        
        self._metrics['filter_duration'] = Histogram(
            'memfuse_filter_duration_seconds',
            'Filter execution duration',
            ['filter_name', 'direction'],
            registry=self.registry,
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5]
        )
        
        self._metrics['filter_violations'] = Counter(
            'memfuse_filter_violations_total',
            'Filter violations detected',
            ['filter_name', 'violation_type', 'severity'],
            registry=self.registry
        )
        
        # Cache metrics
        self._metrics['cache_operations'] = Counter(
            'memfuse_cache_operations_total',
            'Cache operations',
            ['cache_type', 'operation', 'result'],
            registry=self.registry
        )
        
        self._metrics['cache_hit_rate'] = Gauge(
            'memfuse_cache_hit_rate',
            'Cache hit rate percentage',
            ['cache_type'],
            registry=self.registry
        )
        
        self._metrics['cache_size'] = Gauge(
            'memfuse_cache_size_bytes',
            'Cache size in bytes',
            ['cache_type'],
            registry=self.registry
        )
        
        # RAG system metrics
        self._metrics['rag_operations'] = Counter(
            'memfuse_rag_operations_total',
            'RAG system operations',
            ['operation', 'store_type', 'status'],
            registry=self.registry
        )
        
        self._metrics['rag_retrieval_time'] = Histogram(
            'memfuse_rag_retrieval_seconds',
            'RAG retrieval duration',
            ['store_type'],
            registry=self.registry,
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
        )
        
        self._metrics['rag_results_count'] = Histogram(
            'memfuse_rag_results_count',
            'Number of RAG results returned',
            ['store_type'],
            registry=self.registry,
            buckets=[1, 5, 10, 25, 50, 100, 250, 500]
        )
        
        # Error metrics
        self._metrics['errors_total'] = Counter(
            'memfuse_errors_total',
            'Total errors by type',
            ['error_type', 'component', 'severity'],
            registry=self.registry
        )
        
        # System resource metrics
        self._metrics['active_sessions'] = Gauge(
            'memfuse_active_sessions',
            'Number of active user sessions',
            registry=self.registry
        )
        
        self._metrics['buffer_size'] = Gauge(
            'memfuse_buffer_size',
            'Buffer size by type',
            ['buffer_type', 'user_id'],
            registry=self.registry
        )
        
        # Semantic validation metrics
        self._metrics['semantic_validations'] = Counter(
            'memfuse_semantic_validations_total',
            'Semantic validation operations',
            ['validation_type', 'result'],
            registry=self.registry
        )
        
        self._metrics['semantic_similarity_scores'] = Histogram(
            'memfuse_semantic_similarity_scores',
            'Semantic similarity scores',
            ['comparison_type'],
            registry=self.registry,
            buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )
        
        # System info
        self._metrics['system_info'] = Info(
            'memfuse_system_info',
            'System information',
            registry=self.registry
        )
        
        # Set system info
        self._metrics['system_info'].info({
            'version': '1.0.0',
            'python_version': '3.11+',
            'prometheus_enabled': str(PROMETHEUS_AVAILABLE)
        })
    
    def record_request(self, operation: str, status: str, user_id: Optional[str] = None):
        """Record a request metric."""
        if not self.enabled or not PROMETHEUS_AVAILABLE:
            return
        
        try:
            self._metrics['requests_total'].labels(
                operation=operation,
                status=status,
                user_id=user_id or 'anonymous'
            ).inc()
        except Exception as e:
            logger.warning(f"Failed to record request metric: {e}")
    
    @contextmanager
    def time_request(self, operation: str, layer: Optional[str] = None):
        """Context manager to time request duration."""
        if not self.enabled or not PROMETHEUS_AVAILABLE:
            yield
            return
        
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            try:
                self._metrics['request_duration'].labels(
                    operation=operation,
                    layer=layer or 'unknown'
                ).observe(duration)
            except Exception as e:
                logger.warning(f"Failed to record request duration: {e}")
    
    def record_memory_operation(self, layer: str, operation: str, status: str):
        """Record memory layer operation."""
        if not self.enabled or not PROMETHEUS_AVAILABLE:
            return
        
        try:
            self._metrics['memory_operations'].labels(
                layer=layer,
                operation=operation,
                status=status
            ).inc()
        except Exception as e:
            logger.warning(f"Failed to record memory operation: {e}")
    
    def update_memory_records(self, layer: str, count: int, user_id: Optional[str] = None):
        """Update memory records count."""
        if not self.enabled or not PROMETHEUS_AVAILABLE:
            return
        
        try:
            self._metrics['memory_records'].labels(
                layer=layer,
                user_id=user_id or 'global'
            ).set(count)
        except Exception as e:
            logger.warning(f"Failed to update memory records: {e}")
    
    def record_filter_execution(self, filter_name: str, direction: str, status: str, duration: float):
        """Record filter execution metrics."""
        if not self.enabled or not PROMETHEUS_AVAILABLE:
            return
        
        try:
            self._metrics['filter_executions'].labels(
                filter_name=filter_name,
                direction=direction,
                status=status
            ).inc()
            
            self._metrics['filter_duration'].labels(
                filter_name=filter_name,
                direction=direction
            ).observe(duration)
        except Exception as e:
            logger.warning(f"Failed to record filter execution: {e}")
    
    def record_filter_violation(self, filter_name: str, violation_type: str, severity: str):
        """Record filter violation."""
        if not self.enabled or not PROMETHEUS_AVAILABLE:
            return
        
        try:
            self._metrics['filter_violations'].labels(
                filter_name=filter_name,
                violation_type=violation_type,
                severity=severity
            ).inc()
        except Exception as e:
            logger.warning(f"Failed to record filter violation: {e}")
    
    def record_cache_operation(self, cache_type: str, operation: str, result: str):
        """Record cache operation."""
        if not self.enabled or not PROMETHEUS_AVAILABLE:
            return
        
        try:
            self._metrics['cache_operations'].labels(
                cache_type=cache_type,
                operation=operation,
                result=result
            ).inc()
        except Exception as e:
            logger.warning(f"Failed to record cache operation: {e}")
    
    def update_cache_hit_rate(self, cache_type: str, hit_rate: float):
        """Update cache hit rate."""
        if not self.enabled or not PROMETHEUS_AVAILABLE:
            return
        
        try:
            self._metrics['cache_hit_rate'].labels(cache_type=cache_type).set(hit_rate)
        except Exception as e:
            logger.warning(f"Failed to update cache hit rate: {e}")
    
    def record_error(self, error_type: str, component: str, severity: str = "error"):
        """Record an error."""
        if not self.enabled or not PROMETHEUS_AVAILABLE:
            return
        
        try:
            self._metrics['errors_total'].labels(
                error_type=error_type,
                component=component,
                severity=severity
            ).inc()
        except Exception as e:
            logger.warning(f"Failed to record error: {e}")
    
    def record_semantic_validation(self, validation_type: str, result: str, similarity_score: Optional[float] = None):
        """Record semantic validation metrics."""
        if not self.enabled or not PROMETHEUS_AVAILABLE:
            return
        
        try:
            self._metrics['semantic_validations'].labels(
                validation_type=validation_type,
                result=result
            ).inc()
            
            if similarity_score is not None:
                self._metrics['semantic_similarity_scores'].labels(
                    comparison_type=validation_type
                ).observe(similarity_score)
        except Exception as e:
            logger.warning(f"Failed to record semantic validation: {e}")
    
    def get_metrics(self) -> str:
        """Get metrics in Prometheus format."""
        if not self.enabled or not PROMETHEUS_AVAILABLE:
            return ""
        
        try:
            return generate_latest(self.registry).decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to generate metrics: {e}")
            return ""
    
    def get_content_type(self) -> str:
        """Get Prometheus content type."""
        return CONTENT_TYPE_LATEST


# Global metrics instance
_metrics: Optional[MemFuseMetrics] = None


def get_metrics() -> MemFuseMetrics:
    """Get global metrics instance."""
    global _metrics
    if _metrics is None:
        _metrics = MemFuseMetrics()
    return _metrics


def initialize_metrics():
    """Initialize metrics system."""
    metrics = get_metrics()
    logger.info(f"Metrics system initialized (enabled: {metrics.enabled})")
