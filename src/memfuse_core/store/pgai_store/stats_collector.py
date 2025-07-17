"""
Statistics collection for multi-layer PgAI system.

This module provides centralized statistics collection, aggregation,
and reporting for all PgAI components.
"""

import time
from collections import defaultdict, deque
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class OperationStats:
    """Statistics for a specific operation type."""
    count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    errors: int = 0
    last_operation: Optional[float] = None
    
    def add_operation(self, duration: float, success: bool = True):
        """Add an operation to the statistics."""
        self.count += 1
        self.total_time += duration
        self.min_time = min(self.min_time, duration)
        self.max_time = max(self.max_time, duration)
        self.last_operation = time.time()
        
        if not success:
            self.errors += 1
    
    @property
    def average_time(self) -> float:
        """Calculate average operation time."""
        return self.total_time / self.count if self.count > 0 else 0.0
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate as percentage."""
        return (self.errors / self.count * 100) if self.count > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'count': self.count,
            'total_time': self.total_time,
            'average_time': self.average_time,
            'min_time': self.min_time if self.min_time != float('inf') else 0.0,
            'max_time': self.max_time,
            'errors': self.errors,
            'error_rate': self.error_rate,
            'last_operation': self.last_operation
        }


class StatsCollector:
    """
    Statistics collector for multi-layer PgAI system.

    Provides centralized collection, aggregation, and reporting
    of performance and operational statistics.
    """
    
    def __init__(self, max_recent_operations: int = 100):
        """Initialize statistics collector.
        
        Args:
            max_recent_operations: Maximum number of recent operations to track
        """
        self.max_recent_operations = max_recent_operations
        
        # Operation statistics by type
        self.operations: Dict[str, OperationStats] = defaultdict(OperationStats)
        
        # Layer-specific statistics
        self.layer_stats: Dict[str, Dict[str, OperationStats]] = defaultdict(
            lambda: defaultdict(OperationStats)
        )
        
        # Recent operations for trend analysis
        self.recent_operations: deque = deque(maxlen=max_recent_operations)
        
        # Component-specific counters
        self.counters: Dict[str, int] = defaultdict(int)
        
        # Start time for uptime calculation
        self.start_time = time.time()
        
        logger.info("StatsCollector initialized")
    
    def record_operation(self, operation_type: str, duration: float, 
                        success: bool = True, layer: Optional[str] = None,
                        metadata: Optional[Dict[str, Any]] = None):
        """Record an operation with timing and success information.
        
        Args:
            operation_type: Type of operation (e.g., 'write', 'query', 'fact_extraction')
            duration: Operation duration in seconds
            success: Whether the operation succeeded
            layer: Optional layer name for layer-specific stats
            metadata: Optional additional metadata
        """
        # Record global operation stats
        self.operations[operation_type].add_operation(duration, success)
        
        # Record layer-specific stats if layer provided
        if layer:
            self.layer_stats[layer][operation_type].add_operation(duration, success)
        
        # Add to recent operations
        operation_record = {
            'timestamp': time.time(),
            'type': operation_type,
            'duration': duration,
            'success': success,
            'layer': layer,
            'metadata': metadata or {}
        }
        self.recent_operations.append(operation_record)
        
        # Update counters
        self.counters[f'{operation_type}_total'] += 1
        if success:
            self.counters[f'{operation_type}_success'] += 1
        else:
            self.counters[f'{operation_type}_error'] += 1
        
        if layer:
            self.counters[f'{layer}_{operation_type}_total'] += 1
    
    def increment_counter(self, counter_name: str, value: int = 1):
        """Increment a named counter.
        
        Args:
            counter_name: Name of the counter
            value: Value to increment by (default: 1)
        """
        self.counters[counter_name] += value
    
    def get_operation_stats(self, operation_type: str) -> Dict[str, Any]:
        """Get statistics for a specific operation type.
        
        Args:
            operation_type: Type of operation
            
        Returns:
            Dictionary containing operation statistics
        """
        if operation_type in self.operations:
            return self.operations[operation_type].to_dict()
        else:
            return OperationStats().to_dict()
    
    def get_layer_stats(self, layer: str) -> Dict[str, Any]:
        """Get statistics for a specific layer.
        
        Args:
            layer: Layer name
            
        Returns:
            Dictionary containing layer statistics
        """
        if layer in self.layer_stats:
            return {
                operation_type: stats.to_dict()
                for operation_type, stats in self.layer_stats[layer].items()
            }
        else:
            return {}
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics summary.
        
        Returns:
            Dictionary containing all collected statistics
        """
        return {
            'uptime_seconds': time.time() - self.start_time,
            'total_operations': sum(stats.count for stats in self.operations.values()),
            'total_errors': sum(stats.errors for stats in self.operations.values()),
            'operations': {
                op_type: stats.to_dict()
                for op_type, stats in self.operations.items()
            },
            'layers': {
                layer: {
                    op_type: stats.to_dict()
                    for op_type, stats in layer_ops.items()
                }
                for layer, layer_ops in self.layer_stats.items()
            },
            'counters': dict(self.counters),
            'recent_operations_count': len(self.recent_operations)
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary with key metrics.
        
        Returns:
            Dictionary containing performance summary
        """
        total_ops = sum(stats.count for stats in self.operations.values())
        total_errors = sum(stats.errors for stats in self.operations.values())
        uptime = time.time() - self.start_time
        
        # Calculate operations per second
        ops_per_second = total_ops / uptime if uptime > 0 else 0
        
        # Get average response times
        avg_times = {
            op_type: stats.average_time
            for op_type, stats in self.operations.items()
            if stats.count > 0
        }
        
        return {
            'uptime_seconds': uptime,
            'total_operations': total_ops,
            'total_errors': total_errors,
            'overall_error_rate': (total_errors / total_ops * 100) if total_ops > 0 else 0,
            'operations_per_second': ops_per_second,
            'average_response_times': avg_times,
            'active_layers': list(self.layer_stats.keys())
        }
    
    def get_recent_operations(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get recent operations for trend analysis.
        
        Args:
            limit: Maximum number of operations to return
            
        Returns:
            List of recent operation records
        """
        operations = list(self.recent_operations)
        if limit:
            operations = operations[-limit:]
        return operations
    
    def reset_stats(self):
        """Reset all statistics."""
        self.operations.clear()
        self.layer_stats.clear()
        self.recent_operations.clear()
        self.counters.clear()
        self.start_time = time.time()
        
        logger.info("Statistics reset")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status based on recent operations.
        
        Returns:
            Dictionary containing health status information
        """
        recent_ops = list(self.recent_operations)
        if not recent_ops:
            return {
                'status': 'unknown',
                'message': 'No recent operations',
                'last_activity': None
            }
        
        # Check recent error rate
        recent_errors = sum(1 for op in recent_ops[-10:] if not op['success'])
        recent_error_rate = recent_errors / min(10, len(recent_ops)) * 100
        
        # Determine health status
        if recent_error_rate == 0:
            status = 'healthy'
            message = 'All recent operations successful'
        elif recent_error_rate < 10:
            status = 'warning'
            message = f'Recent error rate: {recent_error_rate:.1f}%'
        else:
            status = 'critical'
            message = f'High error rate: {recent_error_rate:.1f}%'
        
        return {
            'status': status,
            'message': message,
            'recent_error_rate': recent_error_rate,
            'last_activity': recent_ops[-1]['timestamp'] if recent_ops else None,
            'total_operations': len(recent_ops)
        }
