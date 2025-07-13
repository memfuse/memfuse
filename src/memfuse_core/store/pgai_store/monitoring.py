"""
Monitoring and logging utilities for pgai embedding processing.

This module provides comprehensive monitoring capabilities for both traditional
polling-based and event-driven embedding processing systems.
"""

import time
import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json

logger = logging.getLogger(__name__)


@dataclass
class ProcessingEvent:
    """Represents a single embedding processing event."""
    record_id: str
    event_type: str  # 'start', 'success', 'failure', 'retry'
    timestamp: float
    duration: Optional[float] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    worker_id: Optional[str] = None


class EmbeddingMonitor:
    """Comprehensive monitoring for embedding processing operations."""
    
    def __init__(self, store_name: str = "pgai_store"):
        self.store_name = store_name
        self.events: deque = deque(maxlen=10000)  # Keep last 10k events
        self.metrics = {
            "total_processed": 0,
            "success_count": 0,
            "failure_count": 0,
            "retry_count": 0,
            "total_processing_time": 0.0,
            "start_time": time.time()
        }
        self.active_processing: Dict[str, ProcessingEvent] = {}
        self.error_patterns: Dict[str, int] = defaultdict(int)
        self.performance_history: deque = deque(maxlen=1000)
        
    def start_processing(self, record_id: str, worker_id: Optional[str] = None, retry_count: int = 0):
        """Record the start of embedding processing."""
        event = ProcessingEvent(
            record_id=record_id,
            event_type='start',
            timestamp=time.time(),
            retry_count=retry_count,
            worker_id=worker_id
        )
        
        self.events.append(event)
        self.active_processing[record_id] = event
        
        if retry_count > 0:
            self.metrics["retry_count"] += 1
            
        logger.debug(f"Started processing {record_id} (worker: {worker_id}, retry: {retry_count})")
        
    def complete_processing(self, record_id: str, success: bool, error_message: Optional[str] = None):
        """Record the completion of embedding processing."""
        if record_id not in self.active_processing:
            logger.warning(f"Completing processing for unknown record: {record_id}")
            return
            
        start_event = self.active_processing.pop(record_id)
        duration = time.time() - start_event.timestamp
        
        event = ProcessingEvent(
            record_id=record_id,
            event_type='success' if success else 'failure',
            timestamp=time.time(),
            duration=duration,
            error_message=error_message,
            retry_count=start_event.retry_count,
            worker_id=start_event.worker_id
        )
        
        self.events.append(event)
        self.metrics["total_processed"] += 1
        self.metrics["total_processing_time"] += duration
        
        if success:
            self.metrics["success_count"] += 1
            logger.debug(f"Completed processing {record_id} in {duration:.3f}s")
        else:
            self.metrics["failure_count"] += 1
            if error_message:
                self.error_patterns[error_message] += 1
            logger.warning(f"Failed processing {record_id} after {duration:.3f}s: {error_message}")
            
        # Record performance metrics
        self.performance_history.append({
            "timestamp": time.time(),
            "duration": duration,
            "success": success,
            "retry_count": start_event.retry_count
        })
        
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current monitoring statistics."""
        uptime = time.time() - self.metrics["start_time"]
        avg_processing_time = (
            self.metrics["total_processing_time"] / max(self.metrics["total_processed"], 1)
        )
        
        # Calculate recent performance (last 100 events)
        recent_events = list(self.performance_history)[-100:]
        recent_success_rate = sum(1 for e in recent_events if e["success"]) / max(len(recent_events), 1)
        recent_avg_time = sum(e["duration"] for e in recent_events) / max(len(recent_events), 1)
        
        return {
            "store_name": self.store_name,
            "uptime_seconds": uptime,
            "total_processed": self.metrics["total_processed"],
            "success_count": self.metrics["success_count"],
            "failure_count": self.metrics["failure_count"],
            "retry_count": self.metrics["retry_count"],
            "success_rate": self.metrics["success_count"] / max(self.metrics["total_processed"], 1),
            "retry_rate": self.metrics["retry_count"] / max(self.metrics["total_processed"], 1),
            "avg_processing_time": avg_processing_time,
            "processing_rate": self.metrics["total_processed"] / max(uptime, 1),
            "active_processing_count": len(self.active_processing),
            "recent_success_rate": recent_success_rate,
            "recent_avg_processing_time": recent_avg_time,
            "top_errors": dict(sorted(self.error_patterns.items(), key=lambda x: x[1], reverse=True)[:5])
        }
        
    def get_performance_trends(self, window_minutes: int = 60) -> Dict[str, Any]:
        """Get performance trends over specified time window."""
        cutoff_time = time.time() - (window_minutes * 60)
        recent_events = [e for e in self.performance_history if e["timestamp"] > cutoff_time]
        
        if not recent_events:
            return {"no_data": True, "window_minutes": window_minutes}
            
        # Group by time buckets (5-minute intervals)
        bucket_size = 300  # 5 minutes
        buckets = defaultdict(list)
        
        for event in recent_events:
            bucket = int(event["timestamp"] // bucket_size) * bucket_size
            buckets[bucket].append(event)
            
        trends = []
        for bucket_time in sorted(buckets.keys()):
            bucket_events = buckets[bucket_time]
            trends.append({
                "timestamp": bucket_time,
                "count": len(bucket_events),
                "success_rate": sum(1 for e in bucket_events if e["success"]) / len(bucket_events),
                "avg_duration": sum(e["duration"] for e in bucket_events) / len(bucket_events),
                "retry_rate": sum(1 for e in bucket_events if e["retry_count"] > 0) / len(bucket_events)
            })
            
        return {
            "window_minutes": window_minutes,
            "bucket_size_seconds": bucket_size,
            "trends": trends,
            "total_events": len(recent_events)
        }
        
    def get_worker_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get per-worker statistics."""
        worker_stats = defaultdict(lambda: {
            "processed": 0,
            "success": 0,
            "failure": 0,
            "total_time": 0.0,
            "active": 0
        })
        
        # Analyze completed events
        for event in self.events:
            if event.worker_id and event.event_type in ['success', 'failure']:
                stats = worker_stats[event.worker_id]
                stats["processed"] += 1
                if event.event_type == 'success':
                    stats["success"] += 1
                else:
                    stats["failure"] += 1
                if event.duration:
                    stats["total_time"] += event.duration
                    
        # Count active processing
        for event in self.active_processing.values():
            if event.worker_id:
                worker_stats[event.worker_id]["active"] += 1
                
        # Calculate derived metrics
        result = {}
        for worker_id, stats in worker_stats.items():
            result[worker_id] = {
                **stats,
                "success_rate": stats["success"] / max(stats["processed"], 1),
                "avg_processing_time": stats["total_time"] / max(stats["processed"], 1)
            }
            
        return result
        
    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in specified format."""
        data = {
            "current_stats": self.get_current_stats(),
            "performance_trends": self.get_performance_trends(),
            "worker_stats": self.get_worker_stats(),
            "export_timestamp": time.time()
        }
        
        if format == "json":
            return json.dumps(data, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
            
    def reset_metrics(self):
        """Reset all metrics and events."""
        self.events.clear()
        self.active_processing.clear()
        self.error_patterns.clear()
        self.performance_history.clear()
        self.metrics = {
            "total_processed": 0,
            "success_count": 0,
            "failure_count": 0,
            "retry_count": 0,
            "total_processing_time": 0.0,
            "start_time": time.time()
        }
        logger.info(f"Reset metrics for {self.store_name}")


class HealthChecker:
    """Health checking utilities for embedding processing systems."""
    
    def __init__(self, monitor: EmbeddingMonitor):
        self.monitor = monitor
        
    async def check_health(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        stats = self.monitor.get_current_stats()
        health_report = {
            "overall_health": "healthy",
            "checks": {},
            "timestamp": time.time()
        }
        
        # Check success rate
        if stats["success_rate"] < 0.8:
            health_report["checks"]["success_rate"] = {
                "status": "warning" if stats["success_rate"] > 0.5 else "critical",
                "value": stats["success_rate"],
                "message": f"Success rate is {stats['success_rate']:.2%}"
            }
        else:
            health_report["checks"]["success_rate"] = {
                "status": "healthy",
                "value": stats["success_rate"]
            }
            
        # Check processing time
        if stats["avg_processing_time"] > 10.0:
            health_report["checks"]["processing_time"] = {
                "status": "warning" if stats["avg_processing_time"] < 30.0 else "critical",
                "value": stats["avg_processing_time"],
                "message": f"Average processing time is {stats['avg_processing_time']:.2f}s"
            }
        else:
            health_report["checks"]["processing_time"] = {
                "status": "healthy",
                "value": stats["avg_processing_time"]
            }
            
        # Check for stuck processing
        stuck_count = len([e for e in self.monitor.active_processing.values() 
                          if time.time() - e.timestamp > 300])  # 5 minutes
        if stuck_count > 0:
            health_report["checks"]["stuck_processing"] = {
                "status": "warning",
                "value": stuck_count,
                "message": f"{stuck_count} records stuck in processing"
            }
        else:
            health_report["checks"]["stuck_processing"] = {
                "status": "healthy",
                "value": 0
            }
            
        # Determine overall health
        check_statuses = [check["status"] for check in health_report["checks"].values()]
        if "critical" in check_statuses:
            health_report["overall_health"] = "critical"
        elif "warning" in check_statuses:
            health_report["overall_health"] = "warning"
            
        return health_report
        
    async def get_recommendations(self) -> List[str]:
        """Get performance and configuration recommendations."""
        stats = self.monitor.get_current_stats()
        recommendations = []
        
        if stats["retry_rate"] > 0.2:
            recommendations.append("High retry rate detected. Consider investigating embedding model performance.")
            
        if stats["avg_processing_time"] > 5.0:
            recommendations.append("High processing time. Consider optimizing embedding model or increasing worker count.")
            
        if stats["active_processing_count"] > 50:
            recommendations.append("High number of active processing tasks. Consider increasing queue size or worker count.")
            
        if stats["success_rate"] < 0.9:
            recommendations.append("Low success rate. Check error logs and model availability.")
            
        return recommendations
