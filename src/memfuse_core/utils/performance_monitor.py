"""Performance monitoring utilities for MemFuse."""

import time
import statistics
from typing import Dict, List, Any, Optional


class PerformanceMonitor:
    """Performance monitoring class for MemFuse.

    This class provides utilities for monitoring and reporting the performance
    of MemFuse operations.
    """

    def __init__(self, enabled: bool = True):
        """Initialize the performance monitor.

        Args:
            enabled: Whether the monitor is enabled
        """
        self.enabled = enabled
        self.timings: Dict[str, List[Dict[str, float]]] = {}
        self.counters: Dict[str, int] = {}
        self.start_time = time.time()

    def start_timer(self, operation: str) -> None:
        """Start a timer for an operation.

        Args:
            operation: Name of the operation
        """
        if not self.enabled:
            return

        if operation not in self.timings:
            self.timings[operation] = []

        self.timings[operation].append({"start": time.time()})

    def end_timer(self, operation: str) -> Optional[float]:
        """End a timer for an operation.

        Args:
            operation: Name of the operation

        Returns:
            Duration of the operation in seconds, or None if no timer was started
        """
        if not self.enabled:
            return None

        if operation in self.timings and self.timings[operation]:
            last_timing = self.timings[operation][-1]
            if "start" in last_timing and "end" not in last_timing:
                last_timing["end"] = time.time()
                last_timing["duration"] = last_timing["end"] - \
                    last_timing["start"]
                return last_timing["duration"]

        return None

    def increment_counter(self, counter: str, amount: int = 1) -> None:
        """Increment a counter.

        Args:
            counter: Name of the counter
            amount: Amount to increment by
        """
        if not self.enabled:
            return

        if counter not in self.counters:
            self.counters[counter] = 0

        self.counters[counter] += amount

    def get_counter(self, counter: str) -> int:
        """Get the value of a counter.

        Args:
            counter: Name of the counter

        Returns:
            Value of the counter
        """
        if not self.enabled or counter not in self.counters:
            return 0

        return self.counters[counter]

    def get_stats(self, operation: str) -> Optional[Dict[str, Any]]:
        """Get statistics for an operation.

        Args:
            operation: Name of the operation

        Returns:
            Dictionary of statistics, or None if no timings are available
        """
        if not self.enabled or operation not in self.timings:
            return None

        durations = [t.get("duration", 0)
                     for t in self.timings[operation] if "duration" in t]
        if not durations:
            return None

        return {
            "count": len(durations),
            "min": min(durations) * 1000,  # Convert to milliseconds
            "max": max(durations) * 1000,
            "avg": statistics.mean(durations) * 1000,
            "median": statistics.median(durations) * 1000,
            "p95": sorted(durations)[int(len(durations) * 0.95)] * 1000 if len(durations) >= 20 else None,
            "total": sum(durations) * 1000
        }

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all operations.

        Returns:
            Dictionary of operation names to statistics
        """
        if not self.enabled:
            return {}

        stats = {}
        for operation in self.timings:
            op_stats = self.get_stats(operation)
            if op_stats:
                stats[operation] = op_stats

        return stats

    def print_stats(self) -> None:
        """Print statistics for all operations."""
        if not self.enabled:
            print("Performance monitoring is disabled.")
            return

        print("\n===== Performance Statistics =====")

        # Print timings
        for operation, stats in self.get_all_stats().items():
            print(f"{operation}:")
            print(f"  Count: {stats['count']}")
            print(f"  Min: {stats['min']:.2f} ms")
            print(f"  Max: {stats['max']:.2f} ms")
            print(f"  Avg: {stats['avg']:.2f} ms")
            print(f"  Median: {stats['median']:.2f} ms")
            if stats.get('p95'):
                print(f"  P95: {stats['p95']:.2f} ms")
            print(f"  Total: {stats['total']:.2f} ms")
            print()

        # Print counters
        if self.counters:
            print("Counters:")
            for counter, value in self.counters.items():
                print(f"  {counter}: {value}")
            print()

        # Print total runtime
        runtime = time.time() - self.start_time
        print(f"Total runtime: {runtime:.2f} seconds")

    def reset(self) -> None:
        """Reset all timings and counters."""
        if not self.enabled:
            return

        self.timings = {}
        self.counters = {}
        self.start_time = time.time()

    def enable(self) -> None:
        """Enable the performance monitor."""
        self.enabled = True

    def disable(self) -> None:
        """Disable the performance monitor."""
        self.enabled = False


# Global performance monitor instance
monitor = PerformanceMonitor()


def start_timer(operation: str) -> None:
    """Start a timer for an operation.

    Args:
        operation: Name of the operation
    """
    monitor.start_timer(operation)


def end_timer(operation: str) -> Optional[float]:
    """End a timer for an operation.

    Args:
        operation: Name of the operation

    Returns:
        Duration of the operation in seconds, or None if no timer was started
    """
    return monitor.end_timer(operation)


def increment_counter(counter: str, amount: int = 1) -> None:
    """Increment a counter.

    Args:
        counter: Name of the counter
        amount: Amount to increment by
    """
    monitor.increment_counter(counter, amount)


def get_counter(counter: str) -> int:
    """Get the value of a counter.

    Args:
        counter: Name of the counter

    Returns:
        Value of the counter
    """
    return monitor.get_counter(counter)


def get_stats(operation: str) -> Optional[Dict[str, Any]]:
    """Get statistics for an operation.

    Args:
        operation: Name of the operation

    Returns:
        Dictionary of statistics, or None if no timings are available
    """
    return monitor.get_stats(operation)


def get_all_stats() -> Dict[str, Dict[str, Any]]:
    """Get statistics for all operations.

    Returns:
        Dictionary of operation names to statistics
    """
    return monitor.get_all_stats()


def print_stats() -> None:
    """Print statistics for all operations."""
    monitor.print_stats()


def reset() -> None:
    """Reset all timings and counters."""
    monitor.reset()


def enable() -> None:
    """Enable the performance monitor."""
    monitor.enable()


def disable() -> None:
    """Disable the performance monitor."""
    monitor.disable()
