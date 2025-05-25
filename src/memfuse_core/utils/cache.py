"""Enhanced caching utilities for MemFuse.

This module provides advanced caching implementations with configurable
eviction strategies, monitoring, and thread-safety features.
"""

from typing import Any, Dict, TypeVar, Generic, Callable, Optional
from collections import OrderedDict, Counter
import time
import threading
from loguru import logger
import random

K = TypeVar('K')  # Key type
V = TypeVar('V')  # Value type


class Cache(Generic[K, V]):
    """Enhanced cache implementation with configurable eviction strategies and monitoring.

    Features:
    - Multiple eviction strategies (LRU, FIFO, LFU, ARC, Random)
    - Time-based expiration (TTL)
    - Size-based limits
    - Cache statistics and monitoring
    - Cache preloading/warming
    - Thread-safe operations
    """

    VALID_STRATEGIES = {"lru", "fifo", "lfu", "arc", "random", "tlru"}

    def __init__(
        self,
        max_size: int = 100,
        ttl: Optional[float] = None,
        eviction_strategy: str = "lru",
        monitor_interval: Optional[float] = None,
        thread_safe: bool = True
    ):
        """Initialize the cache.

        Args:
            max_size: Maximum number of items in the cache
            ttl: Time-to-live in seconds (None for no expiration)
            eviction_strategy: Eviction strategy ("lru", "fifo", "lfu", "arc", "random", "tlru")
            monitor_interval: Interval in seconds for cache monitoring (None to disable)
            thread_safe: Whether to make cache operations thread-safe
        """
        self.max_size = max_size
        self.ttl = ttl
        self.eviction_strategy = eviction_strategy.lower()
        self.monitor_interval = monitor_interval
        self.thread_safe = thread_safe

        if self.eviction_strategy not in self.VALID_STRATEGIES:
            raise ValueError(f"Unknown eviction strategy: {eviction_strategy}. "
                             f"Valid strategies: {', '.join(self.VALID_STRATEGIES)}")

        # Initialize lock if thread-safe
        self._lock = threading.RLock() if thread_safe else None

        # Cache storage
        if self.eviction_strategy == "lru":
            self.cache = OrderedDict()  # LRU cache
        elif self.eviction_strategy == "fifo":
            self.cache = OrderedDict()  # FIFO cache
        elif self.eviction_strategy == "lfu":
            self.cache = {}  # LFU cache
            self.access_count = {}  # Access count for LFU
        elif self.eviction_strategy == "arc":
            # Adaptive Replacement Cache (ARC)
            self.t1 = OrderedDict()  # Recently used items
            self.t2 = OrderedDict()  # Frequently used items
            self.b1 = OrderedDict()  # Ghost entries for recently used items
            self.b2 = OrderedDict()  # Ghost entries for frequently used items
            self.p = 0  # Target size for t1
            self.cache = {}  # Combined cache (t1 + t2)
        elif self.eviction_strategy == "random":
            self.cache = {}  # Random eviction
        elif self.eviction_strategy == "tlru":
            # Time-aware Least Recently Used
            self.cache = OrderedDict()  # TLRU cache
            self.access_time = {}  # Last access time

        # Cache timestamps for TTL
        self.timestamps = {}

        # Cache statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.expirations = 0
        self.total_lookup_time = 0
        self.lookups = 0

        # Key frequency tracking
        self.key_frequency = Counter()

    def _acquire_lock(self):
        """Acquire the lock if thread-safe is enabled."""
        if self._lock:
            self._lock.acquire()

    def _release_lock(self):
        """Release the lock if thread-safe is enabled."""
        if self._lock:
            self._lock.release()

    def _remove_item(self, key: K):
        """Remove an item from the cache.

        Args:
            key: Key to remove
        """
        if self.eviction_strategy == "lru" or self.eviction_strategy == "fifo" or self.eviction_strategy == "tlru":
            if key in self.cache:
                del self.cache[key]
                if self.eviction_strategy == "tlru" and key in self.access_time:
                    del self.access_time[key]
        elif self.eviction_strategy == "lfu":
            if key in self.cache:
                del self.cache[key]
                if key in self.access_count:
                    del self.access_count[key]
        elif self.eviction_strategy == "arc":
            if key in self.t1:
                del self.t1[key]
            if key in self.t2:
                del self.t2[key]
            if key in self.cache:
                del self.cache[key]
        elif self.eviction_strategy == "random":
            if key in self.cache:
                del self.cache[key]

        # Remove from timestamps
        if key in self.timestamps:
            del self.timestamps[key]

    def _evict_item(self):
        """Evict an item from the cache based on the eviction strategy."""
        if not self.cache:
            return

        if self.eviction_strategy == "lru":
            # LRU: Remove least recently used item (first item in OrderedDict)
            key, _ = self.cache.popitem(last=False)
        elif self.eviction_strategy == "fifo":
            # FIFO: Remove oldest item (first item in OrderedDict)
            key, _ = self.cache.popitem(last=False)
        elif self.eviction_strategy == "lfu":
            # LFU: Remove least frequently used item
            key = min(self.access_count, key=self.access_count.get)
            del self.cache[key]
            del self.access_count[key]
        elif self.eviction_strategy == "arc":
            # ARC: Eviction based on ARC algorithm
            if len(self.t1) >= max(1, self.p):
                # Evict from t1
                key, _ = self.t1.popitem(last=False)
                # Add to b1 (ghost entries)
                self.b1[key] = True
                if len(self.b1) > self.max_size:
                    self.b1.popitem(last=False)
            else:
                # Evict from t2
                key, _ = self.t2.popitem(last=False)
                # Add to b2 (ghost entries)
                self.b2[key] = True
                if len(self.b2) > self.max_size:
                    self.b2.popitem(last=False)
            del self.cache[key]
        elif self.eviction_strategy == "random":
            # Random: Remove a random item
            key = random.choice(list(self.cache.keys()))
            del self.cache[key]
        elif self.eviction_strategy == "tlru":
            # TLRU: Remove item with oldest access time
            oldest_key = min(self.access_time, key=self.access_time.get)
            del self.cache[oldest_key]
            del self.access_time[oldest_key]
            key = oldest_key

        # Remove from timestamps
        if key in self.timestamps:
            del self.timestamps[key]

        self.evictions += 1

    def get(self, key: K, default: V = None) -> Optional[V]:
        """Get an item from the cache.

        Args:
            key: Key to get
            default: Default value to return if key not found

        Returns:
            Value or default if not found
        """
        try:
            self._acquire_lock()

            start_time = time.time()
            self.lookups += 1
            self.key_frequency[key] += 1

            # Check if key exists
            if key not in self.cache:
                self.misses += 1
                return default

            # Check if expired
            if self.ttl is not None:
                timestamp = self.timestamps.get(key, 0)
                if time.time() - timestamp > self.ttl:
                    # Remove expired item
                    self._remove_item(key)
                    self.misses += 1
                    self.expirations += 1
                    return default

            # Get value
            if self.eviction_strategy == "lru":
                # Move to end (most recently used)
                value = self.cache.pop(key)
                self.cache[key] = value
            elif self.eviction_strategy == "fifo":
                # Don't change order for FIFO
                value = self.cache[key]
            elif self.eviction_strategy == "lfu":
                # Increment access count
                value = self.cache[key]
                self.access_count[key] = self.access_count.get(key, 0) + 1
            elif self.eviction_strategy == "arc":
                # ARC: Update access patterns
                value = self.cache[key]
                if key in self.t1:
                    # Move from t1 to t2 (recently used to frequently used)
                    del self.t1[key]
                    self.t2[key] = True
                elif key in self.t2:
                    # Move to end of t2
                    del self.t2[key]
                    self.t2[key] = True
            elif self.eviction_strategy == "random":
                # No special handling for random
                value = self.cache[key]
            elif self.eviction_strategy == "tlru":
                # Update access time
                value = self.cache[key]
                self.access_time[key] = time.time()

            self.hits += 1
            self.total_lookup_time += time.time() - start_time
            return value
        finally:
            self._release_lock()

    def set(self, key: K, value: V) -> None:
        """Set an item in the cache.

        Args:
            key: Key to set
            value: Value to set
        """
        try:
            self._acquire_lock()

            # Check if we need to evict an item
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._evict_item()

            # Set value
            if self.eviction_strategy == "lru":
                self.cache[key] = value
            elif self.eviction_strategy == "fifo":
                self.cache[key] = value
            elif self.eviction_strategy == "lfu":
                self.cache[key] = value
                self.access_count[key] = self.access_count.get(key, 0)
            elif self.eviction_strategy == "arc":
                # ARC: Handle new items
                self.cache[key] = value
                if key in self.b1:
                    # Item was recently evicted from t1
                    # Increase p (target size for t1)
                    self.p = min(self.max_size, self.p + max(1, len(self.b2) // len(self.b1)))
                    # Move to t2 (frequently used)
                    self.t2[key] = True
                    # Remove from b1
                    del self.b1[key]
                elif key in self.b2:
                    # Item was recently evicted from t2
                    # Decrease p (target size for t1)
                    self.p = max(0, self.p - max(1, len(self.b1) // len(self.b2)))
                    # Move to t2 (frequently used)
                    self.t2[key] = True
                    # Remove from b2
                    del self.b2[key]
                else:
                    # New item, add to t1 (recently used)
                    self.t1[key] = True
            elif self.eviction_strategy == "random":
                self.cache[key] = value
            elif self.eviction_strategy == "tlru":
                self.cache[key] = value
                self.access_time[key] = time.time()

            # Update timestamp
            self.timestamps[key] = time.time()
        finally:
            self._release_lock()

    def delete(self, key: K) -> bool:
        """Delete an item from the cache.

        Args:
            key: Key to delete

        Returns:
            Whether the key was found and deleted
        """
        try:
            self._acquire_lock()

            if key in self.cache:
                self._remove_item(key)
                return True
            return False
        finally:
            self._release_lock()

    def clear(self) -> None:
        """Clear the cache."""
        try:
            self._acquire_lock()

            if self.eviction_strategy == "lru" or self.eviction_strategy == "fifo" or self.eviction_strategy == "tlru":
                self.cache.clear()
                if self.eviction_strategy == "tlru":
                    self.access_time.clear()
            elif self.eviction_strategy == "lfu":
                self.cache.clear()
                self.access_count.clear()
            elif self.eviction_strategy == "arc":
                self.t1.clear()
                self.t2.clear()
                self.b1.clear()
                self.b2.clear()
                self.cache.clear()
                self.p = 0
            elif self.eviction_strategy == "random":
                self.cache.clear()

            self.timestamps.clear()
            self.key_frequency.clear()
        finally:
            self._release_lock()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        try:
            self._acquire_lock()

            stats = {
                "size": len(self.cache),
                "max_size": self.max_size,
                "ttl": self.ttl,
                "eviction_strategy": self.eviction_strategy,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0,
                "evictions": self.evictions,
                "expirations": self.expirations,
                "lookups": self.lookups,
                "avg_lookup_time": self.total_lookup_time / self.lookups if self.lookups > 0 else 0,
                "most_frequent_keys": dict(self.key_frequency.most_common(10)),
            }

            return stats
        finally:
            self._release_lock()

    def preload(self, items: Dict[K, V]) -> None:
        """Preload items into the cache.

        Args:
            items: Dictionary of items to preload
        """
        for key, value in items.items():
            self.set(key, value)

    def memoize(self, func: Callable[..., V]) -> Callable[..., V]:
        """Decorator to memoize a function using this cache.

        Args:
            func: Function to memoize

        Returns:
            Memoized function
        """
        def wrapper(*args, **kwargs):
            # Create a key from the function name and arguments
            key_parts = [func.__name__]
            key_parts.extend([str(arg) for arg in args])
            key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
            key = "_".join(key_parts)

            # Check if result is in cache
            result = self.get(key)
            if result is not None:
                return result

            # Call function and cache result
            result = func(*args, **kwargs)
            self.set(key, result)
            return result

        return wrapper
