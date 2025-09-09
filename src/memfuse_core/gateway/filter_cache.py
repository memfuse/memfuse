"""
Performance-optimized caching for gateway filters.

This module provides specialized caching mechanisms for expensive filter operations:
- Compiled regex pattern caching for sensitive word filters
- Content hash-based result caching for composite filters
- Quality score caching for content quality assessments
"""

import hashlib
import re
import time
from typing import Any, Dict, List, Optional, Pattern, Tuple, Union
from ..utils.cache import Cache
from ..utils.global_config_manager import get_global_config_manager


class RegexPatternCache:
    """
    Thread-safe cache for compiled regex patterns with performance monitoring.
    
    Optimizes sensitive word filtering by pre-compiling and caching regex patterns
    instead of recompiling them for every text processing operation.
    """
    
    def __init__(self, max_patterns: int = 1000):
        """Initialize regex pattern cache.
        
        Args:
            max_patterns: Maximum number of compiled patterns to cache
        """
        self.cache = Cache[str, Pattern](
            max_size=max_patterns,
            ttl=3600,  # 1 hour TTL for patterns
            eviction_strategy="lru",
            thread_safe=True
        )
        self.compilation_stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "compilation_time": 0.0
        }
    
    def get_pattern(self, pattern_str: str, flags: int = 0) -> Pattern:
        """Get compiled regex pattern from cache or compile and cache it.
        
        Args:
            pattern_str: Regex pattern string
            flags: Regex compilation flags
            
        Returns:
            Compiled regex pattern
        """
        cache_key = f"{pattern_str}|{flags}"
        
        # Try to get from cache first
        cached_pattern = self.cache.get(cache_key)
        if cached_pattern is not None:
            self.compilation_stats["cache_hits"] += 1
            return cached_pattern
        
        # Compile and cache the pattern
        start_time = time.perf_counter()
        try:
            compiled_pattern = re.compile(pattern_str, flags)
            compilation_time = time.perf_counter() - start_time
            
            self.cache.set(cache_key, compiled_pattern)
            self.compilation_stats["cache_misses"] += 1
            self.compilation_stats["compilation_time"] += compilation_time
            
            return compiled_pattern
        except re.error:
            # Return a pattern that never matches for invalid regex
            return re.compile(r"(?!.*)")
    
    def get_word_pattern(self, word: str, case_insensitive: bool = True) -> Pattern:
        """Get compiled pattern for exact word matching.
        
        Args:
            word: Word to create pattern for
            case_insensitive: Whether to use case-insensitive matching
            
        Returns:
            Compiled regex pattern for word boundary matching
        """
        escaped_word = re.escape(word)
        pattern_str = rf"\b{escaped_word}\b"
        flags = re.IGNORECASE if case_insensitive else 0
        return self.get_pattern(pattern_str, flags)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        cache_stats = self.cache.get_stats()
        total_requests = self.compilation_stats["cache_hits"] + self.compilation_stats["cache_misses"]
        hit_rate = (self.compilation_stats["cache_hits"] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "pattern_cache": cache_stats,
            "compilation_stats": self.compilation_stats,
            "hit_rate_percent": round(hit_rate, 2),
            "avg_compilation_time_ms": round(
                (self.compilation_stats["compilation_time"] / max(1, self.compilation_stats["cache_misses"])) * 1000, 3
            )
        }
    
    def clear(self) -> None:
        """Clear the pattern cache and reset statistics."""
        self.cache.clear()
        self.compilation_stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "compilation_time": 0.0
        }


class ContentFilterCache:
    """
    Content-based caching for expensive filter operations.
    
    Caches filter results based on content hashes to avoid reprocessing
    identical content through expensive validation operations.
    """
    
    def __init__(self, max_entries: int = 5000, ttl: float = 1800):
        """Initialize content filter cache.
        
        Args:
            max_entries: Maximum number of cached results
            ttl: Time-to-live in seconds (default: 30 minutes)
        """
        self.cache = Cache[str, Dict[str, Any]](
            max_size=max_entries,
            ttl=ttl,
            eviction_strategy="lru",
            thread_safe=True
        )
        self.hash_stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "hash_collisions": 0
        }
    
    def _compute_content_hash(self, content: str, filter_config: Dict[str, Any]) -> str:
        """Compute hash for content and filter configuration.
        
        Args:
            content: Content to hash
            filter_config: Filter configuration to include in hash
            
        Returns:
            SHA-256 hash string
        """
        # Create a deterministic string representation of config
        config_str = str(sorted(filter_config.items()))
        combined = f"{content}|{config_str}"
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()[:16]  # Use first 16 chars
    
    def get_cached_result(self, content: str, filter_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get cached filter result for content.
        
        Args:
            content: Content to check
            filter_config: Filter configuration
            
        Returns:
            Cached result or None if not found
        """
        if not content:
            return None
        
        cache_key = self._compute_content_hash(content, filter_config)
        cached_result = self.cache.get(cache_key)
        
        if cached_result is not None:
            self.hash_stats["cache_hits"] += 1
            # Deep copy to avoid mutation of nested structures
            import copy
            return copy.deepcopy(cached_result)
        
        self.hash_stats["cache_misses"] += 1
        return None
    
    def cache_result(self, content: str, filter_config: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Cache filter result for content.
        
        Args:
            content: Content that was processed
            filter_config: Filter configuration used
            result: Filter result to cache
        """
        if not content or not result:
            return
        
        cache_key = self._compute_content_hash(content, filter_config)
        
        # Check for hash collision (very unlikely but good to track)
        existing = self.cache.get(cache_key)
        if existing is not None:
            self.hash_stats["hash_collisions"] += 1
        
        self.cache.set(cache_key, result.copy())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        cache_stats = self.cache.get_stats()
        total_requests = self.hash_stats["cache_hits"] + self.hash_stats["cache_misses"]
        hit_rate = (self.hash_stats["cache_hits"] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "content_cache": cache_stats,
            "hash_stats": self.hash_stats,
            "hit_rate_percent": round(hit_rate, 2)
        }
    
    def clear(self) -> None:
        """Clear the content cache and reset statistics."""
        self.cache.clear()
        self.hash_stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "hash_collisions": 0
        }


class QualityScoreCache:
    """
    Specialized cache for content quality scores.
    
    Caches quality assessment results to avoid expensive scoring computations
    for content that has been evaluated before.
    """
    
    def __init__(self, max_entries: int = 10000, ttl: float = 3600):
        """Initialize quality score cache.
        
        Args:
            max_entries: Maximum number of cached scores
            ttl: Time-to-live in seconds (default: 1 hour)
        """
        self.cache = Cache[str, Tuple[float, Dict[str, float]]](
            max_size=max_entries,
            ttl=ttl,
            eviction_strategy="lru",
            thread_safe=True
        )
        self.score_stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "score_computations_saved": 0
        }
    
    def _compute_score_key(self, content: str, query: str, weights: Dict[str, float]) -> str:
        """Compute cache key for quality score.
        
        Args:
            content: Content to score
            query: Query context
            weights: Quality dimension weights
            
        Returns:
            Cache key string
        """
        # Include content length and first/last words for quick differentiation
        content_signature = f"{len(content)}:{content[:50]}:{content[-50:]}" if len(content) > 100 else content
        weights_str = str(sorted(weights.items()))
        combined = f"{content_signature}|{query}|{weights_str}"
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()[:16]
    
    def get_cached_score(self, content: str, query: str, weights: Dict[str, float]) -> Optional[Tuple[float, Dict[str, float]]]:
        """Get cached quality score.
        
        Args:
            content: Content to score
            query: Query context
            weights: Quality dimension weights
            
        Returns:
            Tuple of (overall_score, dimension_scores) or None if not cached
        """
        if not content:
            return None
        
        cache_key = self._compute_score_key(content, query, weights)
        cached_score = self.cache.get(cache_key)
        
        if cached_score is not None:
            self.score_stats["cache_hits"] += 1
            self.score_stats["score_computations_saved"] += 1
            return cached_score
        
        self.score_stats["cache_misses"] += 1
        return None
    
    def cache_score(self, content: str, query: str, weights: Dict[str, float], 
                   overall_score: float, dimension_scores: Dict[str, float]) -> None:
        """Cache quality score result.
        
        Args:
            content: Content that was scored
            query: Query context
            weights: Quality dimension weights
            overall_score: Overall quality score
            dimension_scores: Individual dimension scores
        """
        if not content:
            return
        
        cache_key = self._compute_score_key(content, query, weights)
        self.cache.set(cache_key, (overall_score, dimension_scores.copy()))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        cache_stats = self.cache.get_stats()
        total_requests = self.score_stats["cache_hits"] + self.score_stats["cache_misses"]
        hit_rate = (self.score_stats["cache_hits"] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "quality_cache": cache_stats,
            "score_stats": self.score_stats,
            "hit_rate_percent": round(hit_rate, 2)
        }
    
    def clear(self) -> None:
        """Clear the quality score cache and reset statistics."""
        self.cache.clear()
        self.score_stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "score_computations_saved": 0
        }


# Global cache instances
_regex_cache: Optional[RegexPatternCache] = None
_content_cache: Optional[ContentFilterCache] = None
_quality_cache: Optional[QualityScoreCache] = None


def get_regex_cache() -> RegexPatternCache:
    """Get global regex pattern cache instance."""
    global _regex_cache
    if _regex_cache is None:
        gcm = get_global_config_manager()
        cache_cfg = gcm.get_section("filter_cache") if gcm.is_initialized() else {}
        regex_cfg = cache_cfg.get("regex", {}) or {}
        max_patterns = int(regex_cfg.get("max_patterns", 1000))
        _regex_cache = RegexPatternCache(max_patterns=max_patterns)
    return _regex_cache


def get_content_cache() -> ContentFilterCache:
    """Get global content filter cache instance."""
    global _content_cache
    if _content_cache is None:
        gcm = get_global_config_manager()
        cache_cfg = gcm.get_section("filter_cache") if gcm.is_initialized() else {}
        content_cfg = cache_cfg.get("content", {}) or {}
        max_entries = int(content_cfg.get("max_entries", 5000))
        ttl = float(content_cfg.get("ttl", 1800))
        _content_cache = ContentFilterCache(max_entries=max_entries, ttl=ttl)
    return _content_cache


def get_quality_cache() -> QualityScoreCache:
    """Get global quality score cache instance."""
    global _quality_cache
    if _quality_cache is None:
        gcm = get_global_config_manager()
        cache_cfg = gcm.get_section("filter_cache") if gcm.is_initialized() else {}
        quality_cfg = cache_cfg.get("quality", {}) or {}
        max_entries = int(quality_cfg.get("max_entries", 10000))
        ttl = float(quality_cfg.get("ttl", 3600))
        _quality_cache = QualityScoreCache(max_entries=max_entries, ttl=ttl)
    return _quality_cache


def get_all_cache_stats() -> Dict[str, Any]:
    """Get statistics from all filter caches."""
    return {
        "regex_cache": get_regex_cache().get_stats(),
        "content_cache": get_content_cache().get_stats(),
        "quality_cache": get_quality_cache().get_stats()
    }


def clear_all_caches() -> None:
    """Clear all filter caches."""
    get_regex_cache().clear()
    get_content_cache().clear()
    get_quality_cache().clear()
