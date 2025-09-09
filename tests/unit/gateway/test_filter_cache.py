import pytest
import re
import time
from typing import Dict, Any

from memfuse_core.gateway.filter_cache import (
    RegexPatternCache, 
    ContentFilterCache, 
    QualityScoreCache,
    get_regex_cache,
    get_content_cache,
    get_quality_cache,
    get_all_cache_stats,
    clear_all_caches
)


def test_regex_pattern_cache_basic_functionality():
    """Test basic regex pattern caching functionality."""
    cache = RegexPatternCache(max_patterns=10)
    
    # Test pattern compilation and caching
    pattern1 = cache.get_pattern(r"test\d+", re.IGNORECASE)
    pattern2 = cache.get_pattern(r"test\d+", re.IGNORECASE)  # Should be cached
    
    # Should be the same object (cached)
    assert pattern1 is pattern2
    
    # Test different flags create different cache entries
    pattern3 = cache.get_pattern(r"test\d+", 0)  # No flags
    assert pattern3 is not pattern1
    
    # Test word pattern helper
    word_pattern = cache.get_word_pattern("hello", case_insensitive=True)
    assert word_pattern.search("say hello world") is not None
    assert word_pattern.search("hellothere") is None  # Word boundary
    
    # Test statistics
    stats = cache.get_stats()
    assert stats["compilation_stats"]["cache_hits"] >= 1
    assert stats["compilation_stats"]["cache_misses"] >= 2


def test_regex_pattern_cache_invalid_patterns():
    """Test handling of invalid regex patterns."""
    cache = RegexPatternCache(max_patterns=10)
    
    # Invalid regex should return a pattern that never matches
    invalid_pattern = cache.get_pattern(r"[invalid", 0)
    assert invalid_pattern.search("any text") is None
    
    # Should still cache the "never match" pattern
    invalid_pattern2 = cache.get_pattern(r"[invalid", 0)
    assert invalid_pattern is invalid_pattern2


def test_content_filter_cache_basic_functionality():
    """Test content-based filter result caching."""
    cache = ContentFilterCache(max_entries=100, ttl=60)
    
    content = "This is test content for caching"
    filter_config = {"enabled": True, "min_length": 10}
    result = {"violations": [], "passed": True}
    
    # Should not be cached initially
    cached_result = cache.get_cached_result(content, filter_config)
    assert cached_result is None
    
    # Cache the result
    cache.cache_result(content, filter_config, result)
    
    # Should now be cached
    cached_result = cache.get_cached_result(content, filter_config)
    assert cached_result is not None
    assert cached_result["passed"] is True
    
    # Different config should not match
    different_config = {"enabled": True, "min_length": 20}
    cached_result2 = cache.get_cached_result(content, different_config)
    assert cached_result2 is None
    
    # Test statistics
    stats = cache.get_stats()
    assert stats["hash_stats"]["cache_hits"] >= 1
    assert stats["hash_stats"]["cache_misses"] >= 1


def test_content_filter_cache_mutation_safety():
    """Test that cached results are safe from mutation."""
    cache = ContentFilterCache(max_entries=100, ttl=60)
    
    content = "Test content"
    filter_config = {"test": True}
    original_result = {"violations": ["test"], "score": 0.5}
    
    cache.cache_result(content, filter_config, original_result)
    
    # Get cached result and modify it
    cached_result = cache.get_cached_result(content, filter_config)
    cached_result["violations"].append("modified")
    cached_result["score"] = 0.9
    
    # Get again - should not be affected by previous modification
    cached_result2 = cache.get_cached_result(content, filter_config)
    assert len(cached_result2["violations"]) == 1
    assert cached_result2["score"] == 0.5


def test_quality_score_cache_basic_functionality():
    """Test quality score caching functionality."""
    cache = QualityScoreCache(max_entries=1000, ttl=120)
    
    content = "This is a comprehensive test content with good quality"
    query = "test quality"
    weights = {"completeness": 0.3, "relevance": 0.3, "clarity": 0.2, "accuracy": 0.2}
    overall_score = 0.85
    dimension_scores = {"completeness": 0.9, "relevance": 0.8, "clarity": 0.8, "accuracy": 0.9}
    
    # Should not be cached initially
    cached_score = cache.get_cached_score(content, query, weights)
    assert cached_score is None
    
    # Cache the score
    cache.cache_score(content, query, weights, overall_score, dimension_scores)
    
    # Should now be cached
    cached_score = cache.get_cached_score(content, query, weights)
    assert cached_score is not None
    assert cached_score[0] == overall_score  # Overall score
    assert cached_score[1]["completeness"] == 0.9  # Dimension scores
    
    # Different weights should not match
    different_weights = {"completeness": 0.4, "relevance": 0.3, "clarity": 0.2, "accuracy": 0.1}
    cached_score2 = cache.get_cached_score(content, query, different_weights)
    assert cached_score2 is None
    
    # Test statistics
    stats = cache.get_stats()
    assert stats["score_stats"]["cache_hits"] >= 1
    assert stats["score_stats"]["score_computations_saved"] >= 1


def test_quality_score_cache_content_differentiation():
    """Test that quality cache properly differentiates content."""
    cache = QualityScoreCache(max_entries=1000, ttl=120)
    
    content1 = "Short content"
    content2 = "This is a much longer content that should have different quality characteristics"
    query = "test"
    weights = {"completeness": 0.25, "relevance": 0.25, "clarity": 0.25, "accuracy": 0.25}
    
    # Cache different scores for different content
    cache.cache_score(content1, query, weights, 0.3, {"completeness": 0.2})
    cache.cache_score(content2, query, weights, 0.8, {"completeness": 0.9})
    
    # Should get different cached results
    cached1 = cache.get_cached_score(content1, query, weights)
    cached2 = cache.get_cached_score(content2, query, weights)
    
    assert cached1[0] == 0.3
    assert cached2[0] == 0.8


def test_cache_ttl_expiration():
    """Test that cache entries expire after TTL."""
    cache = ContentFilterCache(max_entries=100, ttl=0.1)  # 100ms TTL
    
    content = "Test content"
    filter_config = {"test": True}
    result = {"passed": True}
    
    # Cache the result
    cache.cache_result(content, filter_config, result)
    
    # Should be cached immediately
    cached_result = cache.get_cached_result(content, filter_config)
    assert cached_result is not None
    
    # Wait for expiration
    time.sleep(0.15)
    
    # Should be expired now
    cached_result = cache.get_cached_result(content, filter_config)
    assert cached_result is None


def test_global_cache_instances():
    """Test global cache instance management."""
    # Clear all caches first
    clear_all_caches()
    
    # Get global instances
    regex_cache = get_regex_cache()
    content_cache = get_content_cache()
    quality_cache = get_quality_cache()
    
    # Should be singleton instances
    assert get_regex_cache() is regex_cache
    assert get_content_cache() is content_cache
    assert get_quality_cache() is quality_cache
    
    # Test some basic functionality
    pattern = regex_cache.get_pattern(r"test", 0)
    assert pattern is not None
    
    content_cache.cache_result("test", {"config": True}, {"result": True})
    cached = content_cache.get_cached_result("test", {"config": True})
    assert cached is not None
    
    quality_cache.cache_score("test", "query", {"w": 1.0}, 0.5, {"dim": 0.5})
    cached_score = quality_cache.get_cached_score("test", "query", {"w": 1.0})
    assert cached_score is not None


def test_all_cache_stats():
    """Test getting statistics from all caches."""
    clear_all_caches()
    
    # Use all caches
    regex_cache = get_regex_cache()
    content_cache = get_content_cache()
    quality_cache = get_quality_cache()

    regex_cache.get_pattern("test", 0)
    content_cache.cache_result("test", {"config": "value"}, {"result": "value"})
    content_cache.get_cached_result("test", {"config": "different"})  # Force a miss
    quality_cache.cache_score("test", "q", {"weight": 1.0}, 0.5, {"dim": 0.5})
    quality_cache.get_cached_score("test", "different_q", {"weight": 1.0})  # Force a miss
    
    # Get all stats
    all_stats = get_all_cache_stats()
    
    assert "regex_cache" in all_stats
    assert "content_cache" in all_stats
    assert "quality_cache" in all_stats
    
    # Each should have some activity
    assert all_stats["regex_cache"]["compilation_stats"]["cache_misses"] > 0
    assert all_stats["content_cache"]["hash_stats"]["cache_misses"] > 0
    assert all_stats["quality_cache"]["score_stats"]["cache_misses"] > 0


def test_cache_clear_functionality():
    """Test cache clearing functionality."""
    regex_cache = get_regex_cache()
    content_cache = get_content_cache()
    quality_cache = get_quality_cache()
    
    # Add some data
    regex_cache.get_pattern("test", 0)
    content_cache.cache_result("test", {}, {})
    quality_cache.cache_score("test", "q", {}, 0.5, {})
    
    # Clear all caches
    clear_all_caches()
    
    # Stats should be reset
    all_stats = get_all_cache_stats()
    assert all_stats["regex_cache"]["compilation_stats"]["cache_hits"] == 0
    assert all_stats["content_cache"]["hash_stats"]["cache_hits"] == 0
    assert all_stats["quality_cache"]["score_stats"]["cache_hits"] == 0


def test_cache_performance_under_load():
    """Test cache performance with many operations."""
    regex_cache = get_regex_cache()
    
    # Test many pattern compilations
    patterns = [f"pattern{i}" for i in range(100)]
    
    start_time = time.perf_counter()
    
    # First pass - should be cache misses
    for pattern in patterns:
        regex_cache.get_pattern(pattern, 0)
    
    first_pass_time = time.perf_counter() - start_time
    
    # Second pass - should be cache hits
    start_time = time.perf_counter()
    for pattern in patterns:
        regex_cache.get_pattern(pattern, 0)
    
    second_pass_time = time.perf_counter() - start_time
    
    # Second pass should be significantly faster
    assert second_pass_time < first_pass_time * 0.5  # At least 50% faster
    
    # Check hit rate
    stats = regex_cache.get_stats()
    hit_rate = stats["hit_rate_percent"]
    assert hit_rate >= 50  # Should have good hit rate
