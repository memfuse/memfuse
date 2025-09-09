import pytest
from typing import Any, Dict

from memfuse_core.gateway.composite_filters import CompositeContentFilter, ContentQualityFilter
from memfuse_core.interfaces.gateway_interface import RequestContext
from memfuse_core.utils.global_config_manager import get_global_config_manager


@pytest.mark.asyncio
async def test_composite_content_filter_length_validation():
    """Test composite filter length validation with different actions."""
    gcm = get_global_config_manager()
    await gcm.hot_reload({
        "guardrail": {
            "composite": {
                "enabled": True,
                "strategy": "custom",
                "length": {
                    "min_length": 10,
                    "max_length": 100,
                    "action": "truncate"
                },
                "semantic": {"action": "flag"},
                "structural": {"action": "flag"}
            }
        }
    })

    filter_instance = CompositeContentFilter()
    context = RequestContext(user_id="u", agent_id="a", session_id="s", query="test")
    
    response = {
        "status": "success",
        "data": {
            "results": [
                {"id": "1", "content": "Short"},  # too short
                {"id": "2", "content": "This is a good length content that should pass validation"},
                {"id": "3", "content": "This is way too long content that exceeds the maximum length limit and should be truncated automatically by the filter"}  # too long
            ]
        }
    }

    result = filter_instance.apply(response, context)
    results = result["data"]["results"]
    
    # Check that short content is flagged
    assert len(results) == 3
    short_result = results[0]
    violations = short_result.get("metadata", {}).get("composite_violations", [])
    assert len(violations) == 1
    assert violations[0]["type"] == "length"
    assert violations[0]["subtype"] == "too_short"
    
    # Check that long content is truncated
    long_result = results[2]
    assert len(long_result["content"]) <= 103  # 100 + "..."
    assert long_result["content"].endswith("...")


@pytest.mark.asyncio
async def test_composite_content_filter_semantic_validation():
    """Test semantic validation with required keywords and forbidden patterns."""
    gcm = get_global_config_manager()
    await gcm.hot_reload({
        "guardrail": {
            "composite": {
                "enabled": True,
                "strategy": "lenient",
                "semantic": {
                    "required_keywords": ["important", "data"],
                    "forbidden_patterns": ["forbidden", "bad.*word"],
                    "action": "flag"
                },
                "length": {"action": "flag"},
                "structural": {"action": "flag"}
            }
        }
    })

    filter_instance = CompositeContentFilter()
    context = RequestContext(user_id="u", agent_id="a", session_id="s", query="test")
    
    response = {
        "status": "success",
        "data": {
            "results": [
                {"id": "1", "content": "This contains important data"},  # good
                {"id": "2", "content": "Missing required keywords"},  # missing keywords
                {"id": "3", "content": "This has forbidden content and important data"},  # forbidden pattern but has required keywords
                {"id": "4", "content": "This has bad word in it with important data"}  # forbidden pattern (regex)
            ]
        }
    }

    result = filter_instance.apply(response, context)
    results = result["data"]["results"]
    
    # Check good content has no violations
    good_result = results[0]
    violations = good_result.get("metadata", {}).get("composite_violations", [])
    assert len(violations) == 0
    
    # Check missing keywords violation
    missing_result = results[1]
    violations = missing_result.get("metadata", {}).get("composite_violations", [])
    assert len(violations) == 1
    assert violations[0]["type"] == "semantic"
    assert violations[0]["subtype"] == "missing_keywords"
    
    # Check forbidden pattern violations
    forbidden_result = results[2]
    violations = forbidden_result.get("metadata", {}).get("composite_violations", [])
    assert len(violations) >= 1
    # Find the forbidden pattern violation
    forbidden_violation = next((v for v in violations if v["subtype"] == "forbidden_pattern"), None)
    assert forbidden_violation is not None
    assert forbidden_violation["type"] == "semantic"


@pytest.mark.asyncio
async def test_composite_content_filter_structural_validation():
    """Test structural validation for required fields and metadata depth."""
    gcm = get_global_config_manager()
    await gcm.hot_reload({
        "guardrail": {
            "composite": {
                "enabled": True,
                "strategy": "custom",
                "structural": {
                    "required_fields": ["content", "score"],
                    "max_metadata_depth": 2,
                    "action": "flag"
                },
                "length": {"action": "flag"},
                "semantic": {"action": "flag"}
            }
        }
    })

    filter_instance = CompositeContentFilter()
    context = RequestContext(user_id="u", agent_id="a", session_id="s", query="test")
    
    response = {
        "status": "success",
        "data": {
            "results": [
                {"id": "1", "content": "good", "score": 0.8},  # good
                {"id": "2", "content": "missing score"},  # missing required field
                {
                    "id": "3", 
                    "content": "deep metadata",
                    "score": 0.7,
                    "metadata": {
                        "level1": {
                            "level2": {
                                "level3": "too deep"  # exceeds max depth
                            }
                        }
                    }
                }
            ]
        }
    }

    result = filter_instance.apply(response, context)
    results = result["data"]["results"]
    
    # Check good result has no violations
    good_result = results[0]
    violations = good_result.get("metadata", {}).get("composite_violations", [])
    assert len(violations) == 0
    
    # Check missing field violation
    missing_result = results[1]
    violations = missing_result.get("metadata", {}).get("composite_violations", [])
    assert len(violations) == 1
    assert violations[0]["type"] == "structural"
    assert violations[0]["subtype"] == "missing_fields"
    
    # Check metadata depth violation
    deep_result = results[2]
    violations = deep_result.get("metadata", {}).get("composite_violations", [])
    assert len(violations) == 1
    assert violations[0]["type"] == "structural"
    assert violations[0]["subtype"] == "metadata_too_deep"


@pytest.mark.asyncio
async def test_composite_content_filter_strategy_strict():
    """Test strict strategy drops results on any violation."""
    gcm = get_global_config_manager()
    await gcm.hot_reload({
        "guardrail": {
            "composite": {
                "enabled": True,
                "strategy": "strict",
                "length": {
                    "min_length": 20,
                    "action": "flag"
                },
                "semantic": {"action": "flag"},
                "structural": {"action": "flag"}
            }
        }
    })

    filter_instance = CompositeContentFilter()
    context = RequestContext(user_id="u", agent_id="a", session_id="s", query="test")
    
    response = {
        "status": "success",
        "data": {
            "results": [
                {"id": "1", "content": "This is long enough content to pass"},
                {"id": "2", "content": "Short"},  # will be dropped
                {"id": "3", "content": "This is also long enough to pass"}
            ]
        }
    }

    result = filter_instance.apply(response, context)
    results = result["data"]["results"]
    
    # Should have only 2 results (short one dropped)
    assert len(results) == 2
    assert results[0]["id"] == "1"
    assert results[1]["id"] == "3"


@pytest.mark.asyncio
async def test_content_quality_filter_scoring():
    """Test content quality scoring and filtering."""
    gcm = get_global_config_manager()
    await gcm.hot_reload({
        "guardrail": {
            "quality": {
                "enabled": True,
                "min_score": 0.5,
                "action": "flag",
                "weights": {
                    "completeness": 0.4,
                    "relevance": 0.3,
                    "clarity": 0.2,
                    "accuracy": 0.1
                }
            }
        }
    })

    filter_instance = ContentQualityFilter()
    context = RequestContext(user_id="u", agent_id="a", session_id="s", query="machine learning algorithms")
    
    response = {
        "status": "success",
        "data": {
            "results": [
                {
                    "id": "1", 
                    "content": "Machine learning algorithms are powerful tools for data analysis. Research shows that they can improve prediction accuracy significantly. These algorithms process large datasets to identify patterns and make predictions."
                },
                {
                    "id": "2",
                    "content": "Short text"  # low completeness and relevance
                },
                {
                    "id": "3",
                    "content": "This is about completely different topic like cooking recipes and has nothing to do with the query about machine learning or algorithms."
                }
            ]
        }
    }

    result = filter_instance.apply(response, context)
    results = result["data"]["results"]
    
    # All results should have quality scores
    for res in results:
        quality_score = res.get("metadata", {}).get("quality_score")
        assert quality_score is not None
        assert 0.0 <= quality_score <= 1.0
    
    # High quality result should have good score
    high_quality = results[0]
    assert high_quality["metadata"]["quality_score"] > 0.5
    
    # Low quality results should be flagged
    low_quality = results[1]
    assert low_quality["metadata"].get("low_quality") is True


@pytest.mark.asyncio
async def test_content_quality_filter_drop_action():
    """Test content quality filter with drop action."""
    gcm = get_global_config_manager()
    await gcm.hot_reload({
        "guardrail": {
            "quality": {
                "enabled": True,
                "min_score": 0.6,
                "action": "drop"
            }
        }
    })

    filter_instance = ContentQualityFilter()
    context = RequestContext(user_id="u", agent_id="a", session_id="s", query="detailed analysis")
    
    response = {
        "status": "success",
        "data": {
            "results": [
                {
                    "id": "1",
                    "content": "This is a comprehensive detailed analysis that covers multiple aspects of the topic with clear explanations and evidence-based conclusions."
                },
                {"id": "2", "content": "Bad"},  # very low quality
                {
                    "id": "3",
                    "content": "This provides a detailed analysis with good structure and relevant information for the query."
                }
            ]
        }
    }

    result = filter_instance.apply(response, context)
    results = result["data"]["results"]
    
    # Low quality result should be removed
    assert len(results) == 2
    assert results[0]["id"] == "1"
    assert results[1]["id"] == "3"


@pytest.mark.asyncio
async def test_composite_filters_disabled():
    """Test that filters are no-op when disabled."""
    gcm = get_global_config_manager()
    await gcm.hot_reload({
        "guardrail": {
            "composite": {"enabled": False},
            "quality": {"enabled": False}
        }
    })

    composite_filter = CompositeContentFilter()
    quality_filter = ContentQualityFilter()
    context = RequestContext(user_id="u", agent_id="a", session_id="s", query="test")
    
    response = {
        "status": "success",
        "data": {
            "results": [
                {"id": "1", "content": "X"}  # would normally fail validation
            ]
        }
    }

    # Both filters should be no-op
    result1 = composite_filter.apply(response, context)
    result2 = quality_filter.apply(result1, context)
    
    assert result2 == response  # unchanged
