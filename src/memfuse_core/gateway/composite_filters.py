"""
Composite guardrail filters that combine multiple validation strategies.

These filters implement more sophisticated content validation by combining
length, semantic, and structural checks in configurable ways.
"""

import re
from typing import Any, Dict, List, Optional
from ..utils.global_config_manager import get_global_config_manager
from ..interfaces.gateway_interface import RequestContext


class CompositeContentFilter:
    """
    Advanced content filter that combines multiple validation strategies.
    
    Supports:
    - Length constraints (min/max with different actions)
    - Semantic validation (keyword patterns, sentiment heuristics)
    - Structural validation (format requirements, field presence)
    - Configurable action policies per violation type
    """
    
    def __init__(self):
        gcm = get_global_config_manager()
        cfg = gcm.get_section("guardrail") if gcm.is_initialized() else {}
        comp_cfg = cfg.get("composite", {}) or {}
        
        self.enabled = bool(comp_cfg.get("enabled", False))
        self.strategy = str(comp_cfg.get("strategy", "lenient")).lower()  # lenient|strict|custom
        
        # Length validation
        length_cfg = comp_cfg.get("length", {}) or {}
        self.min_length = int(length_cfg.get("min_length", 0))
        self.max_length = int(length_cfg.get("max_length", 10000))
        self.length_action = str(length_cfg.get("action", "flag")).lower()  # flag|truncate|drop
        
        # Semantic validation
        semantic_cfg = comp_cfg.get("semantic", {}) or {}
        self.required_keywords = [w for w in (semantic_cfg.get("required_keywords") or []) if isinstance(w, str)]
        self.forbidden_patterns = [p for p in (semantic_cfg.get("forbidden_patterns") or []) if isinstance(p, str)]
        self.semantic_action = str(semantic_cfg.get("action", "flag")).lower()
        
        # Structural validation
        struct_cfg = comp_cfg.get("structural", {}) or {}
        self.required_fields = [f for f in (struct_cfg.get("required_fields") or []) if isinstance(f, str)]
        self.max_metadata_depth = int(struct_cfg.get("max_metadata_depth", 5))
        self.structural_action = str(struct_cfg.get("action", "flag")).lower()
        
        # Policy configuration
        policy_cfg = comp_cfg.get("policy", {}) or {}
        self.fail_fast = bool(policy_cfg.get("fail_fast", False))  # stop on first violation
        self.aggregate_violations = bool(policy_cfg.get("aggregate_violations", True))
        
        # Use cached regex patterns for better performance
        self._pattern_cache_enabled = True
    
    def apply(self, response: Dict[str, Any], context: RequestContext) -> Dict[str, Any]:
        """Apply composite content validation."""
        if not self.enabled:
            return response
        
        data = response.get("data", {})
        results = data.get("results", [])
        results_to_remove = []
        
        for i, result in enumerate(results):
            violations = []
            
            # Length validation
            length_violation = self._validate_length(result)
            if length_violation:
                violations.append(length_violation)
                if self.fail_fast:
                    self._apply_violation_action(result, length_violation, i, results_to_remove)
                    continue
            
            # Semantic validation
            semantic_violation = self._validate_semantics(result)
            if semantic_violation:
                violations.append(semantic_violation)
                if self.fail_fast:
                    self._apply_violation_action(result, semantic_violation, i, results_to_remove)
                    continue
            
            # Structural validation
            structural_violation = self._validate_structure(result)
            if structural_violation:
                violations.append(structural_violation)
                if self.fail_fast:
                    self._apply_violation_action(result, structural_violation, i, results_to_remove)
                    continue
            
            # Apply strategy-based actions for accumulated violations
            if violations and not self.fail_fast:
                self._apply_strategy_action(result, violations, i, results_to_remove)
        
        # Remove flagged results
        for i in reversed(results_to_remove):
            results.pop(i)
        
        return response
    
    def _validate_length(self, result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Validate content length constraints."""
        content = result.get("content", "")
        if not isinstance(content, str):
            return None
        
        length = len(content)
        if length < self.min_length:
            return {
                "type": "length",
                "subtype": "too_short",
                "actual": length,
                "expected": f">= {self.min_length}",
                "action": self.length_action
            }
        elif length > self.max_length:
            return {
                "type": "length", 
                "subtype": "too_long",
                "actual": length,
                "expected": f"<= {self.max_length}",
                "action": self.length_action
            }
        return None
    
    def _validate_semantics(self, result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Validate semantic content requirements."""
        content = result.get("content", "")
        if not isinstance(content, str):
            return None
        
        # Check required keywords
        if self.required_keywords:
            content_lower = content.lower()
            missing_keywords = [kw for kw in self.required_keywords if kw.lower() not in content_lower]
            if missing_keywords:
                return {
                    "type": "semantic",
                    "subtype": "missing_keywords", 
                    "missing": missing_keywords,
                    "action": self.semantic_action
                }
        
        # Check forbidden patterns using cached regex
        if self._pattern_cache_enabled:
            from .filter_cache import get_regex_cache
            regex_cache = get_regex_cache()

            for pattern_str in self.forbidden_patterns:
                if not pattern_str:
                    continue
                try:
                    pattern = regex_cache.get_pattern(pattern_str, re.IGNORECASE)
                    if pattern.search(content):
                        return {
                            "type": "semantic",
                            "subtype": "forbidden_pattern",
                            "pattern": pattern_str,
                            "action": self.semantic_action
                        }
                except Exception:
                    # Skip invalid patterns
                    continue
        
        return None
    
    def _validate_structure(self, result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Validate structural requirements."""
        # Check required fields
        if self.required_fields:
            missing_fields = [field for field in self.required_fields if field not in result]
            if missing_fields:
                return {
                    "type": "structural",
                    "subtype": "missing_fields",
                    "missing": missing_fields,
                    "action": self.structural_action
                }
        
        # Check metadata depth
        metadata = result.get("metadata")
        if isinstance(metadata, dict):
            depth = self._calculate_dict_depth(metadata)
            if depth > self.max_metadata_depth:
                return {
                    "type": "structural",
                    "subtype": "metadata_too_deep",
                    "actual": depth,
                    "expected": f"<= {self.max_metadata_depth}",
                    "action": self.structural_action
                }
        
        return None
    
    def _calculate_dict_depth(self, d: Dict[str, Any], current_depth: int = 1) -> int:
        """Calculate maximum nesting depth of a dictionary."""
        if not isinstance(d, dict):
            return current_depth
        
        max_depth = current_depth
        for value in d.values():
            if isinstance(value, dict):
                depth = self._calculate_dict_depth(value, current_depth + 1)
                max_depth = max(max_depth, depth)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        depth = self._calculate_dict_depth(item, current_depth + 1)
                        max_depth = max(max_depth, depth)
        
        return max_depth
    
    def _apply_violation_action(self, result: Dict[str, Any], violation: Dict[str, Any], 
                               index: int, results_to_remove: List[int]) -> None:
        """Apply action for a single violation."""
        action = violation.get("action", "flag")
        
        if action == "drop":
            results_to_remove.append(index)
        elif action == "truncate" and violation.get("type") == "length":
            content = result.get("content", "")
            if isinstance(content, str) and len(content) > self.max_length:
                result["content"] = content[:self.max_length] + "..."
        
        # Always mark violation in metadata
        self._mark_violation(result, violation)
    
    def _apply_strategy_action(self, result: Dict[str, Any], violations: List[Dict[str, Any]], 
                              index: int, results_to_remove: List[int]) -> None:
        """Apply strategy-based action for multiple violations."""
        if self.strategy == "strict":
            # Drop on any violation
            results_to_remove.append(index)
        elif self.strategy == "lenient":
            # Only flag violations
            pass
        elif self.strategy == "custom":
            # Apply individual violation actions
            for violation in violations:
                self._apply_violation_action(result, violation, index, results_to_remove)
                if index in results_to_remove:
                    break  # Don't process further if already marked for removal
        
        # Mark all violations
        for violation in violations:
            self._mark_violation(result, violation)
    
    def _mark_violation(self, result: Dict[str, Any], violation: Dict[str, Any]) -> None:
        """Mark violation in result metadata."""
        md = result.setdefault("metadata", {})
        if isinstance(md, dict):
            violations = md.setdefault("composite_violations", [])
            if isinstance(violations, list):
                # Check if this violation type already exists to avoid duplicates
                violation_key = (violation.get("type"), violation.get("subtype"))
                existing = any((v.get("type"), v.get("subtype")) == violation_key for v in violations)
                if not existing:
                    violations.append({
                        "type": violation.get("type"),
                        "subtype": violation.get("subtype"),
                        "details": {k: v for k, v in violation.items() if k not in ("type", "subtype", "action")}
                    })
            md["composite_validation_failed"] = True


class ContentQualityFilter:
    """
    Content quality assessment filter with scoring and thresholds.
    
    Evaluates content based on multiple quality dimensions:
    - Completeness (length, required elements)
    - Relevance (keyword matching, topic alignment) 
    - Clarity (readability heuristics, structure)
    - Accuracy (fact-checking patterns, confidence indicators)
    """
    
    def __init__(self):
        gcm = get_global_config_manager()
        cfg = gcm.get_section("guardrail") if gcm.is_initialized() else {}
        quality_cfg = cfg.get("quality", {}) or {}
        
        self.enabled = bool(quality_cfg.get("enabled", False))
        self.min_quality_score = float(quality_cfg.get("min_score", 0.6))
        self.action = str(quality_cfg.get("action", "flag")).lower()  # flag|drop|annotate
        
        # Quality dimension weights
        weights_cfg = quality_cfg.get("weights", {}) or {}
        self.completeness_weight = float(weights_cfg.get("completeness", 0.3))
        self.relevance_weight = float(weights_cfg.get("relevance", 0.3))
        self.clarity_weight = float(weights_cfg.get("clarity", 0.2))
        self.accuracy_weight = float(weights_cfg.get("accuracy", 0.2))
        
        # Normalize weights
        total_weight = (self.completeness_weight + self.relevance_weight + 
                       self.clarity_weight + self.accuracy_weight)
        if total_weight > 0:
            self.completeness_weight /= total_weight
            self.relevance_weight /= total_weight
            self.clarity_weight /= total_weight
            self.accuracy_weight /= total_weight
    
    def apply(self, response: Dict[str, Any], context: RequestContext) -> Dict[str, Any]:
        """Apply content quality assessment."""
        if not self.enabled:
            return response
        
        data = response.get("data", {})
        results = data.get("results", [])
        results_to_remove = []
        
        for i, result in enumerate(results):
            quality_score = self._calculate_quality_score(result, context)
            
            # Mark quality score in metadata
            md = result.setdefault("metadata", {})
            if isinstance(md, dict):
                md["quality_score"] = round(quality_score, 3)
            
            # Apply action based on score
            if quality_score < self.min_quality_score:
                if self.action == "drop":
                    results_to_remove.append(i)
                elif self.action == "flag":
                    if isinstance(md, dict):
                        md["low_quality"] = True
                elif self.action == "annotate":
                    content = result.get("content", "")
                    if isinstance(content, str):
                        result["content"] = f"[Quality: {quality_score:.2f}] {content}"
        
        # Remove low-quality results
        for i in reversed(results_to_remove):
            results.pop(i)
        
        return response
    
    def _calculate_quality_score(self, result: Dict[str, Any], context: RequestContext) -> float:
        """Calculate overall quality score for a result."""
        content = result.get("content", "")
        query = getattr(context, 'query', '') or ""

        # Check cache first
        from .filter_cache import get_quality_cache
        quality_cache = get_quality_cache()
        weights = {
            "completeness": self.completeness_weight,
            "relevance": self.relevance_weight,
            "clarity": self.clarity_weight,
            "accuracy": self.accuracy_weight
        }

        cached_result = quality_cache.get_cached_score(content, query, weights)
        if cached_result is not None:
            return cached_result[0]  # Return overall score

        # Calculate scores
        completeness = self._score_completeness(result)
        relevance = self._score_relevance(result, context)
        clarity = self._score_clarity(result)
        accuracy = self._score_accuracy(result)

        overall_score = (completeness * self.completeness_weight +
                        relevance * self.relevance_weight +
                        clarity * self.clarity_weight +
                        accuracy * self.accuracy_weight)

        # Cache the result
        dimension_scores = {
            "completeness": completeness,
            "relevance": relevance,
            "clarity": clarity,
            "accuracy": accuracy
        }
        quality_cache.cache_score(content, query, weights, overall_score, dimension_scores)

        return overall_score
    
    def _score_completeness(self, result: Dict[str, Any]) -> float:
        """Score content completeness (0.0 to 1.0)."""
        content = result.get("content", "")
        if not isinstance(content, str):
            return 0.0
        
        # Basic length-based completeness
        length = len(content.strip())
        if length == 0:
            return 0.0
        elif length < 50:
            return 0.3
        elif length < 200:
            return 0.7
        else:
            return 1.0
    
    def _score_relevance(self, result: Dict[str, Any], context: RequestContext) -> float:
        """Score content relevance to query (0.0 to 1.0)."""
        content = result.get("content", "")
        query = getattr(context, 'query', '') or ""
        
        if not isinstance(content, str) or not isinstance(query, str):
            return 0.5  # neutral score
        
        # Simple keyword overlap heuristic
        content_words = set(content.lower().split())
        query_words = set(query.lower().split())
        
        if not query_words:
            return 0.5
        
        overlap = len(content_words.intersection(query_words))
        return min(1.0, overlap / len(query_words))
    
    def _score_clarity(self, result: Dict[str, Any]) -> float:
        """Score content clarity and readability (0.0 to 1.0)."""
        content = result.get("content", "")
        if not isinstance(content, str):
            return 0.0
        
        # Simple readability heuristics
        sentences = content.count('.') + content.count('!') + content.count('?')
        words = len(content.split())
        
        if words == 0:
            return 0.0
        
        # Prefer moderate sentence length
        avg_sentence_length = words / max(1, sentences)
        if 10 <= avg_sentence_length <= 25:
            clarity_score = 1.0
        elif avg_sentence_length < 5 or avg_sentence_length > 40:
            clarity_score = 0.3
        else:
            clarity_score = 0.7
        
        return clarity_score
    
    def _score_accuracy(self, result: Dict[str, Any]) -> float:
        """Score content accuracy indicators (0.0 to 1.0)."""
        content = result.get("content", "")
        if not isinstance(content, str):
            return 0.5
        
        # Look for confidence indicators
        confidence_indicators = ['according to', 'research shows', 'studies indicate', 
                               'data suggests', 'evidence', 'confirmed']
        uncertainty_indicators = ['maybe', 'possibly', 'might', 'could be', 'uncertain']
        
        content_lower = content.lower()
        confidence_count = sum(1 for indicator in confidence_indicators if indicator in content_lower)
        uncertainty_count = sum(1 for indicator in uncertainty_indicators if indicator in content_lower)
        
        # Simple scoring based on indicator balance
        if confidence_count > uncertainty_count:
            return 0.8
        elif uncertainty_count > confidence_count:
            return 0.4
        else:
            return 0.6
