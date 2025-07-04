"""
Conflict Detection Engine for MemFuse L1 Layer.

This module provides sophisticated conflict detection and resolution
capabilities for facts in the semantic memory layer.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from loguru import logger

from ..llm.base import LLMProvider, LLMRequest
from ..llm.prompts.manager import get_prompt_manager


class ConflictType(Enum):
    """Types of conflicts that can be detected."""
    DIRECT_CONTRADICTION = "DIRECT_CONTRADICTION"
    SEMANTIC_CONFLICT = "SEMANTIC_CONFLICT"
    TEMPORAL_INCONSISTENCY = "TEMPORAL_INCONSISTENCY"
    VALUE_MISMATCH = "VALUE_MISMATCH"
    PREFERENCE_CHANGE = "PREFERENCE_CHANGE"


class ConflictSeverity(Enum):
    """Severity levels for conflicts."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass
class Conflict:
    """Represents a detected conflict between facts."""
    type: ConflictType
    severity: ConflictSeverity
    confidence: float
    existing_fact_id: str
    description: str
    resolution_suggestion: str
    evidence: Dict[str, str]


@dataclass
class ConflictDetectionResult:
    """Result of conflict detection analysis."""
    conflicts_detected: bool
    conflicts: List[Conflict]
    overall_assessment: str


class ConflictDetectionEngine:
    """
    Advanced conflict detection engine for L1 semantic memory.
    
    Detects and analyzes conflicts between new facts and existing facts
    using LLM-based semantic analysis and rule-based detection.
    """
    
    def __init__(self, llm_service: Any, config: Optional[Dict[str, Any]] = None):
        """Initialize the conflict detection engine.
        
        Args:
            llm_service: Advanced LLM service instance
            config: Configuration dictionary
        """
        self.llm_service = llm_service
        self.config = config or {}
        self.prompt_manager = get_prompt_manager()
        
        # Configuration
        self.semantic_similarity_threshold = self.config.get("semantic_similarity_threshold", 0.85)
        self.auto_resolve_threshold = self.config.get("auto_resolve_threshold", 0.9)
        self.escalation_threshold = self.config.get("escalation_threshold", 0.5)
        self.resolution_strategies = self.config.get("resolution_strategies", [
            "evidence_based", "source_credibility", "temporal_priority"
        ])
        
        logger.info("ConflictDetectionEngine: Initialized with threshold %.2f", 
                   self.semantic_similarity_threshold)
    
    async def detect_conflicts(
        self,
        new_fact: Dict[str, Any],
        existing_facts: List[Dict[str, Any]],
        user_id: Optional[str] = None
    ) -> ConflictDetectionResult:
        """
        Detect conflicts between a new fact and existing facts.
        
        Args:
            new_fact: New fact to check for conflicts
            existing_facts: List of existing facts to compare against
            user_id: User identifier
            
        Returns:
            Conflict detection result
        """
        try:
            logger.debug("ConflictDetectionEngine: Checking conflicts for new fact")
            
            if not existing_facts:
                return ConflictDetectionResult(
                    conflicts_detected=False,
                    conflicts=[],
                    overall_assessment="No existing facts to compare against"
                )
            
            # Prepare prompt for LLM-based conflict detection
            prompt = self.prompt_manager.get_prompt(
                "conflict_detection",
                new_fact=str(new_fact),
                existing_facts=str(existing_facts),
                user_id=user_id or "unknown",
                threshold=self.semantic_similarity_threshold
            )
            
            # Create LLM request
            request = LLMRequest(
                prompt=prompt,
                model=self.config.get("llm_model", "grok-3-mini"),
                temperature=0.1,  # Low temperature for consistent analysis
                max_tokens=1000
            )
            
            # Call LLM through the advanced LLM service
            response = await self.llm_service.llm_provider.generate(request)
            
            if not response.success:
                logger.error("ConflictDetectionEngine: LLM call failed: %s", response.error)
                return ConflictDetectionResult(
                    conflicts_detected=False,
                    conflicts=[],
                    overall_assessment=f"Analysis failed: {response.error}"
                )
            
            # Parse conflict detection response
            result = self._parse_conflict_response(response.content)
            
            # Apply rule-based filtering and enhancement
            enhanced_result = await self._enhance_with_rules(result, new_fact, existing_facts)
            
            logger.info("ConflictDetectionEngine: Detected %d conflicts", 
                       len(enhanced_result.conflicts))
            
            return enhanced_result
            
        except Exception as e:
            logger.error("ConflictDetectionEngine: Conflict detection failed: %s", e)
            return ConflictDetectionResult(
                conflicts_detected=False,
                conflicts=[],
                overall_assessment=f"Detection failed: {str(e)}"
            )
    
    async def resolve_conflicts(
        self,
        conflicts: List[Conflict],
        resolution_strategy: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Resolve detected conflicts using specified strategy.
        
        Args:
            conflicts: List of conflicts to resolve
            resolution_strategy: Strategy to use for resolution
            
        Returns:
            List of resolution actions
        """
        try:
            strategy = resolution_strategy or self.resolution_strategies[0]
            logger.debug("ConflictDetectionEngine: Resolving %d conflicts using %s strategy", 
                        len(conflicts), strategy)
            
            resolutions = []
            
            for conflict in conflicts:
                if conflict.confidence >= self.auto_resolve_threshold:
                    # Auto-resolve high-confidence conflicts
                    resolution = self._auto_resolve_conflict(conflict, strategy)
                elif conflict.confidence >= self.escalation_threshold:
                    # Escalate medium-confidence conflicts
                    resolution = self._escalate_conflict(conflict)
                else:
                    # Ignore low-confidence conflicts
                    resolution = self._ignore_conflict(conflict)
                
                resolutions.append(resolution)
            
            logger.info("ConflictDetectionEngine: Generated %d resolutions", len(resolutions))
            return resolutions
            
        except Exception as e:
            logger.error("ConflictDetectionEngine: Conflict resolution failed: %s", e)
            return []
    
    def _parse_conflict_response(self, response_content: str) -> ConflictDetectionResult:
        """Parse LLM response for conflict detection."""
        try:
            import json
            
            if response_content.strip().startswith('{'):
                data = json.loads(response_content)
                
                conflicts = []
                for conflict_data in data.get("conflicts", []):
                    conflict = Conflict(
                        type=ConflictType(conflict_data.get("type", "SEMANTIC_CONFLICT")),
                        severity=ConflictSeverity(conflict_data.get("severity", "MEDIUM")),
                        confidence=float(conflict_data.get("confidence", 0.5)),
                        existing_fact_id=conflict_data.get("existing_fact_id", "unknown"),
                        description=conflict_data.get("description", ""),
                        resolution_suggestion=conflict_data.get("resolution_suggestion", ""),
                        evidence=conflict_data.get("evidence", {})
                    )
                    conflicts.append(conflict)
                
                return ConflictDetectionResult(
                    conflicts_detected=data.get("conflicts_detected", False),
                    conflicts=conflicts,
                    overall_assessment=data.get("overall_assessment", "")
                )
            else:
                # Simple text-based parsing
                conflicts_detected = "conflict" in response_content.lower()
                return ConflictDetectionResult(
                    conflicts_detected=conflicts_detected,
                    conflicts=[],
                    overall_assessment=response_content[:200]
                )
                
        except Exception as e:
            logger.warning("ConflictDetectionEngine: Failed to parse conflict response: %s", e)
            return ConflictDetectionResult(
                conflicts_detected=False,
                conflicts=[],
                overall_assessment=f"Parse error: {str(e)}"
            )
    
    async def _enhance_with_rules(
        self,
        result: ConflictDetectionResult,
        new_fact: Dict[str, Any],
        existing_facts: List[Dict[str, Any]]
    ) -> ConflictDetectionResult:
        """Enhance conflict detection with rule-based analysis."""
        # Apply additional rule-based checks
        # This is a placeholder for more sophisticated rule-based conflict detection
        
        # For now, just return the original result
        return result
    
    def _auto_resolve_conflict(self, conflict: Conflict, strategy: str) -> Dict[str, Any]:
        """Auto-resolve a high-confidence conflict."""
        return {
            "action": "auto_resolve",
            "conflict_id": conflict.existing_fact_id,
            "strategy": strategy,
            "resolution": conflict.resolution_suggestion,
            "confidence": conflict.confidence
        }
    
    def _escalate_conflict(self, conflict: Conflict) -> Dict[str, Any]:
        """Escalate a medium-confidence conflict for manual review."""
        return {
            "action": "escalate",
            "conflict_id": conflict.existing_fact_id,
            "reason": "Requires manual review",
            "description": conflict.description,
            "confidence": conflict.confidence
        }
    
    def _ignore_conflict(self, conflict: Conflict) -> Dict[str, Any]:
        """Ignore a low-confidence conflict."""
        return {
            "action": "ignore",
            "conflict_id": conflict.existing_fact_id,
            "reason": "Low confidence conflict",
            "confidence": conflict.confidence
        }
