"""
Advanced LLM Service for MemFuse M1 Layer.

This module provides sophisticated LLM-based services for fact extraction,
validation, and processing in the M1 semantic memory layer.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from loguru import logger

from ..llm.base import LLMProvider, LLMRequest
from ..llm.prompts.manager import get_prompt_manager


@dataclass
class ExtractedFact:
    """Represents an extracted fact from content."""
    content: str
    type: str  # Flexible fact type, no longer constrained to specific values
    confidence: float
    entities: List[str]
    temporal_info: Optional[Dict[str, Any]] = None
    source_context: Optional[str] = None
    category: Optional[Dict[str, Any]] = None  # Flexible categorization system


@dataclass
class FormedEpisode:
    """Represents a formed episode from raw content."""
    episode_content: str
    episode_type: str  # Flexible episode type, no constraints for extensibility
    confidence: float
    entities: Optional[List[str]] = None
    temporal_info: Optional[Dict[str, Any]] = None
    source_context: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ValidationResult:
    """Result of fact validation."""
    is_valid: bool
    validation_score: float
    issues: List[Dict[str, Any]]
    corrected_fact: Optional[Dict[str, Any]] = None
    recommendation: str = "ACCEPT"  # ACCEPT, REJECT, REVISE


class AdvancedLLMService:
    """
    Advanced LLM service for M1 semantic memory operations.
    
    Provides sophisticated fact extraction, validation, and conflict resolution
    using LLM capabilities with structured prompts and response parsing.
    """
    
    def __init__(self, llm_provider: LLMProvider, config: Optional[Dict[str, Any]] = None):
        """Initialize the advanced LLM service.
        
        Args:
            llm_provider: LLM provider instance
            config: Configuration dictionary
        """
        self.llm_provider = llm_provider
        self.config = config or {}
        self.prompt_manager = get_prompt_manager()
        
        # Configuration
        self.model = self.config.get("llm_model", "grok-3-mini")
        self.max_facts_per_chunk = self.config.get("max_facts_per_chunk", 10)
        self.min_confidence_threshold = self.config.get("min_confidence_threshold", 0.7)
        self.temperature = self.config.get("temperature", 0.3)
        self.max_tokens = self.config.get("max_tokens", 1000)
        
        logger.info("AdvancedLLMService: Initialized with model %s", self.model)
    
    async def extract_facts_with_context(
        self,
        content: str,
        context: Optional[List[str]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> List[ExtractedFact]:
        """
        Extract facts from content with contextual awareness.
        
        Args:
            content: Content to extract facts from
            context: Optional context information
            user_id: User identifier
            session_id: Session identifier
            
        Returns:
            List of extracted facts
        """
        try:
            logger.debug("AdvancedLLMService: Extracting facts from content (length: %d)", len(content))
            
            # Prepare prompt
            prompt = self.prompt_manager.get_prompt(
                "fact_extraction",
                content=content,
                user_id=user_id or "unknown",
                session_id=session_id or "unknown",
                timestamp=time.time()
            )
            
            # Create LLM request
            request = LLMRequest(
                prompt=prompt,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Call LLM
            response = await self.llm_provider.generate(request)
            
            if not response.success:
                logger.error("AdvancedLLMService: LLM call failed: %s", response.error)
                return []
            
            # Parse response
            facts = self._parse_fact_extraction_response(response.content)
            
            # Filter by confidence threshold
            filtered_facts = [
                fact for fact in facts 
                if fact.confidence >= self.min_confidence_threshold
            ]
            
            logger.info("AdvancedLLMService: Extracted %d facts (%d after filtering)", 
                       len(facts), len(filtered_facts))
            
            return filtered_facts
            
        except Exception as e:
            logger.error("AdvancedLLMService: Fact extraction failed: %s", e)
            return []
    
    async def validate_fact_consistency(
        self,
        fact: Dict[str, Any],
        original_context: str,
        user_id: Optional[str] = None
    ) -> ValidationResult:
        """
        Validate a fact for consistency and quality.
        
        Args:
            fact: Fact to validate
            original_context: Original context the fact was extracted from
            user_id: User identifier
            
        Returns:
            Validation result
        """
        try:
            logger.debug("AdvancedLLMService: Validating fact: %s", fact.get("content", "")[:50])
            
            # Prepare prompt
            prompt = self.prompt_manager.get_prompt(
                "fact_validation",
                fact=str(fact),
                original_context=original_context,
                user_id=user_id or "unknown",
                min_confidence=self.min_confidence_threshold,
                quality_standards="high"
            )
            
            # Create LLM request
            request = LLMRequest(
                prompt=prompt,
                model=self.model,
                temperature=0.1,  # Lower temperature for validation
                max_tokens=500
            )
            
            # Call LLM
            response = await self.llm_provider.generate(request)
            
            if not response.success:
                logger.error("AdvancedLLMService: Validation LLM call failed: %s", response.error)
                return ValidationResult(
                    is_valid=False,
                    validation_score=0.0,
                    issues=[{"type": "LLM_ERROR", "description": response.error}]
                )
            
            # Parse validation response
            validation_result = self._parse_validation_response(response.content)
            
            logger.debug("AdvancedLLMService: Validation result: %s (score: %.2f)", 
                        validation_result.recommendation, validation_result.validation_score)
            
            return validation_result
            
        except Exception as e:
            logger.error("AdvancedLLMService: Fact validation failed: %s", e)
            return ValidationResult(
                is_valid=False,
                validation_score=0.0,
                issues=[{"type": "VALIDATION_ERROR", "description": str(e)}]
            )
    
    def _parse_fact_extraction_response(self, response_content: str) -> List[ExtractedFact]:
        """Parse LLM response for fact extraction."""
        try:
            import json
            
            # Try to parse as JSON
            if response_content.strip().startswith('{'):
                data = json.loads(response_content)
                facts_data = data.get("facts", [])
            else:
                # Fallback: simple text parsing
                return self._parse_text_facts(response_content)
            
            facts = []
            for fact_data in facts_data:
                fact = ExtractedFact(
                    content=fact_data.get("content", ""),
                    type=fact_data.get("type", "general"),
                    confidence=float(fact_data.get("confidence", 0.5)),
                    entities=fact_data.get("entities", []),
                    temporal_info=fact_data.get("temporal_info"),
                    source_context=fact_data.get("source_context")
                )
                facts.append(fact)
            
            return facts
            
        except Exception as e:
            logger.warning("AdvancedLLMService: Failed to parse fact extraction response: %s", e)
            return self._parse_text_facts(response_content)
    
    def _parse_text_facts(self, text: str) -> List[ExtractedFact]:
        """Fallback text parsing for facts."""
        facts = []
        lines = text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line and len(line) > 10:  # Basic filtering
                fact = ExtractedFact(
                    content=line,
                    type="general",
                    confidence=0.6,  # Default confidence
                    entities=[]
                )
                facts.append(fact)
        
        return facts[:self.max_facts_per_chunk]  # Limit number of facts
    
    def _parse_validation_response(self, response_content: str) -> ValidationResult:
        """Parse LLM response for fact validation."""
        try:
            import json
            
            if response_content.strip().startswith('{'):
                data = json.loads(response_content)
                
                return ValidationResult(
                    is_valid=data.get("is_valid", False),
                    validation_score=float(data.get("validation_score", 0.0)),
                    issues=data.get("issues", []),
                    corrected_fact=data.get("corrected_fact"),
                    recommendation=data.get("recommendation", "REJECT")
                )
            else:
                # Simple text-based validation
                is_valid = "valid" in response_content.lower() or "accept" in response_content.lower()
                return ValidationResult(
                    is_valid=is_valid,
                    validation_score=0.7 if is_valid else 0.3,
                    issues=[],
                    recommendation="ACCEPT" if is_valid else "REJECT"
                )
                
        except Exception as e:
            logger.warning("AdvancedLLMService: Failed to parse validation response: %s", e)
            return ValidationResult(
                is_valid=False,
                validation_score=0.0,
                issues=[{"type": "PARSE_ERROR", "description": str(e)}],
                recommendation="REJECT"
            )
