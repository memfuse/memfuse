"""
Advanced semantic validation for MemFuse Gateway.

This module provides semantic-level content validation using embedding-based
similarity analysis, semantic conflict detection, and contextual relevance scoring.
"""

import asyncio
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from loguru import logger

from ..utils.global_config_manager import get_global_config_manager
from ..rag.encode.MiniLM import MiniLMEncoder
from ..interfaces.gateway_interface import RequestContext


@dataclass
class SemanticViolation:
    """Represents a semantic validation violation."""
    type: str  # similarity, conflict, relevance, coherence
    severity: float  # 0.0 to 1.0
    description: str
    confidence: float  # 0.0 to 1.0
    metadata: Dict[str, Any]


@dataclass
class SemanticValidationResult:
    """Result of semantic validation."""
    passed: bool
    violations: List[SemanticViolation]
    similarity_scores: Dict[str, float]
    relevance_score: float
    coherence_score: float
    metadata: Dict[str, Any]


class SemanticValidator:
    """
    Advanced semantic validator using embedding-based analysis.
    
    Features:
    - Semantic similarity detection for duplicate content
    - Contextual relevance scoring
    - Content coherence analysis
    - Cross-lingual semantic understanding
    - Configurable thresholds and actions
    """
    
    def __init__(self):
        self.encoder: Optional[MiniLMEncoder] = None
        self.similarity_threshold = 0.85
        self.relevance_threshold = 0.3
        self.coherence_threshold = 0.4
        self.enabled = False
        self._reference_embeddings: Dict[str, np.ndarray] = {}
        self._load_config()
    
    def _load_config(self):
        """Load configuration from global config manager."""
        try:
            gcm = get_global_config_manager()
            if gcm.is_initialized():
                semantic_cfg = gcm.get_section("semantic_validation") or {}
                self.enabled = bool(semantic_cfg.get("enabled", False))
                self.similarity_threshold = float(semantic_cfg.get("similarity_threshold", 0.85))
                self.relevance_threshold = float(semantic_cfg.get("relevance_threshold", 0.3))
                self.coherence_threshold = float(semantic_cfg.get("coherence_threshold", 0.4))
                
                # Load reference patterns for comparison
                patterns = semantic_cfg.get("reference_patterns", [])
                if patterns and self.enabled:
                    asyncio.create_task(self._load_reference_patterns(patterns))
        except Exception as e:
            logger.warning(f"Failed to load semantic validation config: {e}")
            self.enabled = False
    
    async def initialize(self):
        """Initialize the semantic validator."""
        if not self.enabled:
            return
        
        try:
            # Initialize encoder
            self.encoder = MiniLMEncoder(model_name="all-MiniLM-L6-v2")
            logger.info("SemanticValidator: Initialized with MiniLM encoder")
        except Exception as e:
            logger.error(f"Failed to initialize semantic validator: {e}")
            self.enabled = False
    
    async def _load_reference_patterns(self, patterns: List[str]):
        """Load and encode reference patterns for similarity comparison."""
        if not self.encoder:
            await self.initialize()
        
        if not self.encoder:
            return
        
        try:
            for i, pattern in enumerate(patterns):
                embedding = await self._encode_text(pattern)
                self._reference_embeddings[f"pattern_{i}"] = embedding
            
            logger.info(f"Loaded {len(patterns)} reference patterns for semantic validation")
        except Exception as e:
            logger.error(f"Failed to load reference patterns: {e}")
    
    async def _encode_text(self, text: str) -> np.ndarray:
        """Encode text to embedding vector."""
        if not self.encoder:
            raise ValueError("Encoder not initialized")
        
        # Use the encoder's encode method
        embeddings = self.encoder.encode([text])
        return np.array(embeddings[0])
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    async def validate_content(
        self,
        content: str,
        context: Optional[RequestContext] = None,
        reference_content: Optional[List[str]] = None
    ) -> SemanticValidationResult:
        """
        Validate content using semantic analysis.
        
        Args:
            content: Content to validate
            context: Request context for contextual validation
            reference_content: Reference content for similarity comparison
            
        Returns:
            Semantic validation result
        """
        if not self.enabled or not content.strip():
            return SemanticValidationResult(
                passed=True,
                violations=[],
                similarity_scores={},
                relevance_score=1.0,
                coherence_score=1.0,
                metadata={"enabled": self.enabled}
            )
        
        try:
            # Encode the content
            content_embedding = await self._encode_text(content)
            
            violations = []
            similarity_scores = {}
            
            # 1. Similarity analysis with reference patterns
            for pattern_id, pattern_embedding in self._reference_embeddings.items():
                similarity = self._cosine_similarity(content_embedding, pattern_embedding)
                similarity_scores[pattern_id] = similarity
                
                if similarity > self.similarity_threshold:
                    violations.append(SemanticViolation(
                        type="similarity",
                        severity=similarity,
                        description=f"High similarity to reference pattern {pattern_id}",
                        confidence=0.9,
                        metadata={"pattern_id": pattern_id, "similarity": similarity}
                    ))
            
            # 2. Similarity analysis with reference content
            if reference_content:
                for i, ref_text in enumerate(reference_content):
                    ref_embedding = await self._encode_text(ref_text)
                    similarity = self._cosine_similarity(content_embedding, ref_embedding)
                    similarity_scores[f"reference_{i}"] = similarity
                    
                    if similarity > self.similarity_threshold:
                        violations.append(SemanticViolation(
                            type="similarity",
                            severity=similarity,
                            description=f"High similarity to reference content {i}",
                            confidence=0.8,
                            metadata={"reference_index": i, "similarity": similarity}
                        ))
            
            # 3. Contextual relevance analysis
            relevance_score = await self._calculate_relevance_score(
                content, content_embedding, context
            )
            
            if relevance_score < self.relevance_threshold:
                violations.append(SemanticViolation(
                    type="relevance",
                    severity=1.0 - relevance_score,
                    description=f"Low contextual relevance score: {relevance_score:.3f}",
                    confidence=0.7,
                    metadata={"relevance_score": relevance_score}
                ))
            
            # 4. Content coherence analysis
            coherence_score = await self._calculate_coherence_score(content, content_embedding)
            
            if coherence_score < self.coherence_threshold:
                violations.append(SemanticViolation(
                    type="coherence",
                    severity=1.0 - coherence_score,
                    description=f"Low content coherence score: {coherence_score:.3f}",
                    confidence=0.6,
                    metadata={"coherence_score": coherence_score}
                ))
            
            # 5. Semantic conflict detection
            conflict_violations = await self._detect_semantic_conflicts(content, content_embedding)
            violations.extend(conflict_violations)
            
            return SemanticValidationResult(
                passed=len(violations) == 0,
                violations=violations,
                similarity_scores=similarity_scores,
                relevance_score=relevance_score,
                coherence_score=coherence_score,
                metadata={
                    "content_length": len(content),
                    "embedding_dimension": len(content_embedding),
                    "validation_time": "calculated"
                }
            )
            
        except Exception as e:
            logger.error(f"Semantic validation failed: {e}")
            return SemanticValidationResult(
                passed=True,  # Fail open for safety
                violations=[],
                similarity_scores={},
                relevance_score=1.0,
                coherence_score=1.0,
                metadata={"error": str(e)}
            )
    
    async def _calculate_relevance_score(
        self,
        content: str,
        content_embedding: np.ndarray,
        context: Optional[RequestContext]
    ) -> float:
        """Calculate contextual relevance score."""
        if not context or not context.query:
            return 1.0  # No context to compare against
        
        try:
            # Encode the query/context
            query_embedding = await self._encode_text(context.query)
            
            # Calculate similarity between content and query
            relevance = self._cosine_similarity(content_embedding, query_embedding)
            
            # Adjust based on context metadata
            if context.request_metadata:
                # Boost relevance if content matches expected topics
                expected_topics = context.request_metadata.get("expected_topics", [])
                if expected_topics:
                    topic_boost = await self._calculate_topic_relevance(content, expected_topics)
                    relevance = min(1.0, relevance + topic_boost * 0.2)
            
            return max(0.0, relevance)
            
        except Exception as e:
            logger.warning(f"Failed to calculate relevance score: {e}")
            return 1.0
    
    async def _calculate_topic_relevance(self, content: str, expected_topics: List[str]) -> float:
        """Calculate relevance to expected topics."""
        if not expected_topics:
            return 0.0
        
        try:
            content_embedding = await self._encode_text(content)
            topic_similarities = []
            
            for topic in expected_topics:
                topic_embedding = await self._encode_text(topic)
                similarity = self._cosine_similarity(content_embedding, topic_embedding)
                topic_similarities.append(similarity)
            
            # Return the maximum similarity to any expected topic
            return max(topic_similarities) if topic_similarities else 0.0
            
        except Exception as e:
            logger.warning(f"Failed to calculate topic relevance: {e}")
            return 0.0
    
    async def _calculate_coherence_score(self, content: str, content_embedding: np.ndarray) -> float:
        """Calculate content coherence score."""
        try:
            # Split content into sentences
            sentences = [s.strip() for s in content.split('.') if s.strip()]
            
            if len(sentences) < 2:
                return 1.0  # Single sentence is coherent by definition
            
            # Calculate pairwise similarities between sentences
            similarities = []
            for i in range(len(sentences) - 1):
                try:
                    sent1_embedding = await self._encode_text(sentences[i])
                    sent2_embedding = await self._encode_text(sentences[i + 1])
                    similarity = self._cosine_similarity(sent1_embedding, sent2_embedding)
                    similarities.append(similarity)
                except Exception:
                    continue
            
            if not similarities:
                return 1.0
            
            # Average similarity indicates coherence
            coherence = np.mean(similarities)
            return max(0.0, min(1.0, coherence))
            
        except Exception as e:
            logger.warning(f"Failed to calculate coherence score: {e}")
            return 1.0
    
    async def _detect_semantic_conflicts(
        self,
        content: str,
        content_embedding: np.ndarray
    ) -> List[SemanticViolation]:
        """Detect semantic conflicts within content."""
        violations = []
        
        try:
            # Split content into logical segments
            segments = [s.strip() for s in content.split('\n') if s.strip()]
            
            if len(segments) < 2:
                return violations
            
            # Look for contradictory statements
            for i in range(len(segments)):
                for j in range(i + 1, len(segments)):
                    try:
                        seg1_embedding = await self._encode_text(segments[i])
                        seg2_embedding = await self._encode_text(segments[j])
                        
                        # Check for semantic opposition (very low similarity might indicate conflict)
                        similarity = self._cosine_similarity(seg1_embedding, seg2_embedding)
                        
                        # Detect potential conflicts (this is a simplified heuristic)
                        if similarity < 0.1 and self._contains_opposing_keywords(segments[i], segments[j]):
                            violations.append(SemanticViolation(
                                type="conflict",
                                severity=1.0 - similarity,
                                description=f"Potential semantic conflict between segments {i} and {j}",
                                confidence=0.5,
                                metadata={
                                    "segment1": segments[i][:100],
                                    "segment2": segments[j][:100],
                                    "similarity": similarity
                                }
                            ))
                    except Exception:
                        continue
            
        except Exception as e:
            logger.warning(f"Failed to detect semantic conflicts: {e}")
        
        return violations
    
    def _contains_opposing_keywords(self, text1: str, text2: str) -> bool:
        """Check if two text segments contain opposing keywords."""
        # Simple keyword-based conflict detection
        opposing_pairs = [
            ("yes", "no"), ("true", "false"), ("correct", "incorrect"),
            ("right", "wrong"), ("good", "bad"), ("positive", "negative"),
            ("increase", "decrease"), ("more", "less"), ("higher", "lower")
        ]
        
        text1_lower = text1.lower()
        text2_lower = text2.lower()
        
        for word1, word2 in opposing_pairs:
            if word1 in text1_lower and word2 in text2_lower:
                return True
            if word2 in text1_lower and word1 in text2_lower:
                return True
        
        return False


# Global semantic validator instance
_semantic_validator: Optional[SemanticValidator] = None


def get_semantic_validator() -> SemanticValidator:
    """Get global semantic validator instance."""
    global _semantic_validator
    if _semantic_validator is None:
        _semantic_validator = SemanticValidator()
    return _semantic_validator


async def initialize_semantic_validation():
    """Initialize semantic validation system."""
    validator = get_semantic_validator()
    await validator.initialize()
