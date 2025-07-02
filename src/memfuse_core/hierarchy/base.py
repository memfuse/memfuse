"""
Legacy data structures for backward compatibility.

This module provides the data structures that were previously in the old base module,
ensuring compatibility with existing L1 components.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime
from enum import Enum


# Enums
class FactType(str, Enum):
    """Types of facts that can be extracted."""
    GENERAL = "general"
    PERSONAL = "personal"
    TEMPORAL = "temporal"
    CAUSAL = "causal"
    DESCRIPTIVE = "descriptive"
    QUANTITATIVE = "quantitative"


class ValidationStatus(str, Enum):
    """Validation status for facts."""
    PENDING = "pending"
    VALIDATED = "validated"
    REJECTED = "rejected"
    CONFLICTED = "conflicted"


class ConflictType(str, Enum):
    """Types of conflicts between facts."""
    SEMANTIC = "semantic"
    TEMPORAL = "temporal"
    LOGICAL = "logical"
    SOURCE = "source"


class TimeGranularity(str, Enum):
    """Time granularity levels."""
    SECOND = "second"
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"


# Data structures
@dataclass
class TemporalInfo:
    """Temporal information for facts."""
    timestamp: Optional[datetime] = None
    time_expression: str = ""
    uncertainty: float = 0.0
    granularity: TimeGranularity = TimeGranularity.DAY
    is_relative: bool = False


@dataclass
class EntityMention:
    """Entity mention in text."""
    text: str
    entity_type: str
    start_pos: int
    end_pos: int
    confidence: float = 1.0


@dataclass
class Fact:
    """Fact data structure."""
    id: str
    content: str
    confidence: float
    source_ids: List[str] = field(default_factory=list)
    temporal_info: Optional[TemporalInfo] = None
    entity_mentions: List[EntityMention] = field(default_factory=list)
    fact_type: FactType = FactType.GENERAL
    validation_status: ValidationStatus = ValidationStatus.PENDING
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class Entity:
    """Entity data structure."""
    id: str
    name: str
    entity_type: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    source_fact_ids: List[str] = field(default_factory=list)
    session_ids: List[str] = field(default_factory=list)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class Relationship:
    """Relationship data structure."""
    id: str
    subject_id: str
    predicate: str
    object_id: str
    weight: float = 1.0
    confidence: float = 1.0
    temporal_info: Optional[TemporalInfo] = None
    source_fact_ids: List[str] = field(default_factory=list)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class Conflict:
    """Conflict between facts."""
    id: str
    fact_ids: List[str]
    conflict_type: ConflictType
    description: str
    confidence: float
    resolution_strategy: Optional[str] = None
    resolved: bool = False
    created_at: Optional[datetime] = None


@dataclass
class ValidationResult:
    """Result of fact validation."""
    fact_id: str
    status: ValidationStatus
    confidence: float
    reasons: List[str] = field(default_factory=list)
    conflicts: List[Conflict] = field(default_factory=list)
    validated_at: Optional[datetime] = None


@dataclass
class SourceLineage:
    """Source lineage for facts."""
    fact_id: str
    source_type: str
    source_id: str
    extraction_method: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None


@dataclass
class QualityReport:
    """Quality report for facts."""
    fact_id: str
    quality_score: float
    quality_factors: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    generated_at: Optional[datetime] = None


@dataclass
class ConsolidationResult:
    """Result of fact consolidation."""
    consolidated_facts: List[Fact]
    removed_duplicates: List[str]
    resolved_conflicts: List[Conflict]
    quality_improvements: Dict[str, float] = field(default_factory=dict)
    consolidation_time: Optional[datetime] = None


# Legacy base class for compatibility
class MemoryItem:
    """Legacy base class for memory items."""
    
    def __init__(self, id: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        self.id = id
        self.content = content
        self.metadata = metadata or {}
        self.created_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
