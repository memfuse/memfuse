"""Pydantic models for M2 fact extraction responses."""

from typing import List, Optional
from pydantic import BaseModel, Field


class ExtractedFact(BaseModel):
    """Single fact extracted from a memory chunk."""
    
    content: str = Field(
        ..., 
        description="Clear, self-contained factual statement",
        min_length=10,
        max_length=200
    )
    source_chunk_ids: List[str] = Field(
        ..., 
        description="List of chunk IDs that support this fact"
    )
    confidence: Optional[float] = Field(
        0.8, 
        ge=0.0, 
        le=1.0, 
        description="Confidence score between 0.0 and 1.0"
    )


class FactExtractionResponse(BaseModel):
    """Response model for LLM-based fact extraction."""
    
    facts: List[ExtractedFact] = Field(
        ..., 
        description="List of extracted facts from the memory chunk",
        max_length=20
    )