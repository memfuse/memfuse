"""
Unit tests for FactExtractionProcessor.

Tests the fact extraction functionality including LLM integration,
confidence filtering, and batch processing.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List

from src.memfuse_core.store.pgai_store.fact_extraction_processor import (
    FactExtractionProcessor, FactExtractionResult
)
from src.memfuse_core.rag.chunk.base import ChunkData
from src.memfuse_core.hierarchy.llm_service import ExtractedFact


class TestFactExtractionProcessor:
    """Test cases for FactExtractionProcessor."""
    
    @pytest.fixture
    def sample_config(self) -> Dict[str, Any]:
        """Sample configuration for testing."""
        return {
            'llm_model': 'grok-3-mini',
            'temperature': 0.3,
            'max_tokens': 1000,
            'max_facts_per_chunk': 10,
            'min_confidence_threshold': 0.7,
            'batch_size': 5,
            'context_window': 2,
            'max_retries': 3,
            'retry_delay': 1.0,
            'enable_validation': True
        }
    
    @pytest.fixture
    def sample_chunks(self) -> List[ChunkData]:
        """Sample chunk data for testing."""
        return [
            ChunkData(
                content="I really like pizza and pasta. They are my favorite Italian foods.",
                metadata={'session_id': 'test_session', 'user_id': 'test_user'}
            ),
            ChunkData(
                content="I decided to learn Python programming this year to advance my career.",
                metadata={'session_id': 'test_session', 'user_id': 'test_user'}
            ),
            ChunkData(
                content="My name is John Smith and I work as a software engineer at TechCorp.",
                metadata={'session_id': 'test_session', 'user_id': 'test_user'}
            ),
            ChunkData(
                content="Tomorrow I will start my new project on machine learning.",
                metadata={'session_id': 'test_session', 'user_id': 'test_user'}
            ),
            ChunkData(
                content="Short text.",  # This should be filtered out
                metadata={'session_id': 'test_session', 'user_id': 'test_user'}
            )
        ]
    
    def test_initialization(self, sample_config):
        """Test FactExtractionProcessor initialization."""
        processor = FactExtractionProcessor(sample_config)
        
        assert processor.config == sample_config
        assert processor.llm_model == 'grok-3-mini'
        assert processor.temperature == 0.3
        assert processor.max_facts_per_chunk == 10
        assert processor.min_confidence_threshold == 0.7
        assert processor.batch_size == 5
        assert processor.context_window == 2
        assert processor.llm_service is None  # Not initialized yet
    
    def test_initialization_with_defaults(self):
        """Test initialization with default configuration."""
        processor = FactExtractionProcessor()
        
        assert processor.llm_model == 'grok-3-mini'
        assert processor.temperature == 0.3
        assert processor.max_tokens == 1000
        assert processor.min_confidence_threshold == 0.7
        assert processor.batch_size == 5
    
    @pytest.mark.asyncio
    async def test_initialize(self, sample_config):
        """Test processor initialization."""
        processor = FactExtractionProcessor(sample_config)
        
        # Test initialization
        result = await processor.initialize()
        
        assert result is True
        # Note: LLM service would be initialized in real implementation
    
    def test_extract_mock_entities(self, sample_config):
        """Test mock entity extraction."""
        processor = FactExtractionProcessor(sample_config)
        
        # Test entity extraction from text with proper nouns
        text = "John Smith works at TechCorp in San Francisco"
        entities = processor._extract_mock_entities(text)
        
        assert 'John' in entities
        assert 'Smith' in entities
        assert 'TechCorp' in entities
        assert 'San' in entities
        assert 'Francisco' in entities
        assert len(entities) <= 5  # Should be limited to 5
    
    def test_extract_mock_temporal(self, sample_config):
        """Test mock temporal information extraction."""
        processor = FactExtractionProcessor(sample_config)
        
        # Test temporal extraction
        text_with_temporal = "I will do this tomorrow"
        temporal = processor._extract_mock_temporal(text_with_temporal)
        
        assert temporal is not None
        assert temporal['time_expression'] == 'tomorrow'
        assert temporal['is_relative'] is True
        assert temporal['timestamp'] is None
        
        # Test text without temporal information
        text_without_temporal = "I like programming"
        temporal = processor._extract_mock_temporal(text_without_temporal)
        
        assert temporal is None
    
    @pytest.mark.asyncio
    async def test_mock_fact_extraction_preference(self, sample_config):
        """Test mock fact extraction for preference facts."""
        processor = FactExtractionProcessor(sample_config)
        
        content = "I really like pizza and pasta. They are my favorite foods."
        facts = await processor._mock_fact_extraction(content, None, {})
        
        assert len(facts) > 0
        
        preference_facts = [f for f in facts if f.type == 'preference']
        assert len(preference_facts) > 0
        
        fact = preference_facts[0]
        assert fact.confidence >= 0.7
        assert 'pizza' in fact.content.lower()
        assert len(fact.entities) > 0
    
    @pytest.mark.asyncio
    async def test_mock_fact_extraction_decision(self, sample_config):
        """Test mock fact extraction for decision facts."""
        processor = FactExtractionProcessor(sample_config)
        
        content = "I decided to learn Python programming this year."
        facts = await processor._mock_fact_extraction(content, None, {})
        
        decision_facts = [f for f in facts if f.type == 'decision']
        assert len(decision_facts) > 0
        
        fact = decision_facts[0]
        assert fact.type == 'decision'
        assert fact.confidence >= 0.7
        assert 'python' in fact.content.lower()
        assert fact.temporal_info is None  # 'this year' not in temporal keywords
    
    @pytest.mark.asyncio
    async def test_mock_fact_extraction_personal(self, sample_config):
        """Test mock fact extraction for personal facts."""
        processor = FactExtractionProcessor(sample_config)
        
        content = "My name is John and I work as a software engineer."
        facts = await processor._mock_fact_extraction(content, None, {})
        
        personal_facts = [f for f in facts if f.type == 'personal']
        assert len(personal_facts) > 0
        
        fact = personal_facts[0]
        assert fact.type == 'personal'
        assert fact.confidence >= 0.7
        assert 'john' in fact.content.lower()
    
    @pytest.mark.asyncio
    async def test_mock_fact_extraction_temporal(self, sample_config):
        """Test mock fact extraction with temporal information."""
        processor = FactExtractionProcessor(sample_config)
        
        content = "Tomorrow I will start my new project."
        facts = await processor._mock_fact_extraction(content, None, {})
        
        decision_facts = [f for f in facts if f.type == 'decision']
        assert len(decision_facts) > 0
        
        fact = decision_facts[0]
        assert fact.temporal_info is not None
        assert fact.temporal_info['time_expression'] == 'tomorrow'
        assert fact.temporal_info['is_relative'] is True
    
    @pytest.mark.asyncio
    async def test_mock_fact_extraction_short_content(self, sample_config):
        """Test that very short content is filtered out."""
        processor = FactExtractionProcessor(sample_config)
        
        short_content = "Yes."
        facts = await processor._mock_fact_extraction(short_content, None, {})
        
        assert len(facts) == 0  # Should be filtered out
    
    def test_filter_facts_by_confidence(self, sample_config):
        """Test confidence-based fact filtering."""
        processor = FactExtractionProcessor(sample_config)
        
        # Create mock facts with different confidence levels
        facts = [
            ExtractedFact(content="High confidence fact", type="general", confidence=0.9, entities=[]),
            ExtractedFact(content="Medium confidence fact", type="general", confidence=0.75, entities=[]),
            ExtractedFact(content="Low confidence fact", type="general", confidence=0.5, entities=[]),
            ExtractedFact(content="Very low confidence fact", type="general", confidence=0.3, entities=[])
        ]
        
        # Filter with threshold 0.7
        filtered = processor._filter_facts_by_confidence(facts)
        
        assert len(filtered) == 2  # Only facts with confidence >= 0.7
        assert all(fact.confidence >= 0.7 for fact in filtered)
    
    def test_convert_to_storage_format(self, sample_config, sample_chunks):
        """Test conversion of facts to storage format."""
        processor = FactExtractionProcessor(sample_config)
        
        # Create mock extracted facts
        facts = [
            ExtractedFact(
                content="User likes pizza",
                type="preference",
                confidence=0.85,
                entities=["pizza"],
                temporal_info=None,
                source_context="conversation about food"
            )
        ]
        
        source_chunk = sample_chunks[0]
        metadata = {'session_id': 'test_session', 'user_id': 'test_user'}
        
        storage_facts = processor._convert_to_storage_format(facts, source_chunk, metadata)
        
        assert len(storage_facts) == 1
        
        storage_fact = storage_facts[0]
        assert 'id' in storage_fact
        assert storage_fact['source_id'] == source_chunk.chunk_id
        assert storage_fact['fact_content'] == "User likes pizza"
        assert storage_fact['fact_type'] == "preference"
        assert storage_fact['confidence'] == 0.85
        assert storage_fact['needs_embedding'] is True
        assert storage_fact['retry_count'] == 0
        assert storage_fact['retry_status'] == 'pending'
    
    def test_prepare_context(self, sample_config, sample_chunks):
        """Test context preparation from chunks."""
        processor = FactExtractionProcessor(sample_config)
        
        context_chunks = sample_chunks[:2]
        context = processor._prepare_context(context_chunks)
        
        assert isinstance(context, str)
        assert len(context) > 0
        assert '|' in context  # Should be joined with |
        
        # Test with empty context
        empty_context = processor._prepare_context([])
        assert empty_context == ""
    
    def test_get_context_chunks(self, sample_config, sample_chunks):
        """Test context chunk extraction."""
        processor = FactExtractionProcessor(sample_config)
        
        # Test context for middle chunk (index 2)
        context_chunks = processor._get_context_chunks(sample_chunks, 2)
        
        # Should include chunks before and after (within context window of 2)
        assert len(context_chunks) <= 4  # 2 before + 2 after
        assert sample_chunks[0] in context_chunks  # Before
        assert sample_chunks[1] in context_chunks  # Before
        assert sample_chunks[3] in context_chunks  # After
        assert sample_chunks[4] in context_chunks  # After
        assert sample_chunks[2] not in context_chunks  # Current chunk excluded
        
        # Test context for first chunk
        context_chunks = processor._get_context_chunks(sample_chunks, 0)
        assert len(context_chunks) <= 2  # Only after chunks
        assert sample_chunks[1] in context_chunks
        assert sample_chunks[2] in context_chunks
    
    @pytest.mark.asyncio
    async def test_extract_facts_from_chunk(self, sample_config, sample_chunks):
        """Test fact extraction from a single chunk."""
        processor = FactExtractionProcessor(sample_config)
        
        chunk = sample_chunks[0]  # Preference chunk
        context_chunks = sample_chunks[1:3]
        metadata = {'session_id': 'test_session'}
        
        result = await processor.extract_facts_from_chunk(chunk, context_chunks, metadata)
        
        assert isinstance(result, FactExtractionResult)
        assert result.success is True
        assert result.source_chunk_id == chunk.chunk_id
        assert len(result.extracted_facts) > 0
        assert result.processing_time > 0
        
        # Check that facts are in storage format
        fact = result.extracted_facts[0]
        assert 'id' in fact
        assert 'source_id' in fact
        assert 'fact_content' in fact
        assert 'fact_type' in fact
        assert 'confidence' in fact
    
    @pytest.mark.asyncio
    async def test_extract_facts_batch(self, sample_config, sample_chunks):
        """Test batch fact extraction."""
        processor = FactExtractionProcessor(sample_config)
        
        # Use first 3 chunks (excluding short one)
        chunks = sample_chunks[:3]
        metadata = {'session_id': 'test_session'}
        
        results = await processor.extract_facts_batch(chunks, metadata)
        
        assert len(results) == len(chunks)
        
        successful_results = [r for r in results if r.success]
        assert len(successful_results) > 0
        
        # Check that each result has the correct source chunk ID
        for i, result in enumerate(results):
            assert result.source_chunk_id == chunks[i].chunk_id
    
    def test_get_stats(self, sample_config):
        """Test statistics collection."""
        processor = FactExtractionProcessor(sample_config)
        
        # Update some stats manually
        processor.stats['total_chunks_processed'] = 10
        processor.stats['total_facts_extracted'] = 25
        processor.stats['total_facts_filtered'] = 20
        processor.stats['processing_time_total'] = 50.0
        processor.stats['errors'] = 1
        
        stats = processor.get_stats()
        
        assert stats['total_chunks_processed'] == 10
        assert stats['total_facts_extracted'] == 25
        assert stats['total_facts_filtered'] == 20
        assert stats['average_facts_per_chunk'] == 2.5
        assert stats['average_processing_time'] == 5.0
        assert stats['success_rate'] == 0.9  # (10-1)/10
        assert stats['filter_rate'] == 0.8  # 20/25
    
    def test_reset_stats(self, sample_config):
        """Test statistics reset."""
        processor = FactExtractionProcessor(sample_config)
        
        # Set some stats
        processor.stats['total_chunks_processed'] = 10
        processor.stats['errors'] = 5
        
        # Reset stats
        processor.reset_stats()
        
        assert processor.stats['total_chunks_processed'] == 0
        assert processor.stats['errors'] == 0
        assert processor.stats['total_facts_extracted'] == 0


if __name__ == '__main__':
    pytest.main([__file__])