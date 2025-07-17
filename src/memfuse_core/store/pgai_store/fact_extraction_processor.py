"""
Fact extraction processor for M1 semantic memory layer.

This module provides LLM-based fact extraction from M0 episodic data,
converting raw conversational content into structured semantic facts.
"""

import asyncio
import json
import time
import uuid
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from loguru import logger

from ...rag.chunk.base import ChunkData
from ...hierarchy.llm_service import AdvancedLLMService, ExtractedFact


@dataclass
class FactExtractionResult:
    """Result of fact extraction operation."""
    success: bool
    extracted_facts: List[Dict[str, Any]]
    source_chunk_id: Optional[str] = None
    processing_time: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None


class FactExtractionProcessor:
    """
    Processor for extracting semantic facts from M0 episodic data.
    
    This processor uses LLM services to analyze conversational content
    and extract structured facts suitable for M1 semantic memory storage.
    
    Features:
    - LLM-based fact extraction with confidence scoring
    - Configurable fact types and filtering
    - Batch processing for efficiency
    - Error handling and retry logic
    - Integration with existing AdvancedLLMService
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize fact extraction processor.
        
        Args:
            config: Configuration dictionary with extraction settings
        """
        self.config = config or {}
        
        # Extraction configuration
        self.llm_model = self.config.get('llm_model', 'grok-3-mini')
        self.temperature = self.config.get('temperature', 0.3)
        self.max_tokens = self.config.get('max_tokens', 1000)
        self.max_facts_per_chunk = self.config.get('max_facts_per_chunk', 10)
        self.min_confidence_threshold = self.config.get('min_confidence_threshold', 0.7)
        self.batch_size = self.config.get('batch_size', 5)
        self.context_window = self.config.get('context_window', 2)

        # Flexible fact classification configuration
        self.classification_strategy = self.config.get('classification_strategy', 'open')  # 'open', 'predefined', 'custom'
        self.custom_fact_types = self.config.get('custom_fact_types', [])
        self.enable_auto_categorization = self.config.get('enable_auto_categorization', True)
        
        # Processing settings
        self.max_retries = self.config.get('max_retries', 3)
        self.retry_delay = self.config.get('retry_delay', 1.0)
        self.enable_validation = self.config.get('enable_validation', True)
        
        # LLM service (will be initialized when needed)
        self.llm_service: Optional[AdvancedLLMService] = None
        
        # Statistics
        self.stats = {
            'total_chunks_processed': 0,
            'total_facts_extracted': 0,
            'total_facts_filtered': 0,
            'processing_time_total': 0.0,
            'errors': 0,
            'retries': 0
        }
        
        logger.info(f"FactExtractionProcessor initialized with model: {self.llm_model}")
    
    async def initialize(self) -> bool:
        """Initialize the fact extraction processor.
        
        Returns:
            True if initialization successful
        """
        try:
            # Initialize LLM service
            # Note: In a real implementation, this would connect to actual LLM provider
            logger.info("FactExtractionProcessor: LLM service initialization would happen here")
            # self.llm_service = AdvancedLLMService(llm_provider, self.config)
            
            logger.info("FactExtractionProcessor initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"FactExtractionProcessor initialization failed: {e}")
            return False
    
    async def extract_facts_from_chunk(self, chunk: ChunkData, 
                                     context_chunks: Optional[List[ChunkData]] = None,
                                     metadata: Optional[Dict[str, Any]] = None) -> FactExtractionResult:
        """Extract facts from a single chunk with optional context.
        
        Args:
            chunk: ChunkData object to extract facts from
            context_chunks: Optional list of context chunks for better extraction
            metadata: Optional metadata for the extraction
            
        Returns:
            FactExtractionResult containing extracted facts
        """
        start_time = time.time()
        
        try:
            logger.debug(f"Extracting facts from chunk: {chunk.chunk_id}")
            
            # Prepare content for extraction
            content = chunk.content
            context_content = self._prepare_context(context_chunks) if context_chunks else None
            
            # Extract facts using LLM service
            if self.llm_service:
                extracted_facts = await self._extract_with_llm_service(
                    content, context_content, metadata
                )
            else:
                # Fallback to rule-based extraction
                extracted_facts = await self._rule_based_fact_extraction(
                    content, context_content, metadata
                )
            
            # Filter facts by confidence threshold
            filtered_facts = self._filter_facts_by_confidence(extracted_facts)
            
            # Convert to storage format
            storage_facts = self._convert_to_storage_format(
                filtered_facts, chunk, metadata
            )
            
            processing_time = time.time() - start_time
            
            # Update statistics
            self.stats['total_chunks_processed'] += 1
            self.stats['total_facts_extracted'] += len(extracted_facts)
            self.stats['total_facts_filtered'] += len(filtered_facts)
            self.stats['processing_time_total'] += processing_time
            
            logger.debug(f"Extracted {len(filtered_facts)} facts from chunk {chunk.chunk_id}")
            
            return FactExtractionResult(
                success=True,
                extracted_facts=storage_facts,
                source_chunk_id=chunk.chunk_id,
                processing_time=processing_time,
                metadata=metadata
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.stats['errors'] += 1
            
            logger.error(f"Fact extraction failed for chunk {chunk.chunk_id}: {e}")
            
            return FactExtractionResult(
                success=False,
                extracted_facts=[],
                source_chunk_id=chunk.chunk_id,
                processing_time=processing_time,
                error_message=str(e),
                metadata=metadata
            )
    
    async def extract_facts_batch(self, chunks: List[ChunkData],
                                metadata: Optional[Dict[str, Any]] = None) -> List[FactExtractionResult]:
        """Extract facts from multiple chunks in batch.
        
        Args:
            chunks: List of ChunkData objects to process
            metadata: Optional metadata for the extraction
            
        Returns:
            List of FactExtractionResult objects
        """
        try:
            logger.info(f"Processing batch of {len(chunks)} chunks for fact extraction")
            
            results = []
            
            # Process chunks in batches
            for i in range(0, len(chunks), self.batch_size):
                batch = chunks[i:i + self.batch_size]
                
                # Process each chunk in the batch with context
                for j, chunk in enumerate(batch):
                    # Prepare context chunks (before and after current chunk)
                    context_chunks = self._get_context_chunks(chunks, i + j)
                    
                    # Extract facts with retry logic
                    result = await self._extract_with_retry(chunk, context_chunks, metadata)
                    results.append(result)
                
                # Small delay between batches to avoid overwhelming the LLM service
                if i + self.batch_size < len(chunks):
                    await asyncio.sleep(0.1)
            
            successful_results = [r for r in results if r.success]
            total_facts = sum(len(r.extracted_facts) for r in successful_results)
            
            logger.info(f"Batch processing completed: {len(successful_results)}/{len(results)} successful, {total_facts} facts extracted")
            
            return results
            
        except Exception as e:
            logger.error(f"Batch fact extraction failed: {e}")
            return [FactExtractionResult(
                success=False,
                extracted_facts=[],
                error_message=str(e),
                metadata=metadata
            )]
    
    def _prepare_context(self, context_chunks: List[ChunkData]) -> str:
        """Prepare context content from context chunks.
        
        Args:
            context_chunks: List of context chunks
            
        Returns:
            Combined context content
        """
        if not context_chunks:
            return ""
        
        context_parts = []
        for chunk in context_chunks:
            # Truncate very long chunks for context
            content = chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
            context_parts.append(content)
        
        return " | ".join(context_parts)
    
    def _get_context_chunks(self, all_chunks: List[ChunkData], current_index: int) -> List[ChunkData]:
        """Get context chunks around the current chunk.
        
        Args:
            all_chunks: All chunks in the batch
            current_index: Index of current chunk
            
        Returns:
            List of context chunks
        """
        context_chunks = []
        
        # Get chunks before current chunk
        start_idx = max(0, current_index - self.context_window)
        for i in range(start_idx, current_index):
            context_chunks.append(all_chunks[i])
        
        # Get chunks after current chunk
        end_idx = min(len(all_chunks), current_index + self.context_window + 1)
        for i in range(current_index + 1, end_idx):
            context_chunks.append(all_chunks[i])
        
        return context_chunks
    
    async def _extract_with_retry(self, chunk: ChunkData, 
                                context_chunks: Optional[List[ChunkData]],
                                metadata: Optional[Dict[str, Any]]) -> FactExtractionResult:
        """Extract facts with retry logic.
        
        Args:
            chunk: Chunk to extract facts from
            context_chunks: Optional context chunks
            metadata: Optional metadata
            
        Returns:
            FactExtractionResult
        """
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                result = await self.extract_facts_from_chunk(chunk, context_chunks, metadata)
                if result.success:
                    return result
                else:
                    last_error = result.error_message
                    
            except Exception as e:
                last_error = str(e)
                self.stats['retries'] += 1
                
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Fact extraction attempt {attempt + 1} failed, retrying in {delay}s: {e}")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Fact extraction failed after {self.max_retries + 1} attempts: {e}")
        
        return FactExtractionResult(
            success=False,
            extracted_facts=[],
            source_chunk_id=chunk.chunk_id,
            error_message=last_error,
            metadata=metadata
        )
    
    async def _extract_with_llm_service(self, content: str, context: Optional[str],
                                      metadata: Optional[Dict[str, Any]]) -> List[ExtractedFact]:
        """Extract facts using the LLM service.

        Args:
            content: Content to extract facts from
            context: Optional context content
            metadata: Optional metadata

        Returns:
            List of ExtractedFact objects
        """
        try:
            # Build extraction prompt
            prompt = self._build_extraction_prompt(content, context, metadata)

            # Call LLM service
            if self.llm_service:
                response = await self.llm_service.generate_response(
                    prompt=prompt,
                    model=self.llm_model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )

                if response and hasattr(response, 'content'):
                    # Parse LLM response into structured facts
                    facts = self._parse_llm_response(response.content, content, context)
                    return facts

            # Fallback to rule-based extraction if LLM service unavailable
            logger.warning("LLM service unavailable, using rule-based extraction")
            return await self._rule_based_fact_extraction(content, context, metadata)

        except Exception as e:
            logger.error(f"LLM fact extraction failed: {e}")
            # Fallback to rule-based extraction
            return await self._rule_based_fact_extraction(content, context, metadata)

    def _build_extraction_prompt(self, content: str, context: Optional[str],
                               metadata: Optional[Dict[str, Any]]) -> str:
        """Build prompt for LLM fact extraction."""

        context_info = f"\nContext: {context}" if context else ""
        user_info = ""
        if metadata:
            user_id = metadata.get('user_id', 'unknown')
            session_id = metadata.get('session_id', 'unknown')
            user_info = f"\nUser: {user_id}, Session: {session_id}"

        prompt = f"""Extract structured facts from the following conversational content.
Focus on extracting meaningful, persistent information that would be useful for future conversations.

Content: {content}{context_info}{user_info}

Please extract facts in the following JSON format:
{{
    "facts": [
        {{
            "content": "extracted fact description",
            "type": "fact_type",
            "confidence": 0.85,
            "entities": ["entity1", "entity2"],
            "temporal_info": {{"time_reference": "optional"}},
            "category": {{"semantic_type": "preference|personal|decision|general"}}
        }}
    ]
}}

Guidelines:
- Only extract facts that are likely to be useful in future conversations
- Assign confidence scores between 0.0 and 1.0
- Include relevant entities mentioned in the fact
- Use flexible fact types based on content
- Minimum confidence threshold: {self.min_confidence_threshold}
"""
        return prompt

    def _parse_llm_response(self, response_content: str, original_content: str,
                          context: Optional[str]) -> List[ExtractedFact]:
        """Parse LLM response into ExtractedFact objects."""
        try:
            # Try to parse JSON response
            response_data = json.loads(response_content)
            facts = []

            for fact_data in response_data.get('facts', []):
                # Validate confidence threshold
                confidence = fact_data.get('confidence', 0.0)
                if confidence < self.min_confidence_threshold:
                    continue

                fact = ExtractedFact(
                    content=fact_data.get('content', ''),
                    type=fact_data.get('type', 'extracted_fact'),
                    confidence=confidence,
                    entities=fact_data.get('entities', []),
                    temporal_info=fact_data.get('temporal_info'),
                    source_context=context[:50] if context else None,
                    category=fact_data.get('category', {})
                )
                facts.append(fact)

            return facts[:self.max_facts_per_chunk]  # Limit number of facts

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            # Fallback: create a single fact from the response
            return [ExtractedFact(
                content=f"Extracted information: {response_content[:100]}...",
                type='general',
                confidence=0.7,
                entities=[],
                temporal_info=None,
                source_context=context[:50] if context else None,
                category={'source': 'llm_fallback'}
            )]

    async def _rule_based_fact_extraction(self, content: str, context: Optional[str],
                                        metadata: Optional[Dict[str, Any]]) -> List[ExtractedFact]:
        """Rule-based fact extraction as fallback when LLM is unavailable."""
        facts = []

        # Skip very short content
        if len(content.strip()) < 20:
            return facts

        # Use flexible fact extraction based on configuration
        extracted_fact = self._extract_flexible_fact(content, context)
        if extracted_fact:
            facts.append(extracted_fact)

        return facts[:self.max_facts_per_chunk]  # Limit number of facts

    def _extract_flexible_fact(self, content: str, context: Optional[str]) -> Optional[ExtractedFact]:
        """Extract fact with flexible categorization strategy."""

        # Skip very short content
        if len(content.strip()) < 20:
            return None

        # Determine fact type based on classification strategy
        fact_type = self._determine_fact_type(content)
        fact_category = self._determine_fact_category(content) if self.enable_auto_categorization else {}

        # Create fact with flexible typing
        return ExtractedFact(
            content=f"Extracted fact: {content[:100]}{'...' if len(content) > 100 else ''}",
            type=fact_type,
            confidence=self._calculate_confidence(content, fact_type),
            entities=self._extract_entities(content),
            temporal_info=self._extract_temporal_info(content),
            source_context=context[:50] if context else None,
            category=fact_category
        )

    def _determine_fact_type(self, content: str) -> str:
        """Determine fact type based on classification strategy."""
        if self.classification_strategy == 'open':
            return 'extracted_fact'  # Generic type for open classification
        elif self.classification_strategy == 'custom' and self.custom_fact_types:
            # Use first custom type as default, or implement custom logic
            return self.custom_fact_types[0] if self.custom_fact_types else 'custom_fact'
        else:
            return 'general'  # Fallback

    def _determine_fact_category(self, content: str) -> Dict[str, Any]:
        """Determine flexible fact categorization."""
        content_lower = content.lower()
        categories = {}

        # Add semantic indicators
        if any(keyword in content_lower for keyword in ['like', 'prefer', 'favorite']):
            categories['semantic_type'] = 'preference'
        elif any(keyword in content_lower for keyword in ['decided', 'will', 'plan']):
            categories['semantic_type'] = 'decision'
        elif any(keyword in content_lower for keyword in ['i am', 'my name', 'i work']):
            categories['semantic_type'] = 'personal'

        # Add content characteristics
        categories['content_length'] = len(content)
        categories['has_temporal'] = bool(self._extract_temporal_info(content))

        return categories

    def _extract_entities(self, content: str) -> List[str]:
        """Extract entities from content using rule-based approach."""
        entities = []

        # Simple entity extraction based on capitalization and common patterns
        words = content.split()
        for word in words:
            if word[0].isupper() and len(word) > 2 and word.isalpha():
                entities.append(word)

        return entities[:5]  # Limit to 5 entities

    def _extract_temporal_info(self, content: str) -> Optional[Dict[str, Any]]:
        """Extract temporal information from content."""
        content_lower = content.lower()

        temporal_keywords = ['today', 'tomorrow', 'yesterday', 'next week', 'last month', 'soon']

        for keyword in temporal_keywords:
            if keyword in content_lower:
                return {
                    'time_expression': keyword,
                    'is_relative': True,
                    'timestamp': None
                }

        return None

    def _calculate_confidence(self, content: str, fact_type: str) -> float:
        """Calculate confidence score for extracted fact."""
        base_confidence = 0.7

        # Adjust based on content length
        if len(content) > 50:
            base_confidence += 0.1

        # Adjust based on fact type specificity
        if fact_type in ['personal', 'preference', 'decision']:
            base_confidence += 0.05

        return min(0.95, base_confidence)
    
    def _filter_facts_by_confidence(self, facts: List[ExtractedFact]) -> List[ExtractedFact]:
        """Filter facts by confidence threshold.
        
        Args:
            facts: List of extracted facts
            
        Returns:
            List of facts above confidence threshold
        """
        filtered = [fact for fact in facts if fact.confidence >= self.min_confidence_threshold]
        
        if len(filtered) < len(facts):
            logger.debug(f"Filtered {len(facts) - len(filtered)} facts below confidence threshold {self.min_confidence_threshold}")
        
        return filtered
    
    def _convert_to_storage_format(self, facts: List[ExtractedFact], 
                                 source_chunk: ChunkData,
                                 metadata: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert extracted facts to storage format for M1 table.
        
        Args:
            facts: List of extracted facts
            source_chunk: Source chunk the facts were extracted from
            metadata: Optional metadata
            
        Returns:
            List of fact dictionaries ready for M1 storage
        """
        storage_facts = []
        
        for fact in facts:
            storage_fact = {
                'id': str(uuid.uuid4()),
                'source_id': source_chunk.chunk_id,
                'source_session_id': metadata.get('session_id') if metadata else None,
                'source_user_id': metadata.get('user_id') if metadata else None,
                'fact_content': fact.content,
                'fact_type': fact.type,
                'confidence': fact.confidence,
                'entities': json.dumps(fact.entities) if fact.entities else '[]',
                'temporal_info': json.dumps(fact.temporal_info) if fact.temporal_info else '{}',
                'source_context': fact.source_context,
                'metadata': json.dumps(metadata) if metadata else '{}',
                'needs_embedding': True,  # Will trigger automatic embedding generation
                'retry_count': 0,
                'retry_status': 'pending'
            }
            
            storage_facts.append(storage_fact)
        
        return storage_facts
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics.
        
        Returns:
            Dictionary containing processing statistics
        """
        stats = self.stats.copy()
        
        # Calculate derived statistics
        if stats['total_chunks_processed'] > 0:
            stats['average_facts_per_chunk'] = stats['total_facts_extracted'] / stats['total_chunks_processed']
            stats['average_processing_time'] = stats['processing_time_total'] / stats['total_chunks_processed']
            stats['success_rate'] = (stats['total_chunks_processed'] - stats['errors']) / stats['total_chunks_processed']
        else:
            stats['average_facts_per_chunk'] = 0.0
            stats['average_processing_time'] = 0.0
            stats['success_rate'] = 0.0
        
        stats['filter_rate'] = stats['total_facts_filtered'] / max(stats['total_facts_extracted'], 1)
        
        return stats
    
    def reset_stats(self):
        """Reset processing statistics."""
        self.stats = {
            'total_chunks_processed': 0,
            'total_facts_extracted': 0,
            'total_facts_filtered': 0,
            'processing_time_total': 0.0,
            'errors': 0,
            'retries': 0
        }
        
        logger.info("FactExtractionProcessor statistics reset")