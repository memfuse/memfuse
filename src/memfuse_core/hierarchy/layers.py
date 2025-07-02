"""
Optimized memory layer implementations for the MemFuse hierarchy.

This module provides clean, efficient implementations of L0, L1, and L2
memory layers with unified interfaces and event-driven processing.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional
from datetime import datetime

from .core import (
    MemoryLayer, LayerType, LayerConfig, ProcessingResult,
    StorageType, StorageManager
)

logger = logging.getLogger(__name__)


class L0EpisodicLayer(MemoryLayer):
    """
    L0 (Episodic Memory) Layer - Raw data storage.
    
    Stores raw data in its original form using multiple storage backends:
    - Vector Store: For semantic similarity search
    - Keyword Store: For keyword-based search
    - SQL Store: For structured metadata
    
    Features:
    - Immediate storage of incoming data
    - Multi-backend redundancy
    - Event emission for downstream processing
    """
    
    def __init__(
        self,
        layer_type: LayerType,
        config: LayerConfig,
        user_id: str,
        storage_manager: Optional[StorageManager] = None
    ):
        super().__init__(layer_type, config, user_id, storage_manager)
        
        # L0-specific configuration
        self.storage_backends = config.storage_backends or ["vector", "keyword"]
        
        logger.info(f"L0EpisodicLayer: Initialized for user {user_id}")
    
    async def initialize(self) -> bool:
        """Initialize the L0 layer."""
        try:
            if self.storage_manager:
                await self.storage_manager.initialize()

            self.initialized = True
            logger.info(f"L0EpisodicLayer: Initialized successfully for user {self.user_id}")
            return True

        except Exception as e:
            logger.error(f"L0EpisodicLayer: Initialization failed: {e}")
            return False
    
    async def process_data(self, data: Any, metadata: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """
        Process data through L0 layer.
        
        Args:
            data: Raw data to process
            metadata: Optional metadata
            
        Returns:
            ProcessingResult with storage IDs and status
        """
        start_time = time.time()
        
        try:
            if not self.initialized:
                await self.initialize()
            
            # Store data in configured backends
            storage_results = {}
            if self.storage_manager:
                storage_results = await self.storage_manager.write_to_all(data, metadata)
            
            # Collect successful storage IDs
            processed_items = [
                item_id for item_id in storage_results.values() 
                if item_id is not None
            ]
            
            processing_time = time.time() - start_time
            success = len(processed_items) > 0
            
            # Update statistics
            self._update_stats(processing_time, success)
            
            # Create result
            result = ProcessingResult(
                success=success,
                layer_type=self.layer_type,
                processed_items=processed_items,
                metadata=metadata or {},
                processing_time=processing_time
            )
            

            
            logger.debug(f"L0EpisodicLayer: Processed data with {len(processed_items)} successful stores")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_stats(processing_time, False)
            
            logger.error(f"L0EpisodicLayer: Processing failed: {e}")
            return ProcessingResult(
                success=False,
                layer_type=self.layer_type,
                errors=[str(e)],
                processing_time=processing_time
            )
    
    async def query(self, query: str, **kwargs) -> List[Any]:
        """
        Query data from L0 layer.
        
        Args:
            query: Query string
            **kwargs: Additional query parameters
            
        Returns:
            List of matching results
        """
        try:
            if not self.initialized:
                await self.initialize()
            
            self.total_queries += 1
            
            # Query from all available backends
            all_results = []
            
            if self.storage_manager:
                # Query vector store
                if StorageType.VECTOR in self.storage_manager.get_available_backends():
                    vector_results = await self.storage_manager.read_from_backend(
                        StorageType.VECTOR, query, **kwargs
                    )
                    all_results.extend(vector_results)
                
                # Query keyword store
                if StorageType.KEYWORD in self.storage_manager.get_available_backends():
                    keyword_results = await self.storage_manager.read_from_backend(
                        StorageType.KEYWORD, query, **kwargs
                    )
                    all_results.extend(keyword_results)
            
            logger.debug(f"L0EpisodicLayer: Query returned {len(all_results)} results")
            return all_results
            
        except Exception as e:
            logger.error(f"L0EpisodicLayer: Query failed: {e}")
            return []


class L1SemanticLayer(MemoryLayer):
    """
    L1 (Semantic Memory) Layer - Facts and concepts.
    
    Extracts and stores facts from raw data using LLM processing:
    - Fact extraction from L0 data
    - Fact storage and indexing
    - Semantic search over facts
    
    Features:
    - Event-driven processing (triggered by L0 events)
    - LLM-based fact extraction
    - Fact deduplication and validation
    """
    
    def __init__(
        self,
        layer_type: LayerType,
        config: LayerConfig,
        user_id: str,
        storage_manager: Optional[StorageManager] = None
    ):
        super().__init__(layer_type, config, user_id, storage_manager)
        
        # L1-specific configuration
        self.llm_config = config.custom_config.get("llm_config", {})
        self.fact_extraction_enabled = config.custom_config.get("fact_extraction_enabled", True)
        
        logger.info(f"L1SemanticLayer: Initialized for user {user_id}")
    
    async def initialize(self) -> bool:
        """Initialize the L1 layer."""
        try:
            if self.storage_manager:
                await self.storage_manager.initialize()

            self.initialized = True
            logger.info(f"L1SemanticLayer: Initialized successfully for user {self.user_id}")
            return True

        except Exception as e:
            logger.error(f"L1SemanticLayer: Initialization failed: {e}")
            return False
    
    async def process_data(self, data: Any, metadata: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """
        Process data through L1 layer (fact extraction).
        
        Args:
            data: Data to extract facts from
            metadata: Optional metadata
            
        Returns:
            ProcessingResult with extracted facts
        """
        start_time = time.time()
        
        try:
            if not self.initialized:
                await self.initialize()
            
            # Extract facts from data (placeholder for LLM integration)
            facts = await self._extract_facts(data, metadata)
            
            # Store facts
            processed_items = []
            if facts and self.storage_manager:
                for fact in facts:
                    fact_id = await self.storage_manager.write_to_backend(
                        StorageType.VECTOR, fact, metadata
                    )
                    if fact_id:
                        processed_items.append(fact_id)
            
            processing_time = time.time() - start_time
            success = len(processed_items) > 0
            
            # Update statistics
            self._update_stats(processing_time, success)
            
            # Create result
            result = ProcessingResult(
                success=success,
                layer_type=self.layer_type,
                processed_items=processed_items,
                metadata={"facts_extracted": len(facts)},
                processing_time=processing_time
            )
            
            # Emit event for L2 processing
            if success and self.event_bus:
                await self._emit_event(
                    EventType.DATA_PROCESSED,
                    data={
                        "facts": facts,
                        "processed_items": processed_items
                    },
                    metadata=metadata
                )
            
            logger.debug(f"L1SemanticLayer: Extracted {len(facts)} facts")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_stats(processing_time, False)
            
            logger.error(f"L1SemanticLayer: Processing failed: {e}")
            return ProcessingResult(
                success=False,
                layer_type=self.layer_type,
                errors=[str(e)],
                processing_time=processing_time
            )
    
    async def query(self, query: str, **kwargs) -> List[Any]:
        """Query facts from L1 layer."""
        try:
            if not self.initialized:
                await self.initialize()
            
            self.total_queries += 1
            
            # Query facts from vector store
            results = []
            if self.storage_manager:
                results = await self.storage_manager.read_from_backend(
                    StorageType.VECTOR, query, **kwargs
                )
            
            logger.debug(f"L1SemanticLayer: Query returned {len(results)} facts")
            return results
            
        except Exception as e:
            logger.error(f"L1SemanticLayer: Query failed: {e}")
            return []
    

    
    async def process_new_data(self, data: Any, user_id: str, session_id: Optional[str] = None) -> ProcessingResult:
        """
        Process new data for fact extraction (compatibility method for MemoryService).

        Args:
            data: Data to process (chunks from L0)
            user_id: User identifier
            session_id: Session identifier

        Returns:
            ProcessingResult with extracted facts
        """
        logger.info(f"L1SemanticLayer: Processing new data for user {user_id}, session {session_id}")

        # Add metadata for context
        metadata = {
            "user_id": user_id,
            "session_id": session_id,
            "trigger_source": "memory_service_direct"
        }

        return await self.process_data(data, metadata)

    async def query_facts(self, query: str, top_k: int = 10, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Query facts from L1 layer (compatibility method for MemoryService).

        Args:
            query: Search query
            top_k: Maximum number of results
            filters: Optional filters

        Returns:
            List of fact dictionaries
        """
        logger.debug(f"L1SemanticLayer: Querying facts with query '{query[:50]}...', top_k={top_k}")

        try:
            results = await self.query(query, top_k=top_k, filters=filters)

            # Convert to expected format for MemoryService
            formatted_results = []
            for result in results:
                if isinstance(result, dict):
                    formatted_results.append(result)
                else:
                    formatted_results.append({"content": str(result)})

            logger.debug(f"L1SemanticLayer: Returning {len(formatted_results)} formatted facts")
            return formatted_results

        except Exception as e:
            logger.error(f"L1SemanticLayer: Query facts failed: {e}")
            return []

    @property
    def enabled(self) -> bool:
        """Check if L1 layer is enabled."""
        return self.fact_extraction_enabled and self.initialized

    async def _extract_facts(self, data: Any, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Extract facts from data using LLM integration."""
        try:
            facts = []

            # Handle different data types
            if isinstance(data, list):
                # Process list of chunks
                for item in data:
                    item_facts = await self._extract_facts_from_item(item, metadata)
                    facts.extend(item_facts)
            else:
                # Process single item
                item_facts = await self._extract_facts_from_item(data, metadata)
                facts.extend(item_facts)

            logger.info(f"L1SemanticLayer: Extracted {len(facts)} facts from data")
            return facts

        except Exception as e:
            logger.error(f"L1SemanticLayer: Fact extraction failed: {e}")
            return []

    async def _extract_facts_from_item(self, item: Any, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Extract facts from a single data item."""
        try:
            # Extract content based on item type
            content = ""
            if isinstance(item, dict):
                content = item.get("content", "") or item.get("text", "") or str(item)
            else:
                content = str(item)

            if not content.strip():
                return []

            # Simple fact extraction (placeholder for LLM integration)
            # TODO: Integrate with actual LLM service for sophisticated fact extraction
            facts = []

            # Extract basic facts
            if len(content) > 50:  # Only extract facts from substantial content
                fact = {
                    "content": content[:200] + "..." if len(content) > 200 else content,
                    "type": "extracted_fact",
                    "source": "l1_semantic_layer",
                    "timestamp": time.time(),
                    "metadata": metadata or {}
                }
                facts.append(fact)

            return facts

        except Exception as e:
            logger.error(f"L1SemanticLayer: Failed to extract facts from item: {e}")
            return []


class L2RelationalLayer(MemoryLayer):
    """
    L2 (Relational Memory) Layer - Knowledge graph.
    
    Constructs and maintains a knowledge graph from facts:
    - Entity extraction from facts
    - Relationship identification
    - Graph construction and updates
    - Graph-based querying
    
    Features:
    - Event-driven processing (triggered by L1 events)
    - Graph database integration
    - Entity resolution and linking
    """
    
    def __init__(
        self,
        layer_type: LayerType,
        config: LayerConfig,
        user_id: str,
        storage_manager: Optional[StorageManager] = None
    ):
        super().__init__(layer_type, config, user_id, storage_manager)
        
        # L2-specific configuration
        self.graph_config = config.custom_config.get("graph_config", {})
        self.entity_extraction_enabled = config.custom_config.get("entity_extraction_enabled", True)
        
        logger.info(f"L2RelationalLayer: Initialized for user {user_id}")
    
    async def initialize(self) -> bool:
        """Initialize the L2 layer."""
        try:
            if self.storage_manager:
                await self.storage_manager.initialize()

            self.initialized = True
            logger.info(f"L2RelationalLayer: Initialized successfully for user {self.user_id}")
            return True
            
        except Exception as e:
            logger.error(f"L2RelationalLayer: Initialization failed: {e}")
            return False
    
    async def process_data(self, data: Any, metadata: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """
        Process data through L2 layer (graph construction).
        
        Args:
            data: Facts to process into graph
            metadata: Optional metadata
            
        Returns:
            ProcessingResult with graph updates
        """
        start_time = time.time()
        
        try:
            if not self.initialized:
                await self.initialize()
            
            # Extract entities and relationships (placeholder)
            entities, relationships = await self._extract_entities_and_relationships(data, metadata)
            
            # Update graph
            processed_items = []
            if self.storage_manager and (entities or relationships):
                # Store entities and relationships in graph store
                for entity in entities:
                    entity_id = await self.storage_manager.write_to_backend(
                        StorageType.GRAPH, entity, metadata
                    )
                    if entity_id:
                        processed_items.append(entity_id)
                
                for relationship in relationships:
                    rel_id = await self.storage_manager.write_to_backend(
                        StorageType.GRAPH, relationship, metadata
                    )
                    if rel_id:
                        processed_items.append(rel_id)
            
            processing_time = time.time() - start_time
            success = len(processed_items) > 0
            
            # Update statistics
            self._update_stats(processing_time, success)
            
            # Create result
            result = ProcessingResult(
                success=success,
                layer_type=self.layer_type,
                processed_items=processed_items,
                metadata={
                    "entities_extracted": len(entities),
                    "relationships_extracted": len(relationships)
                },
                processing_time=processing_time
            )
            
            logger.debug(f"L2RelationalLayer: Processed {len(entities)} entities, {len(relationships)} relationships")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_stats(processing_time, False)
            
            logger.error(f"L2RelationalLayer: Processing failed: {e}")
            return ProcessingResult(
                success=False,
                layer_type=self.layer_type,
                errors=[str(e)],
                processing_time=processing_time
            )
    
    async def query(self, query: str, **kwargs) -> List[Any]:
        """Query the knowledge graph."""
        try:
            if not self.initialized:
                await self.initialize()
            
            self.total_queries += 1
            
            # Query graph store
            results = []
            if self.storage_manager:
                results = await self.storage_manager.read_from_backend(
                    StorageType.GRAPH, query, **kwargs
                )
            
            logger.debug(f"L2RelationalLayer: Query returned {len(results)} graph results")
            return results
            
        except Exception as e:
            logger.error(f"L2RelationalLayer: Query failed: {e}")
            return []
    

    
    async def _extract_entities_and_relationships(self, data: Any, metadata: Optional[Dict[str, Any]] = None) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Extract entities and relationships from facts (placeholder)."""
        # This would integrate with the existing L2 graph construction logic
        # For now, return simple representations
        entities = []
        relationships = []
        
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "fact" in item:
                    # Simple entity extraction
                    entities.append({"entity": f"Entity from {item['fact'][:50]}..."})
                    relationships.append({"relationship": "extracted_from", "source": "fact", "target": "entity"})
        
        return entities, relationships
