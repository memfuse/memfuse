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
from ..rag.chunk.base import ChunkData

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
            
            # Convert data to ChunkData objects for storage
            chunks = self._convert_to_chunks(data, metadata)

            # Store chunks in configured backends
            storage_results = {}
            if self.storage_manager and chunks:
                storage_results = await self.storage_manager.write_to_all(chunks, metadata)
            
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

    def _convert_to_chunks(self, data: Any, metadata: Optional[Dict[str, Any]] = None) -> List[ChunkData]:
        """Convert raw data to ChunkData objects for storage."""
        chunks = []

        try:
            if isinstance(data, list):
                # Process list of items
                for i, item in enumerate(data):
                    chunk = self._create_chunk_from_item(item, metadata, f"l0_item_{i}")
                    if chunk:
                        chunks.append(chunk)
            else:
                # Process single item
                chunk = self._create_chunk_from_item(data, metadata, "l0_single")
                if chunk:
                    chunks.append(chunk)

        except Exception as e:
            logger.error(f"L0EpisodicLayer: Failed to convert data to chunks: {e}")

        return chunks

    def _create_chunk_from_item(self, item: Any, metadata: Optional[Dict[str, Any]], prefix: str) -> Optional[ChunkData]:
        """Create a ChunkData object from a single data item."""
        try:
            # Extract content based on item type
            content = ""
            if isinstance(item, dict):
                content = item.get("content", "") or item.get("text", "") or str(item)
            else:
                content = str(item)

            if not content.strip():
                return None

            # Create chunk metadata
            chunk_metadata = {
                "layer": "L0",
                "source": "episodic_layer",
                "timestamp": time.time(),
                **(metadata or {})
            }

            # Create ChunkData object
            return ChunkData(
                content=content,
                metadata=chunk_metadata
            )

        except Exception as e:
            logger.error(f"L0EpisodicLayer: Failed to create chunk from item: {e}")
            return None


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
            
            # Convert facts to ChunkData objects and store
            processed_items = []
            if facts and self.storage_manager:
                fact_chunks = self._convert_facts_to_chunks(facts, metadata)
                for chunk in fact_chunks:
                    fact_id = await self.storage_manager.write_to_backend(
                        StorageType.VECTOR, chunk, metadata
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
            
            # Note: Event emission for L2 processing would be handled by the parallel manager
            # No direct event bus access needed in individual layers
            
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

    def _convert_facts_to_chunks(self, facts: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None) -> List[ChunkData]:
        """Convert extracted facts to ChunkData objects for storage."""
        chunks = []

        try:
            for i, fact in enumerate(facts):
                # Extract content from fact
                content = fact.get("content", "")
                if not content.strip():
                    continue

                # Create chunk metadata combining fact metadata and layer metadata
                chunk_metadata = {
                    "layer": "L1",
                    "source": "semantic_layer",
                    "fact_type": fact.get("type", "unknown"),
                    "extraction_timestamp": fact.get("timestamp", time.time()),
                    **(fact.get("metadata", {})),
                    **(metadata or {})
                }

                # Create ChunkData object
                chunk = ChunkData(
                    content=content,
                    metadata=chunk_metadata
                )
                chunks.append(chunk)

        except Exception as e:
            logger.error(f"L1SemanticLayer: Failed to convert facts to chunks: {e}")

        return chunks


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
            logger.debug(f"L2RelationalLayer: Extracted {len(entities)} entities, {len(relationships)} relationships from data: {type(data)}")

            # Convert entities and relationships to ChunkData objects and store
            processed_items = []
            if self.storage_manager and (entities or relationships):
                # Convert and store entities
                entity_chunks = self._convert_entities_to_chunks(entities, metadata)
                for chunk in entity_chunks:
                    # Try graph storage first, fallback to vector storage if graph is not available
                    entity_id = None
                    try:
                        entity_id = await self.storage_manager.write_to_backend(
                            StorageType.GRAPH, chunk, metadata
                        )
                        logger.debug(f"L2RelationalLayer: Successfully stored entity to graph storage: {entity_id}")
                    except Exception as e:
                        logger.debug(f"L2RelationalLayer: Graph storage not available, using vector storage: {e}")
                        try:
                            entity_id = await self.storage_manager.write_to_backend(
                                StorageType.VECTOR, chunk, metadata
                            )
                            logger.debug(f"L2RelationalLayer: Successfully stored entity to vector storage: {entity_id}")
                        except Exception as ve:
                            logger.error(f"L2RelationalLayer: Failed to store entity to vector storage: {ve}")

                    if entity_id:
                        processed_items.append(entity_id)
                    else:
                        logger.warning(f"L2RelationalLayer: Failed to store entity chunk: {chunk.chunk_id}")

                # Convert and store relationships
                relationship_chunks = self._convert_relationships_to_chunks(relationships, metadata)
                for chunk in relationship_chunks:
                    # Try graph storage first, fallback to vector storage if graph is not available
                    rel_id = None
                    try:
                        rel_id = await self.storage_manager.write_to_backend(
                            StorageType.GRAPH, chunk, metadata
                        )
                        logger.debug(f"L2RelationalLayer: Successfully stored relationship to graph storage: {rel_id}")
                    except Exception as e:
                        logger.debug(f"L2RelationalLayer: Graph storage not available, using vector storage: {e}")
                        try:
                            rel_id = await self.storage_manager.write_to_backend(
                                StorageType.VECTOR, chunk, metadata
                            )
                            logger.debug(f"L2RelationalLayer: Successfully stored relationship to vector storage: {rel_id}")
                        except Exception as ve:
                            logger.error(f"L2RelationalLayer: Failed to store relationship to vector storage: {ve}")

                    if rel_id:
                        processed_items.append(rel_id)
                    else:
                        logger.warning(f"L2RelationalLayer: Failed to store relationship chunk: {chunk.chunk_id}")
            
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

            logger.debug(f"L2RelationalLayer: Processed {len(entities)} entities, {len(relationships)} relationships, success={success}, processed_items={len(processed_items)}")
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
        """Extract entities and relationships from data (placeholder)."""
        # This would integrate with the existing L2 graph construction logic
        # For now, return simple representations based on actual data format
        entities = []
        relationships = []

        try:
            if isinstance(data, list):
                for i, item in enumerate(data):
                    # Handle different data formats
                    if isinstance(item, dict):
                        # Check for fact format first
                        if "fact" in item:
                            fact_content = item["fact"]
                            entities.append({"entity": f"Entity from fact: {fact_content[:50]}..."})
                            relationships.append({"relationship": "extracted_from", "source": "fact", "target": "entity"})
                        # Handle message format
                        elif "content" in item:
                            content = item["content"]
                            entities.append({"entity": f"Entity from message: {content[:50]}..."})
                            relationships.append({"relationship": "mentioned_in", "source": "message", "target": "entity"})
                        # Handle other dict formats
                        else:
                            content = str(item)[:100]
                            entities.append({"entity": f"Entity from data: {content}..."})
                            relationships.append({"relationship": "derived_from", "source": "data", "target": "entity"})
                    elif isinstance(item, str):
                        # Handle string data
                        entities.append({"entity": f"Entity from text: {item[:50]}..."})
                        relationships.append({"relationship": "extracted_from", "source": "text", "target": "entity"})
                    else:
                        # Handle other types
                        content = str(item)[:50]
                        entities.append({"entity": f"Entity from item: {content}..."})
                        relationships.append({"relationship": "derived_from", "source": "item", "target": "entity"})
            elif isinstance(data, dict):
                # Handle single dict
                if "fact" in data:
                    fact_content = data["fact"]
                    entities.append({"entity": f"Entity from fact: {fact_content[:50]}..."})
                    relationships.append({"relationship": "extracted_from", "source": "fact", "target": "entity"})
                elif "content" in data:
                    content = data["content"]
                    entities.append({"entity": f"Entity from message: {content[:50]}..."})
                    relationships.append({"relationship": "mentioned_in", "source": "message", "target": "entity"})
                else:
                    content = str(data)[:100]
                    entities.append({"entity": f"Entity from data: {content}..."})
                    relationships.append({"relationship": "derived_from", "source": "data", "target": "entity"})
            elif isinstance(data, str):
                # Handle string data
                entities.append({"entity": f"Entity from text: {data[:50]}..."})
                relationships.append({"relationship": "extracted_from", "source": "text", "target": "entity"})
            else:
                # Handle other types
                content = str(data)[:50]
                entities.append({"entity": f"Entity from data: {content}..."})
                relationships.append({"relationship": "derived_from", "source": "data", "target": "entity"})

            logger.debug(f"L2RelationalLayer: Extracted {len(entities)} entities and {len(relationships)} relationships from data")

        except Exception as e:
            logger.error(f"L2RelationalLayer: Entity extraction failed: {e}")
            # Return empty lists on error
            entities = []
            relationships = []

        return entities, relationships

    def _convert_entities_to_chunks(self, entities: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None) -> List[ChunkData]:
        """Convert extracted entities to ChunkData objects for storage."""
        chunks = []

        try:
            for i, entity in enumerate(entities):
                # Extract content from entity
                content = entity.get("entity", "") or str(entity)
                if not content.strip():
                    continue

                # Create chunk metadata
                chunk_metadata = {
                    "layer": "L2",
                    "source": "relational_layer",
                    "data_type": "entity",
                    "extraction_timestamp": time.time(),
                    **(metadata or {})
                }

                # Create ChunkData object
                chunk = ChunkData(
                    content=content,
                    metadata=chunk_metadata
                )
                chunks.append(chunk)

        except Exception as e:
            logger.error(f"L2RelationalLayer: Failed to convert entities to chunks: {e}")

        return chunks

    def _convert_relationships_to_chunks(self, relationships: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None) -> List[ChunkData]:
        """Convert extracted relationships to ChunkData objects for storage."""
        chunks = []

        try:
            for i, relationship in enumerate(relationships):
                # Extract content from relationship
                rel_type = relationship.get("relationship", "")
                source = relationship.get("source", "")
                target = relationship.get("target", "")
                content = f"{source} -> {rel_type} -> {target}" if all([source, rel_type, target]) else str(relationship)

                if not content.strip():
                    continue

                # Create chunk metadata
                chunk_metadata = {
                    "layer": "L2",
                    "source": "relational_layer",
                    "data_type": "relationship",
                    "relationship_type": rel_type,
                    "extraction_timestamp": time.time(),
                    **(metadata or {})
                }

                # Create ChunkData object
                chunk = ChunkData(
                    content=content,
                    metadata=chunk_metadata
                )
                chunks.append(chunk)

        except Exception as e:
            logger.error(f"L2RelationalLayer: Failed to convert relationships to chunks: {e}")

        return chunks
