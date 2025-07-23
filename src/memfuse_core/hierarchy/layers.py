"""
Optimized memory layer implementations for the MemFuse hierarchy.

This module provides clean, efficient implementations of M0, M1, and M2
memory layers with unified interfaces and event-driven processing.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional
from datetime import datetime
from loguru import logger

from .core import (
    MemoryLayer, LayerType, LayerConfig, ProcessingResult,
    StorageType, StorageManager
)
from ..rag.chunk.base import ChunkData


class M0RawDataLayer(MemoryLayer):
    """
    M0 (Raw Data) Layer - Original data storage.

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
        
        # M0-specific configuration
        self.storage_backends = config.storage_backends or ["vector", "keyword"]

        logger.info(f"M0RawDataLayer: Initialized for user {user_id}")

    async def initialize(self) -> bool:
        """Initialize the M0 layer."""
        try:
            if self.storage_manager:
                await self.storage_manager.initialize()

            self.initialized = True
            logger.info(f"M0RawDataLayer: Initialized successfully for user {self.user_id}")
            return True

        except Exception as e:
            logger.error(f"M0RawDataLayer: Initialization failed: {e}")
            return False
    
    async def process_data(self, data: Any, metadata: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """
        Process data through M0 layer.

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
            

            
            logger.debug(f"M0RawDataLayer: Processed data with {len(processed_items)} successful stores")
            return result

        except Exception as e:
            processing_time = time.time() - start_time
            self._update_stats(processing_time, False)

            logger.error(f"M0RawDataLayer: Processing failed: {e}")
            return ProcessingResult(
                success=False,
                layer_type=self.layer_type,
                errors=[str(e)],
                processing_time=processing_time
            )
    
    async def query(self, query: str, **kwargs) -> List[Any]:
        """
        Query data from M0 layer.

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
            
            logger.debug(f"M0RawDataLayer: Query returned {len(all_results)} results")
            return all_results

        except Exception as e:
            logger.error(f"M0RawDataLayer: Query failed: {e}")
            return []

    def _convert_to_chunks(self, data: Any, metadata: Optional[Dict[str, Any]] = None) -> List[ChunkData]:
        """Convert raw data to ChunkData objects for storage."""
        chunks = []

        try:
            if isinstance(data, list):
                # Process list of items
                for i, item in enumerate(data):
                    chunk = self._create_chunk_from_item(item, metadata, f"m0_item_{i}")
                    if chunk:
                        chunks.append(chunk)
            else:
                # Process single item
                chunk = self._create_chunk_from_item(data, metadata, "m0_single")
                if chunk:
                    chunks.append(chunk)

        except Exception as e:
            logger.error(f"M0RawDataLayer: Failed to convert data to chunks: {e}")

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
                "layer": "M0",
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
            logger.error(f"M0RawDataLayer: Failed to create chunk from item: {e}")
            return None


class M1EpisodicLayer(MemoryLayer):
    """
    M1 (Episodic Memory) Layer - Event-centered experiences.

    Stores personal experiences and specific events with context:
    - Episode formation from M0 raw data
    - Episode storage and indexing
    - Contextual search over episodes

    Features:
    - Event-driven processing (triggered by M0 events)
    - Episode formation and contextualization
    - Temporal and contextual metadata preservation
    """

    def __init__(
        self,
        layer_type: LayerType,
        config: LayerConfig,
        user_id: str,
        storage_manager: Optional[StorageManager] = None
    ):
        super().__init__(layer_type, config, user_id, storage_manager)

        # M1-specific configuration
        self.llm_config = config.custom_config.get("llm_config", {})
        self.episode_formation_enabled = config.custom_config.get("episode_formation_enabled", True)

        logger.info(f"M1EpisodicLayer: Initialized for user {user_id}")
    
    async def initialize(self) -> bool:
        """Initialize the M1 layer."""
        try:
            if self.storage_manager:
                await self.storage_manager.initialize()

            self.initialized = True
            logger.info(f"M1EpisodicLayer: Initialized successfully for user {self.user_id}")
            return True

        except Exception as e:
            logger.error(f"M1EpisodicLayer: Initialization failed: {e}")
            return False
    
    async def process_data(self, data: Any, metadata: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """
        Process data through M1 layer (episode formation).

        Args:
            data: Data to form episodes from
            metadata: Optional metadata

        Returns:
            ProcessingResult with formed episodes
        """
        start_time = time.time()
        
        try:
            if not self.initialized:
                await self.initialize()
            
            # Form episodes from data (placeholder for LLM integration)
            episodes = await self._form_episodes(data, metadata)

            # Convert episodes to ChunkData objects and store
            processed_items = []
            if episodes and self.storage_manager:
                episode_chunks = self._convert_episodes_to_chunks(episodes, metadata)
                for chunk in episode_chunks:
                    episode_id = await self.storage_manager.write_to_backend(
                        StorageType.VECTOR, chunk, metadata
                    )
                    if episode_id:
                        processed_items.append(episode_id)
            
            processing_time = time.time() - start_time
            success = len(processed_items) > 0
            
            # Update statistics
            self._update_stats(processing_time, success)
            
            # Create result
            result = ProcessingResult(
                success=success,
                layer_type=self.layer_type,
                processed_items=processed_items,
                metadata={"episodes_formed": len(episodes)},
                processing_time=processing_time
            )
            
            # Note: Event emission for M2 processing would be handled by the parallel manager
            # No direct event bus access needed in individual layers

            logger.debug(f"M1EpisodicLayer: Formed {len(episodes)} episodes")
            return result

        except Exception as e:
            processing_time = time.time() - start_time
            self._update_stats(processing_time, False)

            logger.error(f"M1EpisodicLayer: Processing failed: {e}")
            return ProcessingResult(
                success=False,
                layer_type=self.layer_type,
                errors=[str(e)],
                processing_time=processing_time
            )
    
    async def query(self, query: str, **kwargs) -> List[Any]:
        """Query episodes from M1 layer."""
        try:
            if not self.initialized:
                await self.initialize()

            self.total_queries += 1

            # Query episodes from vector store
            results = []
            if self.storage_manager:
                results = await self.storage_manager.read_from_backend(
                    StorageType.VECTOR, query, **kwargs
                )

            logger.debug(f"M1EpisodicLayer: Query returned {len(results)} episodes")
            return results

        except Exception as e:
            logger.error(f"M1EpisodicLayer: Query failed: {e}")
            return []
    

    
    async def process_new_data(self, data: Any, user_id: str, session_id: Optional[str] = None) -> ProcessingResult:
        """
        Process new data for episode formation (compatibility method for MemoryService).

        Args:
            data: Data to process (chunks from M0)
            user_id: User identifier
            session_id: Session identifier

        Returns:
            ProcessingResult with formed episodes
        """
        logger.info(f"M1EpisodicLayer: Processing new data for user {user_id}, session {session_id}")

        # Add metadata for context
        metadata = {
            "user_id": user_id,
            "session_id": session_id,
            "trigger_source": "memory_service_direct"
        }

        return await self.process_data(data, metadata)

    async def query_facts(self, query: str, top_k: int = 10, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Query facts from M1 layer (compatibility method for MemoryService).

        Args:
            query: Search query
            top_k: Maximum number of results
            filters: Optional filters

        Returns:
            List of fact dictionaries
        """
        logger.debug(f"M1EpisodicLayer: Querying episodes with query '{query[:50]}...', top_k={top_k}")

        try:
            results = await self.query(query, top_k=top_k, filters=filters)

            # Convert to expected format for MemoryService
            formatted_results = []
            for result in results:
                if isinstance(result, dict):
                    formatted_results.append(result)
                else:
                    formatted_results.append({"content": str(result)})

            logger.debug(f"M1EpisodicLayer: Returning {len(formatted_results)} formatted episodes")
            return formatted_results

        except Exception as e:
            logger.error(f"M1EpisodicLayer: Query episodes failed: {e}")
            return []

    @property
    def enabled(self) -> bool:
        """Check if M1 layer is enabled."""
        return self.fact_extraction_enabled and self.initialized

    async def _form_episodes(self, data: Any, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Form episodes from data using LLM integration."""
        try:
            episodes = []

            # Handle different data types
            if isinstance(data, list):
                # Process list of chunks
                for item in data:
                    item_episodes = await self._form_episodes_from_item(item, metadata)
                    episodes.extend(item_episodes)
            else:
                # Process single item
                item_episodes = await self._form_episodes_from_item(data, metadata)
                episodes.extend(item_episodes)

            logger.info(f"M1EpisodicLayer: Formed {len(episodes)} episodes from data")
            return episodes

        except Exception as e:
            logger.error(f"M1EpisodicLayer: Episode extraction failed: {e}")
            return []

    async def _form_episodes_from_item(self, item: Any, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Form episodes from a single data item."""
        try:
            # Extract content based on item type
            content = ""
            if isinstance(item, dict):
                content = item.get("content", "") or item.get("text", "") or str(item)
            else:
                content = str(item)

            if not content.strip():
                return []

            # Simple episode formation (placeholder for LLM integration)
            # TODO: Integrate with actual LLM service for sophisticated episode formation
            episodes = []

            # Form basic episodes
            if len(content) > 50:  # Only form episodes from substantial content
                episode = {
                    "episode_content": content[:200] + "..." if len(content) > 200 else content,
                    "episode_type": "formed_episode",
                    "source": "m1_episodic_layer",
                    "timestamp": time.time(),
                    "metadata": metadata or {}
                }
                episodes.append(episode)

            return episodes

        except Exception as e:
            logger.error(f"M1EpisodicLayer: Failed to form episodes from item: {e}")
            return []

    def _convert_episodes_to_chunks(self, episodes: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None) -> List[ChunkData]:
        """Convert episodes to ChunkData objects for storage."""
        chunks = []
        for episode in episodes:
            try:
                # Create chunk metadata
                chunk_metadata = {
                    "layer": "M1",
                    "source": "episodic_layer",
                    "episode_type": episode.get("episode_type", "unknown"),
                    "timestamp": episode.get("timestamp", time.time()),
                    **(metadata or {}),
                    **(episode.get("metadata", {}))
                }

                # Create ChunkData object
                chunk = ChunkData(
                    content=episode.get("episode_content", ""),
                    metadata=chunk_metadata
                )
                chunks.append(chunk)

            except Exception as e:
                logger.error(f"M1EpisodicLayer: Failed to convert episode to chunk: {e}")
                continue

        return chunks

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
                    "layer": "M1",
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
            logger.error(f"M1EpisodicLayer: Failed to convert episodes to chunks: {e}")

        return chunks


class M2SemanticLayer(MemoryLayer):
    """
    M2 (Semantic Memory) Layer - Facts and concepts.

    Extracts and stores semantic knowledge from episodic experiences:
    - Fact extraction from M1 episodes
    - Fact storage and indexing
    - Semantic search over facts
    - Knowledge validation and conflict resolution

    Features:
    - Event-driven processing (triggered by M1 events)
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

        # M2-specific configuration
        self.llm_config = config.custom_config.get("llm_config", {})
        self.fact_extraction_enabled = config.custom_config.get("fact_extraction_enabled", True)

        logger.info(f"M2SemanticLayer: Initialized for user {user_id}")
    
    async def initialize(self) -> bool:
        """Initialize the M2 layer."""
        try:
            if self.storage_manager:
                await self.storage_manager.initialize()

            self.initialized = True
            logger.info(f"M2SemanticLayer: Initialized successfully for user {self.user_id}")
            return True

        except Exception as e:
            logger.error(f"M2SemanticLayer: Initialization failed: {e}")
            return False
    
    async def process_data(self, data: Any, metadata: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """
        Process data through M2 layer (graph construction).

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
            logger.debug(f"M2SemanticLayer: Extracted {len(entities)} facts, {len(relationships)} relationships from data: {type(data)}")

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
                        logger.debug(f"M2SemanticLayer: Successfully stored fact to graph storage: {entity_id}")
                    except Exception as e:
                        logger.debug(f"M2SemanticLayer: Graph storage not available, using vector storage: {e}")
                        try:
                            entity_id = await self.storage_manager.write_to_backend(
                                StorageType.VECTOR, chunk, metadata
                            )
                            logger.debug(f"M2SemanticLayer: Successfully stored fact to vector storage: {entity_id}")
                        except Exception as ve:
                            logger.error(f"M2SemanticLayer: Failed to store fact to vector storage: {ve}")

                    if entity_id:
                        processed_items.append(entity_id)
                    else:
                        logger.warning(f"M2SemanticLayer: Failed to store fact chunk: {chunk.chunk_id}")

                # Convert and store relationships
                relationship_chunks = self._convert_relationships_to_chunks(relationships, metadata)
                for chunk in relationship_chunks:
                    # Try graph storage first, fallback to vector storage if graph is not available
                    rel_id = None
                    try:
                        rel_id = await self.storage_manager.write_to_backend(
                            StorageType.GRAPH, chunk, metadata
                        )
                        logger.debug(f"M2SemanticLayer: Successfully stored relationship to graph storage: {rel_id}")
                    except Exception as e:
                        logger.debug(f"M2SemanticLayer: Graph storage not available, using vector storage: {e}")
                        try:
                            rel_id = await self.storage_manager.write_to_backend(
                                StorageType.VECTOR, chunk, metadata
                            )
                            logger.debug(f"M2SemanticLayer: Successfully stored relationship to vector storage: {rel_id}")
                        except Exception as ve:
                            logger.error(f"M2SemanticLayer: Failed to store relationship to vector storage: {ve}")

                    if rel_id:
                        processed_items.append(rel_id)
                    else:
                        logger.warning(f"M2SemanticLayer: Failed to store relationship chunk: {chunk.chunk_id}")
            
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

            logger.debug(f"M2SemanticLayer: Processed {len(entities)} facts, {len(relationships)} relationships, success={success}, processed_items={len(processed_items)}")
            return result

        except Exception as e:
            processing_time = time.time() - start_time
            self._update_stats(processing_time, False)

            logger.error(f"M2SemanticLayer: Processing failed: {e}")
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
            
            logger.debug(f"M2SemanticLayer: Query returned {len(results)} semantic results")
            return results

        except Exception as e:
            logger.error(f"M2SemanticLayer: Query failed: {e}")
            return []
    

    
    async def _extract_entities_and_relationships(self, data: Any, metadata: Optional[Dict[str, Any]] = None) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Extract entities and relationships from data (placeholder)."""
        # This would integrate with the existing M2 graph construction logic
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

            logger.debug(f"M2SemanticLayer: Extracted {len(entities)} facts and {len(relationships)} relationships from data")

        except Exception as e:
            logger.error(f"M2SemanticLayer: Fact extraction failed: {e}")
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
                    "layer": "M2",
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
            logger.error(f"M2SemanticLayer: Failed to convert facts to chunks: {e}")

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
                    "layer": "M2",
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
            logger.error(f"M2SemanticLayer: Failed to convert relationships to chunks: {e}")

        return chunks


class M3ProceduralLayer(MemoryLayer):
    """
    M3 (Procedural Memory) Layer - Learned patterns and procedures.

    Stores learned patterns, procedures, and behavioral knowledge:
    - Pattern recognition from semantic knowledge
    - Skill and behavior storage
    - Procedural knowledge indexing

    Features:
    - Event-driven processing (triggered by M2 events)
    - Pattern recognition and learning
    - Skill library management

    Note: This is a placeholder implementation for future development.
    """

    def __init__(
        self,
        layer_type: LayerType,
        config: LayerConfig,
        user_id: str,
        storage_manager: Optional[StorageManager] = None
    ):
        super().__init__(layer_type, config, user_id, storage_manager)

        # M3-specific configuration
        self.pattern_config = config.custom_config.get("pattern_config", {})
        self.pattern_recognition_enabled = config.custom_config.get("pattern_recognition_enabled", True)

        logger.info(f"M3ProceduralLayer: Initialized for user {user_id} (Placeholder implementation)")

    async def initialize(self) -> bool:
        """Initialize the M3 layer."""
        try:
            if self.storage_manager:
                await self.storage_manager.initialize()

            self.initialized = True
            logger.info(f"M3ProceduralLayer: Initialized successfully for user {self.user_id}")
            return True

        except Exception as e:
            logger.error(f"M3ProceduralLayer: Initialization failed: {e}")
            return False

    async def process_data(self, data: Any, metadata: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """
        Process data for procedural memory (placeholder implementation).

        In the future, this will:
        - Recognize patterns from semantic data
        - Extract procedural knowledge
        - Store skills and behaviors
        """
        start_time = time.time()

        try:
            logger.info(f"M3ProceduralLayer: Processing data (placeholder): {type(data)}")

            # Placeholder: For now, just return success without actual processing
            processed_items = []
            success = True

            # In future implementation, this would:
            # 1. Analyze semantic data for patterns
            # 2. Extract procedural knowledge
            # 3. Store in procedural memory tables

            processing_time = time.time() - start_time
            self._update_stats(processing_time, success)

            result = ProcessingResult(
                layer_type=self.layer_type,
                success=success,
                processed_items=processed_items,
                processing_time=processing_time,
                metadata={"layer": "M3", "placeholder": True}
            )

            logger.debug("M3ProceduralLayer: Placeholder processing completed")
            return result

        except Exception as e:
            processing_time = time.time() - start_time
            self._update_stats(processing_time, False)

            logger.error(f"M3ProceduralLayer: Processing failed: {e}")
            return ProcessingResult(
                layer_type=self.layer_type,
                success=False,
                processed_items=[],
                processing_time=processing_time,
                metadata={"error": str(e)}
            )

    async def query(self, query: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Query procedural memory (placeholder implementation).

        In the future, this will search for relevant procedures, patterns, and skills.
        """
        try:
            logger.debug(f"M3ProceduralLayer: Querying procedures with query '{query[:50]}...', top_k={top_k}")

            # Placeholder: Return empty results for now
            results = []

            # In future implementation, this would:
            # 1. Search procedural knowledge base using query and filters
            # 2. Find relevant patterns and skills
            # 3. Return top_k ranked results

            # Note: filters parameter will be used in future implementation
            _ = filters  # Acknowledge parameter for future use

            logger.debug(f"M3ProceduralLayer: Query returned {len(results)} procedural results")
            return results

        except Exception as e:
            logger.error(f"M3ProceduralLayer: Query failed: {e}")
            return []

    def _extract_patterns_from_data(self, data: Any) -> List[Dict[str, Any]]:
        """
        Extract patterns from semantic data (placeholder).

        Future implementation will use ML/AI to recognize patterns.
        """
        try:
            # Placeholder: Return empty patterns for now
            patterns = []

            # In future implementation, this would:
            # 1. Analyze semantic data for recurring patterns
            # 2. Identify procedural sequences
            # 3. Extract skill-based knowledge

            # Note: data parameter will be analyzed in future implementation
            _ = data  # Acknowledge parameter for future use

            logger.info(f"M3ProceduralLayer: Extracted {len(patterns)} patterns from data (placeholder)")
            return patterns

        except Exception as e:
            logger.error(f"M3ProceduralLayer: Failed to extract patterns from data: {e}")
            return []

    def _convert_patterns_to_chunks(self, patterns: List[Dict[str, Any]], session_id: Optional[str] = None) -> List[ChunkData]:
        """Convert patterns to chunks for storage (placeholder)."""
        chunks = []

        try:
            for i, pattern in enumerate(patterns):
                chunk_id = f"m3_pattern_{session_id}_{i}" if session_id else f"m3_pattern_{i}"

                chunk = ChunkData(
                    chunk_id=chunk_id,
                    content=pattern.get("description", ""),
                    metadata={
                        "layer": "M3",
                        "type": "pattern",
                        "pattern_type": pattern.get("type", "unknown"),
                        "confidence": pattern.get("confidence", 0.0),
                        "session_id": session_id,
                        "placeholder": True
                    }
                )
                chunks.append(chunk)

        except Exception as e:
            logger.error(f"M3ProceduralLayer: Failed to convert patterns to chunks: {e}")

        return chunks
