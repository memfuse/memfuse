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
            
            # Convert data to raw records for storage (no chunking, no embedding)
            raw_records = self._convert_to_raw_records(data, metadata)

            # Store raw records directly to SQL backend only (no vector storage)
            storage_results = {}
            processed_items = []

            if self.storage_manager and raw_records:
                logger.debug(f"M0RawDataLayer: Writing {len(raw_records)} records to SQL storage")
                # Only use SQL backend for M0 raw data storage
                from ..hierarchy.storage import StorageType

                # Write each record individually to SQL backend
                successful_ids = []
                for record in raw_records:
                    try:
                        sql_result = await self.storage_manager.write_to_backend(
                            StorageType.SQL, record, metadata
                        )
                        if sql_result:
                            successful_ids.append(record["message_id"])
                            logger.debug(f"M0RawDataLayer: Successfully wrote record {record['message_id']}")
                        else:
                            logger.warning(f"M0RawDataLayer: Failed to write record {record['message_id']}")
                    except Exception as e:
                        logger.error(f"M0RawDataLayer: Error writing record {record['message_id']}: {e}")

                if successful_ids:
                    storage_results[StorageType.SQL] = successful_ids
                    processed_items = successful_ids

            processing_time = time.time() - start_time
            success = len(processed_items) > 0

            # Add error information if no items were processed
            errors = []
            if not success and storage_results:
                failed_backends = [
                    backend.value for backend, result in storage_results.items()
                    if result is None
                ]
                if failed_backends:
                    errors.append(f"Failed to write to backends: {', '.join(failed_backends)}")
            elif not success and not raw_records:
                errors.append("No raw records generated from input data")
            elif not success:
                errors.append("Failed to write to SQL backend")
            
            # Update statistics
            self._update_stats(processing_time, success)
            
            # Create result
            result = ProcessingResult(
                success=success,
                layer_type=self.layer_type,
                processed_items=processed_items,
                errors=errors,
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
        Query data from M0 layer - only SQL/keyword search, no vector search.

        M0 layer stores raw data without embeddings, so vector search is not supported.
        For semantic/vector search, use M1 layer instead.

        Args:
            query: Query string
            **kwargs: Additional query parameters

        Returns:
            List of matching results from SQL/keyword backends only
        """
        try:
            if not self.initialized:
                await self.initialize()

            self.total_queries += 1

            # Query from non-vector backends only (M0 has no embeddings)
            all_results = []

            if self.storage_manager:
                # Query SQL store for exact matches
                if StorageType.SQL in self.storage_manager.get_available_backends():
                    sql_results = await self.storage_manager.read_from_backend(
                        StorageType.SQL, query, **kwargs
                    )
                    all_results.extend(sql_results)

                # Query keyword store for text search
                if StorageType.KEYWORD in self.storage_manager.get_available_backends():
                    keyword_results = await self.storage_manager.read_from_backend(
                        StorageType.KEYWORD, query, **kwargs
                    )
                    all_results.extend(keyword_results)

            logger.debug(f"M0RawDataLayer: Query returned {len(all_results)} results (no vector search)")
            return all_results

        except Exception as e:
            logger.error(f"M0RawDataLayer: Query failed: {e}")
            return []

    def _convert_to_raw_records(self, data: Any, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Convert input data to raw records for M0 storage (no chunking, no processing)."""
        records = []

        try:
            logger.info(f"M0RawDataLayer: Converting data to raw records, data type: {type(data)}, data: {data}")
            logger.info(f"M0RawDataLayer: Metadata: {metadata}")

            if isinstance(data, list):
                logger.info(f"M0RawDataLayer: Processing list with {len(data)} items")
                # Process list of message items
                for i, item in enumerate(data):
                    logger.info(f"M0RawDataLayer: Processing item {i}: {type(item)} - {item}")
                    record = self._create_raw_record_from_item(item, metadata, f"m0_raw_{i}")
                    if record:
                        records.append(record)
                        logger.info(f"M0RawDataLayer: Successfully created record {i}")
                    else:
                        logger.warning(f"M0RawDataLayer: Failed to create record from item {i}")
            else:
                logger.info(f"M0RawDataLayer: Processing single item: {type(data)} - {data}")
                # Process single item
                record = self._create_raw_record_from_item(data, metadata, "m0_raw_single")
                if record:
                    records.append(record)
                    logger.info(f"M0RawDataLayer: Successfully created single record")
                else:
                    logger.warning(f"M0RawDataLayer: Failed to create record from single item")

        except Exception as e:
            logger.error(f"M0RawDataLayer: Failed to convert data to raw records: {e}")

        logger.info(f"M0RawDataLayer: Generated {len(records)} raw records total")
        return records

    def _create_raw_record_from_item(self, item: Any, metadata: Optional[Dict[str, Any]], item_id: str) -> Optional[Dict[str, Any]]:
        """Create a raw record from a single item for M0 storage."""
        try:
            import uuid

            # Extract content and basic fields based on item type
            if isinstance(item, dict):
                # Handle message-like objects
                content = item.get("content", str(item))
                role = item.get("role", "unknown")
                item_metadata = {k: v for k, v in item.items() if k not in ["content", "role"]}
            elif isinstance(item, str):
                content = item
                role = "unknown"
                item_metadata = {}
            else:
                content = str(item)
                role = "unknown"
                item_metadata = {}

            # Ensure we have valid content
            if not content or not content.strip():
                logger.warning(f"M0RawDataLayer: Empty content for item {item_id}, skipping")
                return None

            # Generate unique message ID
            message_id = str(uuid.uuid4())

            # Get sequence number from item or metadata
            sequence_number = 1  # Default value
            if isinstance(item, dict):
                # Try to get message_index from the item itself first
                sequence_number = item.get("message_index", 0) + 1
            if sequence_number == 1 and metadata:
                # Fallback to metadata if not found in item
                sequence_number = metadata.get("message_index", 0) + 1

            # Create raw record for M0 table (demo schema compatible)
            record = {
                "message_id": message_id,
                "content": content.strip(),
                "role": role,
                "conversation_id": metadata.get("session_id") if metadata else str(uuid.uuid4()),
                "sequence_number": sequence_number,
                "token_count": max(1, len(content.strip()) // 4),  # Rough token estimate
                "processing_status": "pending"
            }

            logger.debug(f"M0RawDataLayer: Created raw record {message_id} with content: {content[:50]}...")
            return record

        except Exception as e:
            logger.error(f"M0RawDataLayer: Failed to create raw record from item: {e}")
            return None

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
        Process data through M1 layer - chunking and embedding generation from M0 raw data.

        Args:
            data: M0 raw data IDs or raw data to process
            metadata: Optional metadata including source information

        Returns:
            ProcessingResult with chunked and embedded data
        """
        start_time = time.time()

        try:
            if not self.initialized:
                await self.initialize()

            # Process M0 raw data into chunks with embeddings
            chunks = await self._process_m0_to_chunks(data, metadata)

            # Store chunks with automatic embedding generation
            processed_items = []
            if chunks and self.storage_manager:
                logger.debug(f"M1EpisodicLayer: Writing {len(chunks)} chunks to storage with embedding")

                # Store each chunk individually to all configured backends
                for i, chunk in enumerate(chunks):
                    try:
                        logger.debug(f"M1EpisodicLayer: Writing chunk {i+1}/{len(chunks)}: {chunk.metadata.get('chunk_id', 'unknown')}")

                        # Store single chunk to all configured backends
                        storage_results = await self.storage_manager.write_to_all(chunk, metadata)

                        # Collect successful storage IDs for this chunk
                        chunk_processed_items = [
                            item_id for item_id in storage_results.values()
                            if item_id is not None
                        ]

                        if chunk_processed_items:
                            processed_items.extend(chunk_processed_items)
                            logger.debug(f"M1EpisodicLayer: Chunk {i+1} stored successfully to {len(chunk_processed_items)} backends")
                        else:
                            logger.warning(f"M1EpisodicLayer: Chunk {i+1} failed to store to any backend")

                    except Exception as e:
                        logger.error(f"M1EpisodicLayer: Failed to store chunk {i+1}: {e}")

                logger.info(f"M1EpisodicLayer: Chunk storage completed, "
                           f"successful items: {len(processed_items)}, chunks processed: {len(chunks)}")
            
            processing_time = time.time() - start_time
            success = len(processed_items) > 0 or len(chunks) == 0  # Success if stored or no chunks to store

            # Add error information if needed
            errors = []
            if not success:
                errors.append("Failed to store chunks to any backend")

            # Update statistics
            self._update_stats(processing_time, success)

            # Create result
            result = ProcessingResult(
                success=success,
                layer_type=self.layer_type,
                processed_items=processed_items,
                errors=errors,
                metadata={"chunks_processed": len(chunks)},
                processing_time=processing_time
            )

            logger.debug(f"M1EpisodicLayer: Processed {len(chunks)} chunks, stored {len(processed_items)}")
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

    async def _process_m0_to_chunks(self, data: Any, metadata: Optional[Dict[str, Any]] = None) -> List[ChunkData]:
        """Process M0 raw data into chunks for M1 storage with embeddings."""
        chunks = []

        try:
            logger.info(f"M1EpisodicLayer: Processing M0 data to chunks, data type: {type(data)}, data: {data}")

            if isinstance(data, list):
                # Process list of M0 raw data items
                for i, item in enumerate(data):
                    logger.info(f"M1EpisodicLayer: Processing M0 item {i}: {type(item)} - {item}")
                    chunk = self._create_chunk_from_m0_item(item, metadata, f"m1_chunk_{i}")
                    if chunk:
                        chunks.append(chunk)
                        logger.info(f"M1EpisodicLayer: Successfully created chunk {i}")
                    else:
                        logger.warning(f"M1EpisodicLayer: Failed to create chunk from M0 item {i}")
            else:
                # Process single M0 item
                logger.info(f"M1EpisodicLayer: Processing single M0 item: {type(data)} - {data}")
                chunk = self._create_chunk_from_m0_item(data, metadata, "m1_chunk_single")
                if chunk:
                    chunks.append(chunk)
                    logger.info(f"M1EpisodicLayer: Successfully created single chunk")
                else:
                    logger.warning(f"M1EpisodicLayer: Failed to create chunk from single M0 item")

        except Exception as e:
            logger.error(f"M1EpisodicLayer: Failed to process M0 data to chunks: {e}")

        logger.info(f"M1EpisodicLayer: Generated {len(chunks)} chunks from M0 data")
        return chunks

    def _create_chunk_from_m0_item(self, item: Any, metadata: Optional[Dict[str, Any]], chunk_id: str) -> Optional[ChunkData]:
        """Create a ChunkData object from M0 raw data item."""
        try:
            # Handle M0 reference data (need to fetch actual content)
            if isinstance(item, dict) and item.get("source_layer") == "M0":
                # This is a reference to M0 data, extract the reference info
                source_id = item.get("source_id")
                operation_metadata = item.get("metadata", {})

                # Extract actual content from the item if available
                content = item.get("content", "")
                if not content:
                    # Try to get content from nested data
                    if "message_ids" in operation_metadata:
                        message_ids = operation_metadata["message_ids"]
                        content = f"Episode from {len(message_ids)} messages"
                    else:
                        content = f"M0 Reference: {source_id}"

                session_id = operation_metadata.get("session_id")
                user_id = operation_metadata.get("user_id")
                message_role = operation_metadata.get("message_role", "unknown")

                logger.info(f"M1EpisodicLayer: Processing M0 reference {source_id}")

            elif isinstance(item, dict):
                # Direct M0 data
                content = item.get("content", str(item))
                source_id = item.get("id")
                session_id = item.get("session_id")
                user_id = item.get("user_id")
                message_role = item.get("message_role", "unknown")
            elif isinstance(item, str):
                content = item
                source_id = None
                session_id = metadata.get("session_id") if metadata else None
                user_id = metadata.get("user_id") if metadata else None
                message_role = "unknown"
            elif hasattr(item, 'content') and hasattr(item, 'metadata'):
                # Handle ChunkData objects
                content = item.content
                source_id = item.chunk_id if hasattr(item, 'chunk_id') else None
                item_metadata = item.metadata or {}
                session_id = item_metadata.get("session_id") or item_metadata.get("conversation_id")
                user_id = item_metadata.get("user_id")
                message_role = item_metadata.get("message_role", "unknown")

                # Merge item metadata with passed metadata, giving priority to item metadata
                if metadata:
                    merged_metadata = {**metadata, **item_metadata}
                else:
                    merged_metadata = item_metadata
                metadata = merged_metadata

                logger.info(f"M1EpisodicLayer: Processing ChunkData object with conversation_id: {item_metadata.get('conversation_id')}")
            else:
                content = str(item)
                source_id = None
                session_id = metadata.get("session_id") if metadata else None
                user_id = metadata.get("user_id") if metadata else None
                message_role = "unknown"

            # Ensure we have valid content
            if not content or not content.strip():
                logger.warning(f"M1EpisodicLayer: Empty content for chunk {chunk_id}, skipping")
                return None

            # Create chunk metadata for M1
            # Start with passed metadata, then override with M1-specific fields
            chunk_metadata = {
                **(metadata or {}),
                "layer": "M1",  # Always M1 for this layer
                "source_layer": "M0",
                "source_id": source_id,
                "session_id": session_id,
                "user_id": user_id,
                "message_role": message_role,
                "chunk_type": "episodic",
                "chunk_id": chunk_id,
                "confidence": 0.8,  # Default confidence for M1 chunks
                "timestamp": time.time()
            }

            # Ensure conversation_id is preserved from original metadata or use session_id
            if metadata and 'conversation_id' in metadata:
                chunk_metadata['conversation_id'] = metadata['conversation_id']
            elif session_id:
                chunk_metadata['conversation_id'] = session_id

            return ChunkData(
                content=content.strip(),
                metadata=chunk_metadata
            )

        except Exception as e:
            logger.error(f"M1EpisodicLayer: Failed to create chunk from M0 item: {e}")
            return None
    
    async def query(self, query: str, **kwargs) -> List[Any]:
        """
        Query chunks from M1 layer - primary vector search layer.

        M1 layer contains chunked data with embeddings, making it the main layer
        for semantic/vector similarity search. This is where most query operations
        should be directed for meaningful semantic retrieval.

        Args:
            query: Query string for semantic search
            **kwargs: Additional query parameters (top_k, filters, etc.)

        Returns:
            List of matching chunks with embeddings and similarity scores
        """
        try:
            if not self.initialized:
                await self.initialize()

            self.total_queries += 1

            # Query chunks from vector store (primary search method for M1)
            results = []
            if self.storage_manager:
                # M1 layer primarily uses vector search since it has embeddings
                if StorageType.VECTOR in self.storage_manager.get_available_backends():
                    results = await self.storage_manager.read_from_backend(
                        StorageType.VECTOR, query, **kwargs
                    )
                    logger.debug(f"M1EpisodicLayer: Vector search returned {len(results)} chunks")

                # Fallback to other backends if needed
                if not results and StorageType.SQL in self.storage_manager.get_available_backends():
                    sql_results = await self.storage_manager.read_from_backend(
                        StorageType.SQL, query, **kwargs
                    )
                    results.extend(sql_results)
                    logger.debug(f"M1EpisodicLayer: SQL fallback returned {len(sql_results)} additional chunks")

            logger.debug(f"M1EpisodicLayer: Total query returned {len(results)} chunks")
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
                    "content": content[:200] + "..." if len(content) > 200 else content,  # Use 'content' instead of 'episode_content'
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
                    "confidence": episode.get("confidence", 0.8),  # Default confidence for episodes
                    "timestamp": episode.get("timestamp", time.time()),
                    **(metadata or {}),
                    **(episode.get("metadata", {}))
                }

                # Create ChunkData object
                chunk = ChunkData(
                    content=episode.get("content", ""),  # Use 'content' instead of 'episode_content'
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

            # Convert entities and relationships to ChunkData objects and store in parallel
            # P3 OPTIMIZATION: Parallel entity and relationship storage to eliminate sequential bottleneck
            processed_items = []
            if self.storage_manager and (entities or relationships):
                # P3 OPTIMIZATION: Create parallel storage tasks for entities and relationships
                async def store_entity_chunk(chunk: Any) -> Optional[str]:
                    """Store a single entity chunk with fallback storage strategy."""
                    try:
                        # Try graph storage first, fallback to vector storage if graph is not available
                        try:
                            entity_id = await self.storage_manager.write_to_backend(
                                StorageType.GRAPH, chunk, metadata
                            )
                            logger.debug(f"M2SemanticLayer: Successfully stored fact to graph storage: {entity_id}")
                            return entity_id
                        except Exception as e:
                            logger.debug(f"M2SemanticLayer: Graph storage not available, using vector storage: {e}")
                            entity_id = await self.storage_manager.write_to_backend(
                                StorageType.VECTOR, chunk, metadata
                            )
                            logger.debug(f"M2SemanticLayer: Successfully stored fact to vector storage: {entity_id}")
                            return entity_id
                    except Exception as e:
                        logger.error(f"M2SemanticLayer: Failed to store entity chunk: {e}")
                        return None

                async def store_relationship_chunk(chunk: Any) -> Optional[str]:
                    """Store a single relationship chunk with fallback storage strategy."""
                    try:
                        # Try graph storage first, fallback to vector storage if graph is not available
                        try:
                            rel_id = await self.storage_manager.write_to_backend(
                                StorageType.GRAPH, chunk, metadata
                            )
                            logger.debug(f"M2SemanticLayer: Successfully stored relationship to graph storage: {rel_id}")
                            return rel_id
                        except Exception as e:
                            logger.debug(f"M2SemanticLayer: Graph storage not available, using vector storage: {e}")
                            rel_id = await self.storage_manager.write_to_backend(
                                StorageType.VECTOR, chunk, metadata
                            )
                            logger.debug(f"M2SemanticLayer: Successfully stored relationship to vector storage: {rel_id}")
                            return rel_id
                    except Exception as e:
                        logger.error(f"M2SemanticLayer: Failed to store relationship chunk: {e}")
                        return None

                # Create parallel tasks for all entities and relationships
                storage_tasks = []

                # Add entity storage tasks
                entity_chunks = self._convert_entities_to_chunks(entities, metadata)
                storage_tasks.extend([store_entity_chunk(chunk) for chunk in entity_chunks])

                # Add relationship storage tasks
                relationship_chunks = self._convert_relationships_to_chunks(relationships, metadata)
                storage_tasks.extend([store_relationship_chunk(chunk) for chunk in relationship_chunks])

                # Execute all storage tasks in parallel
                if storage_tasks:
                    storage_results = await asyncio.gather(*storage_tasks, return_exceptions=True)

                    # Process results and collect successful IDs
                    for result in storage_results:
                        if isinstance(result, Exception):
                            logger.error(f"M2SemanticLayer: Storage task failed: {result}")
                            continue
                        if result:
                            processed_items.append(result)

                    logger.info(f"M2SemanticLayer: Parallel storage completed, "
                               f"successful: {len(processed_items)}/{len(storage_tasks)} "
                               f"(entities: {len(entity_chunks)}, relationships: {len(relationship_chunks)})")
            
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
