"""
Unified storage management for the MemFuse memory hierarchy.

This module provides a centralized storage manager that coordinates
access to multiple storage backends (vector, graph, keyword, SQL).
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
from loguru import logger

from .core import StorageManager, StorageBackend, StorageType
from ..rag.chunk.base import ChunkData


class StoreBackendAdapter(StorageBackend):
    """
    Adapter that wraps existing Store classes to implement StorageBackend interface.
    
    This allows us to use existing vector_store, graph_store, keyword_store
    implementations without modification.
    """
    
    def __init__(self, store: Any, storage_type: StorageType):
        """
        Initialize the adapter.
        
        Args:
            store: Existing store instance (e.g., QdrantStore, NetworkXStore)
            storage_type: Type of storage this adapter represents
        """
        self.store = store
        self.storage_type = storage_type
        self.initialized = False
    
    async def initialize(self) -> bool:
        """Initialize the underlying store."""
        try:
            if hasattr(self.store, 'ensure_initialized'):
                await self.store.ensure_initialized()
            elif hasattr(self.store, 'initialize'):
                await self.store.initialize()
            
            self.initialized = True
            logger.info(f"StoreBackendAdapter: Initialized {self.storage_type.value} store")
            return True
            
        except Exception as e:
            logger.error(f"StoreBackendAdapter: Failed to initialize {self.storage_type.value} store: {e}")
            return False
    
    async def write(self, data: Any, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Write data to the store using unified add interface."""
        if not self.initialized:
            await self.initialize()

        try:
            # All storage types now use 'add' method - just need proper data conversion
            if self.storage_type == StorageType.VECTOR:
                # Vector stores: add(chunks: List[ChunkData])
                data_list = self._prepare_data_for_store(data)
                if not data_list:
                    logger.warning("StoreBackendAdapter: No valid data to write to vector store")
                    return ""

                # Handle dynamic table names for multi-layer architecture
                table_metadata = metadata
                if hasattr(data, 'metadata') and data.metadata:
                    table_metadata = data.metadata
                elif isinstance(data, dict) and 'metadata' in data:
                    table_metadata = data['metadata']

                # Get the appropriate table name for this layer
                table_name = self._get_table_name_for_storage_type(table_metadata)

                # Check if we need to use a different table for this layer
                if hasattr(self.store, 'add_to_table'):
                    # Store supports dynamic table names
                    result_ids = await self.store.add_to_table(table_name, data_list)
                elif hasattr(self.store, 'pgai_store') and hasattr(self.store.pgai_store, 'table_name'):
                    # For PgaiVectorWrapper, temporarily change the table name
                    original_table_name = self.store.pgai_store.table_name
                    original_embedding_view = self.store.pgai_store.embedding_view
                    original_vectorizer_name = self.store.pgai_store.vectorizer_name

                    try:
                        # Temporarily change table configuration
                        self.store.pgai_store.table_name = table_name
                        self.store.pgai_store.embedding_view = f"{table_name}_embedding"
                        self.store.pgai_store.vectorizer_name = f"{table_name}_vectorizer"
                        logger.info(f"StoreBackendAdapter: Temporarily changed vector store table to: {table_name}")

                        result_ids = await self.store.add(data_list)
                    finally:
                        # Restore original table configuration
                        self.store.pgai_store.table_name = original_table_name
                        self.store.pgai_store.embedding_view = original_embedding_view
                        self.store.pgai_store.vectorizer_name = original_vectorizer_name
                        logger.info(f"StoreBackendAdapter: Restored vector store table to: {original_table_name}")
                else:
                    # Fallback to default behavior
                    result_ids = await self.store.add(data_list)

                return result_ids[0] if result_ids else ""

            elif self.storage_type == StorageType.KEYWORD:
                # Keyword stores: add(chunks: List[ChunkData])
                data_list = self._prepare_data_for_store(data)
                if not data_list:
                    logger.warning("StoreBackendAdapter: No valid data to write to keyword store")
                    return ""
                result_ids = await self.store.add(data_list)
                return result_ids[0] if result_ids else ""

            elif self.storage_type == StorageType.GRAPH:
                # Graph stores: add(chunks: List[ChunkData])
                data_list = self._prepare_data_for_store(data)
                if not data_list:
                    logger.warning("StoreBackendAdapter: No valid data to write to graph store")
                    return ""
                result_ids = await self.store.add(data_list)
                return result_ids[0] if result_ids else ""

            elif self.storage_type == StorageType.SQL:
                # Database: add(table, data)
                db_data = self._prepare_data_for_database(data, metadata)
                if not db_data:
                    logger.warning("StoreBackendAdapter: No valid data to write to database")
                    return ""

                # Use ChunkData metadata if available, otherwise use passed metadata
                table_metadata = metadata
                logger.info(f"StorageAdapter: Data type: {type(data)}, hasattr metadata: {hasattr(data, 'metadata')}")
                if hasattr(data, 'metadata') and data.metadata:
                    table_metadata = data.metadata
                    logger.info(f"StorageAdapter: Using ChunkData metadata: {table_metadata}")
                elif isinstance(data, dict) and 'metadata' in data:
                    table_metadata = data['metadata']
                    logger.info(f"StorageAdapter: Using dict metadata: {table_metadata}")
                else:
                    logger.info(f"StorageAdapter: Using passed metadata: {table_metadata}")
                    if hasattr(data, 'metadata'):
                        logger.info(f"StorageAdapter: ChunkData metadata exists but is: {data.metadata}")

                table_name = self._get_table_name_for_storage_type(table_metadata)
                return await self.store.add(table_name, db_data)

            else:
                raise NotImplementedError(f"Storage type {self.storage_type.value} not supported")

        except Exception as e:
            logger.error(f"StoreBackendAdapter: Write failed for {self.storage_type.value}: {e}")
            raise



    def _prepare_data_for_database(self, data: Any, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Prepare data for database insertion."""
        import uuid
        from datetime import datetime

        # Determine layer from metadata to decide whether to add updated_at
        layer = "M0"  # Default to M0
        if isinstance(data, ChunkData):
            if data.metadata and "layer" in data.metadata:
                layer = data.metadata["layer"]
            elif metadata and "layer" in metadata:
                layer = metadata["layer"]
        elif metadata and "layer" in metadata:
            layer = metadata["layer"]

        # Handle different data types
        if isinstance(data, ChunkData):
            # ChunkData object (from M1, M2 layers)
            # Determine content field name based on layer

            # Map layer to content field name and primary key field name
            content_field_mapping = {
                "M0": "content",
                "M1": "content",  # Use 'content' instead of 'episode_content' to match database schema
                "M2": "semantic_content",
                "M3": "procedural_content"
            }
            primary_key_mapping = {
                "M0": "message_id",
                "M1": "chunk_id",
                "M2": "chunk_id",
                "M3": "chunk_id"
            }
            content_field = content_field_mapping.get(layer, "content")
            primary_key_field = primary_key_mapping.get(layer, "id")

            db_data = {
                primary_key_field: data.chunk_id or str(uuid.uuid4()),
                content_field: data.content,
                'created_at': datetime.now()
            }

            # Only add updated_at for tables that have this field
            # M1 episodic table doesn't have updated_at field
            if layer != "M1":
                db_data['updated_at'] = datetime.now()

            # Extract specific fields from metadata for M1 layer
            if layer == "M1" and data.metadata:
                # Map confidence to chunk_quality_score for M1 episodic table
                if 'confidence' in data.metadata:
                    db_data['chunk_quality_score'] = data.metadata['confidence']
                else:
                    db_data['chunk_quality_score'] = 0.8  # Default quality score

                # Set conversation_id for M1 table (required field)
                if 'session_id' in data.metadata:
                    db_data['conversation_id'] = data.metadata['session_id']
                elif 'conversation_id' in data.metadata:
                    db_data['conversation_id'] = data.metadata['conversation_id']
                else:
                    # Generate new conversation_id if not provided
                    conversation_id = str(uuid.uuid4())
                    db_data['conversation_id'] = conversation_id
                    logger.warning(f"StoreBackendAdapter: No session_id/conversation_id found in metadata, generated: {conversation_id}")

                # Set m0_message_ids for lineage tracking (required field)
                if 'message_ids' in data.metadata:
                    db_data['m0_message_ids'] = data.metadata['message_ids']
                elif 'source_message_ids' in data.metadata:
                    db_data['m0_message_ids'] = data.metadata['source_message_ids']
                else:
                    # Default to empty array if no message IDs provided
                    db_data['m0_message_ids'] = []
                    logger.warning(f"StoreBackendAdapter: No message_ids found in metadata for M1 chunk")

                # Set other M1-specific fields that exist in the table
                if 'chunking_strategy' in data.metadata:
                    db_data['chunking_strategy'] = data.metadata['chunking_strategy']
                else:
                    db_data['chunking_strategy'] = 'semantic'  # Default for M1 layer (episodic processing)

                if 'token_count' in data.metadata:
                    db_data['token_count'] = data.metadata['token_count']
                else:
                    # Estimate token count if not provided
                    db_data['token_count'] = len(data.content.split()) if data.content else 0

            # Note: M1 table doesn't have a metadata column, so we don't add it

            logger.info(f"StoreBackendAdapter: Prepared ChunkData for database: {primary_key_field}={db_data[primary_key_field]}, layer={layer}, content_field={content_field}, content_length={len(data.content)}")
            logger.info(f"StoreBackendAdapter: db_data keys: {list(db_data.keys())}")

        elif isinstance(data, dict):
            # Dictionary data (e.g., from M0 raw records)
            # Check if this is a new demo schema record (has message_id)
            if 'message_id' in data:
                # New demo schema - use fields as-is
                db_data = data.copy()
                # Ensure created_at is set if not present
                if 'created_at' not in db_data:
                    db_data['created_at'] = datetime.now()
            else:
                # Legacy schema - convert to old format
                # Default to using 'id' for legacy compatibility
                primary_key_field = 'id'
                db_data = {
                    primary_key_field: data.get('id', str(uuid.uuid4())),
                    'content': data.get('content', ''),
                    'created_at': datetime.now()
                }
                # Only add updated_at for tables that have this field
                # M1 episodic table doesn't have updated_at field
                if layer != "M1":
                    db_data['updated_at'] = datetime.now()

                # Add M0-specific fields if present
                if 'session_id' in data:
                    db_data['session_id'] = data['session_id']
                if 'user_id' in data:
                    db_data['user_id'] = data['user_id']
                if 'message_role' in data:
                    db_data['message_role'] = data['message_role']
                if 'round_id' in data:
                    db_data['round_id'] = data['round_id']

                # Add metadata
                if 'metadata' in data:
                    db_data['metadata'] = data['metadata']
                elif metadata:
                    db_data['metadata'] = metadata

        elif hasattr(data, 'id') and hasattr(data, 'content'):
            # Item-like object - use 'id' for legacy compatibility
            db_data = {
                'id': data.id or str(uuid.uuid4()),
                'content': data.content,
                'created_at': datetime.now()
            }
            # Only add updated_at for tables that have this field
            # M1 episodic table doesn't have updated_at field
            if layer != "M1":
                db_data['updated_at'] = datetime.now()

            # Add metadata if available
            if hasattr(data, 'metadata') and data.metadata:
                db_data['metadata'] = data.metadata
            elif metadata:
                db_data['metadata'] = metadata

        elif isinstance(data, dict):
            # Dictionary data
            db_data = data.copy()
            if 'id' not in db_data:
                db_data['id'] = str(uuid.uuid4())
            if 'created_at' not in db_data:
                db_data['created_at'] = datetime.now()
            # Only add updated_at for tables that have this field
            # M1 episodic table doesn't have updated_at field
            if 'updated_at' not in db_data and layer != "M1":
                db_data['updated_at'] = datetime.now()
            if metadata and 'metadata' not in db_data:
                db_data['metadata'] = metadata

        elif isinstance(data, str):
            # String content
            db_data = {
                'id': str(uuid.uuid4()),
                'content': data,
                'created_at': datetime.now()
            }
            # Only add updated_at for tables that have this field
            # M1 episodic table doesn't have updated_at field
            if layer != "M1":
                db_data['updated_at'] = datetime.now()
            if metadata:
                db_data['metadata'] = metadata
        else:
            # Convert to string
            db_data = {
                'id': str(uuid.uuid4()),
                'content': str(data),
                'created_at': datetime.now()
            }
            # Only add updated_at for tables that have this field
            # M1 episodic table doesn't have updated_at field
            if layer != "M1":
                db_data['updated_at'] = datetime.now()
            if metadata:
                db_data['metadata'] = metadata

        return db_data

    def _get_table_name_for_storage_type(self, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Get the appropriate table name for the storage type based on layer."""
        # Determine layer from metadata
        layer = "M0"  # Default to M0
        if metadata:
            # Check for layer in metadata
            if "layer" in metadata:
                layer = metadata["layer"]
            # Check for layer in nested metadata
            elif "metadata" in metadata and isinstance(metadata["metadata"], dict):
                layer = metadata["metadata"].get("layer", "M0")

        # Map storage types and layers to table names
        if self.storage_type == StorageType.SQL:
            layer_table_mapping = {
                "M0": "m0_raw",
                "M1": "m1_episodic",
                "M2": "m2_semantic",
                "M3": "m3_procedural"
            }
            table_name = layer_table_mapping.get(layer, "m0_raw")

            # Debug logging
            logger.info(f"StorageAdapter: Selecting table for layer '{layer}' -> '{table_name}' (metadata: {metadata})")

            return table_name
        elif self.storage_type == StorageType.VECTOR:
            # Vector storage also needs layer-specific table names
            layer_table_mapping = {
                "M0": "m0_raw",
                "M1": "m1_episodic",
                "M2": "m2_semantic",
                "M3": "m3_procedural"
            }
            table_name = layer_table_mapping.get(layer, "m0_raw")

            # Debug logging
            logger.info(f"StorageAdapter: Selecting vector table for layer '{layer}' -> '{table_name}' (metadata: {metadata})")

            return table_name
        else:
            # For other storage types, use generic names
            table_mapping = {
                StorageType.KEYWORD: "keyword_data",
                StorageType.GRAPH: "graph_data"
            }
            return table_mapping.get(self.storage_type, "data")

    def _prepare_data_for_store(self, data: Any) -> List[ChunkData]:
        """Prepare data for store interface, ensuring it's a list of ChunkData objects."""
        from ..rag.chunk.base import ChunkData

        data_list = []

        try:
            logger.debug(f"StoreBackendAdapter: Preparing data of type {type(data)} for {self.storage_type.value}")

            if isinstance(data, ChunkData):
                # Single ChunkData object
                data_list = [data]
                logger.debug(f"StoreBackendAdapter: Single ChunkData object prepared")
            elif isinstance(data, list):
                # List of items - validate each one
                logger.debug(f"StoreBackendAdapter: Processing list with {len(data)} items")
                for i, item in enumerate(data):
                    if isinstance(item, ChunkData):
                        data_list.append(item)
                        logger.debug(f"StoreBackendAdapter: Added ChunkData item {i}")
                    elif isinstance(item, list):
                        # Nested list - this might be the issue!
                        logger.warning(f"StoreBackendAdapter: Found nested list at index {i}, flattening...")
                        for nested_item in item:
                            if isinstance(nested_item, ChunkData):
                                data_list.append(nested_item)
                            elif isinstance(nested_item, dict):
                                chunk = self._dict_to_chunk_data(nested_item)
                                if chunk:
                                    data_list.append(chunk)
                    elif isinstance(item, dict):
                        # Try to convert dict to ChunkData
                        chunk = self._dict_to_chunk_data(item)
                        if chunk:
                            data_list.append(chunk)
                            logger.debug(f"StoreBackendAdapter: Converted dict to ChunkData at index {i}")
                    else:
                        logger.warning(f"StoreBackendAdapter: Skipping invalid data item of type {type(item)} at index {i}: {item}")
            elif isinstance(data, dict):
                # Single dict - try to convert to ChunkData
                chunk = self._dict_to_chunk_data(data)
                if chunk:
                    data_list = [chunk]
                    logger.debug(f"StoreBackendAdapter: Single dict converted to ChunkData")
            else:
                logger.error(f"StoreBackendAdapter: Unsupported data type {type(data)} for {self.storage_type.value}")

            logger.debug(f"StoreBackendAdapter: Prepared {len(data_list)} ChunkData objects for {self.storage_type.value}")

        except Exception as e:
            logger.error(f"StoreBackendAdapter: Failed to prepare data for {self.storage_type.value}: {e}")

        return data_list

    def _dict_to_chunk_data(self, item: dict) -> Optional[ChunkData]:
        """Convert a dictionary to ChunkData object."""
        from ..rag.chunk.base import ChunkData

        try:
            # Extract content from various possible keys
            content = item.get("content") or item.get("text") or item.get("fact") or str(item)

            if not content or not content.strip():
                return None

            # Extract metadata
            metadata = item.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}

            # Create ChunkData object
            return ChunkData(
                content=content.strip(),
                metadata=metadata
            )

        except Exception as e:
            logger.error(f"StoreBackendAdapter: Failed to convert dict to ChunkData: {e}")
            return None
    
    async def read(self, query: str, **kwargs) -> List[Any]:
        """Read data from the store."""
        if not self.initialized:
            await self.initialize()
        
        try:
            # Most stores have a 'query' or 'search' method
            if hasattr(self.store, 'query'):
                return await self.store.query(query, **kwargs)
            elif hasattr(self.store, 'search'):
                return await self.store.search(query, **kwargs)
            else:
                raise NotImplementedError(f"Store {type(self.store)} does not support read operations")
                
        except Exception as e:
            logger.error(f"StoreBackendAdapter: Read failed for {self.storage_type.value}: {e}")
            raise
    
    async def update(self, item_id: str, data: Any) -> bool:
        """Update existing data."""
        if not self.initialized:
            await self.initialize()
        
        try:
            if hasattr(self.store, 'update'):
                return await self.store.update(item_id, data)
            else:
                logger.warning(f"Store {type(self.store)} does not support update operations")
                return False
                
        except Exception as e:
            logger.error(f"StoreBackendAdapter: Update failed for {self.storage_type.value}: {e}")
            return False
    
    async def delete(self, item_id: str) -> bool:
        """Delete data by ID."""
        if not self.initialized:
            await self.initialize()
        
        try:
            if hasattr(self.store, 'delete'):
                return await self.store.delete(item_id)
            else:
                logger.warning(f"Store {type(self.store)} does not support delete operations")
                return False
                
        except Exception as e:
            logger.error(f"StoreBackendAdapter: Delete failed for {self.storage_type.value}: {e}")
            return False
    
    async def batch_write(self, items: List[Any]) -> List[str]:
        """Write multiple items in batch."""
        if not self.initialized:
            await self.initialize()
        
        try:
            # Check if store supports batch operations
            if hasattr(self.store, 'add_batch'):
                return await self.store.add_batch(items)
            else:
                # Fallback to individual writes
                ids = []
                for item in items:
                    item_id = await self.write(item)
                    ids.append(item_id)
                return ids
                
        except Exception as e:
            logger.error(f"StoreBackendAdapter: Batch write failed for {self.storage_type.value}: {e}")
            raise


class UnifiedStorageManager(StorageManager):
    """
    Unified storage manager that coordinates access to multiple storage backends.
    
    This manager provides a single interface for accessing vector, graph, keyword,
    and SQL storage backends, with automatic initialization, error handling,
    and statistics tracking.
    """
    
    def __init__(self, config: Dict[str, Any], user_id: str):
        """
        Initialize the storage manager.
        
        Args:
            config: Storage configuration dictionary
            user_id: User ID for user-specific storage
        """
        self.config = config
        self.user_id = user_id
        self.initialized = False
        
        # Storage backends
        self.backends: Dict[StorageType, StorageBackend] = {}
        
        # Statistics
        self.write_count = 0
        self.read_count = 0
        self.error_count = 0
        self.last_operation_time: Optional[datetime] = None
    
    async def initialize(self) -> bool:
        """Initialize all configured storage backends."""
        try:
            logger.info(f"UnifiedStorageManager: Initializing storage for user {self.user_id}")

            # Validate configuration structure
            self._validate_storage_config()

            # Get global multi_path configuration to determine which stores to enable
            from ..utils.config import config_manager
            global_config = config_manager.get_config()
            multi_path_config = global_config.get("store", {}).get("multi_path", {})
            
            # Determine which storage types should be enabled based on multi_path config
            enabled_storage_types = set()
            if multi_path_config.get("use_vector", True):
                enabled_storage_types.add(StorageType.VECTOR)
            if multi_path_config.get("use_keyword", False):
                enabled_storage_types.add(StorageType.KEYWORD)
            if multi_path_config.get("use_graph", False):
                enabled_storage_types.add(StorageType.GRAPH)
            
            # Always enable SQL storage for database operations
            enabled_storage_types.add(StorageType.SQL)
            
            logger.info(f"UnifiedStorageManager: Enabled storage types based on multi_path config: {[st.value for st in enabled_storage_types]}")

            # Initialize each configured storage backend
            for storage_name, storage_config in self.config.items():
                try:
                    # Validate that this is a supported storage type
                    storage_type = StorageType(storage_name)
                    
                    # Skip initialization if this storage type is not enabled
                    if storage_type not in enabled_storage_types:
                        logger.info(f"UnifiedStorageManager: Skipping {storage_type.value} backend (disabled by multi_path configuration)")
                        continue
                    
                    backend = await self._create_backend(storage_type, storage_config)

                    if backend and await backend.initialize():
                        self.backends[storage_type] = backend
                        logger.info(f"UnifiedStorageManager: Initialized {storage_type.value} backend")
                    else:
                        logger.error(f"UnifiedStorageManager: Failed to initialize {storage_type.value} backend")

                except ValueError as e:
                    # This is a configuration error - should be explicit
                    valid_types = [t.value for t in StorageType]
                    error_msg = (
                        f"Invalid storage type '{storage_name}' in configuration. "
                        f"Valid storage types are: {valid_types}. "
                        f"If '{storage_name}' is a layer-specific configuration, "
                        f"it should be moved to the appropriate layer configuration section."
                    )
                    logger.error(f"UnifiedStorageManager: {error_msg}")
                    raise ValueError(error_msg) from e
                except Exception as e:
                    logger.error(f"UnifiedStorageManager: Error initializing {storage_name}: {e}")
                    raise e

            self.initialized = True
            logger.info(f"UnifiedStorageManager: Initialized {len(self.backends)} storage backends")
            return True

        except Exception as e:
            logger.error(f"UnifiedStorageManager: Initialization failed: {e}")
            return False

    def _validate_storage_config(self) -> None:
        """Validate storage configuration structure."""
        valid_storage_types = {t.value for t in StorageType}
        invalid_configs = []

        for storage_name in self.config.keys():
            if storage_name not in valid_storage_types:
                invalid_configs.append(storage_name)

        if invalid_configs:
            error_msg = (
                f"Invalid storage configuration detected. "
                f"Found invalid storage types: {invalid_configs}. "
                f"Valid storage types are: {list(valid_storage_types)}. "
                f"\n\nConfiguration structure should be:"
                f"\nstorage:"
                f"\n  vector: {{...}}"
                f"\n  keyword: {{...}}"
                f"\n  graph: {{...}}"
                f"\n  sql: {{...}}"
                f"\n\nLayer-specific configurations like 'l1_storage' should be placed under:"
                f"\nlayers:"
                f"\n  l1:"
                f"\n    storage: {{...}}"
            )
            logger.error(f"UnifiedStorageManager: {error_msg}")
            raise ValueError(error_msg)

    async def get_backend(self, storage_type: StorageType) -> Optional[StorageBackend]:
        """Get a storage backend by type."""
        return self.backends.get(storage_type)
    
    async def write_to_all(self, data: Any, metadata: Optional[Dict[str, Any]] = None) -> Dict[StorageType, Optional[str]]:
        """Write data to all available storage backends in parallel.

        P3 OPTIMIZATION: Parallel storage backend writes to eliminate synchronization bottlenecks.
        """
        if not self.backends:
            logger.warning("UnifiedStorageManager: No backends available for write_to_all")
            return {}

        # P3 OPTIMIZATION: Create parallel tasks for all storage backends
        async def write_to_backend_task(storage_type: StorageType, backend: StoreBackendAdapter) -> tuple[StorageType, Optional[str]]:
            """Write to a single backend with error handling."""
            try:
                item_id = await backend.write(data, metadata)
                logger.debug(f"UnifiedStorageManager: Wrote to {storage_type.value}: {item_id}")
                return storage_type, item_id
            except Exception as e:
                logger.error(f"UnifiedStorageManager: Write failed for {storage_type.value}: {e}")
                return storage_type, None

        # Create tasks for all backends
        tasks = [
            write_to_backend_task(storage_type, backend)
            for storage_type, backend in self.backends.items()
        ]

        # Execute all writes in parallel
        start_time = asyncio.get_event_loop().time()
        task_results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = asyncio.get_event_loop().time()

        # Process results
        results = {}
        error_count = 0

        for result in task_results:
            if isinstance(result, Exception):
                logger.error(f"UnifiedStorageManager: Backend write task failed: {result}")
                error_count += 1
                continue

            storage_type, item_id = result
            results[storage_type] = item_id
            if item_id is None:
                error_count += 1

        # Update statistics
        self.write_count += 1
        self.error_count += error_count
        self.last_operation_time = datetime.utcnow()

        processing_time = end_time - start_time
        logger.info(f"UnifiedStorageManager: Parallel write to {len(self.backends)} backends completed in {processing_time:.3f}s, "
                   f"successful: {len([r for r in results.values() if r is not None])}/{len(self.backends)}")

        return results
    
    async def write_to_backend(self, storage_type: StorageType, data: Any, metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Write data to a specific storage backend."""
        backend = self.backends.get(storage_type)
        if not backend:
            logger.warning(f"UnifiedStorageManager: Backend {storage_type.value} not available")
            raise ValueError(f"Backend {storage_type.value} not available")

        try:
            item_id = await backend.write(data, metadata)
            self.write_count += 1
            self.last_operation_time = datetime.utcnow()
            return item_id

        except Exception as e:
            logger.error(f"UnifiedStorageManager: Write failed for {storage_type.value}: {e}")
            self.error_count += 1
            raise e
    
    async def read_from_backend(self, storage_type: StorageType, query: str, **kwargs) -> List[Any]:
        """Read data from a specific storage backend."""
        backend = self.backends.get(storage_type)
        if not backend:
            logger.warning(f"UnifiedStorageManager: Backend {storage_type.value} not available")
            return []
        
        try:
            results = await backend.read(query, **kwargs)
            self.read_count += 1
            self.last_operation_time = datetime.utcnow()
            return results
            
        except Exception as e:
            logger.error(f"UnifiedStorageManager: Read failed for {storage_type.value}: {e}")
            self.error_count += 1
            return []
    
    def get_available_backends(self) -> List[StorageType]:
        """Get list of available storage backends."""
        return list(self.backends.keys())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage manager statistics."""
        return {
            "user_id": self.user_id,
            "initialized": self.initialized,
            "available_backends": [st.value for st in self.backends.keys()],
            "total_writes": self.write_count,
            "total_reads": self.read_count,
            "total_errors": self.error_count,
            "last_operation_time": self.last_operation_time
        }
    
    async def shutdown(self) -> None:
        """Shutdown all storage backends."""
        for storage_type, backend in self.backends.items():
            try:
                if hasattr(backend.store, 'close'):
                    await backend.store.close()
                logger.info(f"UnifiedStorageManager: Shutdown {storage_type.value} backend")
            except Exception as e:
                logger.error(f"UnifiedStorageManager: Error shutting down {storage_type.value}: {e}")
        
        self.backends.clear()
        self.initialized = False
        logger.info("UnifiedStorageManager: Shutdown complete")
    
    async def _create_backend(self, storage_type: StorageType, config: Dict[str, Any]) -> Optional[StorageBackend]:
        """Create a storage backend from configuration with unified config hierarchy."""
        try:
            # Check if this storage type should be enabled based on multi_path config
            from ..utils.config import config_manager
            global_config = config_manager.get_config()
            multi_path_config = global_config.get("store", {}).get("multi_path", {})
            
            # Check if this storage type is enabled
            should_enable = False
            if storage_type == StorageType.VECTOR and multi_path_config.get("use_vector", True):
                should_enable = True
            elif storage_type == StorageType.KEYWORD and multi_path_config.get("use_keyword", False):
                should_enable = True
            elif storage_type == StorageType.GRAPH and multi_path_config.get("use_graph", False):
                should_enable = True
            elif storage_type == StorageType.SQL:
                should_enable = True  # Always enable SQL storage
            
            if not should_enable:
                logger.info(f"UnifiedStorageManager: Skipping {storage_type.value} backend creation (disabled by multi_path configuration)")
                return None

            # This would integrate with the existing StoreFactory
            from ..store.factory import StoreFactory
            from ..models.core import StoreBackend

            # Get unified configuration with proper hierarchy
            unified_config = self._get_unified_backend_config(storage_type, config)
            backend = unified_config.get("backend")
            backend_enum = StoreBackend(backend) if backend else None

            # Note: Since we use PostgreSQL, we don't need user-specific data directories
            if storage_type == StorageType.VECTOR:
                # Determine table name based on layer context
                table_name = self._get_table_name_for_layer(storage_type, config)
                store = await StoreFactory.create_vector_store(
                    backend=backend_enum,
                    table_name=table_name
                )
            elif storage_type == StorageType.GRAPH:
                store = await StoreFactory.create_graph_store(
                    backend=backend_enum
                )
            elif storage_type == StorageType.KEYWORD:
                store = await StoreFactory.create_keyword_store(
                    backend=backend_enum
                )
            elif storage_type == StorageType.SQL:
                # SQL storage backend - use existing database service
                from ..services.database_service import DatabaseService
                db_service = await DatabaseService.get_instance()
                store = db_service.backend  # Access the backend directly
                logger.info(f"UnifiedStorageManager: Created SQL backend using database service")
            else:
                logger.warning(f"UnifiedStorageManager: Unsupported storage type: {storage_type}")
                return None

            return StoreBackendAdapter(store, storage_type)

        except Exception as e:
            logger.error(f"UnifiedStorageManager: Failed to create {storage_type.value} backend: {e}")
            return None

    def _get_table_name_for_layer(self, storage_type: StorageType, layer_config: Dict[str, Any]) -> str:
        """Get the appropriate table name for a storage type and layer.

        Args:
            storage_type: Type of storage (VECTOR, SQL, etc.)
            layer_config: Layer configuration containing layer information

        Returns:
            Table name to use for this storage type and layer
        """
        # Since layer_config only contains storage backend config, we need to get layer info from global config
        try:
            from ..utils.config import config_manager
            global_config = config_manager.get_config()
            memory_layers_config = global_config.get("store", {}).get("memory_layers", {})

            # Find the layer that matches the current context by checking which layer is being processed
            # We'll use a simple heuristic: check the call stack to see which layer is calling this
            import inspect
            frame = inspect.currentframe()
            layer_name = "M0"  # Default

            # Walk up the call stack to find layer context
            while frame:
                frame_locals = frame.f_locals
                frame_globals = frame.f_globals

                # Check for layer type in locals
                if 'self' in frame_locals:
                    obj = frame_locals['self']
                    if hasattr(obj, 'layer_type'):
                        layer_type = obj.layer_type
                        if hasattr(layer_type, 'value'):
                            layer_name = layer_type.value.upper()
                            break
                        elif isinstance(layer_type, str):
                            layer_name = layer_type.upper()
                            break

                frame = frame.f_back

            logger.info(f"UnifiedStorageManager: Detected layer context: '{layer_name}'")

            # Map layers to table names for vector storage
            if storage_type == StorageType.VECTOR:
                layer_table_mapping = {
                    "M0": "m0_raw",
                    "M1": "m1_episodic",
                    "M2": "m2_semantic",
                    "M3": "m3_procedural"
                }
                table_name = layer_table_mapping.get(layer_name, "m0_raw")
                logger.info(f"UnifiedStorageManager: Selected vector table '{table_name}' for layer '{layer_name}'")
                return table_name
            elif storage_type == StorageType.SQL:
                # SQL storage uses the same table mapping
                layer_table_mapping = {
                    "M0": "m0_raw",
                    "M1": "m1_episodic",
                    "M2": "m2_semantic",
                    "M3": "m3_procedural"
                }
                return layer_table_mapping.get(layer_name, "m0_raw")
            else:
                # For other storage types, use generic names
                return "data"

        except Exception as e:
            logger.warning(f"UnifiedStorageManager: Error detecting layer context: {e}, defaulting to M0")
            # Fallback to M0 if detection fails
            if storage_type == StorageType.VECTOR:
                return "m0_raw"
            elif storage_type == StorageType.SQL:
                return "m0_raw"
            else:
                return "data"

    def _get_unified_backend_config(self, storage_type: StorageType, layer_config: Dict[str, Any]) -> Dict[str, Any]:
        """Get unified backend configuration with proper hierarchy.

        Configuration precedence (highest to lowest):
        1. Layer-specific config (from memory layer)
        2. Store-specific config (from store config)
        3. Global defaults

        Args:
            storage_type: Type of storage backend
            layer_config: Configuration from memory layer

        Returns:
            Unified configuration dictionary
        """
        from ..utils.config import config_manager

        # Get global configuration
        global_config = config_manager.get_config()

        # Start with global defaults
        unified_config = {}

        # 1. Apply store-level defaults
        store_config = global_config.get("store", {})
        if store_config:
            unified_config.update(store_config)

        # 2. Apply pgai-specific configuration if backend is pgai
        backend = layer_config.get("backend") or store_config.get("backend", "qdrant")
        if backend == "pgai":
            # Try to load pgai-specific configuration
            pgai_config = global_config.get("store", {}).get("pgai", {})
            if pgai_config:
                # Apply general pgai config first
                pgai_general = {k: v for k, v in pgai_config.items() if k != "storage_backends"}
                unified_config.update(pgai_general)

                # Apply storage backend specific settings
                storage_backends = pgai_config.get("storage_backends", {})
                storage_specific = storage_backends.get(storage_type.value, {})
                if storage_specific:
                    unified_config.update(storage_specific)

        # 3. Apply layer-specific overrides (highest priority)
        unified_config.update(layer_config)

        # Ensure backend is set - but allow storage-specific backend to override
        if "backend" not in unified_config:
            unified_config["backend"] = backend

        logger.debug(f"UnifiedStorageManager: Unified config for {storage_type.value}: {unified_config}")
        return unified_config
