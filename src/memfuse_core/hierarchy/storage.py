"""
Unified storage management for the MemFuse memory hierarchy.

This module provides a centralized storage manager that coordinates
access to multiple storage backends (vector, graph, keyword, SQL).
"""

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
                # Vector/Keyword/Graph stores: add(chunks: List[ChunkData])
                data_list = self._prepare_data_for_store(data)
                if not data_list:
                    logger.warning("StoreBackendAdapter: No valid data to write to vector store")
                    return ""
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
                table_name = self._get_table_name_for_storage_type()
                return self.store.add(table_name, db_data)

            else:
                raise NotImplementedError(f"Storage type {self.storage_type.value} not supported")

        except Exception as e:
            logger.error(f"StoreBackendAdapter: Write failed for {self.storage_type.value}: {e}")
            raise



    def _prepare_data_for_database(self, data: Any, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Prepare data for database insertion."""
        import uuid
        from datetime import datetime

        # Handle different data types
        if hasattr(data, 'id') and hasattr(data, 'content'):
            # Item-like object
            db_data = {
                'id': data.id or str(uuid.uuid4()),
                'content': data.content,
                'created_at': datetime.now(),
                'updated_at': datetime.now()
            }

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
            if 'updated_at' not in db_data:
                db_data['updated_at'] = datetime.now()
            if metadata and 'metadata' not in db_data:
                db_data['metadata'] = metadata

        elif isinstance(data, str):
            # String content
            db_data = {
                'id': str(uuid.uuid4()),
                'content': data,
                'created_at': datetime.now(),
                'updated_at': datetime.now()
            }
            if metadata:
                db_data['metadata'] = metadata
        else:
            # Convert to string
            db_data = {
                'id': str(uuid.uuid4()),
                'content': str(data),
                'created_at': datetime.now(),
                'updated_at': datetime.now()
            }
            if metadata:
                db_data['metadata'] = metadata

        return db_data

    def _get_table_name_for_storage_type(self) -> str:
        """Get the appropriate table name for the storage type."""
        # Map storage types to table names
        table_mapping = {
            StorageType.SQL: "m0_raw",  # Default table for SQL storage (M0 layer)
            StorageType.VECTOR: "vector_data",
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

            # Initialize each configured storage backend
            for storage_name, storage_config in self.config.items():
                try:
                    # Validate that this is a supported storage type
                    storage_type = StorageType(storage_name)
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
        """Write data to all available storage backends."""
        results = {}
        
        for storage_type, backend in self.backends.items():
            try:
                item_id = await backend.write(data, metadata)
                results[storage_type] = item_id
                logger.debug(f"UnifiedStorageManager: Wrote to {storage_type.value}: {item_id}")
                
            except Exception as e:
                logger.error(f"UnifiedStorageManager: Write failed for {storage_type.value}: {e}")
                results[storage_type] = None
                self.error_count += 1
        
        self.write_count += 1
        self.last_operation_time = datetime.utcnow()
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
            # This would integrate with the existing StoreFactory
            from ..store.factory import StoreFactory
            from ..models.core import StoreBackend
            from ..utils.config import config_manager

            # Get unified configuration with proper hierarchy
            unified_config = self._get_unified_backend_config(storage_type, config)
            backend = unified_config.get("backend")
            backend_enum = StoreBackend(backend) if backend else None

            if storage_type == StorageType.VECTOR:
                store = await StoreFactory.create_vector_store(
                    backend=backend_enum,
                    data_dir=unified_config.get("data_dir", f"data/{self.user_id}")
                )
            elif storage_type == StorageType.GRAPH:
                store = await StoreFactory.create_graph_store(
                    backend=backend_enum,
                    data_dir=unified_config.get("data_dir", f"data/{self.user_id}")
                )
            elif storage_type == StorageType.KEYWORD:
                store = await StoreFactory.create_keyword_store(
                    backend=backend_enum,
                    data_dir=unified_config.get("data_dir", f"data/{self.user_id}")
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
