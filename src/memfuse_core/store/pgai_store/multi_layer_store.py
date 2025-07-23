"""
Multi-layer PgAI store for multi-layer embedding system.

This module provides an interface for managing M0 (raw data)
and M1 (episodic) memory layers with automatic embedding generation.
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Union
from loguru import logger
from enum import Enum

from .event_driven_store import EventDrivenPgaiStore
from .schema_manager import SchemaManager
from .config_manager import ConfigManager
from .stats_collector import StatsCollector
from ...rag.chunk.base import ChunkData
from ...hierarchy.llm_service import AdvancedLLMService, ExtractedFact
from .episode_formation_processor import EpisodeFormationProcessor


class LayerType(Enum):
    """Memory layer types."""
    M0 = "m0"  # Raw Data Memory
    M1 = "m1"  # Episodic Memory


class MultiLayerPgaiStore:
    """
    Multi-layer PgAI store supporting M0 and M1 memory layers.

    This class coordinates multi-layer processing with automatic embedding
    generation for raw data (M0) and episodic (M1) memory layers.

    Features:
    - Interface for multi-layer operations
    - Configuration-driven layer enable/disable
    - Parallel processing of M0 and M1 layers
    - Independent embedding generation for each layer
    - Backward compatibility with single-layer usage
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize multi-layer PgAI store.

        Args:
            config: Configuration dictionary with layer settings
        """
        # Apply default configuration
        self.config = ConfigManager.apply_defaults(config)
        self.initialized = False

        # Get enabled layers using ConfigManager
        self.enabled_layers = self._get_enabled_layers()

        # Layer configurations
        self.layer_configs = self.config.get('memory_layers', {})

        # Layer stores
        self.layer_stores: Dict[LayerType, EventDrivenPgaiStore] = {}

        # Schema manager
        self.schema_manager: Optional[SchemaManager] = None

        # Episode formation service (for M1)
        self.episode_former: Optional[EpisodeFormationProcessor] = None

        # Unified statistics collector
        self.stats_collector = StatsCollector()

        # Legacy stats for backward compatibility
        self.stats = {
            'total_operations': 0,
            'layer_operations': {layer.value: 0 for layer in LayerType},
            'episode_formations': 0,
            'errors': 0
        }
        
        logger.info(f"MultiLayerPgaiStore initialized with layers: {[l.value for l in self.enabled_layers]}")
    
    def _get_enabled_layers(self) -> List[LayerType]:
        """Get list of enabled memory layers from configuration."""
        enabled = []
        
        # Use ConfigManager to get enabled layers
        enabled_layer_names = ConfigManager.get_enabled_layers(self.config)

        for layer_name in enabled_layer_names:
            try:
                layer_type = LayerType(layer_name)
                enabled.append(layer_type)
            except ValueError:
                logger.warning(f"Unknown layer type: {layer_name}")
        
        # Ensure M0 is always first if enabled (dependency order)
        if LayerType.M0 in enabled and LayerType.M1 in enabled:
            enabled = [LayerType.M0, LayerType.M1]
        
        return enabled
    
    async def initialize(self) -> bool:
        """Initialize all enabled memory layers and their stores.
        
        Returns:
            True if initialization successful
        """
        try:
            if self.initialized:
                logger.info("MultiTablePgaiStore already initialized")
                return True
            
            logger.info("Initializing MultiTablePgaiStore...")
            
            # Initialize layer stores
            for layer_type in self.enabled_layers:
                success = await self._initialize_layer_store(layer_type)
                if not success:
                    logger.error(f"Failed to initialize store for layer {layer_type.value}")
                    return False
            
            # Initialize schema manager using the first available store's pool
            if self.layer_stores:
                first_store = next(iter(self.layer_stores.values()))
                if hasattr(first_store, 'core_store') and first_store.core_store:
                    self.schema_manager = SchemaManager(first_store.core_store.pool)
                    
                    # Initialize schemas for all enabled layers
                    layer_names = [layer.value for layer in self.enabled_layers]
                    schema_success = await self.schema_manager.initialize_all_schemas(layer_names)
                    if not schema_success:
                        logger.error("Schema initialization failed")
                        return False
            
            # Initialize episode formation service if M1 is enabled
            if LayerType.M1 in self.enabled_layers:
                await self._initialize_episode_former()
            
            self.initialized = True
            logger.info("MultiTablePgaiStore initialization completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"MultiTablePgaiStore initialization failed: {e}")
            return False
    
    async def _initialize_layer_store(self, layer_type: LayerType) -> bool:
        """Initialize store for a specific memory layer.
        
        Args:
            layer_type: Type of memory layer to initialize
            
        Returns:
            True if initialization successful
        """
        try:
            layer_config = self.layer_configs.get(layer_type.value, {})
            table_name = layer_config.get('table_name', f"{layer_type.value}_default")
            
            logger.info(f"Initializing {layer_type.value} layer store with table: {table_name}")
            
            # Create layer-specific configuration
            store_config = self._create_store_config(layer_type, layer_config)
            
            # Create and initialize the store
            store = EventDrivenPgaiStore(config=store_config, table_name=table_name)
            success = await store.initialize()
            
            if success:
                self.layer_stores[layer_type] = store
                logger.info(f"{layer_type.value} layer store initialized successfully")
                return True
            else:
                logger.error(f"Failed to initialize {layer_type.value} layer store")
                return False
                
        except Exception as e:
            logger.error(f"Error initializing {layer_type.value} layer store: {e}")
            return False
    
    def _create_store_config(self, layer_type: LayerType, layer_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create store configuration for a specific layer.
        
        Args:
            layer_type: Type of memory layer
            layer_config: Layer-specific configuration
            
        Returns:
            Store configuration dictionary
        """
        # Base configuration from main config
        store_config = {
            'database': self.config.get('database', {}),
            'pgai': {}
        }
        
        # Layer-specific pgai configuration
        layer_pgai_config = layer_config.get('pgai', {})
        performance_config = layer_config.get('performance', {})
        
        store_config['pgai'] = {
            'auto_embedding': layer_pgai_config.get('auto_embedding', True),
            'immediate_trigger': layer_pgai_config.get('immediate_trigger', True),
            'embedding_model': layer_pgai_config.get('embedding_model', 'all-MiniLM-L6-v2'),
            'embedding_dimensions': layer_pgai_config.get('embedding_dimensions', 384),
            'max_retries': performance_config.get('max_retries', 3),
            'retry_interval': performance_config.get('retry_interval', 5.0),
            'worker_count': performance_config.get('worker_count', 2),
            'queue_size': performance_config.get('queue_size', 100),
            'enable_metrics': True
        }
        
        return store_config
    
    async def _initialize_episode_former(self):
        """Initialize episode formation service for M1 layer."""
        try:
            m1_config = self.layer_configs.get('m1', {})
            episode_config = m1_config.get('episode_formation', {})

            if episode_config.get('enabled', True):
                # Initialize EpisodeFormationProcessor
                self.episode_former = EpisodeFormationProcessor(episode_config)
                logger.info("Episode formation service initialized successfully")
            else:
                logger.info("Episode formation disabled in configuration")

        except Exception as e:
            logger.error(f"Failed to initialize episode former: {e}")
    
    async def write_to_layer(self, layer_type: LayerType, data: Union[List[ChunkData], ChunkData], 
                           metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """Write data to a specific memory layer.
        
        Args:
            layer_type: Target memory layer
            data: Data to write (ChunkData or list of ChunkData)
            metadata: Optional metadata
            
        Returns:
            List of record IDs that were created
        """
        try:
            if not self.initialized:
                await self.initialize()
            
            store = self.layer_stores.get(layer_type)
            if not store:
                raise ValueError(f"Layer {layer_type.value} not available or not enabled")
            
            # Ensure data is a list
            if not isinstance(data, list):
                data = [data]
            
            # Write to the layer store
            result_ids = await store.add(data)
            
            # Update statistics
            self.stats['total_operations'] += 1
            self.stats['layer_operations'][layer_type.value] += 1
            
            logger.debug(f"Wrote {len(data)} items to {layer_type.value} layer")
            return result_ids
            
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Failed to write to {layer_type.value} layer: {e}")
            raise
    
    async def write_to_all_layers(self, data: Union[List[ChunkData], ChunkData],
                                metadata: Optional[Dict[str, Any]] = None) -> Dict[str, List[str]]:
        """Write data to all enabled layers in parallel.

        This method supports the architecture where data flows from Memory Layer top
        to M0 and M1 in parallel, rather than M0->M1 dependency.

        Args:
            data: Original data to write to all layers
            metadata: Optional metadata

        Returns:
            Dict with layer names as keys and lists of created record IDs as values
        """
        try:
            results = {}
            tasks = []

            # Prepare parallel tasks for all enabled layers
            if LayerType.M0 in self.enabled_layers:
                tasks.append(self._write_to_m0_layer(data, metadata))

            if LayerType.M1 in self.enabled_layers:
                tasks.append(self._write_to_m1_layer(data, metadata))

            # Execute all layer writes in parallel
            if tasks:
                task_results = await asyncio.gather(*tasks, return_exceptions=True)

                # Process results
                task_index = 0
                if LayerType.M0 in self.enabled_layers:
                    m0_result = task_results[task_index]
                    if isinstance(m0_result, Exception):
                        logger.error(f"M0 write failed: {m0_result}")
                        results['m0'] = []
                    else:
                        results['m0'] = m0_result
                    task_index += 1

                if LayerType.M1 in self.enabled_layers:
                    m1_result = task_results[task_index]
                    if isinstance(m1_result, Exception):
                        logger.error(f"M1 write failed: {m1_result}")
                        results['m1'] = []
                    else:
                        results['m1'] = m1_result

            return results

        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Failed to write to all layers: {e}")
            raise
    
    async def _write_to_m0_layer(self, data: Union[List[ChunkData], ChunkData],
                               metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """Write data directly to M0 layer."""
        return await self.write_to_layer(LayerType.M0, data, metadata)

    async def _write_to_m1_layer(self, data: Union[List[ChunkData], ChunkData],
                               metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """Write data to M1 layer with episode formation from original data."""
        try:
            # Form episodes directly from original data (not from M0)
            m1_data = await self._form_episodes_from_original_data(data, metadata)
            if m1_data:
                result = await self.write_to_layer(LayerType.M1, m1_data, metadata)
                self.stats['episode_formations'] += len(m1_data)
                return result
            return []
        except Exception as e:
            logger.error(f"M1 layer write failed: {e}")
            return []

    async def _form_episodes_from_original_data(self, data: Union[List[ChunkData], ChunkData],
                                              metadata: Optional[Dict[str, Any]] = None) -> List[ChunkData]:
        """Form episodes from original data for M1 storage.

        This method processes original data directly, supporting the parallel
        data->M0 and data->M1 architecture.

        Args:
            data: Original data to form episodes from
            metadata: Optional metadata

        Returns:
            List of ChunkData objects containing formed episodes
        """
        try:
            if not self.episode_former:
                logger.warning("Episode former not available")
                return []

            # Ensure data is a list
            if not isinstance(data, list):
                data = [data]

            episode_chunks = []

            # Use EpisodeFormationProcessor to form episodes
            formation_results = await self.episode_former.form_episodes_batch(data, metadata)

            # Extract episodes from results
            formed_episodes = []
            for result in formation_results:
                if result.success:
                    formed_episodes.extend(result.formed_episodes)

            # Convert formed episodes to ChunkData objects for M1 storage
            for episode in formed_episodes:
                # Handle both dict and FormedEpisode objects
                if isinstance(episode, dict):
                    episode_chunk = ChunkData(
                        content=episode.get('episode_content', ''),
                        metadata={
                            'episode_type': episode.get('episode_type', 'general'),
                            'episode_category': episode.get('episode_category', {}),
                            'confidence': episode.get('confidence', 0.5),
                            'source_chunk_id': episode.get('source_chunk_id'),
                            'entities': episode.get('entities', []),
                            'temporal_info': episode.get('temporal_info', {}),
                            'source_context': episode.get('source_context', ''),
                            **(metadata or {})
                        }
                    )
                else:
                    # FormedEpisode object
                    episode_chunk = ChunkData(
                        content=episode.episode_content,
                        metadata={
                            'episode_type': episode.episode_type,
                            'episode_category': getattr(episode, 'episode_category', {}),
                            'confidence': episode.confidence,
                            'source_chunk_id': getattr(episode, 'source_chunk_id', None),
                            'entities': episode.entities,
                            'temporal_info': episode.temporal_info,
                            'source_context': episode.source_context,
                            **(metadata or {})
                        }
                    )
                episode_chunks.append(episode_chunk)

            logger.debug(f"Formed {len(episode_chunks)} episodes from {len(data)} original chunks")
            return episode_chunks

        except Exception as e:
            logger.error(f"Episode formation from original data failed: {e}")
            return []

    async def query_layer(self, layer_type: LayerType, query: str, top_k: int = 5) -> List[ChunkData]:
        """Query a specific memory layer.
        
        Args:
            layer_type: Layer to query
            query: Query string
            top_k: Maximum number of results
            
        Returns:
            List of matching ChunkData objects
        """
        try:
            if not self.initialized:
                await self.initialize()
            
            store = self.layer_stores.get(layer_type)
            if not store:
                raise ValueError(f"Layer {layer_type.value} not available or not enabled")
            
            results = await store.query(query, top_k)
            
            logger.debug(f"Query returned {len(results)} results from {layer_type.value} layer")
            return results
            
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Failed to query {layer_type.value} layer: {e}")
            raise
    
    async def query_all_layers(self, query: str, top_k: int = 5) -> Dict[str, List[ChunkData]]:
        """Query all enabled memory layers.
        
        Args:
            query: Query string
            top_k: Maximum number of results per layer
            
        Returns:
            Dict mapping layer names to query results
        """
        results = {}
        
        for layer_type in self.enabled_layers:
            try:
                layer_results = await self.query_layer(layer_type, query, top_k)
                results[layer_type.value] = layer_results
            except Exception as e:
                logger.error(f"Query failed for layer {layer_type.value}: {e}")
                results[layer_type.value] = []
        
        return results
    
    async def get_layer_stats(self, layer_type: LayerType) -> Dict[str, Any]:
        """Get statistics for a specific layer.
        
        Args:
            layer_type: Layer to get stats for
            
        Returns:
            Dictionary containing layer statistics
        """
        try:
            store = self.layer_stores.get(layer_type)
            if not store:
                return {'error': f'Layer {layer_type.value} not available'}
            
            # Get store-specific stats
            store_stats = await store.get_processing_stats()
            
            # Add layer-specific stats
            layer_stats = {
                'layer_type': layer_type.value,
                'operations_count': self.stats['layer_operations'][layer_type.value],
                'store_stats': store_stats
            }
            
            return layer_stats
            
        except Exception as e:
            logger.error(f"Failed to get stats for {layer_type.value}: {e}")
            return {'error': str(e)}
    
    async def get_all_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics for all layers.
        
        Returns:
            Dictionary containing all statistics
        """
        try:
            stats = {
                'overall': self.stats.copy(),
                'enabled_layers': [layer.value for layer in self.enabled_layers],
                'layer_stats': {}
            }
            
            # Get stats for each enabled layer
            for layer_type in self.enabled_layers:
                layer_stats = await self.get_layer_stats(layer_type)
                stats['layer_stats'][layer_type.value] = layer_stats
            
            # Add schema information if available
            if self.schema_manager:
                schema_info = await self.schema_manager.get_schema_info()
                stats['schema'] = schema_info
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get comprehensive stats: {e}")
            return {'error': str(e)}
    
    async def cleanup(self):
        """Cleanup all layer stores and resources."""
        try:
            logger.info("Cleaning up MultiTablePgaiStore...")
            
            # Cleanup all layer stores
            for layer_type, store in self.layer_stores.items():
                try:
                    await store.cleanup()
                    logger.info(f"Cleaned up {layer_type.value} layer store")
                except Exception as e:
                    logger.error(f"Error cleaning up {layer_type.value} store: {e}")
            
            self.layer_stores.clear()
            self.initialized = False
            
            logger.info("MultiTablePgaiStore cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    # Convenience methods for backward compatibility
    async def add(self, chunks: List[ChunkData]) -> List[str]:
        """Add chunks to M0 layer (backward compatibility).
        
        Args:
            chunks: List of ChunkData objects
            
        Returns:
            List of created record IDs
        """
        if LayerType.M0 in self.enabled_layers:
            return await self.write_to_layer(LayerType.M0, chunks)
        else:
            raise ValueError("M0 layer not enabled")
    
    async def query(self, query: str, top_k: int = 5) -> List[ChunkData]:
        """Query M0 layer (backward compatibility).
        
        Args:
            query: Query string
            top_k: Maximum number of results
            
        Returns:
            List of matching ChunkData objects
        """
        if LayerType.M0 in self.enabled_layers:
            return await self.query_layer(LayerType.M0, query, top_k)
        else:
            raise ValueError("M0 layer not enabled")