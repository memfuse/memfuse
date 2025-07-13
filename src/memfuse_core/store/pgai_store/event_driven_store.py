"""
Event-driven pgai store implementation with immediate triggers.

This implementation uses composition instead of inheritance and provides
real-time embedding generation through PostgreSQL NOTIFY/LISTEN mechanism.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional

from .pgai_store import PgaiStore
from .immediate_trigger_components import ImmediateTriggerCoordinator
from .error_handling import (
    initialize_error_handling, cleanup_error_handling,
    health_checker
)
from ...rag.chunk.base import ChunkData

logger = logging.getLogger(__name__)


class EventDrivenPgaiStore:
    """Event-driven pgai store using composition and immediate triggers."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, table_name: str = "m0_episodic"):
        """Initialize event-driven store with immediate triggers."""
        self.config = config or {}
        self.table_name = table_name
        
        # Get pgai configuration from multiple sources
        self.pgai_config = self.config.get("pgai", {})

        # If no pgai config in direct config, try to get from global config
        if not self.pgai_config:
            try:
                from ...utils.config import config_manager
                full_config = config_manager.get_config()
                if full_config:
                    # Try store.pgai first, then database.pgai for backward compatibility
                    self.pgai_config = full_config.get("store", {}).get("pgai", {})
                    if not self.pgai_config:
                        self.pgai_config = full_config.get("database", {}).get("pgai", {})
            except Exception as e:
                logger.warning(f"Failed to load global pgai config: {e}")
                self.pgai_config = {}
        
        # Core store for basic operations
        self.core_store = None
        
        # Immediate trigger coordinator
        self.coordinator = None
        
        # State
        self.initialized = False
        
    async def initialize(self) -> bool:
        """Initialize the store and immediate trigger system."""
        try:
            # Initialize core store first
            self.core_store = PgaiStore(self.config, self.table_name)
            success = await self.core_store.initialize()
            
            if not success:
                logger.error("Failed to initialize core pgai store")
                return False
            
            # Initialize error handling system
            initialize_error_handling()

            # Check if immediate trigger is enabled
            if self.pgai_config.get("immediate_trigger", False):
                # Initialize immediate trigger coordinator
                self.coordinator = ImmediateTriggerCoordinator(
                    self.core_store.pool,
                    self.table_name,
                    self.pgai_config
                )

                # Initialize coordinator with embedding processor
                await self.coordinator.initialize(self._process_embedding)
                logger.info(f"Initialized immediate trigger system for {self.table_name}")
            else:
                logger.info(f"Immediate trigger disabled, using traditional polling for {self.table_name}")

            self.initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize EventDrivenPgaiStore: {e}")
            return False
    
    async def _ensure_initialized(self, operation: str) -> bool:
        """Ensure store is initialized, return True if successful."""
        if not self.initialized:
            success = await self.initialize()
            if not success:
                logger.error(f"EventDrivenPgaiStore: Failed to initialize for {operation}")
                return False

        if self.core_store is None:
            logger.error(f"EventDrivenPgaiStore: core_store is None for {operation}")
            return False

        return True

    async def add(self, chunks: List[ChunkData]) -> List[str]:
        """Add chunks to the store."""
        if not await self._ensure_initialized("add"):
            return []

        # Use core store for actual insertion
        return await self.core_store.add(chunks)

    async def query(self, query, top_k: int = 5) -> List[ChunkData]:
        """Query: Semantic search for relevant chunks based on query text."""
        if not await self._ensure_initialized("query"):
            return []

        return await self.core_store.query(query, top_k)
    
    async def get_processing_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics."""
        stats = {
            "immediate_trigger_enabled": self.coordinator is not None,
            "table_name": self.table_name,
            "initialized": self.initialized
        }

        if self.coordinator:
            coordinator_stats = await self.coordinator.get_stats()
            stats.update(coordinator_stats)

        # Add error handling stats
        health_status = await health_checker.check_health()
        performance_stats = resource_monitor.get_performance_stats()

        stats.update({
            "health": health_status,
            "performance": performance_stats
        })

        return stats

    async def get_health_status(self) -> Dict[str, Any]:
        """Get detailed health status."""
        return await health_checker.check_health()
    
    async def force_process_pending(self) -> int:
        """Force process all pending embeddings."""
        if not self.initialized:
            await self.initialize()
        
        if self.coordinator:
            # For immediate trigger mode, get pending records and queue them
            return await self._queue_pending_records()
        else:
            # Fall back to core store polling
            return await self.core_store._process_pending_embeddings()
    
    async def cleanup(self):
        """Cleanup resources with enhanced error handling."""
        try:
            if self.coordinator:
                await self.coordinator.cleanup()

            if self.core_store:
                # Core store cleanup if needed
                pass

            # Cleanup error handling system
            await cleanup_error_handling()

            self.initialized = False
            logger.info("EventDrivenPgaiStore cleaned up successfully")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            # Still mark as uninitialized even if cleanup failed
            self.initialized = False
    
    async def _process_embedding(self, record_id: str) -> bool:
        """Process embedding for a single record."""
        try:
            # Get record content
            conn = await self.core_store.pool.connection()
            try:
                cur = await conn.cursor()
                await cur.execute(f"""
                    SELECT content FROM {self.table_name}
                    WHERE id = %s AND needs_embedding = TRUE
                """, (record_id,))
                
                result = await cur.fetchone()
                if not result:
                    logger.warning(f"Record {record_id} not found or doesn't need embedding")
                    return False
                
                content = result[0]
            finally:
                pass
            
            # Generate embedding using core store's method
            embedding = await self.core_store._generate_embedding(content)
            if not embedding:
                logger.error(f"Failed to generate embedding for record {record_id}")
                return False
            
            # Update record with embedding
            conn = await self.core_store.pool.connection()
            try:
                cur = await conn.cursor()
                await cur.execute(f"""
                    UPDATE {self.table_name}
                    SET embedding = %s
                    WHERE id = %s
                """, (embedding, record_id))
                
                await conn.commit()
            finally:
                pass
            
            logger.debug(f"Successfully generated embedding for record {record_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing embedding for record {record_id}: {e}")
            return False
    
    async def _queue_pending_records(self) -> int:
        """Queue all pending records for immediate processing."""
        conn = await self.core_store.pool.connection()
        try:
            cur = await conn.cursor()
            await cur.execute(f"""
                SELECT id FROM {self.table_name}
                WHERE needs_embedding = TRUE AND retry_status != 'failed'
            """)
            
            pending_records = await cur.fetchall()
        finally:
            pass
        
        processed_count = 0
        for record in pending_records:
            record_id = record[0]
            try:
                await self.coordinator.queue.put(record_id)
                processed_count += 1
            except asyncio.QueueFull:
                logger.warning(f"Queue full, could not queue {record_id}")
                break
        
        logger.info(f"Queued {processed_count} pending records for immediate processing")
        return processed_count
    
    # Delegate other methods to core store
    async def get(self, chunk_id: str) -> Optional[ChunkData]:
        """Get a chunk by ID."""
        if not await self._ensure_initialized("get"):
            return None
        return await self.core_store.get(chunk_id)

    async def delete(self, chunk_ids: List[str]) -> bool:
        """Delete chunks by IDs."""
        if not await self._ensure_initialized("delete"):
            return False
        return await self.core_store.delete(chunk_ids)

    async def update(self, chunks: List[ChunkData]) -> bool:
        """Update existing chunks."""
        if not await self._ensure_initialized("update"):
            return False
        return await self.core_store.update(chunks)

    async def list_chunks(self, limit: int = 100, offset: int = 0) -> List[ChunkData]:
        """List chunks with pagination."""
        if not await self._ensure_initialized("list_chunks"):
            return []
        return await self.core_store.list_chunks(limit, offset)

    async def count(self) -> int:
        """Get total number of chunks."""
        if not await self._ensure_initialized("count"):
            return 0
        return await self.core_store.count()
    
    # Properties for compatibility
    @property
    def pool(self):
        """Get database pool."""
        return self.core_store.pool if self.core_store else None
    
    @property
    def encoder(self):
        """Get encoder."""
        return self.core_store.encoder if self.core_store else None
    
    @encoder.setter
    def encoder(self, value):
        """Set encoder."""
        if self.core_store:
            self.core_store.encoder = value


