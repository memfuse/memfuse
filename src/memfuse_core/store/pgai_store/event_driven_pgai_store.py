"""
Event-driven pgai store implementation with immediate trigger and retry mechanism.

This module implements an enhanced version of PgaiStore that uses PostgreSQL
NOTIFY/LISTEN for immediate embedding generation instead of polling.
"""

import asyncio
import time
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging

from .pgai_store import PgaiStore
from ..rag.chunk.base import ChunkData

logger = logging.getLogger(__name__)





class RetryManager:
    """Manages embedding generation retry logic."""
    
    def __init__(self, store: 'EventDrivenPgaiStore'):
        self.store = store
        self.max_retries = store.pgai_config.get("max_retries", 3)
        self.retry_interval = store.pgai_config.get("retry_interval", 5.0)
        self.retry_timeout = store.pgai_config.get("retry_timeout", 300)
        
    async def should_retry(self, record_id: str) -> bool:
        """Check if record should be retried."""
        conn = await self.store.pool.connection()
        try:
            cur = await conn.cursor()
            await cur.execute(f"""
                SELECT retry_count, last_retry_at, retry_status
                FROM {self.store.table_name}
                WHERE id = %s
            """, (record_id,))

            result = await cur.fetchone()
            if not result:
                return False

            retry_count, last_retry_at, retry_status = result

            # Check if already failed permanently
            if retry_status == 'failed':
                return False

            # Check if exceeded max retries
            if retry_count >= self.max_retries:
                await self.mark_failed(record_id)
                return False

            # Check if enough time has passed since last retry
            if last_retry_at:
                from datetime import datetime
                time_since_retry = datetime.now() - last_retry_at
                if time_since_retry.total_seconds() < self.retry_interval:
                    return False

            return True
        finally:
            # Close connection if needed
            pass
                
    async def mark_retry(self, record_id: str):
        """Mark a retry attempt."""
        async with self.store.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(f"""
                    UPDATE {self.store.table_name}
                    SET retry_count = retry_count + 1,
                        last_retry_at = CURRENT_TIMESTAMP,
                        retry_status = 'processing'
                    WHERE id = %s
                """, (record_id,))
                
            await conn.commit()
            logger.debug(f"Marked retry for record {record_id}")
            
    async def mark_success(self, record_id: str):
        """Mark successful processing and reset retry state."""
        async with self.store.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(f"""
                    UPDATE {self.store.table_name}
                    SET needs_embedding = FALSE,
                        retry_count = 0,
                        retry_status = 'completed',
                        last_retry_at = NULL
                    WHERE id = %s
                """, (record_id,))
                
            await conn.commit()
            logger.debug(f"Marked success for record {record_id}")
            
    async def mark_failed(self, record_id: str):
        """Mark final failure after max retries."""
        async with self.store.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(f"""
                    UPDATE {self.store.table_name}
                    SET retry_status = 'failed'
                    WHERE id = %s
                """, (record_id,))
                
            await conn.commit()
            logger.warning(f"Marked final failure for record {record_id} after max retries")
            
    async def get_retry_stats(self) -> Dict[str, int]:
        """Get retry statistics."""
        async with self.store.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(f"""
                    SELECT 
                        retry_status,
                        COUNT(*) as count
                    FROM {self.store.table_name}
                    WHERE needs_embedding = TRUE
                    GROUP BY retry_status
                """)
                
                results = await cur.fetchall()
                stats = {status: count for status, count in results}
                
                return {
                    'pending': stats.get('pending', 0),
                    'processing': stats.get('processing', 0),
                    'failed': stats.get('failed', 0),
                    'total_needing_retry': sum(stats.values())
                }


class EventDrivenPgaiStore(PgaiStore):
    """Event-driven pgai store with immediate trigger and retry mechanism."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, table_name: str = "m0_episodic"):
        """Initialize event-driven pgai store."""
        super().__init__(config, table_name)
        
        # Initialize async components
        self.embedding_queue = None
        self.workers = []
        self.notification_listener = None
        self.retry_manager = None

        # Import and initialize metrics
        from .monitoring import EmbeddingMonitor
        self.metrics = EmbeddingMonitor(store_name=f"event_driven_{table_name}")
        
        # Configuration
        self.immediate_trigger = self.pgai_config.get("immediate_trigger", False)
        self.worker_count = self.pgai_config.get("worker_count", 3)
        self.queue_size = self.pgai_config.get("queue_size", 1000)
        self.enable_metrics = self.pgai_config.get("enable_metrics", True)
        
    async def initialize(self) -> bool:
        """Initialize event-driven system."""
        # Initialize parent first
        success = await super().initialize()
        if not success:
            return False
            
        try:
            # Initialize retry manager
            self.retry_manager = RetryManager(self)
            
            # Initialize queue
            self.embedding_queue = asyncio.Queue(maxsize=self.queue_size)
            
            # Start immediate trigger system if enabled
            if self.immediate_trigger:
                await self._start_immediate_trigger_system()
                logger.info(f"Event-driven pgai store initialized with immediate trigger for {self.table_name}")
            else:
                logger.info(f"Event-driven pgai store initialized with polling fallback for {self.table_name}")
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize event-driven system: {e}")
            return False
            
    async def _start_immediate_trigger_system(self):
        """Start the immediate trigger system with workers and listener."""
        # Start worker pool
        for i in range(self.worker_count):
            worker = asyncio.create_task(self._embedding_worker(f"worker-{i}"))
            self.workers.append(worker)
            
        # Start notification listener
        self.notification_listener = asyncio.create_task(self._notification_listener())
        
        # Start metrics collection if enabled
        if self.enable_metrics:
            asyncio.create_task(self._metrics_collector())
            
        logger.info(f"Started {self.worker_count} embedding workers and notification listener")

    async def _notification_listener(self):
        """Listen for database notifications and queue work immediately."""
        channel_name = f"embedding_needed_{self.table_name}"

        try:
            async with self.pool.connection() as conn:
                await conn.execute(f"LISTEN {channel_name}")
                logger.info(f"Started listening for notifications on channel: {channel_name}")

                async for notify in conn.notifies():
                    record_id = notify.payload
                    try:
                        # Add to queue for immediate processing
                        await self.embedding_queue.put(record_id)
                        logger.debug(f"Queued embedding for record {record_id}")

                        # Update metrics
                        if self.enable_metrics:
                            await self.metrics.record_queue_size(self.embedding_queue.qsize())

                    except asyncio.QueueFull:
                        logger.warning(f"Embedding queue full, dropping record {record_id}")

        except Exception as e:
            logger.error(f"Notification listener error: {e}")
            # Restart listener after delay
            await asyncio.sleep(10)
            asyncio.create_task(self._notification_listener())

    async def _embedding_worker(self, worker_name: str):
        """Worker to process embedding queue immediately."""
        logger.info(f"Starting embedding worker: {worker_name}")

        while True:
            try:
                # Get record ID from queue
                record_id = await self.embedding_queue.get()
                start_time = time.time()

                # Check if should retry
                if not await self.retry_manager.should_retry(record_id):
                    self.embedding_queue.task_done()
                    continue

                # Mark retry attempt and start processing
                is_retry = await self._is_retry_attempt(record_id)
                if is_retry:
                    await self.retry_manager.mark_retry(record_id)

                # Start metrics tracking
                if self.enable_metrics:
                    self.metrics.start_processing(record_id, worker_name,
                                                 await self._get_retry_count(record_id))

                # Process single embedding
                success = await self._process_single_embedding(record_id)
                duration = time.time() - start_time

                if success:
                    await self.retry_manager.mark_success(record_id)
                    logger.debug(f"Worker {worker_name} successfully processed {record_id} in {duration:.3f}s")
                else:
                    # Schedule retry if applicable
                    if await self.retry_manager.should_retry(record_id):
                        # Add back to queue after delay
                        retry_interval = self.pgai_config.get("retry_interval", 5.0)
                        asyncio.create_task(self._schedule_retry(record_id, retry_interval))
                        logger.warning(f"Worker {worker_name} failed to process {record_id}, scheduled retry")
                    else:
                        await self.retry_manager.mark_failed(record_id)
                        logger.error(f"Worker {worker_name} permanently failed to process {record_id}")

                # Record metrics
                if self.enable_metrics:
                    self.metrics.complete_processing(record_id, success,
                                                   None if success else "Processing failed")

                self.embedding_queue.task_done()

            except Exception as e:
                logger.error(f"Worker {worker_name} error: {e}")
                await asyncio.sleep(1)

    async def _schedule_retry(self, record_id: str, delay: float):
        """Schedule a retry after specified delay."""
        await asyncio.sleep(delay)
        try:
            await self.embedding_queue.put(record_id)
            logger.debug(f"Scheduled retry for record {record_id} after {delay}s delay")
        except asyncio.QueueFull:
            logger.warning(f"Queue full, could not schedule retry for {record_id}")

    async def _is_retry_attempt(self, record_id: str) -> bool:
        """Check if this is a retry attempt."""
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(f"""
                    SELECT retry_count FROM {self.table_name} WHERE id = %s
                """, (record_id,))

                result = await cur.fetchone()
                return result and result[0] > 0

    async def _get_retry_count(self, record_id: str) -> int:
        """Get current retry count for a record."""
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(f"""
                    SELECT retry_count FROM {self.table_name} WHERE id = %s
                """, (record_id,))

                result = await cur.fetchone()
                return result[0] if result else 0

    async def _process_single_embedding(self, record_id: str) -> bool:
        """Process embedding for a single record."""
        try:
            # Get record content
            async with self.pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(f"""
                        SELECT content FROM {self.table_name}
                        WHERE id = %s AND needs_embedding = TRUE
                    """, (record_id,))

                    result = await cur.fetchone()
                    if not result:
                        logger.warning(f"Record {record_id} not found or doesn't need embedding")
                        return False

                    content = result[0]

            # Generate embedding
            embedding = await self._generate_embedding(content)
            if not embedding:
                logger.error(f"Failed to generate embedding for record {record_id}")
                return False

            # Update record with embedding
            async with self.pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(f"""
                        UPDATE {self.table_name}
                        SET embedding = %s
                        WHERE id = %s
                    """, (embedding, record_id))

                await conn.commit()

            logger.debug(f"Successfully generated embedding for record {record_id}")
            return True

        except Exception as e:
            logger.error(f"Error processing embedding for record {record_id}: {e}")
            return False

    async def _metrics_collector(self):
        """Collect and log metrics periodically."""
        while True:
            try:
                await asyncio.sleep(60)  # Collect metrics every minute

                stats = await self.metrics.get_stats()
                retry_stats = await self.retry_manager.get_retry_stats()

                logger.info(f"Embedding metrics: {stats}")
                logger.info(f"Retry stats: {retry_stats}")

            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(60)

    async def add(self, chunks: List[ChunkData]) -> List[str]:
        """Add chunks with immediate trigger support."""
        if not self.initialized:
            await self.initialize()

        # Use parent implementation for actual insertion
        chunk_ids = await super().add(chunks)

        # If immediate trigger is disabled, fall back to polling
        if not self.immediate_trigger and self.pgai_config.get("use_polling_fallback", True):
            # Parent class will handle polling
            pass

        return chunk_ids

    async def get_processing_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics."""
        if not self.enable_metrics:
            return {"metrics_disabled": True}

        stats = await self.metrics.get_stats()
        retry_stats = await self.retry_manager.get_retry_stats()

        return {
            "processing_stats": stats,
            "retry_stats": retry_stats,
            "queue_size": self.embedding_queue.qsize() if self.embedding_queue else 0,
            "worker_count": len(self.workers),
            "immediate_trigger_enabled": self.immediate_trigger
        }

    async def force_process_pending(self) -> int:
        """Force process all pending embeddings (for testing/debugging)."""
        if not self.immediate_trigger:
            # Fall back to parent polling method
            return await super()._process_pending_embeddings()

        # Get all pending records and add to queue
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(f"""
                    SELECT id FROM {self.table_name}
                    WHERE needs_embedding = TRUE AND retry_status != 'failed'
                """)

                pending_records = await cur.fetchall()

        processed_count = 0
        for record in pending_records:
            record_id = record[0]
            try:
                await self.embedding_queue.put(record_id)
                processed_count += 1
            except asyncio.QueueFull:
                logger.warning(f"Queue full, could not queue {record_id}")
                break

        logger.info(f"Queued {processed_count} pending records for immediate processing")
        return processed_count

    async def cleanup(self):
        """Cleanup resources."""
        logger.info("Cleaning up event-driven pgai store")

        # Cancel workers
        for worker in self.workers:
            worker.cancel()

        # Cancel notification listener
        if self.notification_listener:
            self.notification_listener.cancel()

        # Wait for workers to finish
        if self.workers:
            await asyncio.gather(*self.workers, return_exceptions=True)

        # Clear queue
        if self.embedding_queue:
            while not self.embedding_queue.empty():
                try:
                    self.embedding_queue.get_nowait()
                    self.embedding_queue.task_done()
                except asyncio.QueueEmpty:
                    break

        logger.info("Event-driven pgai store cleanup completed")
