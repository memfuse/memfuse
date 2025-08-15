"""
Immediate trigger components for pgai embedding processing.

This module contains the core components for the immediate trigger system,
separated by concern for better maintainability.
"""

import asyncio
import time
from loguru import logger
from typing import Dict, Any, Optional, List
from datetime import datetime


class TriggerManager:
    """Manages PostgreSQL NOTIFY/LISTEN for immediate triggers."""
    
    def __init__(self, pool, table_name: str, queue: asyncio.Queue):
        self.pool = pool
        self.table_name = table_name
        self.queue = queue
        self.listener_task = None
        self.channel_name = f"embedding_needed_{table_name}"
        
    async def setup_triggers(self):
        """Setup database triggers for immediate notification."""
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                # Create notification function
                # Determine the primary key field based on table name
                pk_field = "message_id" if self.table_name == "m0_raw" else "id"

                await cur.execute(f"""
                    CREATE OR REPLACE FUNCTION notify_embedding_needed_{self.table_name}()
                    RETURNS TRIGGER AS $$
                    BEGIN
                        PERFORM pg_notify('{self.channel_name}', NEW.{pk_field}::text);
                        RETURN NEW;
                    END;
                    $$ LANGUAGE plpgsql;
                """)

                # Create trigger
                await cur.execute(f"""
                    DROP TRIGGER IF EXISTS trigger_immediate_embedding_{self.table_name} ON {self.table_name};
                    CREATE TRIGGER trigger_immediate_embedding_{self.table_name}
                        AFTER INSERT ON {self.table_name}
                        FOR EACH ROW
                        WHEN (NEW.needs_embedding = TRUE AND NEW.content IS NOT NULL)
                        EXECUTE FUNCTION notify_embedding_needed_{self.table_name}();
                """)

            await conn.commit()
            logger.info(f"Setup immediate triggers for {self.table_name}")
            
    async def start_listening(self):
        """Start listening for notifications."""
        import os
        
        # Check if notifications are disabled in test environment
        disable_notifications = os.environ.get("DISABLE_PGAI_NOTIFICATIONS", "false").lower() == "true"
        if disable_notifications:
            logger.info("PGAI notifications disabled in test environment, skipping listener startup")
            return
            
        self.listener_task = asyncio.create_task(self._notification_listener())
        # Give the listener a moment to start up, but don't wait indefinitely
        await asyncio.sleep(0.1)
        
    async def stop_listening(self):
        """Stop listening for notifications."""
        if self.listener_task:
            self.listener_task.cancel()
            try:
                await self.listener_task
            except asyncio.CancelledError:
                pass
                
    async def _notification_listener(self):
        """Listen for database notifications with robust error handling."""
        import os
        
        retry_count = 0
        max_retries = 5
        base_delay = 1.0
        
        # Check if we're in test environment and adjust timeout
        is_test_mode = os.environ.get("MEMFUSE_TEST_MODE", "false").lower() == "true"
        conn_timeout = 5.0 if is_test_mode else 10.0  # Even shorter timeout for tests
        max_retries = 2 if is_test_mode else 5  # Fewer retries in tests to fail fast

        while retry_count < max_retries and not self.listener_task.cancelled():
            conn = None
            try:
                # Use connection context manager for proper cleanup
                async with self.pool.connection(timeout=conn_timeout) as conn:
                    await conn.execute(f"LISTEN {self.channel_name}")
                    logger.info(f"Started listening on channel: {self.channel_name}")

                    # Reset retry count on successful connection
                    retry_count = 0

                    # Set autocommit for better notification handling
                    # Need to ensure we're not in a transaction before setting autocommit
                    try:
                        await conn.rollback()  # Clear any existing transaction
                        await conn.set_autocommit(True)
                    except Exception as e:
                        logger.debug(f"Could not set autocommit: {e}")
                        # Continue without autocommit - will still work but may be less responsive
                    
                    # Listen for notifications
                    while not self.listener_task.cancelled():
                        try:
                            # Wait for notifications
                            await asyncio.sleep(0.5)  # Check every 500ms
                            
                            # Get notifications using the correct async API
                            try:
                                notifications = await conn.notifies()
                                for notify in notifications:
                                    if notify.channel == self.channel_name:
                                        record_id = notify.payload
                                        try:
                                            await self.queue.put(record_id)
                                            logger.debug(f"Queued embedding for record {record_id}")
                                        except asyncio.QueueFull:
                                            logger.warning(f"Queue full, dropping record {record_id}")
                            except Exception as notify_error:
                                logger.debug(f"Error getting notifications: {notify_error}")
                                # Alternative approach for older psycopg versions
                                try:
                                    if hasattr(conn, 'get_notifies'):
                                        notifications = conn.get_notifies()
                                        for notify in notifications:
                                            if notify.channel == self.channel_name:
                                                record_id = notify.payload
                                                try:
                                                    await self.queue.put(record_id)
                                                    logger.debug(f"Queued embedding for record {record_id}")
                                                except asyncio.QueueFull:
                                                    logger.warning(f"Queue full, dropping record {record_id}")
                                except Exception as alt_error:
                                    logger.debug(f"Alternative notification method also failed: {alt_error}")
                        except Exception as notify_error:
                            logger.warning(f"Notification iteration error: {notify_error}")
                            break

            except asyncio.CancelledError:
                logger.info("Notification listener cancelled")
                break
            except asyncio.TimeoutError:
                retry_count += 1
                delay = base_delay * (2 ** min(retry_count - 1, 4))  # Exponential backoff
                logger.warning(f"Connection pool timeout (attempt {retry_count}/{max_retries}) - pool may be exhausted")

                if retry_count < max_retries:
                    logger.info(f"Retrying notification listener in {delay}s...")
                    await asyncio.sleep(delay)
                else:
                    logger.error("Max retries exceeded for connection timeout, notification listener stopping")
                    break
            except Exception as e:
                retry_count += 1
                delay = base_delay * (2 ** min(retry_count - 1, 4))  # Exponential backoff
                logger.error(f"Notification listener error (attempt {retry_count}/{max_retries}): {e}")

                if retry_count < max_retries:
                    logger.info(f"Retrying notification listener in {delay}s...")
                    await asyncio.sleep(delay)
                else:
                    logger.error("Max retries exceeded, notification listener stopping")
                    break

        logger.info("Notification listener stopped")


class RetryProcessor:
    """Handles retry logic for failed embeddings."""
    
    def __init__(self, pool, table_name: str, max_retries: int = 3, retry_interval: float = 5.0):
        self.pool = pool
        self.table_name = table_name
        self.max_retries = max_retries
        self.retry_interval = retry_interval
        
    async def should_retry(self, record_id: str) -> bool:
        """Check if record should be retried."""
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                # Determine the primary key field based on table name
                pk_field = "message_id" if self.table_name == "m0_raw" else "id"
                await cur.execute(f"""
                    SELECT retry_count, last_retry_at, retry_status
                    FROM {self.table_name}
                    WHERE {pk_field} = %s
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
                    time_since_retry = datetime.now() - last_retry_at
                    if time_since_retry.total_seconds() < self.retry_interval:
                        return False

                return True
            
    async def mark_retry(self, record_id: str):
        """Mark a retry attempt."""
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                # Determine the primary key field based on table name
                pk_field = "message_id" if self.table_name == "m0_raw" else "id"
                await cur.execute(f"""
                    UPDATE {self.table_name}
                    SET retry_count = retry_count + 1,
                        last_retry_at = CURRENT_TIMESTAMP,
                        retry_status = 'processing'
                    WHERE {pk_field} = %s
                """, (record_id,))
            await conn.commit()

    async def mark_success(self, record_id: str):
        """Mark successful processing."""
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                # Determine the primary key field based on table name
                pk_field = "message_id" if self.table_name == "m0_raw" else "id"
                await cur.execute(f"""
                    UPDATE {self.table_name}
                    SET needs_embedding = FALSE,
                        retry_count = 0,
                        retry_status = 'completed',
                        last_retry_at = NULL
                    WHERE {pk_field} = %s
                """, (record_id,))
            await conn.commit()

    async def mark_failed(self, record_id: str):
        """Mark final failure."""
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                # Determine the primary key field based on table name
                pk_field = "message_id" if self.table_name == "m0_raw" else "id"
                await cur.execute(f"""
                    UPDATE {self.table_name}
                    SET retry_status = 'failed'
                    WHERE {pk_field} = %s
                """, (record_id,))
            await conn.commit()


class WorkerPool:
    """Manages async workers for embedding processing."""
    
    def __init__(self, queue: asyncio.Queue, worker_count: int = 3):
        self.queue = queue
        self.worker_count = worker_count
        self.workers = []
        self.running = False
        
    async def start(self, process_func):
        """Start worker pool."""
        self.running = True
        for i in range(self.worker_count):
            worker = asyncio.create_task(self._worker(f"worker-{i}", process_func))
            self.workers.append(worker)
        logger.info(f"Started {self.worker_count} workers")
        
    async def stop(self):
        """Stop worker pool."""
        self.running = False
        for worker in self.workers:
            worker.cancel()
            
        if self.workers:
            await asyncio.gather(*self.workers, return_exceptions=True)
            
        self.workers.clear()
        logger.info("Stopped worker pool")
        
    async def _worker(self, worker_name: str, process_func):
        """Individual worker process."""
        logger.info(f"Starting worker: {worker_name}")
        
        while self.running:
            try:
                # Get record ID from queue with timeout
                record_id = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                
                # Process the record
                await process_func(record_id, worker_name)
                
                # Mark task as done
                self.queue.task_done()
                
            except asyncio.TimeoutError:
                # No work available, continue
                continue
            except asyncio.CancelledError:
                # Worker cancelled
                break
            except Exception as e:
                logger.error(f"Worker {worker_name} error: {e}")
                await asyncio.sleep(1)
                
        logger.info(f"Worker {worker_name} stopped")


class ImmediateTriggerCoordinator:
    """Coordinates all immediate trigger components."""
    
    def __init__(self, pool, table_name: str, config: Dict[str, Any]):
        self.pool = pool
        self.table_name = table_name
        self.config = config
        
        # Initialize queue
        queue_size = config.get("queue_size", 1000)
        self.queue = asyncio.Queue(maxsize=queue_size)
        
        # Initialize components
        self.trigger_manager = TriggerManager(pool, table_name, self.queue)
        self.retry_processor = RetryProcessor(
            pool, table_name,
            max_retries=config.get("max_retries", 3),
            retry_interval=config.get("retry_interval", 5.0)
        )
        self.worker_pool = WorkerPool(
            self.queue,
            worker_count=config.get("worker_count", 3)
        )
        
        # Monitoring
        if config.get("enable_metrics", True):
            from .monitoring import EmbeddingMonitor
            self.monitor = EmbeddingMonitor(f"immediate_trigger_{table_name}")
        else:
            self.monitor = None
            
    async def initialize(self, embedding_processor):
        """Initialize all components."""
        self.embedding_processor = embedding_processor
        
        # Setup database triggers
        await self.trigger_manager.setup_triggers()
        
        # Start listening for notifications
        await self.trigger_manager.start_listening()
        
        # Start worker pool
        await self.worker_pool.start(self._process_record)
        
        logger.info(f"Immediate trigger coordinator initialized for {self.table_name}")
        
    async def cleanup(self):
        """Cleanup all components."""
        await self.trigger_manager.stop_listening()
        await self.worker_pool.stop()
        
        # Clear queue
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
                self.queue.task_done()
            except asyncio.QueueEmpty:
                break
                
        logger.info(f"Immediate trigger coordinator cleaned up for {self.table_name}")
        
    async def _process_record(self, record_id: str, worker_name: str):
        """Process a single record."""
        start_time = time.time()
        
        try:
            # Check if should retry
            if not await self.retry_processor.should_retry(record_id):
                return
                
            # Get retry count for monitoring
            retry_count = await self._get_retry_count(record_id)
            is_retry = retry_count > 0
            
            # Start monitoring
            if self.monitor:
                self.monitor.start_processing(record_id, worker_name, retry_count)
                
            # Mark retry if needed
            if is_retry:
                await self.retry_processor.mark_retry(record_id)
                
            # Process embedding
            success = await self.embedding_processor(record_id)
            
            if success:
                await self.retry_processor.mark_success(record_id)
                logger.debug(f"Worker {worker_name} successfully processed {record_id}")
            else:
                # Will be retried if within limits
                logger.warning(f"Worker {worker_name} failed to process {record_id}")
                
            # Complete monitoring
            if self.monitor:
                error_msg = None if success else "Processing failed"
                self.monitor.complete_processing(record_id, success, error_msg)
                
        except Exception as e:
            logger.error(f"Error processing record {record_id}: {e}")
            if self.monitor:
                self.monitor.complete_processing(record_id, False, str(e))
                
    async def _get_retry_count(self, record_id: str) -> int:
        """Get current retry count for a record."""
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                # Determine the primary key field based on table name
                pk_field = "message_id" if self.table_name == "m0_raw" else "id"
                await cur.execute(f"""
                    SELECT retry_count FROM {self.table_name} WHERE {pk_field} = %s
                """, (record_id,))

                result = await cur.fetchone()
                return result[0] if result else 0
            
    async def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        stats = {
            "queue_size": self.queue.qsize(),
            "worker_count": len(self.worker_pool.workers),
            "table_name": self.table_name
        }
        
        if self.monitor:
            monitor_stats = self.monitor.get_current_stats()
            stats.update(monitor_stats)
            
        return stats
