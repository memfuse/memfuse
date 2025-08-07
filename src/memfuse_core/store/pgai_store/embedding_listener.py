"""
Embedding listener for immediate trigger processing.
"""

import asyncio
import os
from typing import Callable, Optional
from loguru import logger


class EmbeddingListener:
    """Listens for PostgreSQL NOTIFY messages and processes embedding requests."""

    def __init__(self, pool, table_name: str, process_callback: Callable):
        self.pool = pool
        self.table_name = table_name
        self.process_callback = process_callback
        self.channel_name = f"embedding_needed_{table_name}"
        self.listener_task: Optional[asyncio.Task] = None
        self.running = False

    async def start(self):
        """Start the embedding listener."""
        # Check if notifications are disabled in test environment
        disable_notifications = os.environ.get("DISABLE_PGAI_NOTIFICATIONS", "false").lower() == "true"
        if disable_notifications:
            logger.info("PGAI notifications disabled in test environment, skipping embedding listener startup")
            return

        if self.running:
            logger.warning(f"Embedding listener for {self.table_name} already running")
            return

        self.running = True
        self.listener_task = asyncio.create_task(self._listen_loop())
        logger.info(f"Started embedding listener for channel: {self.channel_name}")

    async def stop(self):
        """Stop the embedding listener."""
        self.running = False
        if self.listener_task:
            self.listener_task.cancel()
            try:
                await self.listener_task
            except asyncio.CancelledError:
                pass
        logger.info(f"Stopped embedding listener for {self.table_name}")

    async def _listen_loop(self):
        """Main listening loop for NOTIFY messages using async connection context manager."""

        retry_count = 0
        max_retries = 5
        base_delay = 1.0

        # Check if we're in test environment and adjust timeout
        is_test_mode = os.environ.get("MEMFUSE_TEST_MODE", "false").lower() == "true"
        conn_timeout = 5.0 if is_test_mode else 30.0  # Increased timeout for stability
        max_retries = 2 if is_test_mode else 5  # Fewer retries in tests to fail fast

        while self.running and retry_count < max_retries:
            try:
                # Use async connection context manager for proper cleanup
                async with self.pool.connection(timeout=conn_timeout) as conn:
                    await conn.execute(f"LISTEN {self.channel_name}")
                    logger.info(f"Started listening on channel: {self.channel_name}")

                    # Reset retry count on successful connection
                    retry_count = 0

                    # Set autocommit for better notification handling
                    try:
                        await conn.rollback()  # Clear any existing transaction
                        await conn.set_autocommit(True)
                    except Exception as e:
                        logger.debug(f"Could not set autocommit: {e}")
                        # Continue without autocommit - will still work but may be less responsive

                    # Listen for notifications
                    while self.running:
                        try:
                            # Wait for notifications
                            await asyncio.sleep(0.5)  # Check every 500ms

                            # Get notifications using the correct async API
                            try:
                                notifications = await conn.notifies()
                                for notify in notifications:
                                    if notify.channel == self.channel_name:
                                        record_id = notify.payload
                                        logger.debug(f"Received embedding notification for record: {record_id}")

                                        # Process the embedding asynchronously
                                        asyncio.create_task(self._safe_process_embedding(record_id))
                            except Exception as notify_error:
                                logger.debug(f"Error getting notifications: {notify_error}")
                                # Alternative approach for older psycopg versions
                                try:
                                    if hasattr(conn, 'get_notifies'):
                                        notifications = conn.get_notifies()
                                        for notify in notifications:
                                            if notify.channel == self.channel_name:
                                                record_id = notify.payload
                                                logger.debug(f"Received embedding notification for record: {record_id}")

                                                # Process the embedding asynchronously
                                                asyncio.create_task(self._safe_process_embedding(record_id))
                                except Exception as alt_error:
                                    logger.debug(f"Alternative notification method also failed: {alt_error}")
                        except Exception as notify_error:
                            logger.warning(f"Notification iteration error: {notify_error}")
                            break

            except asyncio.CancelledError:
                logger.info("Embedding listener cancelled")
                break
            except asyncio.TimeoutError:
                retry_count += 1
                delay = base_delay * (2 ** min(retry_count - 1, 4))  # Exponential backoff
                logger.warning(f"Connection pool timeout (attempt {retry_count}/{max_retries}) - pool may be exhausted")

                if retry_count < max_retries:
                    logger.info(f"Retrying embedding listener in {delay}s...")
                    await asyncio.sleep(delay)
                else:
                    logger.error("Max retries exceeded for connection timeout, embedding listener stopping")
                    break
            except Exception as e:
                retry_count += 1
                delay = base_delay * (2 ** min(retry_count - 1, 4))  # Exponential backoff
                logger.error(f"Embedding listener error (attempt {retry_count}/{max_retries}): {e}")

                if retry_count < max_retries:
                    logger.info(f"Retrying embedding listener in {delay}s...")
                    await asyncio.sleep(delay)
                else:
                    logger.error("Max retries exceeded, embedding listener stopping")
                    break

        logger.info("Embedding listener stopped")
                    
    async def _safe_process_embedding(self, record_id: str):
        """Safely process embedding with error handling."""
        try:
            success = await self.process_callback(record_id)
            if success:
                logger.debug(f"Successfully processed embedding for {record_id}")
            else:
                logger.warning(f"Failed to process embedding for {record_id}")
        except Exception as e:
            logger.error(f"Error processing embedding for {record_id}: {e}")


class EmbeddingProcessor:
    """Fallback polling-based embedding processor."""
    
    def __init__(self, pool, table_name: str, generate_embedding_func: Callable):
        self.pool = pool
        self.table_name = table_name
        self.generate_embedding_func = generate_embedding_func
        self.processor_task: Optional[asyncio.Task] = None
        self.running = False
        self.poll_interval = 5.0  # Poll every 5 seconds
        
    async def start(self):
        """Start the polling processor."""
        if self.running:
            logger.warning(f"Embedding processor for {self.table_name} already running")
            return
            
        self.running = True
        self.processor_task = asyncio.create_task(self._poll_loop())
        logger.info(f"Started polling embedding processor for {self.table_name}")
        
    async def stop(self):
        """Stop the polling processor."""
        self.running = False
        if self.processor_task:
            self.processor_task.cancel()
            try:
                await self.processor_task
            except asyncio.CancelledError:
                pass
        logger.info(f"Stopped embedding processor for {self.table_name}")
        
    async def _poll_loop(self):
        """Main polling loop for pending embeddings."""
        while self.running:
            try:
                await self._process_pending_embeddings()
                await asyncio.sleep(self.poll_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in embedding processor loop: {e}")
                await asyncio.sleep(self.poll_interval)
                
    async def _process_pending_embeddings(self):
        """Process all pending embeddings."""
        try:
            async with self.pool.connection() as conn:
                async with conn.cursor() as cur:
                    # Get pending records
                    await cur.execute(f"""
                        SELECT id, content FROM {self.table_name}
                        WHERE needs_embedding = TRUE AND retry_status != 'failed'
                        LIMIT 10
                    """)
                    
                    pending_records = await cur.fetchall()
                    
                    for record_id, content in pending_records:
                        try:
                            # Generate embedding
                            embedding = await self.generate_embedding_func(content)
                            
                            if embedding:
                                # Update record
                                await cur.execute(f"""
                                    UPDATE {self.table_name}
                                    SET embedding = %s, needs_embedding = FALSE, retry_status = 'completed'
                                    WHERE id = %s
                                """, (embedding, record_id))
                                
                                logger.debug(f"Generated embedding for {record_id}")
                            else:
                                # Mark as failed
                                await cur.execute(f"""
                                    UPDATE {self.table_name}
                                    SET retry_status = 'failed'
                                    WHERE id = %s
                                """, (record_id,))
                                
                        except Exception as e:
                            logger.error(f"Failed to process embedding for {record_id}: {e}")
                    
                    if pending_records:
                        await conn.commit()
                        logger.debug(f"Processed {len(pending_records)} pending embeddings")
                        
        except Exception as e:
            logger.error(f"Error processing pending embeddings: {e}")