"""
Test cases for event-driven pgai store with immediate trigger and retry mechanism.

This module contains comprehensive tests for the new async trigger functionality,
including normal flow, retry mechanism, and error handling scenarios.
"""

import pytest
import asyncio
import time
import os
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any

from memfuse_core.store.pgai_store import EventDrivenPgaiStore, PgaiStoreFactory
from memfuse_core.rag.chunk.base import ChunkData


class TestEventDrivenPgaiStore:
    """Test cases for EventDrivenPgaiStore."""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing."""
        return {
            "database": {
                "postgres": {
                    "host": "localhost",
                    "port": 5432,
                    "database": "test_memfuse",
                    "user": "postgres",
                    "password": "postgres"
                },
                "pgai": {
                    "enabled": True,
                    "auto_embedding": True,
                    "immediate_trigger": True,
                    "max_retries": 3,
                    "retry_interval": 1.0,  # Faster for testing
                    "worker_count": 2,
                    "queue_size": 100,
                    "enable_metrics": True
                }
            }
        }
    
    @pytest.fixture
    def mock_pool(self):
        """Mock database connection pool."""
        pool = AsyncMock()
        conn = AsyncMock()
        cursor = AsyncMock()
        
        pool.connection.return_value.__aenter__.return_value = conn
        conn.cursor.return_value.__aenter__.return_value = cursor
        conn.commit = AsyncMock()
        
        return pool, conn, cursor
    
    @pytest.fixture
    async def store(self, mock_config, mock_pool):
        """Create test store instance."""
        pool, conn, cursor = mock_pool
        
        store = EventDrivenPgaiStore(config=mock_config["database"])
        store.pool = pool
        store.initialized = True
        
        # Mock encoder
        store.encoder = AsyncMock()
        store.encoder.encode.return_value = [0.1] * 384
        
        return store
    
    @pytest.mark.asyncio
    async def test_initialization(self, mock_config):
        """Test store initialization with immediate trigger."""
        with patch('memfuse_core.store.event_driven_pgai_store.AsyncConnectionPool'):
            store = EventDrivenPgaiStore(config=mock_config["database"])
            
            # Mock the parent initialization
            with patch.object(store, '_setup_schema', new_callable=AsyncMock):
                with patch.object(store, '_setup_immediate_trigger', new_callable=AsyncMock):
                    with patch.object(store, '_start_immediate_trigger_system', new_callable=AsyncMock):
                        result = await store.initialize()
                        
                        assert result is True
                        assert store.immediate_trigger is True
                        assert store.worker_count == 2
                        assert store.queue_size == 100
                        assert isinstance(store.retry_manager, RetryManager)
                        assert isinstance(store.metrics, EmbeddingMetrics)
    
    @pytest.mark.asyncio
    async def test_notification_listener(self, store, mock_pool):
        """Test notification listener functionality."""
        pool, conn, cursor = mock_pool
        
        # Mock notifications
        mock_notifications = [
            MagicMock(payload="record1"),
            MagicMock(payload="record2")
        ]
        conn.notifies.return_value.__aiter__.return_value = mock_notifications
        
        # Start listener
        listener_task = asyncio.create_task(store._notification_listener())
        
        # Give it time to process
        await asyncio.sleep(0.1)
        
        # Cancel the task
        listener_task.cancel()
        
        # Verify LISTEN command was executed
        conn.execute.assert_called_with("LISTEN embedding_needed_m0_messages")
    
    @pytest.mark.asyncio
    async def test_embedding_worker(self, store, mock_pool):
        """Test embedding worker processing."""
        pool, conn, cursor = mock_pool
        
        # Setup queue with test record
        store.embedding_queue = asyncio.Queue()
        await store.embedding_queue.put("test_record_1")
        
        # Mock retry manager
        store.retry_manager.should_retry = AsyncMock(return_value=True)
        store.retry_manager.mark_success = AsyncMock()
        
        # Mock successful processing
        store._is_retry_attempt = AsyncMock(return_value=False)
        store._process_single_embedding = AsyncMock(return_value=True)
        
        # Start worker
        worker_task = asyncio.create_task(store._embedding_worker("test-worker"))
        
        # Give it time to process
        await asyncio.sleep(0.1)
        
        # Cancel the task
        worker_task.cancel()
        
        # Verify processing was called
        store._process_single_embedding.assert_called_with("test_record_1")
        store.retry_manager.mark_success.assert_called_with("test_record_1")
    
    @pytest.mark.asyncio
    async def test_retry_mechanism(self, store, mock_pool):
        """Test retry mechanism for failed embeddings."""
        pool, conn, cursor = mock_pool
        
        # Setup queue
        store.embedding_queue = asyncio.Queue()
        await store.embedding_queue.put("failing_record")
        
        # Mock retry manager to allow retries
        store.retry_manager.should_retry = AsyncMock(side_effect=[True, True, False])
        store.retry_manager.mark_retry = AsyncMock()
        store.retry_manager.mark_failed = AsyncMock()
        
        # Mock failed processing
        store._is_retry_attempt = AsyncMock(return_value=True)
        store._process_single_embedding = AsyncMock(return_value=False)
        
        # Start worker
        worker_task = asyncio.create_task(store._embedding_worker("test-worker"))
        
        # Give it time to process
        await asyncio.sleep(0.1)
        
        # Cancel the task
        worker_task.cancel()
        
        # Verify retry was attempted
        store.retry_manager.mark_retry.assert_called()
    
    @pytest.mark.asyncio
    async def test_process_single_embedding_success(self, store, mock_pool):
        """Test successful single embedding processing."""
        pool, conn, cursor = mock_pool
        
        # Mock database responses
        cursor.fetchone.return_value = ("Test content for embedding",)
        
        # Test processing
        result = await store._process_single_embedding("test_record")
        
        assert result is True
        
        # Verify database operations
        assert cursor.execute.call_count == 2  # SELECT and UPDATE
        conn.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_single_embedding_failure(self, store, mock_pool):
        """Test failed single embedding processing."""
        pool, conn, cursor = mock_pool
        
        # Mock database failure
        cursor.fetchone.return_value = None
        
        # Test processing
        result = await store._process_single_embedding("nonexistent_record")
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_schedule_retry(self, store):
        """Test retry scheduling mechanism."""
        store.embedding_queue = asyncio.Queue(maxsize=10)
        
        # Schedule retry with short delay
        retry_task = asyncio.create_task(store._schedule_retry("test_record", 0.1))
        
        # Wait for retry to be scheduled
        await asyncio.sleep(0.2)
        
        # Verify record was added back to queue
        assert store.embedding_queue.qsize() == 1
        record_id = await store.embedding_queue.get()
        assert record_id == "test_record"
    
    @pytest.mark.asyncio
    async def test_add_chunks_with_immediate_trigger(self, store, mock_pool):
        """Test adding chunks with immediate trigger enabled."""
        pool, conn, cursor = mock_pool
        
        # Mock parent add method
        with patch.object(store.__class__.__bases__[0], 'add', new_callable=AsyncMock) as mock_add:
            mock_add.return_value = ["chunk1", "chunk2"]
            
            chunks = [
                ChunkData(chunk_id="chunk1", content="Test content 1"),
                ChunkData(chunk_id="chunk2", content="Test content 2")
            ]
            
            result = await store.add(chunks)
            
            assert result == ["chunk1", "chunk2"]
            mock_add.assert_called_once_with(chunks)
    
    @pytest.mark.asyncio
    async def test_get_processing_stats(self, store):
        """Test processing statistics retrieval."""
        # Mock metrics and retry manager
        store.metrics.get_stats = AsyncMock(return_value={"total_processed": 10})
        store.retry_manager.get_retry_stats = AsyncMock(return_value={"pending": 2})
        store.embedding_queue = asyncio.Queue()
        
        stats = await store.get_processing_stats()
        
        assert "processing_stats" in stats
        assert "retry_stats" in stats
        assert "queue_size" in stats
        assert "worker_count" in stats
        assert "immediate_trigger_enabled" in stats
        assert stats["immediate_trigger_enabled"] is True
    
    @pytest.mark.asyncio
    async def test_force_process_pending(self, store, mock_pool):
        """Test forcing processing of pending records."""
        pool, conn, cursor = mock_pool
        
        # Mock pending records
        cursor.fetchall.return_value = [("record1",), ("record2",), ("record3",)]
        store.embedding_queue = asyncio.Queue(maxsize=10)
        
        result = await store.force_process_pending()
        
        assert result == 3
        assert store.embedding_queue.qsize() == 3
    
    @pytest.mark.asyncio
    async def test_cleanup(self, store):
        """Test resource cleanup."""
        # Mock workers and listener
        worker1 = AsyncMock()
        worker2 = AsyncMock()
        store.workers = [worker1, worker2]
        store.notification_listener = AsyncMock()
        store.embedding_queue = asyncio.Queue()
        
        # Add some items to queue
        await store.embedding_queue.put("item1")
        await store.embedding_queue.put("item2")
        
        await store.cleanup()
        
        # Verify workers were cancelled
        worker1.cancel.assert_called_once()
        worker2.cancel.assert_called_once()
        store.notification_listener.cancel.assert_called_once()
        
        # Verify queue was cleared
        assert store.embedding_queue.empty()


class TestRetryManager:
    """Test cases for RetryManager."""
    
    @pytest.fixture
    def mock_store(self, pgai_test_config):
        """Mock store for retry manager testing."""
        store = MagicMock()
        store.pgai_config = pgai_test_config["database"]["pgai"]
        store.table_name = "test_table"
        
        # Mock pool with proper async context manager
        pool = AsyncMock()
        conn = AsyncMock()
        cursor = AsyncMock()

        # Setup async context managers properly
        async def mock_connection():
            return conn

        async def mock_cursor():
            return cursor

        pool.connection = mock_connection
        conn.cursor = mock_cursor
        conn.commit = AsyncMock()

        store.pool = pool
        return store, pool, conn, cursor
    
    @pytest.fixture
    def retry_manager(self, mock_store):
        """Create retry manager instance."""
        store, pool, conn, cursor = mock_store
        return RetryManager(store), store, pool, conn, cursor
    
    @pytest.mark.asyncio
    async def test_should_retry_new_record(self, retry_manager):
        """Test retry check for new record."""
        manager, store, pool, conn, cursor = retry_manager
        
        # Mock database response for new record
        cursor.fetchone.return_value = (0, None, 'pending')
        
        result = await manager.should_retry("new_record")
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_should_retry_max_retries_exceeded(self, retry_manager):
        """Test retry check when max retries exceeded."""
        manager, store, pool, conn, cursor = retry_manager
        
        # Mock database response for record with max retries
        cursor.fetchone.return_value = (3, None, 'pending')
        
        result = await manager.should_retry("max_retry_record")
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_should_retry_failed_status(self, retry_manager):
        """Test retry check for permanently failed record."""
        manager, store, pool, conn, cursor = retry_manager
        
        # Mock database response for failed record
        cursor.fetchone.return_value = (2, None, 'failed')
        
        result = await manager.should_retry("failed_record")
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_mark_retry(self, retry_manager):
        """Test marking a retry attempt."""
        manager, store, pool, conn, cursor = retry_manager
        
        await manager.mark_retry("test_record")
        
        # Verify UPDATE query was executed
        cursor.execute.assert_called_once()
        conn.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_mark_success(self, retry_manager):
        """Test marking successful processing."""
        manager, store, pool, conn, cursor = retry_manager
        
        await manager.mark_success("test_record")
        
        # Verify UPDATE query was executed
        cursor.execute.assert_called_once()
        conn.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_mark_failed(self, retry_manager):
        """Test marking final failure."""
        manager, store, pool, conn, cursor = retry_manager
        
        await manager.mark_failed("test_record")
        
        # Verify UPDATE query was executed
        cursor.execute.assert_called_once()
        conn.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_retry_stats(self, retry_manager):
        """Test getting retry statistics."""
        manager, store, pool, conn, cursor = retry_manager
        
        # Mock database response
        cursor.fetchall.return_value = [
            ('pending', 5),
            ('processing', 2),
            ('failed', 1)
        ]
        
        stats = await manager.get_retry_stats()
        
        assert stats['pending'] == 5
        assert stats['processing'] == 2
        assert stats['failed'] == 1
        assert stats['total_needing_retry'] == 8


class TestEmbeddingMonitor:
    """Test cases for EmbeddingMonitor."""

    @pytest.fixture
    def metrics(self):
        """Create metrics instance."""
        from memfuse_core.store.monitoring import EmbeddingMonitor
        return EmbeddingMonitor("test_store")
    
    @pytest.mark.asyncio
    async def test_record_processing_success(self, metrics):
        """Test recording successful processing."""
        metrics.start_processing("record1", "worker-1", 0)
        metrics.complete_processing("record1", True, None)

        assert metrics.metrics["total_processed"] == 1
        assert metrics.metrics["success_count"] == 1
        assert metrics.metrics["failure_count"] == 0
        assert metrics.metrics["retry_count"] == 0
    
    @pytest.mark.asyncio
    async def test_record_processing_failure(self, metrics):
        """Test recording failed processing."""
        metrics.start_processing("record1", "worker-1", 0)
        metrics.complete_processing("record1", False, "Test error")

        assert metrics.metrics["total_processed"] == 1
        assert metrics.metrics["success_count"] == 0
        assert metrics.metrics["failure_count"] == 1
        assert metrics.metrics["retry_count"] == 0
    
    @pytest.mark.asyncio
    async def test_record_processing_retry(self, metrics):
        """Test recording retry processing."""
        metrics.start_processing("record1", "worker-1", 1)  # retry_count = 1
        metrics.complete_processing("record1", False, "Test error")

        assert metrics.metrics["total_processed"] == 1
        assert metrics.metrics["retry_count"] == 1
    
    @pytest.mark.asyncio
    async def test_get_stats(self, metrics):
        """Test getting comprehensive statistics."""
        # Record some processing events
        metrics.start_processing("record1", "worker-1", 0)
        metrics.complete_processing("record1", True, None)

        metrics.start_processing("record2", "worker-1", 0)
        metrics.complete_processing("record2", False, "Error")

        metrics.start_processing("record3", "worker-1", 1)  # retry
        metrics.complete_processing("record3", True, None)

        stats = await metrics.get_current_stats()

        assert stats["total_processed"] == 3
        assert stats["success_count"] == 2
        assert stats["failure_count"] == 1
        assert stats["retry_count"] == 1


class TestStoreFactory:
    """Test cases for PgaiStoreFactory."""
    
    def test_create_event_driven_store(self):
        """Test creating event-driven store."""
        config = {
            "pgai": {
                "auto_embedding": True,
                "immediate_trigger": True
            }
        }
        
        store = PgaiStoreFactory.create_store(config)
        
        assert isinstance(store, EventDrivenPgaiStore)
    
    def test_create_traditional_store(self):
        """Test creating traditional store."""
        config = {
            "pgai": {
                "auto_embedding": True,
                "immediate_trigger": False
            }
        }
        
        from memfuse_core.store.pgai_store import PgaiStore
        store = PgaiStoreFactory.create_store(config)
        
        assert isinstance(store, PgaiStore)
        assert not isinstance(store, EventDrivenPgaiStore)
    
    def test_get_store_type(self):
        """Test getting store type."""
        config_event_driven = {
            "pgai": {
                "auto_embedding": True,
                "immediate_trigger": True
            }
        }
        
        config_traditional = {
            "pgai": {
                "auto_embedding": True,
                "immediate_trigger": False
            }
        }
        
        assert PgaiStoreFactory.get_store_type(config_event_driven) == "event_driven"
        assert PgaiStoreFactory.get_store_type(config_traditional) == "traditional"


# Integration test markers
pytestmark = pytest.mark.asyncio


class TestIntegrationEventDrivenPgaiStore:
    """Integration tests for event-driven pgai store.

    These tests require a running PostgreSQL instance with pgvector and pgai extensions.
    Run with: pytest tests/store/test_event_driven_pgai_store.py::TestIntegrationEventDrivenPgaiStore -m integration
    """

    @pytest.mark.integration
    @pytest.mark.skipif(
        os.getenv("SKIP_INTEGRATION", "true").lower() == "true",
        reason="Integration tests require SKIP_INTEGRATION=false"
    )
    async def test_end_to_end_immediate_trigger(self):
        """Test complete end-to-end immediate trigger workflow."""
        # This test requires actual database connection
        # It should be run with real pgai environment

        config = {
            "postgres": {
                "host": "localhost",
                "port": 5432,
                "database": "memfuse",
                "user": "postgres",
                "password": "postgres"
            },
            "pgai": {
                "enabled": True,
                "auto_embedding": True,
                "immediate_trigger": True,
                "max_retries": 3,
                "retry_interval": 2.0,
                "worker_count": 2,
                "queue_size": 100,
                "enable_metrics": True
            }
        }

        # Create store
        store = EventDrivenPgaiStore(config=config, table_name="test_immediate_trigger")

        try:
            # Initialize store
            success = await store.initialize()
            assert success, "Store initialization failed"

            # Create test chunks
            chunks = [
                ChunkData(chunk_id="test_immediate_1", content="This is test content for immediate embedding"),
                ChunkData(chunk_id="test_immediate_2", content="Another test content for immediate processing"),
                ChunkData(chunk_id="test_immediate_3", content="Third test content for verification")
            ]

            # Add chunks (should trigger immediate processing)
            start_time = time.time()
            chunk_ids = await store.add(chunks)
            add_time = time.time() - start_time

            assert len(chunk_ids) == 3
            assert add_time < 1.0, f"Add operation took too long: {add_time:.3f}s"

            # Wait for background processing
            max_wait = 30  # 30 seconds max wait
            wait_start = time.time()

            while time.time() - wait_start < max_wait:
                stats = await store.get_processing_stats()
                if stats["processing_stats"]["total_processed"] >= 3:
                    break
                await asyncio.sleep(1)

            # Verify processing completed
            final_stats = await store.get_processing_stats()
            assert final_stats["processing_stats"]["total_processed"] >= 3
            assert final_stats["processing_stats"]["success_count"] >= 3

            # Verify embeddings were generated
            results = await store.search("test content", top_k=3)
            assert len(results) >= 3

            # Check processing time was reasonable
            processing_time = time.time() - start_time
            assert processing_time < 10.0, f"Total processing took too long: {processing_time:.3f}s"

            print(f"✅ End-to-end test completed in {processing_time:.3f}s")
            print(f"📊 Final stats: {final_stats}")

        finally:
            # Cleanup
            await store.cleanup()

    @pytest.mark.integration
    @pytest.mark.skipif(
        os.getenv("SKIP_INTEGRATION", "true").lower() == "true",
        reason="Integration tests require SKIP_INTEGRATION=false"
    )
    async def test_retry_mechanism_integration(self):
        """Test retry mechanism with actual database."""
        config = {
            "postgres": {
                "host": "localhost",
                "port": 5432,
                "database": "memfuse",
                "user": "postgres",
                "password": "postgres"
            },
            "pgai": {
                "enabled": True,
                "auto_embedding": True,
                "immediate_trigger": True,
                "max_retries": 2,
                "retry_interval": 1.0,
                "worker_count": 1,
                "enable_metrics": True
            }
        }

        store = EventDrivenPgaiStore(config=config, table_name="test_retry_mechanism")

        try:
            await store.initialize()

            # Mock encoder to fail first few times
            original_encode = store.encoder.encode if hasattr(store, 'encoder') else None
            call_count = 0

            async def failing_encode(text):
                nonlocal call_count
                call_count += 1
                if call_count <= 2:  # Fail first 2 attempts
                    raise Exception("Simulated encoding failure")
                return [0.1] * 384  # Success on 3rd attempt

            if hasattr(store, 'encoder'):
                store.encoder.encode = failing_encode

            # Add chunk that will initially fail
            chunks = [ChunkData(chunk_id="retry_test_1", content="Content that will fail initially")]
            chunk_ids = await store.add(chunks)

            # Wait for retries to complete
            await asyncio.sleep(5)

            # Check retry stats
            retry_stats = await store.retry_manager.get_retry_stats()
            stats = await store.get_processing_stats()

            # Should have attempted retries
            assert stats["processing_stats"]["retry_count"] >= 1

            print(f"✅ Retry test completed")
            print(f"📊 Retry stats: {retry_stats}")
            print(f"📊 Processing stats: {stats}")

        finally:
            await store.cleanup()



