"""
Simplified tests for immediate trigger functionality.

This module contains working tests for the refactored immediate trigger system.
"""

import pytest
import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

# Import the simplified components
from memfuse_core.store.pgai_store import (
    EventDrivenPgaiStore, TriggerManager, RetryProcessor,
    WorkerPool, ImmediateTriggerCoordinator, EmbeddingMonitor, PgaiStoreFactory
)
from memfuse_core.rag.chunk.base import ChunkData


class TestTriggerManager:
    """Test cases for TriggerManager component."""
    
    @pytest.fixture
    def mock_pool(self):
        """Mock database pool."""
        pool = AsyncMock()
        conn = AsyncMock()
        cursor = AsyncMock()
        
        async def mock_connection():
            return conn
        
        async def mock_cursor():
            return cursor
        
        pool.connection = mock_connection
        conn.cursor = mock_cursor
        cursor.execute = AsyncMock()
        conn.execute = AsyncMock()
        conn.commit = AsyncMock()
        
        return pool, conn, cursor
    
    @pytest.fixture
    def trigger_manager(self, mock_pool):
        """Create TriggerManager instance."""
        pool, conn, cursor = mock_pool
        queue = asyncio.Queue()
        return TriggerManager(pool, "test_table", queue), pool, conn, cursor
    
    @pytest.mark.asyncio
    async def test_setup_triggers(self, trigger_manager):
        """Test database trigger setup."""
        manager, pool, conn, cursor = trigger_manager
        
        await manager.setup_triggers()
        
        # Verify SQL execution
        assert cursor.execute.call_count >= 2  # Function + trigger creation
        conn.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_start_stop_listening(self, trigger_manager):
        """Test starting and stopping notification listener."""
        manager, pool, conn, cursor = trigger_manager
        
        # Mock notifies
        async def mock_notifies():
            yield MagicMock(payload="test_record_1")
            yield MagicMock(payload="test_record_2")
        
        conn.notifies = mock_notifies
        
        # Start listening
        await manager.start_listening()
        assert manager.listener_task is not None
        
        # Give it a moment to process
        await asyncio.sleep(0.1)
        
        # Stop listening
        await manager.stop_listening()
        # Task should be cancelled or finished
        assert manager.listener_task.cancelled() or manager.listener_task.done()


class TestRetryProcessor:
    """Test cases for RetryProcessor component."""
    
    @pytest.fixture
    def mock_pool(self):
        """Mock database pool."""
        pool = AsyncMock()
        conn = AsyncMock()
        cursor = AsyncMock()
        
        async def mock_connection():
            return conn
        
        async def mock_cursor():
            return cursor
        
        pool.connection = mock_connection
        conn.cursor = mock_cursor
        conn.execute = AsyncMock()
        conn.commit = AsyncMock()
        
        return pool, conn, cursor
    
    @pytest.fixture
    def retry_processor(self, mock_pool):
        """Create RetryProcessor instance."""
        pool, conn, cursor = mock_pool
        return RetryProcessor(pool, "test_table", max_retries=3, retry_interval=1.0), pool, conn, cursor
    
    @pytest.mark.asyncio
    async def test_should_retry_new_record(self, retry_processor):
        """Test retry check for new record."""
        processor, pool, conn, cursor = retry_processor
        
        # Mock database response for new record
        cursor.fetchone.return_value = (0, None, 'pending')
        
        result = await processor.should_retry("new_record")
        
        assert result is True
        cursor.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_should_retry_failed_status(self, retry_processor):
        """Test retry check for permanently failed record."""
        processor, pool, conn, cursor = retry_processor
        
        # Mock database response for failed record
        cursor.fetchone.return_value = (2, None, 'failed')
        
        result = await processor.should_retry("failed_record")
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_mark_success(self, retry_processor):
        """Test marking successful processing."""
        processor, pool, conn, cursor = retry_processor
        
        await processor.mark_success("test_record")
        
        # Verify UPDATE query was executed
        cursor.execute.assert_called_once()
        conn.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_mark_failed(self, retry_processor):
        """Test marking final failure."""
        processor, pool, conn, cursor = retry_processor
        
        await processor.mark_failed("test_record")
        
        # Verify UPDATE query was executed
        cursor.execute.assert_called_once()
        conn.commit.assert_called_once()


class TestWorkerPool:
    """Test cases for WorkerPool component."""
    
    @pytest.fixture
    def worker_pool(self):
        """Create WorkerPool instance."""
        queue = asyncio.Queue()
        return WorkerPool(queue, worker_count=2), queue
    
    @pytest.mark.asyncio
    async def test_start_stop_workers(self, worker_pool):
        """Test starting and stopping worker pool."""
        pool, queue = worker_pool
        
        # Mock process function
        process_func = AsyncMock()
        
        # Start workers
        await pool.start(process_func)
        assert len(pool.workers) == 2
        assert pool.running is True
        
        # Stop workers
        await pool.stop()
        assert len(pool.workers) == 0
        assert pool.running is False
    
    @pytest.mark.asyncio
    async def test_worker_processing(self, worker_pool):
        """Test worker processing functionality."""
        pool, queue = worker_pool
        
        # Mock process function
        process_func = AsyncMock()
        
        # Add work to queue
        await queue.put("test_record")
        
        # Start workers
        await pool.start(process_func)
        
        # Give workers time to process
        await asyncio.sleep(0.1)
        
        # Stop workers
        await pool.stop()
        
        # Verify processing was called
        process_func.assert_called_with("test_record", "worker-0")


class TestEmbeddingMonitor:
    """Test cases for EmbeddingMonitor."""
    
    @pytest.fixture
    def monitor(self):
        """Create EmbeddingMonitor instance."""
        return EmbeddingMonitor("test_store")
    
    def test_start_complete_processing(self, monitor):
        """Test processing tracking."""
        # Start processing
        monitor.start_processing("record1", "worker-1", 0)
        assert "record1" in monitor.active_processing
        
        # Complete processing
        monitor.complete_processing("record1", True, None)
        assert "record1" not in monitor.active_processing
        assert monitor.metrics["success_count"] == 1
    
    def test_failure_tracking(self, monitor):
        """Test failure tracking."""
        monitor.start_processing("record1", "worker-1", 0)
        monitor.complete_processing("record1", False, "Test error")
        
        assert monitor.metrics["failure_count"] == 1
        assert "Test error" in monitor.error_patterns
    
    def test_retry_tracking(self, monitor):
        """Test retry tracking."""
        monitor.start_processing("record1", "worker-1", 1)  # retry_count = 1
        monitor.complete_processing("record1", True, None)
        
        assert monitor.metrics["retry_count"] == 1
    
    @pytest.mark.asyncio
    async def test_get_stats(self, monitor):
        """Test statistics retrieval."""
        # Add some processing events
        monitor.start_processing("record1", "worker-1", 0)
        monitor.complete_processing("record1", True, None)
        
        monitor.start_processing("record2", "worker-1", 0)
        monitor.complete_processing("record2", False, "Error")
        
        stats = monitor.get_current_stats()
        
        assert stats["total_processed"] == 2
        assert stats["success_count"] == 1
        assert stats["failure_count"] == 1


class TestStoreFactory:
    """Test cases for simplified store factory."""
    
    def test_get_store_type_event_driven(self):
        """Test store type detection for event-driven configuration."""
        config = {
            "pgai": {
                "auto_embedding": True,
                "immediate_trigger": True
            }
        }
        
        store_type = PgaiStoreFactory.get_store_type(config)
        assert store_type == "event_driven"
    
    def test_get_store_type_traditional(self):
        """Test store type detection for traditional configuration."""
        config = {
            "pgai": {
                "auto_embedding": True,
                "immediate_trigger": False
            }
        }
        
        store_type = PgaiStoreFactory.get_store_type(config)
        assert store_type == "traditional"
    
    def test_validate_configuration(self):
        """Test configuration validation."""
        invalid_config = {
            "pgai": {
                "worker_count": "invalid",
                "max_retries": -1
            }
        }
        
        validated = PgaiStoreFactory.validate_configuration(invalid_config)
        pgai_config = validated["pgai"]
        
        # Check corrections
        assert isinstance(pgai_config["worker_count"], int)
        assert pgai_config["worker_count"] > 0
        assert isinstance(pgai_config["max_retries"], int)
        assert pgai_config["max_retries"] >= 0


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("SKIP_INTEGRATION", "true").lower() == "true",
    reason="Integration tests require SKIP_INTEGRATION=false"
)
class TestIntegrationSimplified:
    """Simplified integration tests."""
    
    @pytest.mark.asyncio
    async def test_basic_functionality(self, pgai_test_config):
        """Test basic immediate trigger functionality."""
        # This test would require actual database connection
        # For now, just test that the store can be created
        
        with patch('memfuse_core.store.pgai_store.pgai_store.PgaiStore') as mock_pgai:
            mock_instance = AsyncMock()
            mock_instance.initialize.return_value = True
            mock_pgai.return_value = mock_instance
            
            store = EventDrivenPgaiStore(
                config=pgai_test_config,
                table_name="test_simplified"
            )
            
            success = await store.initialize()
            assert success is True
            
            await store.cleanup()


# Test configuration
@pytest.fixture
def test_config():
    """Test configuration."""
    return {
        "pgai": {
            "enabled": True,
            "auto_embedding": True,
            "immediate_trigger": True,
            "max_retries": 2,
            "retry_interval": 1.0,
            "worker_count": 2,
            "queue_size": 100,
            "enable_metrics": True
        }
    }
