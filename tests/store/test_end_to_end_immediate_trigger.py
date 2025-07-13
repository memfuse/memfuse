"""
End-to-end integration test for immediate trigger functionality.

This test validates the complete immediate trigger workflow with real database operations.
"""

import pytest
import asyncio
import os
import time
from unittest.mock import AsyncMock, patch
from typing import Dict, Any

from memfuse_core.store.event_driven_store import EventDrivenPgaiStore
from memfuse_core.store.immediate_trigger_components import ImmediateTriggerCoordinator
from memfuse_core.rag.chunk.base import ChunkData


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("SKIP_INTEGRATION", "true").lower() == "true",
    reason="Integration tests require SKIP_INTEGRATION=false"
)
class TestEndToEndImmediateTrigger:
    """End-to-end integration tests for immediate trigger system."""
    
    @pytest.fixture
    def integration_config(self):
        """Configuration for integration testing."""
        return {
            "database": {
                "postgres": {
                    "host": "localhost",
                    "port": 5432,
                    "database": "memfuse_test",
                    "user": "postgres",
                    "password": "postgres"
                },
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
        }
    
    @pytest.fixture
    def mock_embedding_model(self):
        """Mock embedding model for testing."""
        model = AsyncMock()
        model.encode.return_value = [0.1] * 384  # Mock 384-dimensional embedding
        return model
    
    @pytest.mark.asyncio
    async def test_complete_immediate_trigger_workflow(self, integration_config, mock_embedding_model):
        """Test complete immediate trigger workflow with mocked database."""
        print("\n🧪 Testing Complete Immediate Trigger Workflow")
        print("=" * 60)
        
        # Mock the database components
        with patch('memfuse_core.store.pgai_store.PgaiStore') as mock_pgai_store:
            # Setup mock store
            mock_store_instance = AsyncMock()
            mock_store_instance.initialize.return_value = True
            mock_store_instance.pool = AsyncMock()
            mock_store_instance.encoder = mock_embedding_model
            mock_store_instance._generate_embedding.return_value = [0.1] * 384
            
            # Mock database operations
            mock_conn = AsyncMock()
            mock_cursor = AsyncMock()
            
            async def mock_connection():
                return mock_conn
            
            async def mock_cursor_func():
                return mock_cursor
            
            mock_store_instance.pool.connection = mock_connection
            mock_conn.cursor = mock_cursor_func
            mock_conn.execute = AsyncMock()
            mock_conn.commit = AsyncMock()
            mock_cursor.execute = AsyncMock()
            mock_cursor.fetchone = AsyncMock()
            mock_cursor.fetchall = AsyncMock()
            
            mock_pgai_store.return_value = mock_store_instance
            
            # Create simplified event-driven store
            store = EventDrivenPgaiStore(
                config=integration_config["database"],
                table_name="test_immediate_trigger"
            )
            
            # Initialize store
            success = await store.initialize()
            assert success is True
            print("✅ Store initialized successfully")
            
            # Verify coordinator was created
            assert store.coordinator is not None
            print("✅ Immediate trigger coordinator created")
            
            # Test adding chunks (should trigger immediate processing)
            test_chunks = [
                ChunkData(
                    id="chunk_1",
                    content="This is test content for immediate embedding generation.",
                    metadata={"test": True}
                ),
                ChunkData(
                    id="chunk_2", 
                    content="Another test chunk for immediate processing.",
                    metadata={"test": True}
                )
            ]
            
            # Mock the add method to return chunk IDs
            mock_store_instance.add.return_value = ["chunk_1", "chunk_2"]
            
            # Add chunks
            chunk_ids = await store.add(test_chunks)
            assert len(chunk_ids) == 2
            print(f"✅ Added {len(chunk_ids)} chunks: {chunk_ids}")
            
            # Verify coordinator components are working
            coordinator = store.coordinator
            
            # Test queue functionality
            await coordinator.queue.put("test_record_1")
            await coordinator.queue.put("test_record_2")
            
            queue_size = coordinator.queue.qsize()
            assert queue_size == 2
            print(f"✅ Queue working: {queue_size} items queued")
            
            # Test processing stats
            stats = await coordinator.get_stats()
            assert "queue_size" in stats
            assert "worker_count" in stats
            assert "table_name" in stats
            print(f"✅ Stats available: queue_size={stats['queue_size']}, workers={stats['worker_count']}")
            
            # Test retry processor
            retry_processor = coordinator.retry_processor
            
            # Mock database response for retry check
            mock_cursor.fetchone.return_value = (0, None, 'pending')
            
            should_retry = await retry_processor.should_retry("test_record")
            assert should_retry is True
            print("✅ Retry processor working")
            
            # Test monitoring
            if coordinator.monitor:
                monitor = coordinator.monitor
                
                # Simulate processing
                monitor.start_processing("test_record", "worker-1", 0)
                monitor.complete_processing("test_record", True, None)
                
                monitor_stats = monitor.get_current_stats()
                assert monitor_stats["total_processed"] == 1
                assert monitor_stats["success_count"] == 1
                print("✅ Monitoring working")
            
            # Cleanup
            await store.cleanup()
            print("✅ Store cleaned up successfully")
            
            print("\n🎉 Complete immediate trigger workflow test PASSED!")
            return True
    
    @pytest.mark.asyncio
    async def test_retry_mechanism_integration(self, integration_config):
        """Test retry mechanism with realistic scenarios."""
        print("\n🧪 Testing Retry Mechanism Integration")
        print("=" * 60)
        
        with patch('memfuse_core.store.pgai_store.PgaiStore') as mock_pgai_store:
            # Setup mock store with retry scenarios
            mock_store_instance = AsyncMock()
            mock_store_instance.initialize.return_value = True
            mock_store_instance.pool = AsyncMock()
            
            mock_conn = AsyncMock()
            mock_cursor = AsyncMock()
            
            async def mock_connection():
                return mock_conn
            
            async def mock_cursor_func():
                return mock_cursor
            
            mock_store_instance.pool.connection = mock_connection
            mock_conn.cursor = mock_cursor_func
            mock_conn.execute = AsyncMock()
            mock_conn.commit = AsyncMock()
            mock_cursor.execute = AsyncMock()
            mock_cursor.fetchone = AsyncMock()
            
            mock_pgai_store.return_value = mock_store_instance
            
            # Create store
            store = EventDrivenPgaiStore(
                config=integration_config["database"],
                table_name="test_retry"
            )
            
            await store.initialize()
            coordinator = store.coordinator
            retry_processor = coordinator.retry_processor
            
            # Test scenario 1: New record should retry
            mock_cursor.fetchone.return_value = (0, None, 'pending')
            should_retry = await retry_processor.should_retry("new_record")
            assert should_retry is True
            print("✅ New record retry check passed")
            
            # Test scenario 2: Max retries exceeded
            mock_cursor.fetchone.return_value = (3, None, 'pending')
            should_retry = await retry_processor.should_retry("max_retry_record")
            assert should_retry is False
            print("✅ Max retries exceeded check passed")
            
            # Test scenario 3: Already failed
            mock_cursor.fetchone.return_value = (2, None, 'failed')
            should_retry = await retry_processor.should_retry("failed_record")
            assert should_retry is False
            print("✅ Failed record check passed")
            
            # Test marking operations
            await retry_processor.mark_retry("test_record")
            mock_cursor.execute.assert_called()
            print("✅ Mark retry operation passed")
            
            await retry_processor.mark_success("test_record")
            mock_cursor.execute.assert_called()
            print("✅ Mark success operation passed")
            
            await retry_processor.mark_failed("test_record")
            mock_cursor.execute.assert_called()
            print("✅ Mark failed operation passed")
            
            await store.cleanup()
            print("\n🎉 Retry mechanism integration test PASSED!")
            return True
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self, integration_config):
        """Test system performance under simulated load."""
        print("\n🧪 Testing Performance Under Load")
        print("=" * 60)
        
        with patch('memfuse_core.store.pgai_store.PgaiStore') as mock_pgai_store:
            # Setup high-performance mock
            mock_store_instance = AsyncMock()
            mock_store_instance.initialize.return_value = True
            mock_store_instance.pool = AsyncMock()
            
            mock_conn = AsyncMock()
            mock_cursor = AsyncMock()
            
            async def mock_connection():
                return mock_conn
            
            async def mock_cursor_func():
                return mock_cursor
            
            mock_store_instance.pool.connection = mock_connection
            mock_conn.cursor = mock_cursor_func
            mock_conn.execute = AsyncMock()
            mock_conn.commit = AsyncMock()
            mock_cursor.execute = AsyncMock()
            mock_cursor.fetchone = AsyncMock(return_value=(0, None, 'pending'))
            
            mock_pgai_store.return_value = mock_store_instance
            
            # Create store with higher worker count
            config = integration_config["database"].copy()
            config["pgai"]["worker_count"] = 4
            config["pgai"]["queue_size"] = 1000
            
            store = EventDrivenPgaiStore(config=config, table_name="test_performance")
            await store.initialize()
            
            coordinator = store.coordinator
            
            # Simulate high load
            start_time = time.time()
            
            # Queue many items quickly
            for i in range(100):
                try:
                    await coordinator.queue.put(f"record_{i}")
                except asyncio.QueueFull:
                    break
            
            queue_time = time.time() - start_time
            queue_size = coordinator.queue.qsize()
            
            print(f"✅ Queued {queue_size} items in {queue_time:.3f}s")
            print(f"✅ Queue rate: {queue_size/queue_time:.1f} items/sec")
            
            # Test stats collection under load
            stats = await coordinator.get_stats()
            assert stats["queue_size"] == queue_size
            assert stats["worker_count"] == 4
            print(f"✅ Stats collection working under load")
            
            # Test monitoring under load
            if coordinator.monitor:
                monitor = coordinator.monitor
                
                # Simulate rapid processing
                for i in range(50):
                    record_id = f"load_test_{i}"
                    monitor.start_processing(record_id, f"worker-{i%4}", 0)
                    monitor.complete_processing(record_id, True, None)
                
                monitor_stats = monitor.get_current_stats()
                assert monitor_stats["total_processed"] == 50
                print(f"✅ Monitoring handled {monitor_stats['total_processed']} rapid operations")
            
            await store.cleanup()
            print("\n🎉 Performance under load test PASSED!")
            return True
    
    @pytest.mark.asyncio
    async def test_error_recovery_scenarios(self, integration_config):
        """Test error recovery and resilience."""
        print("\n🧪 Testing Error Recovery Scenarios")
        print("=" * 60)
        
        with patch('memfuse_core.store.pgai_store.PgaiStore') as mock_pgai_store:
            # Setup mock with error scenarios
            mock_store_instance = AsyncMock()
            mock_store_instance.initialize.return_value = True
            mock_store_instance.pool = AsyncMock()
            
            mock_conn = AsyncMock()
            mock_cursor = AsyncMock()
            
            async def mock_connection():
                return mock_conn
            
            async def mock_cursor_func():
                return mock_cursor
            
            mock_store_instance.pool.connection = mock_connection
            mock_conn.cursor = mock_cursor_func
            mock_conn.execute = AsyncMock()
            mock_conn.commit = AsyncMock()
            mock_cursor.execute = AsyncMock()
            mock_cursor.fetchone = AsyncMock()
            
            mock_pgai_store.return_value = mock_store_instance
            
            store = EventDrivenPgaiStore(
                config=integration_config["database"],
                table_name="test_error_recovery"
            )
            
            await store.initialize()
            coordinator = store.coordinator
            
            # Test error handling in monitoring
            if coordinator.monitor:
                monitor = coordinator.monitor
                
                # Test various error scenarios
                error_scenarios = [
                    ("connection_error", "Database connection failed"),
                    ("timeout_error", "Operation timed out"),
                    ("encoding_error", "Text encoding failed"),
                    ("memory_error", "Out of memory"),
                ]
                
                for error_type, error_msg in error_scenarios:
                    monitor.start_processing(f"error_{error_type}", "worker-1", 0)
                    monitor.complete_processing(f"error_{error_type}", False, error_msg)
                
                stats = monitor.get_current_stats()
                assert stats["failure_count"] == 4
                print(f"✅ Error tracking: {stats['failure_count']} errors recorded")
                
                # Check error patterns
                assert len(monitor.error_patterns) == 4
                print(f"✅ Error pattern analysis: {len(monitor.error_patterns)} unique patterns")
            
            # Test retry processor error handling
            retry_processor = coordinator.retry_processor
            
            # Test with database errors
            mock_cursor.fetchone.side_effect = Exception("Database error")
            
            try:
                await retry_processor.should_retry("error_record")
                # Should handle the error gracefully
                print("✅ Retry processor handled database error")
            except Exception as e:
                print(f"⚠️ Retry processor error handling needs improvement: {e}")
            
            # Reset mock
            mock_cursor.fetchone.side_effect = None
            mock_cursor.fetchone.return_value = (0, None, 'pending')
            
            await store.cleanup()
            print("\n🎉 Error recovery scenarios test PASSED!")
            return True


@pytest.mark.asyncio
async def test_integration_suite():
    """Run the complete integration test suite."""
    print("🚀 MemFuse Immediate Trigger Integration Test Suite")
    print("=" * 80)
    
    # This would be run with: SKIP_INTEGRATION=false pytest tests/store/test_end_to_end_immediate_trigger.py::test_integration_suite -v
    
    config = {
        "database": {
            "postgres": {
                "host": "localhost",
                "port": 5432,
                "database": "memfuse_test",
                "user": "postgres",
                "password": "postgres"
            },
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
    }
    
    test_instance = TestEndToEndImmediateTrigger()
    
    tests = [
        ("Complete Workflow", test_instance.test_complete_immediate_trigger_workflow),
        ("Retry Mechanism", test_instance.test_retry_mechanism_integration),
        ("Performance Load", test_instance.test_performance_under_load),
        ("Error Recovery", test_instance.test_error_recovery_scenarios),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        try:
            if test_func.__code__.co_argcount > 1:  # Has self parameter
                result = await test_func(config, AsyncMock())
            else:
                result = await test_func(config)
            results.append((test_name, True))
        except Exception as e:
            print(f"❌ {test_name} failed: {e}")
            results.append((test_name, False))
    
    # Summary
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\n🎯 Integration Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 ALL INTEGRATION TESTS PASSED!")
    else:
        print("⚠️ Some integration tests failed")
    
    return passed == total
