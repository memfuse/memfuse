"""Integration tests for optimized flush mechanism.

This test suite validates the new FlushManager and HybridBufferV2 implementation
to ensure timeout issues are resolved and performance is improved.
"""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, MagicMock

from memfuse_core.buffer.flush_manager import FlushManager, FlushPriority
from memfuse_core.buffer.hybrid_buffer_v2 import HybridBufferV2
from memfuse_core.rag.chunk.base import ChunkData


class TestOptimizedFlush:
    """Test cases for optimized flush mechanism."""
    
    @pytest.fixture
    async def flush_manager(self):
        """Create FlushManager instance for testing."""
        sqlite_handler = AsyncMock()
        qdrant_handler = AsyncMock()
        
        manager = FlushManager(
            max_workers=2,
            max_queue_size=10,
            default_timeout=5.0,
            flush_interval=10.0,
            enable_auto_flush=True,
            sqlite_handler=sqlite_handler,
            qdrant_handler=qdrant_handler
        )
        
        await manager.initialize()
        yield manager
        await manager.shutdown()
    
    @pytest.fixture
    async def hybrid_buffer_v2(self, flush_manager):
        """Create HybridBufferV2 instance for testing."""
        buffer = HybridBufferV2(
            max_size=3,
            chunk_strategy="message",
            embedding_model="all-MiniLM-L6-v2",
            flush_manager=flush_manager,
            auto_flush_interval=5.0,
            enable_auto_flush=True
        )
        
        await buffer.initialize()
        yield buffer
        await buffer.shutdown()
    
    @pytest.mark.asyncio
    async def test_flush_manager_initialization(self, flush_manager):
        """Test FlushManager initialization and shutdown."""
        assert flush_manager is not None
        
        # Check metrics
        metrics = await flush_manager.get_metrics()
        assert metrics.total_flushes == 0
        assert metrics.active_workers >= 0
        assert metrics.queue_size == 0
    
    @pytest.mark.asyncio
    async def test_non_blocking_sqlite_flush(self, flush_manager):
        """Test non-blocking SQLite flush operation."""
        # Create test data
        test_rounds = [
            [{"role": "user", "content": "Hello", "id": "msg1"}],
            [{"role": "assistant", "content": "Hi there!", "id": "msg2"}]
        ]
        
        # Schedule flush
        start_time = time.time()
        task_id = await flush_manager.flush_sqlite(
            rounds=test_rounds,
            priority=FlushPriority.NORMAL,
            timeout=5.0
        )
        end_time = time.time()
        
        # Should return immediately (non-blocking)
        assert end_time - start_time < 0.1  # Should be very fast
        assert task_id is not None
        assert task_id.startswith("sqlite-")
        
        # Wait a bit for processing
        await asyncio.sleep(0.5)
        
        # Check metrics
        metrics = await flush_manager.get_metrics()
        assert metrics.total_flushes >= 1
    
    @pytest.mark.asyncio
    async def test_non_blocking_qdrant_flush(self, flush_manager):
        """Test non-blocking Qdrant flush operation."""
        # Create test data
        test_chunks = [
            ChunkData(content="Test chunk 1", metadata={"id": "chunk1"}),
            ChunkData(content="Test chunk 2", metadata={"id": "chunk2"})
        ]
        test_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        
        # Schedule flush
        start_time = time.time()
        task_id = await flush_manager.flush_qdrant(
            chunks=test_chunks,
            embeddings=test_embeddings,
            priority=FlushPriority.NORMAL,
            timeout=5.0
        )
        end_time = time.time()
        
        # Should return immediately (non-blocking)
        assert end_time - start_time < 0.1
        assert task_id is not None
        assert task_id.startswith("qdrant-")
        
        # Wait for processing
        await asyncio.sleep(0.5)
        
        # Check metrics
        metrics = await flush_manager.get_metrics()
        assert metrics.total_flushes >= 1
    
    @pytest.mark.asyncio
    async def test_hybrid_flush(self, flush_manager):
        """Test hybrid flush operation (both SQLite and Qdrant)."""
        # Create test data
        test_rounds = [[{"role": "user", "content": "Test", "id": "msg1"}]]
        test_chunks = [ChunkData(content="Test chunk", metadata={"id": "chunk1"})]
        test_embeddings = [[0.1, 0.2, 0.3]]
        
        # Schedule hybrid flush
        start_time = time.time()
        task_id = await flush_manager.flush_hybrid(
            rounds=test_rounds,
            chunks=test_chunks,
            embeddings=test_embeddings,
            priority=FlushPriority.HIGH,
            timeout=10.0
        )
        end_time = time.time()
        
        # Should return immediately
        assert end_time - start_time < 0.1
        assert task_id is not None
        assert task_id.startswith("hybrid-")
        
        # Wait for processing
        await asyncio.sleep(1.0)
        
        # Check that both handlers were called
        assert flush_manager.sqlite_handler.called
        assert flush_manager.qdrant_handler.called
    
    @pytest.mark.asyncio
    async def test_hybrid_buffer_v2_non_blocking_add(self, hybrid_buffer_v2):
        """Test that HybridBufferV2 add operations are non-blocking."""
        # Create test rounds
        test_rounds = []
        for i in range(5):  # This should trigger flush (max_size=3)
            test_rounds.append([{
                "role": "user",
                "content": f"Test message {i}",
                "id": f"msg{i}",
                "created_at": "2024-01-01T00:00:00Z"
            }])
        
        # Add rounds and measure time
        start_time = time.time()
        
        for rounds in test_rounds:
            await hybrid_buffer_v2.add_from_rounds(rounds)
        
        end_time = time.time()
        
        # Should complete quickly even with flush operations
        assert end_time - start_time < 2.0  # Should be much faster than before
        
        # Wait for any pending flushes
        await asyncio.sleep(1.0)
        
        # Check stats
        stats = hybrid_buffer_v2.get_stats()
        assert stats["total_rounds_received"] == 5
        assert stats["total_flushes"] >= 1  # Should have triggered at least one flush
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, hybrid_buffer_v2):
        """Test concurrent operations don't block each other."""
        async def add_rounds_task(start_index: int, count: int):
            """Task to add rounds concurrently."""
            for i in range(count):
                rounds = [{
                    "role": "user",
                    "content": f"Concurrent message {start_index}-{i}",
                    "id": f"msg{start_index}-{i}",
                    "created_at": "2024-01-01T00:00:00Z"
                }]
                await hybrid_buffer_v2.add_from_rounds([rounds])
                await asyncio.sleep(0.01)  # Small delay to simulate real usage
        
        # Run multiple concurrent tasks
        start_time = time.time()
        
        tasks = [
            add_rounds_task(1, 3),
            add_rounds_task(2, 3),
            add_rounds_task(3, 3)
        ]
        
        await asyncio.gather(*tasks)
        
        end_time = time.time()
        
        # Should complete in reasonable time
        assert end_time - start_time < 5.0
        
        # Wait for pending operations
        await asyncio.sleep(1.0)
        
        # Check that all data was processed
        stats = hybrid_buffer_v2.get_stats()
        assert stats["total_rounds_received"] == 9
    
    @pytest.mark.asyncio
    async def test_flush_timeout_handling(self, flush_manager):
        """Test that flush operations handle timeouts properly."""
        # Create a slow handler that will timeout
        slow_handler = AsyncMock()
        slow_handler.side_effect = lambda *args: asyncio.sleep(10)  # Longer than timeout
        
        # Replace handler temporarily
        original_handler = flush_manager.sqlite_handler
        flush_manager.sqlite_handler = slow_handler
        
        try:
            # Schedule flush with short timeout
            task_id = await flush_manager.flush_sqlite(
                rounds=[[{"role": "user", "content": "Test", "id": "msg1"}]],
                priority=FlushPriority.NORMAL,
                timeout=1.0  # Short timeout
            )
            
            # Wait for timeout to occur
            await asyncio.sleep(2.0)
            
            # Check metrics for timeout
            metrics = await flush_manager.get_metrics()
            assert metrics.timeout_flushes >= 1
            
        finally:
            # Restore original handler
            flush_manager.sqlite_handler = original_handler
    
    @pytest.mark.asyncio
    async def test_auto_flush_functionality(self, hybrid_buffer_v2):
        """Test auto-flush functionality based on time."""
        # Add some data but not enough to trigger size-based flush
        rounds = [{
            "role": "user",
            "content": "Auto flush test",
            "id": "msg1",
            "created_at": "2024-01-01T00:00:00Z"
        }]
        
        await hybrid_buffer_v2.add_from_rounds([rounds])
        
        # Wait for auto-flush to trigger (interval is 5s in test config)
        await asyncio.sleep(6.0)
        
        # Check that auto-flush occurred
        stats = hybrid_buffer_v2.get_stats()
        assert stats["total_auto_flushes"] >= 1


if __name__ == "__main__":
    # Run a simple test
    async def simple_test():
        print("Testing optimized flush mechanism...")
        
        # Create FlushManager
        manager = FlushManager(
            max_workers=2,
            max_queue_size=10,
            default_timeout=5.0,
            sqlite_handler=AsyncMock(),
            qdrant_handler=AsyncMock()
        )
        
        await manager.initialize()
        
        # Test non-blocking operation
        start_time = time.time()
        task_id = await manager.flush_sqlite(
            rounds=[[{"role": "user", "content": "Test", "id": "msg1"}]],
            priority=FlushPriority.NORMAL
        )
        end_time = time.time()
        
        print(f"Flush scheduled in {end_time - start_time:.4f}s (task_id: {task_id})")
        
        # Wait for processing
        await asyncio.sleep(1.0)
        
        # Check metrics
        metrics = await manager.get_metrics()
        print(f"Metrics: {metrics}")
        
        await manager.shutdown()
        print("Test completed successfully!")
    
    asyncio.run(simple_test())
