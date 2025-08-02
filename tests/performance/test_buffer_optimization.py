"""Performance tests for Phase 2 buffer system optimizations."""

import pytest
import asyncio
import time
from pathlib import Path
import sys
from unittest.mock import AsyncMock, MagicMock

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from memfuse_core.buffer.hybrid_buffer import HybridBuffer
from memfuse_core.buffer.flush_manager import FlushManager
from memfuse_core.interfaces import MessageList


class TestBufferOptimization:
    """Test suite for Phase 2 buffer system performance optimizations."""

    @pytest.fixture
    async def mock_embedding_model(self):
        """Create a mock embedding model for testing."""
        mock_model = AsyncMock()
        mock_model.encode = AsyncMock(return_value=[0.1, 0.2, 0.3, 0.4, 0.5])
        return mock_model

    @pytest.fixture
    async def mock_chunk_strategy(self):
        """Create a mock chunk strategy for testing."""
        mock_strategy = AsyncMock()
        
        # Mock chunk data
        mock_chunk = MagicMock()
        mock_chunk.content = "test chunk content"
        mock_chunk.metadata = {"source": "test"}
        
        mock_strategy.create_chunks = AsyncMock(return_value=[mock_chunk] * 5)  # 5 chunks per round
        return mock_strategy

    @pytest.fixture
    async def mock_flush_manager(self):
        """Create a mock flush manager for testing."""
        mock_manager = AsyncMock()
        mock_manager.flush_buffer_data = AsyncMock(return_value="task_123")
        return mock_manager

    @pytest.fixture
    async def hybrid_buffer(self, mock_flush_manager):
        """Create a HybridBuffer instance for testing."""
        buffer = HybridBuffer(
            max_size=5,
            chunk_strategy="message",
            embedding_model="all-MiniLM-L6-v2",
            flush_manager=mock_flush_manager,
            auto_flush_interval=60.0,
            enable_auto_flush=True
        )
        return buffer

    @pytest.mark.asyncio
    async def test_parallel_embedding_generation_performance(self, hybrid_buffer, mock_embedding_model, mock_chunk_strategy):
        """Test that parallel embedding generation improves performance."""
        # Setup mocks
        hybrid_buffer.embedding_model = mock_embedding_model
        hybrid_buffer.chunk_strategy = mock_chunk_strategy
        
        # Create test data - multiple rounds with multiple chunks each
        test_rounds = []
        for i in range(3):  # 3 rounds
            round_data = [
                {"role": "user", "content": f"Test message {i}-{j}"}
                for j in range(2)  # 2 messages per round
            ]
            test_rounds.append(round_data)
        
        # Measure parallel processing time
        start_time = time.time()
        await hybrid_buffer.add_from_rounds(test_rounds)
        parallel_time = time.time() - start_time
        
        # Verify that chunks and embeddings were created
        assert len(hybrid_buffer.chunks) == 15  # 3 rounds * 5 chunks per round
        assert len(hybrid_buffer.embeddings) == 15
        
        # Verify embedding model was called for each chunk
        assert mock_embedding_model.encode.call_count == 15
        
        # Performance assertion: should complete within reasonable time
        max_expected_time = 2.0  # 2 seconds for 15 embeddings
        assert parallel_time < max_expected_time, f"Parallel processing took {parallel_time:.3f}s, expected <{max_expected_time}s"
        
        print(f"🚀 Parallel embedding generation completed in {parallel_time:.3f}s for 15 chunks")
        print("✅ Parallel embedding generation performance is acceptable")

    @pytest.mark.asyncio
    async def test_concurrent_embedding_generation(self, hybrid_buffer, mock_embedding_model, mock_chunk_strategy):
        """Test that embedding generation handles concurrency correctly."""
        # Setup mocks with delay to simulate real embedding generation
        async def mock_encode_with_delay(content):
            await asyncio.sleep(0.1)  # 100ms delay per embedding
            return [0.1, 0.2, 0.3, 0.4, 0.5]
        
        mock_embedding_model.encode = mock_encode_with_delay
        hybrid_buffer.embedding_model = mock_embedding_model
        hybrid_buffer.chunk_strategy = mock_chunk_strategy
        
        # Create test data
        test_rounds = [[{"role": "user", "content": "Test message"}]]
        
        # Measure time for concurrent processing
        start_time = time.time()
        await hybrid_buffer.add_from_rounds(test_rounds)
        concurrent_time = time.time() - start_time
        
        # With 5 chunks and 100ms delay each:
        # Sequential would take ~500ms
        # Parallel should take ~100ms (limited by slowest embedding)
        expected_max_time = 0.2  # 200ms buffer for overhead
        assert concurrent_time < expected_max_time, f"Concurrent processing took {concurrent_time:.3f}s, expected <{expected_max_time}s"
        
        print(f"🚀 Concurrent embedding generation completed in {concurrent_time:.3f}s")
        print("✅ Concurrent embedding generation provides significant speedup")

    @pytest.mark.asyncio
    async def test_embedding_error_handling(self, hybrid_buffer, mock_embedding_model, mock_chunk_strategy):
        """Test that embedding generation handles errors gracefully."""
        # Setup mock to fail on some embeddings
        call_count = 0
        async def mock_encode_with_failures(content):
            nonlocal call_count
            call_count += 1
            if call_count % 3 == 0:  # Fail every 3rd embedding
                raise Exception("Mock embedding failure")
            return [0.1, 0.2, 0.3, 0.4, 0.5]
        
        mock_embedding_model.encode = mock_encode_with_failures
        hybrid_buffer.embedding_model = mock_embedding_model
        hybrid_buffer.chunk_strategy = mock_chunk_strategy
        
        # Create test data
        test_rounds = [[{"role": "user", "content": "Test message"}]]
        
        # Process should complete without raising exceptions
        await hybrid_buffer.add_from_rounds(test_rounds)
        
        # Should have some successful embeddings (not all failed)
        assert len(hybrid_buffer.chunks) > 0
        assert len(hybrid_buffer.embeddings) > 0
        
        # Should have fewer embeddings than chunks due to failures
        assert len(hybrid_buffer.embeddings) < 5  # Some failed
        
        print(f"🚀 Error handling test completed - {len(hybrid_buffer.embeddings)}/5 embeddings successful")
        print("✅ Embedding error handling works correctly")

    @pytest.mark.asyncio
    async def test_semaphore_concurrency_control(self, hybrid_buffer, mock_embedding_model, mock_chunk_strategy):
        """Test that semaphore properly controls concurrency."""
        # Track concurrent calls
        concurrent_calls = 0
        max_concurrent = 0
        
        async def mock_encode_with_tracking(content):
            nonlocal concurrent_calls, max_concurrent
            concurrent_calls += 1
            max_concurrent = max(max_concurrent, concurrent_calls)
            
            await asyncio.sleep(0.05)  # Small delay
            
            concurrent_calls -= 1
            return [0.1, 0.2, 0.3, 0.4, 0.5]
        
        mock_embedding_model.encode = mock_encode_with_tracking
        hybrid_buffer.embedding_model = mock_embedding_model
        hybrid_buffer.chunk_strategy = mock_chunk_strategy
        
        # Create test data with many chunks
        test_rounds = [[{"role": "user", "content": "Test message"}] for _ in range(5)]  # 25 chunks total
        
        await hybrid_buffer.add_from_rounds(test_rounds)
        
        # Verify concurrency was controlled (should not exceed semaphore limit of 10)
        assert max_concurrent <= 10, f"Max concurrent calls was {max_concurrent}, expected ≤10"
        assert max_concurrent > 1, f"Max concurrent calls was {max_concurrent}, expected >1 for parallel processing"
        
        print(f"🚀 Concurrency control test completed - max concurrent: {max_concurrent}/10")
        print("✅ Semaphore concurrency control works correctly")

    @pytest.mark.asyncio
    async def test_buffer_flush_performance(self, hybrid_buffer, mock_flush_manager):
        """Test that buffer flush operations are non-blocking."""
        # Setup buffer with data
        hybrid_buffer.original_rounds = [
            [{"role": "user", "content": f"Message {i}"}] for i in range(5)
        ]
        
        # Measure flush time
        start_time = time.time()
        await hybrid_buffer.flush_to_storage()
        flush_time = time.time() - start_time
        
        # Flush should be very fast (non-blocking)
        max_flush_time = 0.1  # 100ms
        assert flush_time < max_flush_time, f"Flush took {flush_time:.3f}s, expected <{max_flush_time}s"
        
        # Verify flush manager was called
        mock_flush_manager.flush_buffer_data.assert_called_once()
        
        # Verify buffer was cleared (optimistic clearing)
        assert len(hybrid_buffer.original_rounds) == 0
        assert len(hybrid_buffer.chunks) == 0
        assert len(hybrid_buffer.embeddings) == 0
        
        print(f"🚀 Non-blocking flush completed in {flush_time:.3f}s")
        print("✅ Buffer flush performance is optimal")


if __name__ == "__main__":
    # Run tests directly
    test_instance = TestBufferOptimization()

    async def run_async_tests():
        # Create mocks directly (not as fixtures)
        mock_embedding_model = AsyncMock()
        mock_embedding_model.encode = AsyncMock(return_value=[0.1, 0.2, 0.3, 0.4, 0.5])

        mock_chunk = MagicMock()
        mock_chunk.content = "test chunk content"
        mock_chunk.metadata = {"source": "test"}

        mock_chunk_strategy = AsyncMock()
        mock_chunk_strategy.create_chunks = AsyncMock(return_value=[mock_chunk] * 5)

        mock_flush_manager = AsyncMock()
        mock_flush_manager.flush_buffer_data = AsyncMock(return_value="task_123")

        hybrid_buffer = HybridBuffer(
            max_size=5,
            chunk_strategy="message",
            embedding_model="all-MiniLM-L6-v2",
            flush_manager=mock_flush_manager,
            auto_flush_interval=60.0,
            enable_auto_flush=True
        )

        try:
            await test_instance.test_parallel_embedding_generation_performance(
                hybrid_buffer, mock_embedding_model, mock_chunk_strategy
            )
            await test_instance.test_concurrent_embedding_generation(
                hybrid_buffer, mock_embedding_model, mock_chunk_strategy
            )
            await test_instance.test_embedding_error_handling(
                hybrid_buffer, mock_embedding_model, mock_chunk_strategy
            )
            await test_instance.test_semaphore_concurrency_control(
                hybrid_buffer, mock_embedding_model, mock_chunk_strategy
            )
            await test_instance.test_buffer_flush_performance(
                hybrid_buffer, mock_flush_manager
            )
        except Exception as e:
            print(f"❌ Test failed: {e}")
            raise

    asyncio.run(run_async_tests())
    print("✅ All buffer optimization tests passed!")
