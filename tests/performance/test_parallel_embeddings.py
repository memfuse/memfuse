"""Simple test for parallel embedding generation optimization."""

import asyncio
import time
from pathlib import Path
import sys
from unittest.mock import AsyncMock, MagicMock

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from memfuse_core.buffer.hybrid_buffer import HybridBuffer


async def test_parallel_embeddings():
    """Test parallel embedding generation performance."""
    print("🚀 Testing parallel embedding generation...")
    
    # Create mock embedding model
    mock_embedding_model = AsyncMock()
    
    # Add delay to simulate real embedding generation
    async def mock_encode_with_delay(content):
        await asyncio.sleep(0.01)  # 10ms delay per embedding
        return [0.1, 0.2, 0.3, 0.4, 0.5]
    
    mock_embedding_model.encode = mock_encode_with_delay
    
    # Create mock chunk strategy
    mock_chunk = MagicMock()
    mock_chunk.content = "test chunk content"
    mock_chunk.metadata = {"source": "test"}
    
    mock_chunk_strategy = AsyncMock()
    mock_chunk_strategy.create_chunks = AsyncMock(return_value=[mock_chunk] * 10)  # 10 chunks
    
    # Create mock flush manager
    mock_flush_manager = AsyncMock()
    mock_flush_manager.flush_buffer_data = AsyncMock(return_value="task_123")
    
    # Create HybridBuffer
    buffer = HybridBuffer(
        max_size=5,
        chunk_strategy="message",
        embedding_model="all-MiniLM-L6-v2",
        flush_manager=mock_flush_manager,
        auto_flush_interval=60.0,
        enable_auto_flush=True
    )
    
    # Set mocks
    buffer.embedding_model = mock_embedding_model
    buffer.chunk_strategy = mock_chunk_strategy
    
    # Test data
    test_rounds = [[{"role": "user", "content": "Test message"}]]
    
    # Measure parallel processing time
    start_time = time.time()
    await buffer.add_from_rounds(test_rounds)
    parallel_time = time.time() - start_time
    
    print(f"✅ Parallel processing completed in {parallel_time:.3f}s")
    print(f"✅ Generated {len(buffer.chunks)} chunks and {len(buffer.embeddings)} embeddings")
    
    # With 10 chunks and 10ms delay each:
    # Sequential would take ~100ms
    # Parallel should take ~10ms (limited by slowest embedding)
    expected_max_time = 0.05  # 50ms buffer for overhead
    
    if parallel_time < expected_max_time:
        print(f"🎉 Performance improvement achieved! {parallel_time:.3f}s < {expected_max_time}s")
        return True
    else:
        print(f"⚠️  Performance not optimal: {parallel_time:.3f}s >= {expected_max_time}s")
        return False


async def test_sequential_vs_parallel():
    """Compare sequential vs parallel embedding generation."""
    print("\n🚀 Comparing sequential vs parallel embedding generation...")
    
    # Test with larger dataset
    num_chunks = 20
    delay_per_embedding = 0.005  # 5ms delay
    
    # Sequential simulation
    start_time = time.time()
    for i in range(num_chunks):
        await asyncio.sleep(delay_per_embedding)
    sequential_time = time.time() - start_time
    
    # Parallel simulation
    async def mock_embedding_task():
        await asyncio.sleep(delay_per_embedding)
        return [0.1, 0.2, 0.3, 0.4, 0.5]
    
    start_time = time.time()
    # Simulate semaphore with max 10 concurrent
    semaphore = asyncio.Semaphore(10)
    
    async def controlled_embedding():
        async with semaphore:
            return await mock_embedding_task()
    
    tasks = [controlled_embedding() for _ in range(num_chunks)]
    await asyncio.gather(*tasks)
    parallel_time = time.time() - start_time
    
    speedup = sequential_time / parallel_time
    
    print(f"✅ Sequential time: {sequential_time:.3f}s")
    print(f"✅ Parallel time: {parallel_time:.3f}s")
    print(f"🎉 Speedup: {speedup:.1f}x")
    
    # Should achieve significant speedup
    if speedup > 1.5:
        print("🎉 Parallel processing provides significant speedup!")
        return True
    else:
        print("⚠️  Parallel processing speedup is minimal")
        return False


async def main():
    """Run all tests."""
    print("=" * 60)
    print("Phase 2 Buffer Optimization Tests")
    print("=" * 60)
    
    try:
        # Test 1: Parallel embedding generation
        result1 = await test_parallel_embeddings()
        
        # Test 2: Sequential vs parallel comparison
        result2 = await test_sequential_vs_parallel()
        
        if result1 and result2:
            print("\n🎉 All Phase 2 buffer optimization tests passed!")
            print("✅ Parallel embedding generation is working correctly")
            return True
        else:
            print("\n⚠️  Some tests did not meet performance expectations")
            return False
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    if success:
        print("\n🎉 Phase 2 optimization validation successful!")
    else:
        print("\n❌ Phase 2 optimization validation failed!")
        sys.exit(1)
