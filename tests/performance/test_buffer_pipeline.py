"""Test buffer pipeline optimization performance."""

import asyncio
import time
from pathlib import Path
import sys
from unittest.mock import AsyncMock, MagicMock

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from memfuse_core.buffer.write_buffer import WriteBuffer


async def test_pipeline_optimization():
    """Test buffer pipeline optimization performance."""
    print("🚀 Testing buffer pipeline optimization...")
    
    # Create mock configuration
    config = {
        "round_buffer": {
            "max_rounds": 5,
            "max_tokens": 1000
        },
        "hybrid_buffer": {
            "max_size": 10,
            "chunk_strategy": "message",
            "embedding_model": "all-MiniLM-L6-v2",
            "auto_flush_interval": 60.0,
            "enable_auto_flush": True
        }
    }
    
    # Create WriteBuffer (it will create its own FlushManager internally)
    write_buffer = WriteBuffer(config)
    
    # Mock the token calculation to add delay
    original_calculate_batch_tokens = write_buffer._calculate_batch_tokens
    
    async def mock_calculate_batch_tokens(batch):
        await asyncio.sleep(0.02)  # 20ms delay to simulate token calculation
        return 500  # Mock token count
    
    write_buffer._calculate_batch_tokens = mock_calculate_batch_tokens
    
    # Create test batch data
    message_batch_list = [
        [{"role": "user", "content": f"Test message {i}-{j}"} for j in range(3)]
        for i in range(5)  # 5 batches, 3 messages each
    ]
    
    # Test pipeline processing
    start_time = time.time()
    result = await write_buffer.add_batch(message_batch_list, session_id="test_session")
    pipeline_time = time.time() - start_time
    
    print(f"✅ Pipeline processing completed in {pipeline_time:.3f}s")
    print(f"✅ Processed {result['batch_size']} batches with {result['total_messages']} messages")
    print(f"✅ Strategy: {result.get('strategy', 'unknown')}")
    
    # Pipeline should be faster than sequential processing
    # With 20ms token calculation delay, pipeline should overlap operations
    expected_max_time = 0.05  # 50ms (should be faster than 20ms + processing time)
    
    if pipeline_time < expected_max_time:
        print(f"🎉 Pipeline optimization effective! {pipeline_time:.3f}s < {expected_max_time}s")
        return True
    else:
        print(f"⚠️  Pipeline performance not optimal: {pipeline_time:.3f}s >= {expected_max_time}s")
        return False


async def test_parallel_operations():
    """Test that operations run in parallel in the pipeline."""
    print("\n🚀 Testing parallel operations in pipeline...")
    
    # Track operation timing
    operation_times = {}
    
    async def mock_preprocess_batch(batch, session_id):
        start = time.time()
        await asyncio.sleep(0.01)  # 10ms preprocessing
        operation_times['preprocess'] = time.time() - start
        return batch
    
    async def mock_calculate_batch_tokens(batch):
        start = time.time()
        await asyncio.sleep(0.015)  # 15ms token calculation
        operation_times['tokens'] = time.time() - start
        return 300
    
    def mock_detect_session_changes(batch):
        start = time.time()
        time.sleep(0.005)  # 5ms session analysis (synchronous)
        operation_times['session'] = time.time() - start
        return []
    
    # Create WriteBuffer with mocked methods
    config = {
        "round_buffer": {"max_rounds": 5, "max_tokens": 1000},
        "hybrid_buffer": {"max_size": 10, "chunk_strategy": "message", "embedding_model": "all-MiniLM-L6-v2"}
    }
    
    write_buffer = WriteBuffer(config)
    
    # Replace methods with mocked versions
    write_buffer._preprocess_batch = mock_preprocess_batch
    write_buffer._calculate_batch_tokens = mock_calculate_batch_tokens
    write_buffer._detect_session_changes = mock_detect_session_changes
    
    # Mock other required methods
    write_buffer._plan_transfer_strategy_async = lambda x: {"type": "sequential", "finalized": False}
    write_buffer._finalize_transfer_strategy = lambda x, y: {"type": "sequential", "finalized": True}
    write_buffer._execute_batch_strategy = AsyncMock(return_value={"transfers": 0, "strategy": "test"})
    
    # Test data
    message_batch_list = [
        [{"role": "user", "content": "Test message"}]
    ]
    
    # Run pipeline
    start_time = time.time()
    await write_buffer.add_batch(message_batch_list)
    total_time = time.time() - start_time
    
    print(f"✅ Total pipeline time: {total_time:.3f}s")
    print(f"✅ Preprocessing time: {operation_times.get('preprocess', 0):.3f}s")
    print(f"✅ Token calculation time: {operation_times.get('tokens', 0):.3f}s")
    print(f"✅ Session analysis time: {operation_times.get('session', 0):.3f}s")
    
    # Calculate expected sequential time
    sequential_time = operation_times.get('preprocess', 0) + operation_times.get('tokens', 0) + operation_times.get('session', 0)
    speedup = sequential_time / total_time if total_time > 0 else 1
    
    print(f"✅ Expected sequential time: {sequential_time:.3f}s")
    print(f"🎉 Pipeline speedup: {speedup:.1f}x")
    
    # Pipeline should provide some speedup due to parallel operations
    if speedup > 1.2:
        print("🎉 Pipeline provides significant speedup through parallel operations!")
        return True
    else:
        print("⚠️  Pipeline speedup is minimal")
        return False


async def test_strategy_optimization():
    """Test transfer strategy optimization."""
    print("\n🚀 Testing transfer strategy optimization...")
    
    config = {
        "round_buffer": {"max_rounds": 5, "max_tokens": 1000},
        "hybrid_buffer": {"max_size": 10, "chunk_strategy": "message", "embedding_model": "all-MiniLM-L6-v2"}
    }
    
    write_buffer = WriteBuffer(config)
    
    # Test 1: Small batch should use sequential strategy
    small_session_changes = []
    small_tokens = 500
    
    preliminary_strategy = write_buffer._plan_transfer_strategy_async(small_session_changes)
    final_strategy = write_buffer._finalize_transfer_strategy(preliminary_strategy, small_tokens)
    
    print(f"✅ Small batch strategy: {final_strategy['type']}")
    assert final_strategy['type'] == 'sequential', f"Expected sequential, got {final_strategy['type']}"
    
    # Test 2: Large batch should use bulk transfer
    large_tokens = 2500  # > 2 * max_tokens (1000)
    
    preliminary_strategy = write_buffer._plan_transfer_strategy_async(small_session_changes)
    final_strategy = write_buffer._finalize_transfer_strategy(preliminary_strategy, large_tokens)
    
    print(f"✅ Large batch strategy: {final_strategy['type']}")
    assert final_strategy['type'] == 'bulk_transfer', f"Expected bulk_transfer, got {final_strategy['type']}"
    
    # Test 3: Multiple sessions should use session grouping
    multi_session_changes = [{"session": "s1"}, {"session": "s2"}]
    
    preliminary_strategy = write_buffer._plan_transfer_strategy_async(multi_session_changes)
    final_strategy = write_buffer._finalize_transfer_strategy(preliminary_strategy, small_tokens)
    
    print(f"✅ Multi-session strategy: {final_strategy['type']}")
    # Should remain session_grouped for multi-session even with small tokens
    assert final_strategy['type'] == 'session_grouped', f"Expected session_grouped, got {final_strategy['type']}"
    
    print("🎉 Transfer strategy optimization working correctly!")
    return True


async def main():
    """Run all pipeline optimization tests."""
    print("=" * 60)
    print("Phase 2 Buffer Pipeline Optimization Tests")
    print("=" * 60)
    
    try:
        # Test 1: Pipeline optimization
        result1 = await test_pipeline_optimization()
        
        # Test 2: Parallel operations
        result2 = await test_parallel_operations()
        
        # Test 3: Strategy optimization
        result3 = await test_strategy_optimization()
        
        if result1 and result2 and result3:
            print("\n🎉 All Phase 2 buffer pipeline optimization tests passed!")
            print("✅ Pipeline processing is working correctly")
            return True
        else:
            print("\n⚠️  Some pipeline tests did not meet expectations")
            return False
            
    except Exception as e:
        print(f"\n❌ Pipeline test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    if success:
        print("\n🎉 Phase 2 buffer pipeline optimization validation successful!")
    else:
        print("\n❌ Phase 2 buffer pipeline optimization validation failed!")
        sys.exit(1)
