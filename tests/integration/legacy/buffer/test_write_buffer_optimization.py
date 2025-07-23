"""Test script for WriteBuffer batch processing optimization.

This script tests the enhanced WriteBuffer implementation with optimized
batch processing, including token calculation, session detection, and
transfer strategy optimization.
"""

import asyncio
import time
import uuid
from typing import List, Dict, Any

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from memfuse_core.buffer.write_buffer import WriteBuffer
from memfuse_core.interfaces import MessageList, MessageBatchList


class MockMemoryServiceHandler:
    """Mock memory service handler for testing."""
    
    def __init__(self):
        self.call_count = 0
        self.last_rounds = None
    
    async def __call__(self, rounds: List[MessageList]) -> None:
        """Mock handler that records calls."""
        self.call_count += 1
        self.last_rounds = rounds
        print(f"MockMemoryServiceHandler: Called with {len(rounds)} rounds")


def create_test_message(content: str, session_id: str = "default") -> Dict[str, Any]:
    """Create a test message with required fields."""
    return {
        "id": str(uuid.uuid4()),
        "content": content,
        "role": "user",
        "created_at": time.time(),
        "updated_at": time.time(),
        "metadata": {
            "session_id": session_id
        }
    }


def create_test_message_list(count: int, session_id: str = "default") -> MessageList:
    """Create a test message list."""
    return [create_test_message(f"Test message {i}", session_id) for i in range(count)]


def create_test_batch(batch_size: int, messages_per_list: int, session_id: str = "default") -> MessageBatchList:
    """Create a test message batch."""
    return [create_test_message_list(messages_per_list, session_id) for _ in range(batch_size)]


async def test_basic_batch_processing():
    """Test basic batch processing functionality."""
    print("\n=== Test: Basic Batch Processing ===")
    
    # Create WriteBuffer with test configuration
    config = {
        'round_buffer': {'max_tokens': 800, 'max_size': 5, 'token_model': 'gpt-4o-mini'},
        'hybrid_buffer': {'max_size': 5, 'chunk_strategy': 'message', 'embedding_model': 'all-MiniLM-L6-v2'},
        'flush_manager': {'max_workers': 2, 'max_queue_size': 100, 'default_timeout': 30.0}
    }
    
    mock_handler = MockMemoryServiceHandler()
    write_buffer = WriteBuffer(config=config, memory_service_handler=mock_handler)
    
    # Create test batch
    test_batch = create_test_batch(batch_size=3, messages_per_list=2)
    
    # Test batch processing
    start_time = time.time()
    result = await write_buffer.add_batch(test_batch, session_id="test_session")
    processing_time = time.time() - start_time
    
    print(f"Batch processing result: {result}")
    print(f"Processing time: {processing_time:.3f}s")
    
    # Verify results
    assert result["status"] == "success"
    assert result["batch_size"] == 3
    assert result["total_messages"] == 6
    assert "total_tokens" in result
    assert "strategy_used" in result
    
    print("✅ Basic batch processing test passed")


async def test_token_calculation_optimization():
    """Test optimized token calculation."""
    print("\n=== Test: Token Calculation Optimization ===")
    
    config = {
        'round_buffer': {'max_tokens': 800, 'max_size': 5, 'token_model': 'gpt-4o-mini'},
        'hybrid_buffer': {'max_size': 5, 'chunk_strategy': 'message', 'embedding_model': 'all-MiniLM-L6-v2'},
        'flush_manager': {'max_workers': 2, 'max_queue_size': 100, 'default_timeout': 30.0}
    }
    
    mock_handler = MockMemoryServiceHandler()
    write_buffer = WriteBuffer(config=config, memory_service_handler=mock_handler)
    
    # Create test batch with varying content lengths
    test_batch = [
        [create_test_message("Short message")],
        [create_test_message("This is a much longer message with more content to test token calculation")],
        [create_test_message("Medium length message for testing")]
    ]
    
    # Test token calculation
    total_tokens = await write_buffer._calculate_batch_tokens(test_batch)
    print(f"Total tokens calculated: {total_tokens}")
    
    # Verify token calculation is reasonable
    assert total_tokens > 0
    assert total_tokens < 1000  # Should be reasonable for test content
    
    print("✅ Token calculation optimization test passed")


async def test_session_change_detection():
    """Test session change detection."""
    print("\n=== Test: Session Change Detection ===")
    
    config = {
        'round_buffer': {'max_tokens': 800, 'max_size': 5, 'token_model': 'gpt-4o-mini'},
        'hybrid_buffer': {'max_size': 5, 'chunk_strategy': 'message', 'embedding_model': 'all-MiniLM-L6-v2'},
        'flush_manager': {'max_workers': 2, 'max_queue_size': 100, 'default_timeout': 30.0}
    }
    
    mock_handler = MockMemoryServiceHandler()
    write_buffer = WriteBuffer(config=config, memory_service_handler=mock_handler)
    
    # Create test batch with different sessions
    test_batch = [
        create_test_message_list(2, "session_1"),
        create_test_message_list(2, "session_1"),
        create_test_message_list(2, "session_2"),
        create_test_message_list(2, "session_3"),
        create_test_message_list(2, "session_2")
    ]
    
    # Test session change detection
    session_changes = write_buffer._detect_session_changes(test_batch)
    print(f"Session changes detected: {session_changes}")
    
    # Verify session changes are detected
    assert len(session_changes) > 0
    assert "session_1" in str(session_changes)
    assert "session_2" in str(session_changes)
    assert "session_3" in str(session_changes)
    
    print("✅ Session change detection test passed")


async def test_transfer_strategy_planning():
    """Test transfer strategy planning."""
    print("\n=== Test: Transfer Strategy Planning ===")
    
    config = {
        'round_buffer': {'max_tokens': 100, 'max_size': 5, 'token_model': 'gpt-4o-mini'},  # Low token limit for testing
        'hybrid_buffer': {'max_size': 5, 'chunk_strategy': 'message', 'embedding_model': 'all-MiniLM-L6-v2'},
        'flush_manager': {'max_workers': 2, 'max_queue_size': 100, 'default_timeout': 30.0}
    }
    
    mock_handler = MockMemoryServiceHandler()
    write_buffer = WriteBuffer(config=config, memory_service_handler=mock_handler)
    
    # Test high token count strategy
    high_token_strategy = write_buffer._plan_transfer_strategy(total_tokens=300, session_changes=[])
    print(f"High token strategy: {high_token_strategy}")
    assert high_token_strategy["type"] == "bulk_transfer"
    
    # Test multiple sessions strategy
    multi_session_strategy = write_buffer._plan_transfer_strategy(total_tokens=50, session_changes=["s1", "s2", "s3", "s4"])
    print(f"Multi-session strategy: {multi_session_strategy}")
    assert multi_session_strategy["type"] == "session_grouped"
    
    # Test standard strategy
    standard_strategy = write_buffer._plan_transfer_strategy(total_tokens=50, session_changes=["s1"])
    print(f"Standard strategy: {standard_strategy}")
    assert standard_strategy["type"] == "sequential"
    
    print("✅ Transfer strategy planning test passed")


async def test_batch_performance_comparison():
    """Test performance comparison between old and new batch processing."""
    print("\n=== Test: Batch Performance Comparison ===")
    
    config = {
        'round_buffer': {'max_tokens': 800, 'max_size': 5, 'token_model': 'gpt-4o-mini'},
        'hybrid_buffer': {'max_size': 5, 'chunk_strategy': 'message', 'embedding_model': 'all-MiniLM-L6-v2'},
        'flush_manager': {'max_workers': 2, 'max_queue_size': 100, 'default_timeout': 30.0}
    }
    
    mock_handler = MockMemoryServiceHandler()
    write_buffer = WriteBuffer(config=config, memory_service_handler=mock_handler)
    
    # Create larger test batch
    large_batch = create_test_batch(batch_size=10, messages_per_list=3)
    
    # Test optimized batch processing
    start_time = time.time()
    result = await write_buffer.add_batch(large_batch, session_id="perf_test")
    optimized_time = time.time() - start_time
    
    print(f"Optimized batch processing time: {optimized_time:.3f}s")
    print(f"Strategy used: {result.get('strategy_used')}")
    print(f"Total tokens: {result.get('total_tokens')}")
    print(f"Transfers triggered: {result.get('transfers_triggered')}")
    
    # Verify performance metrics are recorded
    assert result["processing_time"] > 0
    assert result["processing_time"] < 10.0  # Should be reasonable for test data
    
    print("✅ Batch performance comparison test passed")


async def run_all_tests():
    """Run all WriteBuffer optimization tests."""
    print("Starting WriteBuffer Optimization Tests...")
    
    try:
        await test_basic_batch_processing()
        await test_token_calculation_optimization()
        await test_session_change_detection()
        await test_transfer_strategy_planning()
        await test_batch_performance_comparison()
        
        print("\n🎉 All WriteBuffer optimization tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    # Run tests
    success = asyncio.run(run_all_tests())
    
    if success:
        print("\n✅ WriteBuffer optimization implementation is working correctly!")
        exit(0)
    else:
        print("\n❌ WriteBuffer optimization tests failed!")
        exit(1)
