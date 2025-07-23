"""Test script for BufferService simplification.

This script tests the simplified BufferService implementation that delegates
concrete processing to WriteBuffer while maintaining service layer responsibilities.
"""

import asyncio
import time
import uuid
from typing import List, Dict, Any

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from memfuse_core.services.buffer_service import BufferService
from memfuse_core.interfaces import MessageList, MessageBatchList


class MockMemoryService:
    """Mock memory service for testing."""
    
    def __init__(self, user_id: str = "test_user"):
        self._user_id = user_id
        self.add_batch_calls = []
        self.query_calls = []
    
    async def add_batch(self, rounds: List[MessageList]) -> Dict[str, Any]:
        """Mock add_batch method."""
        self.add_batch_calls.append(rounds)
        return {
            "status": "success",
            "message": f"Added {len(rounds)} rounds",
            "data": {"processed": len(rounds)}
        }
    
    async def query(self, query: str, **kwargs) -> Dict[str, Any]:
        """Mock query method."""
        self.query_calls.append((query, kwargs))
        return {
            "status": "success",
            "data": {"results": [{"content": f"Mock result for: {query}"}]}
        }


def create_test_message(content: str, session_id: str = "default") -> Dict[str, Any]:
    """Create a test message."""
    return {
        "content": content,
        "role": "user",
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


async def test_service_metadata_addition():
    """Test that service-level metadata is added correctly."""
    print("\n=== Test: Service Metadata Addition ===")
    
    # Create BufferService with mock memory service
    mock_memory_service = MockMemoryService("test_user_123")
    config = {
        'buffer': {
            'round_buffer': {'max_tokens': 800, 'max_size': 5, 'token_model': 'gpt-4o-mini'},
            'hybrid_buffer': {'max_size': 5, 'chunk_strategy': 'message', 'embedding_model': 'all-MiniLM-L6-v2'},
            'performance': {'max_flush_workers': 2, 'max_flush_queue_size': 100, 'flush_timeout': 30.0}
        }
    }
    
    buffer_service = BufferService(
        memory_service=mock_memory_service,
        user="test_user_123",
        config=config
    )
    
    # Create test batch without user_id or session_id in metadata
    test_batch = [
        [{"content": "Message 1", "role": "user"}],
        [{"content": "Message 2", "role": "user"}]
    ]
    
    # Test service metadata addition
    processed_batch = buffer_service._add_service_metadata(test_batch, "test_session")
    
    # Verify user_id and session_id were added
    for message_list in processed_batch:
        for message in message_list:
            assert 'metadata' in message
            assert message['metadata']['user_id'] == "test_user_123"
            assert message['metadata']['session_id'] == "test_session"
    
    print("✅ Service metadata addition test passed")


async def test_simplified_add_batch():
    """Test simplified add_batch method."""
    print("\n=== Test: Simplified add_batch ===")
    
    # Create BufferService with mock memory service
    mock_memory_service = MockMemoryService("test_user_456")
    config = {
        'buffer': {
            'round_buffer': {'max_tokens': 800, 'max_size': 5, 'token_model': 'gpt-4o-mini'},
            'hybrid_buffer': {'max_size': 5, 'chunk_strategy': 'message', 'embedding_model': 'all-MiniLM-L6-v2'},
            'performance': {'max_flush_workers': 2, 'max_flush_queue_size': 100, 'flush_timeout': 30.0}
        }
    }
    
    buffer_service = BufferService(
        memory_service=mock_memory_service,
        user="test_user_456",
        config=config
    )
    
    # Create test batch
    test_batch = create_test_batch(batch_size=3, messages_per_list=2, session_id="test_session")
    
    # Test simplified add_batch
    result = await buffer_service.add_batch(test_batch, session_id="test_session")
    
    print(f"Add batch result: {result}")
    
    # Verify response structure
    assert result["status"] == "success"
    assert result["code"] == 200
    assert "data" in result
    assert result["data"]["batch_size"] == 3
    assert result["data"]["total_messages"] == 6
    assert "strategy_used" in result["data"]
    assert "processing_time" in result["data"]
    
    print("✅ Simplified add_batch test passed")


async def test_service_statistics_update():
    """Test service statistics are updated correctly."""
    print("\n=== Test: Service Statistics Update ===")
    
    # Create BufferService with mock memory service
    mock_memory_service = MockMemoryService("test_user_789")
    config = {
        'buffer': {
            'round_buffer': {'max_tokens': 800, 'max_size': 5, 'token_model': 'gpt-4o-mini'},
            'hybrid_buffer': {'max_size': 5, 'chunk_strategy': 'message', 'embedding_model': 'all-MiniLM-L6-v2'},
            'performance': {'max_flush_workers': 2, 'max_flush_queue_size': 100, 'flush_timeout': 30.0}
        }
    }
    
    buffer_service = BufferService(
        memory_service=mock_memory_service,
        user="test_user_789",
        config=config
    )
    
    # Record initial statistics
    initial_batch_writes = buffer_service.total_batch_writes
    initial_items_added = buffer_service.total_items_added
    initial_transfers = buffer_service.total_transfers
    
    # Create and process test batch
    test_batch = create_test_batch(batch_size=2, messages_per_list=3, session_id="stats_test")
    result = await buffer_service.add_batch(test_batch, session_id="stats_test")
    
    # Verify statistics were updated
    assert buffer_service.total_batch_writes == initial_batch_writes + 1
    assert buffer_service.total_items_added == initial_items_added + 6  # 2 lists * 3 messages
    # Note: transfers depend on buffer state and may vary
    
    print(f"Statistics updated - batch_writes: {buffer_service.total_batch_writes}, "
          f"items_added: {buffer_service.total_items_added}, transfers: {buffer_service.total_transfers}")
    
    print("✅ Service statistics update test passed")


async def test_response_formatting():
    """Test response formatting functionality."""
    print("\n=== Test: Response Formatting ===")
    
    # Create BufferService with mock memory service
    mock_memory_service = MockMemoryService("test_user_format")
    config = {
        'buffer': {
            'round_buffer': {'max_tokens': 800, 'max_size': 5, 'token_model': 'gpt-4o-mini'},
            'hybrid_buffer': {'max_size': 5, 'chunk_strategy': 'message', 'embedding_model': 'all-MiniLM-L6-v2'},
            'performance': {'max_flush_workers': 2, 'max_flush_queue_size': 100, 'flush_timeout': 30.0}
        }
    }
    
    buffer_service = BufferService(
        memory_service=mock_memory_service,
        user="test_user_format",
        config=config
    )
    
    # Test response formatting with mock WriteBuffer result
    mock_result = {
        "status": "success",
        "total_messages": 5,
        "transfers_triggered": 1,
        "strategy_used": "sequential",
        "processing_time": 0.123
    }
    
    formatted_response = buffer_service._format_write_response(mock_result, batch_size=2)
    
    # Verify response format
    assert formatted_response["status"] == "success"
    assert formatted_response["code"] == 200
    assert formatted_response["data"]["batch_size"] == 2
    assert formatted_response["data"]["total_messages"] == 5
    assert formatted_response["data"]["transfers_triggered"] == 1
    assert formatted_response["data"]["strategy_used"] == "sequential"
    assert formatted_response["data"]["processing_time"] == 0.123
    assert "Added 2 message lists" in formatted_response["message"]
    
    print("✅ Response formatting test passed")


async def test_error_handling():
    """Test error handling in simplified add_batch."""
    print("\n=== Test: Error Handling ===")
    
    # Test with no memory service
    buffer_service = BufferService(
        memory_service=None,
        user="test_user_error",
        config={}
    )
    
    test_batch = create_test_batch(batch_size=1, messages_per_list=1)
    result = await buffer_service.add_batch(test_batch)
    
    # Verify error response
    assert result["status"] == "error"
    assert "No memory service available" in result["message"]
    
    # Test with empty batch
    mock_memory_service = MockMemoryService("test_user_empty")
    buffer_service_with_service = BufferService(
        memory_service=mock_memory_service,
        user="test_user_empty",
        config={}
    )
    
    empty_result = await buffer_service_with_service.add_batch([])
    assert empty_result["status"] == "success"
    assert "No message lists to add" in empty_result["message"]
    
    print("✅ Error handling test passed")


async def run_all_tests():
    """Run all BufferService simplification tests."""
    print("Starting BufferService Simplification Tests...")
    
    try:
        await test_service_metadata_addition()
        await test_simplified_add_batch()
        await test_service_statistics_update()
        await test_response_formatting()
        await test_error_handling()
        
        print("\n🎉 All BufferService simplification tests passed!")
        
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
        print("\n✅ BufferService simplification implementation is working correctly!")
        exit(0)
    else:
        print("\n❌ BufferService simplification tests failed!")
        exit(1)
