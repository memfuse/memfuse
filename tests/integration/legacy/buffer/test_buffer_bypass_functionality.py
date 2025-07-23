"""Test Buffer bypass functionality.

This test verifies that BufferService correctly handles enabled/disabled modes.
"""

import asyncio
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from memfuse_core.services.buffer_service import BufferService


class MockMemoryService:
    """Mock memory service for testing bypass functionality."""
    
    def __init__(self, user_id: str = "test_user"):
        self._user_id = user_id
        self.add_batch_calls = []
        self.query_calls = []
        self.multi_path_retrieval = "initialized"  # Simulate initialized state
    
    async def initialize(self, cfg=None):
        """Mock initialize method."""
        pass
    
    async def add_batch(self, message_batch_list):
        """Mock add_batch method."""
        self.add_batch_calls.append(message_batch_list)
        return {
            "status": "success",
            "message": f"Added {len(message_batch_list)} message lists",
            "data": {
                "processed": len(message_batch_list),
                "total_messages": sum(len(ml) for ml in message_batch_list)
            }
        }
    
    async def query(self, query, **kwargs):
        """Mock query method."""
        self.query_calls.append((query, kwargs))
        return {
            "status": "success",
            "data": {
                "results": [{"content": f"Mock result for: {query}"}],
                "total": 1
            }
        }


def create_test_message(content: str, session_id: str = "default"):
    """Create a test message."""
    return {
        "content": content,
        "role": "user",
        "metadata": {"session_id": session_id}
    }


def create_test_batch(batch_size: int, messages_per_list: int, session_id: str = "default"):
    """Create a test message batch."""
    return [
        [create_test_message(f"Message {i}-{j}", session_id) for j in range(messages_per_list)]
        for i in range(batch_size)
    ]


async def test_buffer_enabled_mode():
    """Test BufferService with buffer enabled (normal mode)."""
    print("\n=== Test: Buffer Enabled Mode ===")
    
    mock_memory_service = MockMemoryService("test_user_enabled")
    
    config = {
        'buffer': {
            'enabled': True,  # Buffer enabled
            'round_buffer': {'max_tokens': 800, 'max_size': 5, 'token_model': 'gpt-4o-mini'},
            'hybrid_buffer': {'max_size': 5, 'chunk_strategy': 'message', 'embedding_model': 'all-MiniLM-L6-v2'},
            'performance': {'max_flush_workers': 2, 'max_flush_queue_size': 100, 'flush_timeout': 30.0}
        }
    }
    
    buffer_service = BufferService(
        memory_service=mock_memory_service,
        user="test_user_enabled",
        config=config
    )
    
    # Verify buffer components are initialized
    assert buffer_service.buffer_enabled == True
    assert buffer_service.write_buffer is not None
    assert buffer_service.query_buffer is not None
    assert buffer_service.speculative_buffer is not None
    
    print("✅ Buffer enabled mode: Components correctly initialized")
    
    # Test add_batch (should go through WriteBuffer)
    test_batch = create_test_batch(batch_size=2, messages_per_list=2)
    result = await buffer_service.add_batch(test_batch, session_id="test_session")
    
    # In normal mode, MemoryService.add_batch should NOT be called directly
    assert len(mock_memory_service.add_batch_calls) == 0
    print("✅ Buffer enabled mode: add_batch correctly uses WriteBuffer")
    
    return True


async def test_buffer_disabled_mode():
    """Test BufferService with buffer disabled (bypass mode)."""
    print("\n=== Test: Buffer Disabled Mode ===")
    
    mock_memory_service = MockMemoryService("test_user_disabled")
    
    config = {
        'buffer': {
            'enabled': False,  # Buffer disabled
            'round_buffer': {'max_tokens': 800, 'max_size': 5},
            'hybrid_buffer': {'max_size': 5}
        }
    }
    
    buffer_service = BufferService(
        memory_service=mock_memory_service,
        user="test_user_disabled",
        config=config
    )
    
    # Verify buffer components are NOT initialized
    assert buffer_service.buffer_enabled == False
    assert buffer_service.write_buffer is None
    assert buffer_service.query_buffer is None
    assert buffer_service.speculative_buffer is None
    
    print("✅ Buffer disabled mode: Components correctly NOT initialized")
    
    # Test add_batch (should go directly to MemoryService)
    test_batch = create_test_batch(batch_size=2, messages_per_list=2)
    result = await buffer_service.add_batch(test_batch, session_id="test_session")
    
    # In bypass mode, MemoryService.add_batch should be called directly
    assert len(mock_memory_service.add_batch_calls) == 1
    assert len(mock_memory_service.add_batch_calls[0]) == 2  # 2 message lists
    assert result["status"] == "success"
    assert result["data"]["mode"] == "bypass"
    
    print("✅ Buffer disabled mode: add_batch correctly bypasses to MemoryService")
    
    # Test query (should go directly to MemoryService)
    query_result = await buffer_service.query("test query", top_k=5)
    
    # In bypass mode, MemoryService.query should be called directly
    assert len(mock_memory_service.query_calls) == 1
    assert mock_memory_service.query_calls[0][0] == "test query"
    assert query_result["status"] == "success"
    assert query_result["data"]["mode"] == "bypass"
    
    print("✅ Buffer disabled mode: query correctly bypasses to MemoryService")
    
    return True


async def test_initialization_bypass():
    """Test initialization in bypass mode."""
    print("\n=== Test: Initialization Bypass ===")
    
    mock_memory_service = MockMemoryService("test_user_init")
    
    config = {
        'buffer': {
            'enabled': False  # Buffer disabled
        }
    }
    
    buffer_service = BufferService(
        memory_service=mock_memory_service,
        user="test_user_init",
        config=config
    )
    
    # Test initialization
    success = await buffer_service.initialize()
    assert success == True
    
    print("✅ Bypass mode initialization successful")
    
    return True


async def test_configuration_edge_cases():
    """Test edge cases in configuration."""
    print("\n=== Test: Configuration Edge Cases ===")
    
    mock_memory_service = MockMemoryService("test_user_edge")
    
    # Test 1: Missing buffer config (should default to enabled)
    config1 = {}
    buffer_service1 = BufferService(
        memory_service=mock_memory_service,
        user="test_user_edge1",
        config=config1
    )
    assert buffer_service1.buffer_enabled == True  # Default to enabled
    print("✅ Missing buffer config defaults to enabled")
    
    # Test 2: Missing enabled field (should default to enabled)
    config2 = {'buffer': {}}
    buffer_service2 = BufferService(
        memory_service=mock_memory_service,
        user="test_user_edge2",
        config=config2
    )
    assert buffer_service2.buffer_enabled == True  # Default to enabled
    print("✅ Missing enabled field defaults to enabled")
    
    # Test 3: Explicit enabled: true
    config3 = {'buffer': {'enabled': True}}
    buffer_service3 = BufferService(
        memory_service=mock_memory_service,
        user="test_user_edge3",
        config=config3
    )
    assert buffer_service3.buffer_enabled == True
    print("✅ Explicit enabled: true works correctly")
    
    return True


async def run_all_tests():
    """Run all bypass functionality tests."""
    print("🔍 Starting Buffer Bypass Functionality Tests...")
    print("=" * 60)
    
    try:
        # Test enabled mode
        await test_buffer_enabled_mode()
        
        # Test disabled mode
        await test_buffer_disabled_mode()
        
        # Test initialization
        await test_initialization_bypass()
        
        # Test edge cases
        await test_configuration_edge_cases()
        
        print("\n" + "=" * 60)
        print("✅ ALL BUFFER BYPASS TESTS PASSED!")
        print("=" * 60)
        print("📊 Test Summary:")
        print("   • Buffer enabled mode: ✅ Working correctly")
        print("   • Buffer disabled mode: ✅ Working correctly")
        print("   • Bypass initialization: ✅ Working correctly")
        print("   • Configuration edge cases: ✅ Working correctly")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run tests
    success = asyncio.run(run_all_tests())
    
    if success:
        print("\n✅ Buffer bypass functionality tests completed successfully!")
        exit(0)
    else:
        print("\n❌ Buffer bypass functionality tests failed!")
        exit(1)
