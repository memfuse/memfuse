"""Comprehensive Buffer Bypass Functionality Tests.

This test suite verifies that BufferService correctly handles enabled/disabled modes
and ensures complete bypass of buffer components when enabled=false.
"""

import asyncio
import sys
import os
import time
from typing import Dict, Any

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from memfuse_core.services.buffer_service import BufferService


class DetailedMockMemoryService:
    """Enhanced mock memory service with detailed tracking for bypass testing."""

    def __init__(self, user_id: str = "test_user"):
        self._user_id = user_id
        self.multi_path_retrieval = "initialized"  # Simulate initialized state

        # Detailed call tracking
        self.add_batch_calls = []
        self.query_calls = []
        self.initialize_calls = []
        self.get_messages_by_session_calls = []

        # Data flow tracking
        self.data_flow_log = []
        self.processing_times = []

        # Configuration tracking
        self.use_parallel_layers = True
        self.memory_layer = MockMemoryLayer()

    def log_data_flow(self, operation: str, data_info: Dict[str, Any]):
        """Log data flow for analysis."""
        self.data_flow_log.append({
            "timestamp": time.time(),
            "operation": operation,
            "data_info": data_info,
            "user_id": self._user_id
        })

    async def initialize(self, cfg=None):
        """Mock initialize method with tracking."""
        self.initialize_calls.append(cfg)
        self.log_data_flow("initialize", {"config": cfg is not None})

    async def add_batch(self, message_batch_list):
        """Mock add_batch method with detailed tracking."""
        start_time = time.time()

        # Track the call
        self.add_batch_calls.append(message_batch_list)

        # Log data flow
        data_info = {
            "batch_count": len(message_batch_list),
            "total_messages": sum(len(ml) for ml in message_batch_list),
            "first_message_content": message_batch_list[0][0]["content"] if message_batch_list and message_batch_list[0] else None,
            "processing_method": "parallel_layers" if self.use_parallel_layers else "traditional"
        }
        self.log_data_flow("add_batch", data_info)

        # Simulate Memory Layer processing
        if self.use_parallel_layers and self.memory_layer:
            await self.memory_layer.write_parallel(message_batch_list)

        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)

        return {
            "status": "success",
            "message": f"Added {len(message_batch_list)} message lists",
            "data": {
                "processed": len(message_batch_list),
                "total_messages": sum(len(ml) for ml in message_batch_list),
                "processing_time": processing_time,
                "method": "memory_service_direct"
            }
        }

    async def query(self, query, **kwargs):
        """Mock query method with detailed tracking."""
        self.query_calls.append((query, kwargs))

        # Log data flow
        self.log_data_flow("query", {
            "query": query,
            "kwargs": list(kwargs.keys()),
            "top_k": kwargs.get("top_k", "not_specified")
        })

        return {
            "status": "success",
            "data": {
                "results": [{"content": f"Mock result for: {query}"}],
                "total": 1,
                "method": "memory_service_direct"
            }
        }

    async def get_messages_by_session(self, session_id, **kwargs):
        """Mock get_messages_by_session method."""
        self.get_messages_by_session_calls.append((session_id, kwargs))

        self.log_data_flow("get_messages_by_session", {
            "session_id": session_id,
            "kwargs": list(kwargs.keys())
        })

        return [
            {"content": f"Mock message for session {session_id}", "role": "user"}
        ]


class MockMemoryLayer:
    """Mock Memory Layer for testing."""

    def __init__(self):
        self.write_parallel_calls = []
        self.query_calls = []

    async def write_parallel(self, message_batch_list, session_id=None, metadata=None):
        """Mock parallel write method."""
        self.write_parallel_calls.append({
            "message_batch_list": message_batch_list,
            "session_id": session_id,
            "metadata": metadata
        })

        return {
            "success": True,
            "message": f"Processed {len(message_batch_list)} batches in parallel",
            "layer_results": {
                "M0": {"status": "success", "processed": len(message_batch_list)},
                "M1": {"status": "success", "processed": len(message_batch_list)},
                "M2": {"status": "success", "processed": len(message_batch_list)}
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

    mock_memory_service = DetailedMockMemoryService("test_user_enabled")

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
    assert buffer_service.config_manager is not None

    print("✅ Buffer enabled mode: All components correctly initialized")

    # Test add_batch (should go through WriteBuffer, NOT directly to MemoryService)
    test_batch = create_test_batch(batch_size=2, messages_per_list=2)
    result = await buffer_service.add_batch(test_batch, session_id="test_session")

    # In normal mode, MemoryService.add_batch should NOT be called directly
    assert len(mock_memory_service.add_batch_calls) == 0
    assert len(mock_memory_service.data_flow_log) == 0  # No direct data flow to MemoryService
    print("✅ Buffer enabled mode: add_batch correctly uses WriteBuffer (bypasses MemoryService)")

    return True


async def test_buffer_disabled_mode():
    """Test BufferService with buffer disabled (bypass mode) - COMPREHENSIVE VERIFICATION."""
    print("\n=== Test: Buffer Disabled Mode (Comprehensive) ===")

    mock_memory_service = DetailedMockMemoryService("test_user_disabled")

    config = {
        'buffer': {
            'enabled': False,  # Buffer disabled - KEY TEST POINT
            'round_buffer': {'max_tokens': 800, 'max_size': 5},
            'hybrid_buffer': {'max_size': 5}
        }
    }

    buffer_service = BufferService(
        memory_service=mock_memory_service,
        user="test_user_disabled",
        config=config
    )

    # CRITICAL VERIFICATION: Buffer components are NOT initialized
    assert buffer_service.buffer_enabled == False
    assert buffer_service.write_buffer is None
    assert buffer_service.query_buffer is None
    assert buffer_service.speculative_buffer is None
    assert buffer_service.config_manager is None
    assert buffer_service.use_rerank is False

    print("✅ Buffer disabled mode: ALL buffer components correctly NOT initialized")

    # CRITICAL TEST: add_batch should go DIRECTLY to MemoryService
    test_batch = create_test_batch(batch_size=2, messages_per_list=3)
    result = await buffer_service.add_batch(test_batch, session_id="test_session_bypass")

    # VERIFY: Direct MemoryService call
    assert len(mock_memory_service.add_batch_calls) == 1
    assert len(mock_memory_service.add_batch_calls[0]) == 2  # 2 message lists
    assert result["status"] == "success"
    assert result["data"]["mode"] == "bypass"

    # VERIFY: Data flow logging shows direct path
    assert len(mock_memory_service.data_flow_log) >= 1
    add_batch_log = next((log for log in mock_memory_service.data_flow_log if log["operation"] == "add_batch"), None)
    assert add_batch_log is not None
    assert add_batch_log["data_info"]["batch_count"] == 2
    assert add_batch_log["data_info"]["total_messages"] == 6  # 2 batches * 3 messages each

    print("✅ Buffer disabled mode: add_batch CORRECTLY bypasses to MemoryService")
    print(f"   📊 Data flow verified: {len(mock_memory_service.data_flow_log)} operations logged")

    # CRITICAL TEST: query should go DIRECTLY to MemoryService
    query_result = await buffer_service.query("test query for bypass", top_k=5)

    # VERIFY: Direct MemoryService query call
    assert len(mock_memory_service.query_calls) == 1
    assert mock_memory_service.query_calls[0][0] == "test query for bypass"
    assert query_result["status"] == "success"
    assert query_result["data"]["mode"] == "bypass"

    # VERIFY: Query data flow logging
    query_log = next((log for log in mock_memory_service.data_flow_log if log["operation"] == "query"), None)
    assert query_log is not None
    assert query_log["data_info"]["query"] == "test query for bypass"
    assert query_log["data_info"]["top_k"] == 5

    print("✅ Buffer disabled mode: query CORRECTLY bypasses to MemoryService")

    # VERIFY: Memory Layer processing (M0/M1/M2 parallel processing)
    assert len(mock_memory_service.memory_layer.write_parallel_calls) == 1
    parallel_call = mock_memory_service.memory_layer.write_parallel_calls[0]
    assert len(parallel_call["message_batch_list"]) == 2

    print("✅ Buffer disabled mode: Memory Layer (M0/M1/M2) processing CORRECTLY triggered")

    return True


async def test_initialization_bypass():
    """Test initialization in bypass mode."""
    print("\n=== Test: Initialization Bypass ===")

    mock_memory_service = DetailedMockMemoryService("test_user_init")

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
    assert success is True

    # Verify MemoryService.initialize was called
    assert len(mock_memory_service.initialize_calls) == 1

    print("✅ Bypass mode initialization successful")

    return True


async def test_configuration_edge_cases():
    """Test edge cases in configuration."""
    print("\n=== Test: Configuration Edge Cases ===")

    mock_memory_service = DetailedMockMemoryService("test_user_edge")

    # Test 1: Missing buffer config (should default to enabled)
    config1 = {}
    buffer_service1 = BufferService(
        memory_service=mock_memory_service,
        user="test_user_edge1",
        config=config1
    )
    assert buffer_service1.buffer_enabled is True  # Default to enabled
    print("✅ Missing buffer config defaults to enabled")

    # Test 2: Missing enabled field (should default to enabled)
    config2 = {'buffer': {}}
    buffer_service2 = BufferService(
        memory_service=mock_memory_service,
        user="test_user_edge2",
        config=config2
    )
    assert buffer_service2.buffer_enabled is True  # Default to enabled
    print("✅ Missing enabled field defaults to enabled")

    # Test 3: Explicit enabled: true
    config3 = {'buffer': {'enabled': True}}
    buffer_service3 = BufferService(
        memory_service=mock_memory_service,
        user="test_user_edge3",
        config=config3
    )
    assert buffer_service3.buffer_enabled is True
    print("✅ Explicit enabled: true works correctly")

    return True


async def test_data_flow_verification():
    """Test comprehensive data flow verification in bypass mode."""
    print("\n=== Test: Data Flow Verification (Bypass Mode) ===")

    mock_memory_service = DetailedMockMemoryService("test_user_flow")

    config = {
        'buffer': {
            'enabled': False  # Buffer disabled - BYPASS MODE
        }
    }

    buffer_service = BufferService(
        memory_service=mock_memory_service,
        user="test_user_flow",
        config=config
    )

    # Clear any existing logs
    mock_memory_service.data_flow_log.clear()

    # Test multiple operations to verify consistent bypass behavior
    test_batch1 = create_test_batch(batch_size=1, messages_per_list=2, session_id="session_1")
    test_batch2 = create_test_batch(batch_size=2, messages_per_list=1, session_id="session_2")

    # Operation 1: add_batch
    result1 = await buffer_service.add_batch(test_batch1, session_id="session_1")

    # Operation 2: query
    result2 = await buffer_service.query("test query 1", top_k=3)

    # Operation 3: another add_batch
    result3 = await buffer_service.add_batch(test_batch2, session_id="session_2")

    # Operation 4: get_messages_by_session
    result4 = await buffer_service.get_messages_by_session("session_1")

    # VERIFY: All operations went directly to MemoryService
    assert len(mock_memory_service.add_batch_calls) == 2  # Two add_batch calls
    assert len(mock_memory_service.query_calls) == 1  # One query call
    assert len(mock_memory_service.get_messages_by_session_calls) == 1  # One session call

    # VERIFY: Data flow logs show all operations
    assert len(mock_memory_service.data_flow_log) >= 4

    operations = [log["operation"] for log in mock_memory_service.data_flow_log]
    assert "add_batch" in operations
    assert "query" in operations
    assert "get_messages_by_session" in operations

    # VERIFY: Memory Layer was triggered for add_batch operations
    assert len(mock_memory_service.memory_layer.write_parallel_calls) == 2

    print("✅ Data flow verification: ALL operations correctly bypass buffer")
    print(f"   📊 Total operations logged: {len(mock_memory_service.data_flow_log)}")
    print(f"   📊 Memory Layer calls: {len(mock_memory_service.memory_layer.write_parallel_calls)}")

    return True


async def run_all_tests():
    """Run all comprehensive bypass functionality tests."""
    print("🔍 Starting Comprehensive Buffer Bypass Functionality Tests...")
    print("=" * 80)

    try:
        # Test enabled mode (normal buffer operation)
        await test_buffer_enabled_mode()

        # Test disabled mode (bypass operation) - CRITICAL TEST
        await test_buffer_disabled_mode()

        # Test initialization in bypass mode
        await test_initialization_bypass()

        # Test configuration edge cases
        await test_configuration_edge_cases()

        # Test comprehensive data flow verification
        await test_data_flow_verification()

        print("\n" + "=" * 80)
        print("✅ ALL COMPREHENSIVE BUFFER BYPASS TESTS PASSED!")
        print("=" * 80)
        print("📊 Test Summary:")
        print("   • Buffer enabled mode: ✅ Components initialized, WriteBuffer used")
        print("   • Buffer disabled mode: ✅ Components NOT initialized, direct MemoryService")
        print("   • Bypass initialization: ✅ Only MemoryService initialized")
        print("   • Configuration edge cases: ✅ Defaults work correctly")
        print("   • Data flow verification: ✅ All operations bypass buffer correctly")
        print("=" * 80)
        print("🎯 CONCLUSION: Buffer bypass functionality is WORKING CORRECTLY")
        print("   When enabled=false, system completely bypasses buffer components")
        print("   and sends data directly through Memory Service → Memory Layer")
        print("=" * 80)

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
