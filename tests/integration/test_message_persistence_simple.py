"""
Simplified integration test for message persistence.

This test demonstrates the persistence issue between BufferService
and immediate reads, simulating the Postman scenario.
"""

import pytest
import asyncio
import time
import uuid
from omegaconf import OmegaConf

from memfuse_core.services.buffer_service import BufferService
from memfuse_core.services.memory_service import MemoryService
from memfuse_core.services.service_factory import ServiceFactory
from memfuse_core.utils.config import config_manager


class TestMessagePersistenceSimple:
    """Simplified integration tests for message persistence."""

    @pytest.fixture
    def user_name(self):
        """Generate unique user name for testing."""
        return f"test-user-{str(uuid.uuid4())[:8]}"

    @pytest.fixture
    def session_id(self):
        """Generate unique session ID for testing."""
        return str(uuid.uuid4())

    @pytest.mark.asyncio
    async def test_buffer_service_immediate_read(self, user_name, session_id):
        """Test that BufferService can read messages immediately after adding them."""
        # Reset service instances
        ServiceFactory.reset()
        
        # Get BufferService instance
        buffer_service = await ServiceFactory.get_buffer_service_for_user(user_name)
        assert buffer_service is not None
        
        # Add messages
        messages = [
            {"role": "user", "content": "Hello from buffer test"},
            {"role": "assistant", "content": "Hi there from buffer!"}
        ]
        
        # Add messages to buffer
        add_result = await buffer_service.add(messages, session_id=session_id)
        assert add_result["status"] == "success"
        message_ids = add_result["data"]["message_ids"]
        assert len(message_ids) == 2
        
        # Read messages immediately (this should work)
        read_result = await buffer_service.read(message_ids)
        print(f"Immediate read result: {read_result}")
        
        # This is the key test - can we read what we just added?
        if read_result["status"] == "success":
            read_messages = read_result["data"]["messages"]
            assert len(read_messages) == 2
            print("✅ PASS: Messages can be read immediately after adding")
        else:
            print("❌ FAIL: Cannot read messages immediately after adding")
            print(f"Error: {read_result}")
            pytest.fail("Cannot read messages immediately after adding")

    @pytest.mark.asyncio
    async def test_buffer_service_delayed_read(self, user_name, session_id):
        """Test that BufferService can read messages after a delay."""
        # Reset service instances
        ServiceFactory.reset()
        
        # Get BufferService instance
        buffer_service = await ServiceFactory.get_buffer_service_for_user(user_name)
        assert buffer_service is not None
        
        # Add messages
        messages = [
            {"role": "user", "content": "Hello from delayed test"},
            {"role": "assistant", "content": "Hi there from delayed test!"}
        ]
        
        # Add messages to buffer
        add_result = await buffer_service.add(messages, session_id=session_id)
        assert add_result["status"] == "success"
        message_ids = add_result["data"]["message_ids"]
        
        # Wait to simulate delay between requests (like Postman)
        print("Waiting 3 seconds to simulate delay...")
        await asyncio.sleep(3)
        
        # Try to read messages after delay
        read_result = await buffer_service.read(message_ids)
        print(f"Delayed read result: {read_result}")
        
        # This test will show if persistence works across delays
        if read_result["status"] == "success":
            read_messages = read_result["data"]["messages"]
            assert len(read_messages) == 2
            print("✅ PASS: Messages persist across delays")
        else:
            print("❌ FAIL: Messages do not persist across delays")
            print(f"Error: {read_result}")
            # This failure would explain the Postman behavior
            pytest.fail("Messages do not persist across delays")

    @pytest.mark.asyncio
    async def test_buffer_service_cross_instance_read(self, user_name, session_id):
        """Test that messages persist across different BufferService instances."""
        # Reset service instances
        ServiceFactory.reset()
        
        # Get first BufferService instance
        buffer_service_1 = await ServiceFactory.get_buffer_service_for_user(user_name)
        assert buffer_service_1 is not None
        
        # Add messages with first instance
        messages = [
            {"role": "user", "content": "Hello from instance 1"},
            {"role": "assistant", "content": "Hi there from instance 1!"}
        ]
        
        add_result = await buffer_service_1.add(messages, session_id=session_id)
        assert add_result["status"] == "success"
        message_ids = add_result["data"]["message_ids"]
        
        # Reset services to simulate new API request
        ServiceFactory.reset()
        
        # Get second BufferService instance (simulating new request)
        buffer_service_2 = await ServiceFactory.get_buffer_service_for_user(user_name)
        assert buffer_service_2 is not None
        
        # Try to read messages with second instance
        read_result = await buffer_service_2.read(message_ids)
        print(f"Cross-instance read result: {read_result}")
        
        # This test simulates the Postman scenario most closely
        if read_result["status"] == "success":
            read_messages = read_result["data"]["messages"]
            assert len(read_messages) == 2
            print("✅ PASS: Messages persist across different service instances")
        else:
            print("❌ FAIL: Messages do not persist across different service instances")
            print(f"Error: {read_result}")
            print("This explains why Postman can't read messages added in previous requests")
            pytest.fail("Messages do not persist across different service instances")

    @pytest.mark.asyncio
    async def test_buffer_service_stats_after_operations(self, user_name, session_id):
        """Test buffer statistics to understand the internal state."""
        # Reset service instances
        ServiceFactory.reset()
        
        # Get BufferService instance
        buffer_service = await ServiceFactory.get_buffer_service_for_user(user_name)
        assert buffer_service is not None
        
        # Get initial stats
        initial_stats = await buffer_service.get_buffer_stats()
        print(f"Initial buffer stats: {initial_stats}")
        
        # Add messages
        messages = [
            {"role": "user", "content": "Stats test message"},
            {"role": "assistant", "content": "Stats test response"}
        ]
        
        add_result = await buffer_service.add(messages, session_id=session_id)
        assert add_result["status"] == "success"
        
        # Get stats after adding
        after_add_stats = await buffer_service.get_buffer_stats()
        print(f"Stats after adding: {after_add_stats}")
        
        # This will show us where the messages are stored
        assert after_add_stats["total_items_added"] > initial_stats["total_items_added"]
        print("✅ Messages were added to buffer system")
        
        # Check if messages are in RoundBuffer or HybridBuffer
        if after_add_stats["round_buffer"]["total_rounds"] > 0:
            print("📝 Messages are in RoundBuffer (short-term storage)")
        if after_add_stats["hybrid_buffer"]["total_chunks"] > 0:
            print("📝 Messages are in HybridBuffer (medium-term storage)")
        
        print("This helps explain the persistence behavior observed in Postman") 