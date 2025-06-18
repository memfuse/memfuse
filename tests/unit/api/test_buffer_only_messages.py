"""Tests for buffer_only parameter in messages API.

This test verifies that when buffer_only=true, messages returned from RoundBuffer
have proper id, created_at, and updated_at fields.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from memfuse_core.services.buffer_service import BufferService
from memfuse_core.services.memory_service import MemoryService
from memfuse_core.buffer.round_buffer import RoundBuffer


@pytest.fixture
def mock_memory_service():
    """Create a mock memory service."""
    mock_service = AsyncMock(spec=MemoryService)
    mock_service._user_id = "test_user_id"

    # Store messages for later retrieval
    stored_messages = []

    async def mock_add_batch(message_batch_list, **kwargs):
        # Store messages for later retrieval
        message_ids = []
        for message_list in message_batch_list:
            for message in message_list:
                # Create a copy with database-like structure
                stored_message = {
                    "id": message.get("id"),
                    "role": message.get("role"),
                    "content": message.get("content"),
                    "created_at": message.get("created_at"),
                    "updated_at": message.get("updated_at"),
                    "round_id": "test_round_id"
                }
                stored_messages.append(stored_message)
                message_ids.append(message.get("id"))

        return {
            "status": "success",
            "data": {"message_ids": message_ids}
        }

    async def mock_get_messages_by_session(session_id, **kwargs):
        # Return stored messages
        return stored_messages.copy()

    mock_service.add_batch = mock_add_batch
    mock_service.get_messages_by_session = mock_get_messages_by_session
    return mock_service


@pytest.fixture
def buffer_service(mock_memory_service):
    """Create a BufferService instance for testing."""
    config = {
        'buffer': {
            'round_buffer': {'max_tokens': 800, 'max_size': 5},
            'hybrid_buffer': {'max_size': 10}
        },
        'retrieval': {'use_rerank': False}
    }
    return BufferService(
        memory_service=mock_memory_service,
        user="test_user",
        config=config
    )


@pytest.fixture
def sample_messages():
    """Sample messages without id, created_at, updated_at fields."""
    return [
        {
            "role": "user",
            "content": "Hello, how are you?",
            "metadata": {"session_id": "session_1"}
        },
        {
            "role": "assistant", 
            "content": "I'm doing well, thank you!",
            "metadata": {"session_id": "session_1"}
        }
    ]


class TestBufferOnlyMessages:
    """Test class for buffer_only message functionality."""
    
    @pytest.mark.asyncio
    async def test_add_messages_generates_required_fields(self, buffer_service, sample_messages):
        """Test that adding messages generates id, created_at, updated_at fields."""
        session_id = "session_1"
        
        # Add messages to buffer service
        result = await buffer_service.add(sample_messages, session_id=session_id)
        
        # Verify the operation was successful
        assert result["status"] == "success"
        
        # Check that messages in RoundBuffer now have required fields
        round_buffer_messages = await buffer_service.round_buffer.get_all_messages_for_read_api()
        
        assert len(round_buffer_messages) == 2
        
        for message in round_buffer_messages:
            # Verify all required fields are present and not empty
            assert "id" in message
            assert message["id"] != ""
            assert message["id"] is not None
            
            assert "created_at" in message
            assert message["created_at"] != ""
            assert message["created_at"] is not None
            
            assert "updated_at" in message
            assert message["updated_at"] != ""
            assert message["updated_at"] is not None
            
            # Verify the timestamp format is valid ISO format
            try:
                datetime.fromisoformat(message["created_at"])
                datetime.fromisoformat(message["updated_at"])
            except ValueError:
                pytest.fail(f"Invalid timestamp format: created_at={message['created_at']}, updated_at={message['updated_at']}")
            
            # Verify metadata is preserved
            assert "metadata" in message
            assert message["metadata"]["session_id"] == session_id
            assert message["metadata"]["source"] == "round_buffer"
    
    @pytest.mark.asyncio
    async def test_buffer_only_returns_complete_messages(self, buffer_service, sample_messages):
        """Test that buffer_only=true returns messages with all required fields."""
        session_id = "session_1"

        # Add messages to buffer service
        await buffer_service.add(sample_messages, session_id=session_id)

        # Get messages with buffer_only=True
        buffer_only_messages = await buffer_service.get_messages_by_session(
            session_id=session_id,
            buffer_only=True
        )

        assert len(buffer_only_messages) == 2

        for message in buffer_only_messages:
            # Verify all required fields are present and not empty
            assert message.get("id") != ""
            assert message.get("created_at") != ""
            assert message.get("updated_at") != ""
            assert message.get("role") in ["user", "assistant"]
            assert message.get("content") != ""
            assert message.get("metadata", {}).get("source") == "round_buffer"

    @pytest.mark.asyncio
    async def test_buffer_and_storage_id_consistency(self, buffer_service, sample_messages):
        """Test that buffer and storage have consistent message IDs and created_at timestamps."""
        session_id = "session_1"

        # Add messages to buffer service
        await buffer_service.add(sample_messages, session_id=session_id)

        # Get messages from buffer
        buffer_messages = await buffer_service.get_messages_by_session(
            session_id=session_id,
            buffer_only=True
        )

        # Get messages from storage
        storage_messages = await buffer_service.get_messages_by_session(
            session_id=session_id,
            buffer_only=False
        )

        assert len(buffer_messages) == 2
        assert len(storage_messages) == 2

        # Sort both lists by content to ensure consistent comparison
        buffer_messages.sort(key=lambda x: x.get("content", ""))
        storage_messages.sort(key=lambda x: x.get("content", ""))

        for i in range(len(buffer_messages)):
            buffer_msg = buffer_messages[i]
            storage_msg = storage_messages[i]

            # Verify same content
            assert buffer_msg["content"] == storage_msg["content"]
            assert buffer_msg["role"] == storage_msg["role"]

            # Verify same ID and created_at timestamp
            assert buffer_msg["id"] == storage_msg["id"]
            assert buffer_msg["created_at"] == storage_msg["created_at"]

            # Storage should have updated_at >= created_at (database storage time)
            assert storage_msg["updated_at"] >= storage_msg["created_at"]
    
    @pytest.mark.asyncio
    async def test_messages_with_existing_fields_preserved(self, buffer_service):
        """Test that messages with existing id, created_at, updated_at are preserved."""
        existing_timestamp = "2023-01-01T12:00:00"
        existing_id = "existing_msg_id"
        
        messages_with_fields = [
            {
                "id": existing_id,
                "role": "user",
                "content": "Test message",
                "created_at": existing_timestamp,
                "updated_at": existing_timestamp,
                "metadata": {"session_id": "session_1"}
            }
        ]
        
        # Add messages to buffer service
        await buffer_service.add(messages_with_fields, session_id="session_1")
        
        # Get messages from buffer
        buffer_messages = await buffer_service.round_buffer.get_all_messages_for_read_api()
        
        assert len(buffer_messages) == 1
        message = buffer_messages[0]
        
        # Verify existing fields are preserved
        assert message["id"] == existing_id
        assert message["created_at"] == existing_timestamp
        assert message["updated_at"] == existing_timestamp
    
    @pytest.mark.asyncio
    async def test_empty_fields_are_populated(self, buffer_service):
        """Test that empty id, created_at, updated_at fields are populated."""
        messages_with_empty_fields = [
            {
                "id": "",
                "role": "user", 
                "content": "Test message",
                "created_at": "",
                "updated_at": "",
                "metadata": {"session_id": "session_1"}
            }
        ]
        
        # Add messages to buffer service
        await buffer_service.add(messages_with_empty_fields, session_id="session_1")
        
        # Get messages from buffer
        buffer_messages = await buffer_service.round_buffer.get_all_messages_for_read_api()
        
        assert len(buffer_messages) == 1
        message = buffer_messages[0]
        
        # Verify empty fields are now populated
        assert message["id"] != ""
        assert message["created_at"] != ""
        assert message["updated_at"] != ""
        
        # Verify the generated values are valid
        assert len(message["id"]) > 0  # UUID should be non-empty
        try:
            datetime.fromisoformat(message["created_at"])
            datetime.fromisoformat(message["updated_at"])
        except ValueError:
            pytest.fail(f"Invalid timestamp format generated")

    @pytest.mark.asyncio
    async def test_no_duplicate_messages_in_buffer_only_false(self, buffer_service, sample_messages):
        """Test that buffer_only=false doesn't return duplicate messages."""
        session_id = "session_1"

        # Add messages to buffer service
        await buffer_service.add(sample_messages, session_id=session_id)

        # Get messages with buffer_only=False (should combine buffer and storage)
        combined_messages = await buffer_service.get_messages_by_session(
            session_id=session_id,
            buffer_only=False
        )

        # Check that there are no duplicate message IDs
        message_ids = [msg.get("id") for msg in combined_messages]
        unique_message_ids = set(message_ids)

        assert len(message_ids) == len(unique_message_ids), f"Found duplicate message IDs: {message_ids}"

        # Verify we have the expected number of unique messages
        assert len(combined_messages) == 2

        # Verify each message has a unique ID
        for i, message in enumerate(combined_messages):
            assert message.get("id") != ""
            assert message.get("content") != ""
            # Check that no other message has the same ID
            other_ids = [msg.get("id") for j, msg in enumerate(combined_messages) if j != i]
            assert message.get("id") not in other_ids

    @pytest.mark.asyncio
    async def test_new_architecture_buffer_flow(self, buffer_service, sample_messages):
        """Test the new architecture: BufferService -> RoundBuffer -> HybridBuffer -> Database."""
        session_id = "session_1"

        # Add messages to buffer service (should only go to buffer, not directly to database)
        result = await buffer_service.add(sample_messages, session_id=session_id)

        # Verify the operation was successful
        assert result["status"] == "success"
        assert "buffer_status" in result  # New architecture should have buffer_status

        # Check that messages are in RoundBuffer initially
        round_buffer_messages = await buffer_service.round_buffer.get_all_messages_for_read_api()
        assert len(round_buffer_messages) == 2

        # Force transfer to HybridBuffer (simulate token limit exceeded)
        await buffer_service.round_buffer.force_transfer()

        # After transfer, RoundBuffer should be empty
        round_buffer_messages_after = await buffer_service.round_buffer.get_all_messages_for_read_api()
        assert len(round_buffer_messages_after) == 0

        # HybridBuffer should have auto-flushed to database and be empty
        hybrid_buffer_messages = await buffer_service.hybrid_buffer.get_all_messages_for_read_api()
        # With auto-flush enabled, HybridBuffer should be empty after flush
        assert len(hybrid_buffer_messages) == 0

        # Messages should now be in database only
        stored_messages = await buffer_service.get_messages_by_session(
            session_id=session_id,
            buffer_only=False
        )
        assert len(stored_messages) == 2

        # Verify messages have proper fields
        for message in stored_messages:
            assert message.get("id") != ""
            assert message.get("created_at") != ""
            assert message.get("updated_at") != ""
            assert message.get("content") != ""


if __name__ == "__main__":
    pytest.main([__file__])
