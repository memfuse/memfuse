"""Unit tests for list_messages API response format consistency.

This module tests that the list_messages API always returns a consistent
response format regardless of the underlying service implementation.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

from memfuse_core.api.messages import normalize_messages_response, list_messages
from memfuse_core.models.core import ApiResponse


class TestNormalizeMessagesResponse:
    """Test cases for the normalize_messages_response function."""

    def test_normalize_valid_list(self):
        """Test normalizing a valid list of message dictionaries."""
        messages = [
            {"id": "1", "role": "user", "content": "Hello"},
            {"id": "2", "role": "assistant", "content": "Hi there"}
        ]
        result = normalize_messages_response(messages)
        assert result == messages
        assert isinstance(result, list)
        assert len(result) == 2

    def test_normalize_empty_list(self):
        """Test normalizing an empty list."""
        result = normalize_messages_response([])
        assert result == []
        assert isinstance(result, list)

    def test_normalize_none(self):
        """Test normalizing None input."""
        result = normalize_messages_response(None)
        assert result == []
        assert isinstance(result, list)

    def test_normalize_single_dict(self):
        """Test normalizing a single message dictionary."""
        message = {"id": "1", "role": "user", "content": "Hello"}
        result = normalize_messages_response(message)
        assert result == [message]
        assert isinstance(result, list)
        assert len(result) == 1

    def test_normalize_nested_messages_key(self):
        """Test normalizing dictionary with 'messages' key."""
        data = {
            "messages": [
                {"id": "1", "role": "user", "content": "Hello"},
                {"id": "2", "role": "assistant", "content": "Hi"}
            ]
        }
        result = normalize_messages_response(data)
        assert result == data["messages"]
        assert isinstance(result, list)
        assert len(result) == 2

    def test_normalize_nested_data_messages(self):
        """Test normalizing dictionary with nested 'data.messages' structure."""
        data = {
            "data": {
                "messages": [
                    {"id": "1", "role": "user", "content": "Hello"}
                ]
            }
        }
        result = normalize_messages_response(data)
        assert result == data["data"]["messages"]
        assert isinstance(result, list)
        assert len(result) == 1

    def test_normalize_pydantic_models(self):
        """Test normalizing list containing Pydantic models."""
        # Mock Pydantic model
        mock_message = MagicMock()
        mock_message.model_dump.return_value = {"id": "1", "role": "user", "content": "Hello"}
        
        messages = [mock_message]
        result = normalize_messages_response(messages)
        
        assert len(result) == 1
        assert result[0] == {"id": "1", "role": "user", "content": "Hello"}
        mock_message.model_dump.assert_called_once()

    def test_normalize_objects_with_dict(self):
        """Test normalizing objects with __dict__ attribute."""
        class MockMessage:
            def __init__(self):
                self.id = "1"
                self.role = "user"
                self.content = "Hello"
        
        mock_obj = MockMessage()
        messages = [mock_obj]
        result = normalize_messages_response(messages)
        
        assert len(result) == 1
        assert result[0]["id"] == "1"
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello"

    def test_normalize_mixed_valid_invalid(self):
        """Test normalizing list with mix of valid and invalid items."""
        messages = [
            {"id": "1", "role": "user", "content": "Hello"},  # Valid
            "invalid_string",  # Invalid - should be skipped
            {"id": "2", "role": "assistant", "content": "Hi"},  # Valid
            123  # Invalid - should be skipped
        ]
        result = normalize_messages_response(messages)
        
        assert len(result) == 2
        assert result[0]["id"] == "1"
        assert result[1]["id"] == "2"

    def test_normalize_invalid_type_raises_error(self):
        """Test that invalid types raise ValueError."""
        with pytest.raises(ValueError, match="Cannot normalize messages of type"):
            normalize_messages_response("invalid_string")

    def test_normalize_iterable_conversion(self):
        """Test normalizing other iterable types."""
        # Test tuple
        messages_tuple = (
            {"id": "1", "role": "user", "content": "Hello"},
            {"id": "2", "role": "assistant", "content": "Hi"}
        )
        result = normalize_messages_response(messages_tuple)
        assert len(result) == 2
        assert isinstance(result, list)


class TestListMessagesAPIConsistency:
    """Test cases for list_messages API response consistency."""

    @pytest.fixture
    def mock_db_service(self):
        """Mock database service."""
        mock_db = MagicMock()
        mock_db.get_messages_by_session.return_value = [
            {"id": "1", "role": "user", "content": "Hello from DB"}
        ]
        return mock_db

    @pytest.fixture
    def mock_memory_service(self):
        """Mock memory service."""
        mock_service = AsyncMock()
        mock_service.get_messages_by_session.return_value = [
            {"id": "1", "role": "user", "content": "Hello from Memory"}
        ]
        return mock_service

    @pytest.fixture
    def mock_buffer_service(self):
        """Mock buffer service."""
        mock_service = AsyncMock()
        mock_service.get_messages_by_session.return_value = [
            {"id": "1", "role": "user", "content": "Hello from Buffer"}
        ]
        return mock_service

    @pytest.mark.asyncio
    async def test_memory_service_response_format(self, mock_db_service, mock_memory_service):
        """Test that MemoryService returns consistent format."""
        with patch('memfuse_core.api.messages.DatabaseService') as mock_db_class, \
             patch('memfuse_core.api.messages.get_service_for_session') as mock_get_service, \
             patch('memfuse_core.api.messages.ensure_session_exists') as mock_ensure_session:
            
            mock_db_class.get_instance.return_value = mock_db_service
            mock_get_service.return_value = mock_memory_service
            mock_ensure_session.return_value = {"id": "session_1"}
            
            response = await list_messages(
                session_id="session_1",
                limit="10",
                sort_by="timestamp",
                order="desc",
                buffer_only=None,
                _api_key_data={}
            )
            
            assert isinstance(response, ApiResponse)
            assert response.status == "success"
            assert "messages" in response.data
            assert isinstance(response.data["messages"], list)
            assert len(response.data["messages"]) == 1
            assert response.data["messages"][0]["content"] == "Hello from Memory"

    @pytest.mark.asyncio
    async def test_buffer_service_response_format(self, mock_db_service, mock_buffer_service):
        """Test that BufferService returns consistent format."""
        with patch('memfuse_core.api.messages.DatabaseService') as mock_db_class, \
             patch('memfuse_core.api.messages.get_service_for_session') as mock_get_service, \
             patch('memfuse_core.api.messages.ensure_session_exists') as mock_ensure_session:
            
            mock_db_class.get_instance.return_value = mock_db_service
            mock_get_service.return_value = mock_buffer_service
            mock_ensure_session.return_value = {"id": "session_1"}
            
            response = await list_messages(
                session_id="session_1",
                limit="10",
                sort_by="timestamp",
                order="desc",
                buffer_only="true",
                _api_key_data={}
            )
            
            assert isinstance(response, ApiResponse)
            assert response.status == "success"
            assert "messages" in response.data
            assert isinstance(response.data["messages"], list)
            assert len(response.data["messages"]) == 1
            assert response.data["messages"][0]["content"] == "Hello from Buffer"

    @pytest.mark.asyncio
    async def test_database_fallback_response_format(self, mock_db_service):
        """Test that database fallback returns consistent format."""
        with patch('memfuse_core.api.messages.DatabaseService') as mock_db_class, \
             patch('memfuse_core.api.messages.get_service_for_session') as mock_get_service, \
             patch('memfuse_core.api.messages.ensure_session_exists') as mock_ensure_session:
            
            mock_db_class.get_instance.return_value = mock_db_service
            mock_get_service.return_value = None  # No service available
            mock_ensure_session.return_value = {"id": "session_1"}
            
            response = await list_messages(
                session_id="session_1",
                limit="10",
                sort_by="timestamp",
                order="desc",
                buffer_only=None,
                _api_key_data={}
            )
            
            assert isinstance(response, ApiResponse)
            assert response.status == "success"
            assert "messages" in response.data
            assert isinstance(response.data["messages"], list)
            assert len(response.data["messages"]) == 1
            assert response.data["messages"][0]["content"] == "Hello from DB"

    @pytest.mark.asyncio
    async def test_service_returns_nested_format(self, mock_db_service):
        """Test handling when service returns nested format."""
        mock_service = AsyncMock()
        # Service returns nested format
        mock_service.get_messages_by_session.return_value = {
            "data": {
                "messages": [
                    {"id": "1", "role": "user", "content": "Nested format"}
                ]
            }
        }

        with patch('memfuse_core.api.messages.DatabaseService') as mock_db_class, \
             patch('memfuse_core.api.messages.get_service_for_session') as mock_get_service, \
             patch('memfuse_core.api.messages.ensure_session_exists') as mock_ensure_session:

            mock_db_class.get_instance.return_value = mock_db_service
            mock_get_service.return_value = mock_service
            mock_ensure_session.return_value = {"id": "session_1"}

            response = await list_messages(
                session_id="session_1",
                limit="10",
                sort_by="timestamp",
                order="desc",
                buffer_only=None,
                _api_key_data={}
            )

            assert isinstance(response, ApiResponse)
            assert response.status == "success"
            assert "messages" in response.data
            assert isinstance(response.data["messages"], list)
            assert len(response.data["messages"]) == 1
            assert response.data["messages"][0]["content"] == "Nested format"

    @pytest.mark.asyncio
    async def test_service_returns_invalid_format(self, mock_db_service):
        """Test handling when service returns invalid format."""
        mock_service = AsyncMock()
        # Service returns invalid format
        mock_service.get_messages_by_session.return_value = "invalid_string_response"

        with patch('memfuse_core.api.messages.DatabaseService') as mock_db_class, \
             patch('memfuse_core.api.messages.get_service_for_session') as mock_get_service, \
             patch('memfuse_core.api.messages.ensure_session_exists') as mock_ensure_session:

            mock_db_class.get_instance.return_value = mock_db_service
            mock_get_service.return_value = mock_service
            mock_ensure_session.return_value = {"id": "session_1"}

            response = await list_messages(
                session_id="session_1",
                limit="10",
                sort_by="timestamp",
                order="desc",
                buffer_only=None,
                _api_key_data={}
            )

            # Should return empty list instead of failing
            assert isinstance(response, ApiResponse)
            assert response.status == "success"
            assert "messages" in response.data
            assert isinstance(response.data["messages"], list)
            assert len(response.data["messages"]) == 0

    @pytest.mark.asyncio
    async def test_buffer_only_parameter_handling(self, mock_db_service):
        """Test buffer_only parameter handling with different service types."""
        # Test with service that supports buffer_only
        mock_buffer_service = AsyncMock()
        mock_buffer_service.get_messages_by_session.return_value = [
            {"id": "1", "role": "user", "content": "Buffer only"}
        ]

        with patch('memfuse_core.api.messages.DatabaseService') as mock_db_class, \
             patch('memfuse_core.api.messages.get_service_for_session') as mock_get_service, \
             patch('memfuse_core.api.messages.ensure_session_exists') as mock_ensure_session:

            mock_db_class.get_instance.return_value = mock_db_service
            mock_get_service.return_value = mock_buffer_service
            mock_ensure_session.return_value = {"id": "session_1"}

            response = await list_messages(
                session_id="session_1",
                limit="10",
                sort_by="timestamp",
                order="desc",
                buffer_only="true",
                _api_key_data={}
            )

            # Verify buffer_only=True was passed
            mock_buffer_service.get_messages_by_session.assert_called_with(
                session_id="session_1",
                limit=10,
                sort_by="timestamp",
                order="desc",
                buffer_only=True
            )

            assert response.data["messages"][0]["content"] == "Buffer only"

    @pytest.mark.asyncio
    async def test_service_without_buffer_only_support(self, mock_db_service):
        """Test handling service that doesn't support buffer_only parameter."""
        mock_service = AsyncMock()

        # First call with buffer_only raises TypeError
        mock_service.get_messages_by_session.side_effect = [
            TypeError("unexpected keyword argument 'buffer_only'"),
            [{"id": "1", "role": "user", "content": "No buffer support"}]
        ]

        with patch('memfuse_core.api.messages.DatabaseService') as mock_db_class, \
             patch('memfuse_core.api.messages.get_service_for_session') as mock_get_service, \
             patch('memfuse_core.api.messages.ensure_session_exists') as mock_ensure_session:

            mock_db_class.get_instance.return_value = mock_db_service
            mock_get_service.return_value = mock_service
            mock_ensure_session.return_value = {"id": "session_1"}

            response = await list_messages(
                session_id="session_1",
                limit="10",
                sort_by="timestamp",
                order="desc",
                buffer_only="true",
                _api_key_data={}
            )

            # Should have been called twice - first with buffer_only, then without
            assert mock_service.get_messages_by_session.call_count == 2
            assert response.data["messages"][0]["content"] == "No buffer support"
