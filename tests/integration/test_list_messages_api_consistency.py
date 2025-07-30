"""Integration tests for list_messages API consistency.

This module provides end-to-end tests to verify that the list_messages API
returns consistent response formats across different service configurations.
"""

import pytest
import asyncio
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock, AsyncMock

from memfuse_core.api.messages import list_messages
from memfuse_core.models.core import ApiResponse
from memfuse_core.services.memory_service import MemoryService
from memfuse_core.services.buffer_service import BufferService


class TestListMessagesAPIIntegration:
    """Integration tests for list_messages API consistency."""

    @pytest.fixture
    def sample_messages(self) -> List[Dict[str, Any]]:
        """Sample messages for testing."""
        return [
            {
                "id": "msg_1",
                "role": "user",
                "content": "Hello, how are you?",
                "created_at": "2024-01-01T10:00:00Z",
                "updated_at": "2024-01-01T10:00:00Z",
                "metadata": {"session_id": "test_session", "user_id": "test_user"}
            },
            {
                "id": "msg_2",
                "role": "assistant",
                "content": "I'm doing well, thank you!",
                "created_at": "2024-01-01T10:01:00Z",
                "updated_at": "2024-01-01T10:01:00Z",
                "metadata": {"session_id": "test_session", "user_id": "test_user"}
            },
            {
                "id": "msg_3",
                "role": "user",
                "content": "What's the weather like?",
                "created_at": "2024-01-01T10:02:00Z",
                "updated_at": "2024-01-01T10:02:00Z",
                "metadata": {"session_id": "test_session", "user_id": "test_user"}
            }
        ]

    def validate_api_response_format(self, response: ApiResponse) -> None:
        """Validate that API response has consistent format."""
        # Check response structure
        assert isinstance(response, ApiResponse)
        assert response.status == "success"
        assert response.code == 200
        assert response.data is not None
        assert response.errors is None
        assert response.message == "Messages retrieved successfully"
        
        # Check data structure
        assert "messages" in response.data
        assert isinstance(response.data["messages"], list)
        
        # Check each message format
        for msg in response.data["messages"]:
            assert isinstance(msg, dict)
            assert "id" in msg
            assert "role" in msg
            assert "content" in msg
            assert isinstance(msg["id"], str)
            assert isinstance(msg["role"], str)
            assert isinstance(msg["content"], str)

    @pytest.mark.asyncio
    async def test_memory_service_integration(self, sample_messages):
        """Test list_messages with MemoryService integration."""
        # Mock database service
        mock_db = MagicMock()
        mock_db.get_messages_by_session.return_value = sample_messages
        
        # Mock memory service
        mock_memory_service = AsyncMock()
        mock_memory_service.get_messages_by_session.return_value = sample_messages
        
        with patch('memfuse_core.api.messages.DatabaseService') as mock_db_class, \
             patch('memfuse_core.api.messages.get_service_for_session') as mock_get_service, \
             patch('memfuse_core.api.messages.ensure_session_exists') as mock_ensure_session:
            
            mock_db_class.get_instance.return_value = mock_db
            mock_get_service.return_value = mock_memory_service
            mock_ensure_session.return_value = {"id": "test_session"}
            
            response = await list_messages(
                session_id="test_session",
                limit="10",
                sort_by="timestamp",
                order="desc",
                buffer_only=None,
                _api_key_data={}
            )
            
            self.validate_api_response_format(response)
            assert len(response.data["messages"]) == 3
            
            # Verify service was called with correct parameters
            mock_memory_service.get_messages_by_session.assert_called_once_with(
                session_id="test_session",
                limit=10,
                sort_by="timestamp",
                order="desc",
                buffer_only=None
            )

    @pytest.mark.asyncio
    async def test_buffer_service_integration(self, sample_messages):
        """Test list_messages with BufferService integration."""
        # Mock database service
        mock_db = MagicMock()
        mock_db.get_messages_by_session.return_value = sample_messages
        
        # Mock buffer service
        mock_buffer_service = AsyncMock()
        mock_buffer_service.get_messages_by_session.return_value = sample_messages
        
        with patch('memfuse_core.api.messages.DatabaseService') as mock_db_class, \
             patch('memfuse_core.api.messages.get_service_for_session') as mock_get_service, \
             patch('memfuse_core.api.messages.ensure_session_exists') as mock_ensure_session:
            
            mock_db_class.get_instance.return_value = mock_db
            mock_get_service.return_value = mock_buffer_service
            mock_ensure_session.return_value = {"id": "test_session"}
            
            response = await list_messages(
                session_id="test_session",
                limit="20",
                sort_by="id",
                order="asc",
                buffer_only="true",
                _api_key_data={}
            )
            
            self.validate_api_response_format(response)
            assert len(response.data["messages"]) == 3
            
            # Verify service was called with correct parameters including buffer_only
            mock_buffer_service.get_messages_by_session.assert_called_once_with(
                session_id="test_session",
                limit=20,
                sort_by="id",
                order="asc",
                buffer_only=True
            )

    @pytest.mark.asyncio
    async def test_database_fallback_integration(self, sample_messages):
        """Test list_messages with database fallback."""
        # Mock database service
        mock_db = MagicMock()
        mock_db.get_messages_by_session.return_value = sample_messages
        
        with patch('memfuse_core.api.messages.DatabaseService') as mock_db_class, \
             patch('memfuse_core.api.messages.get_service_for_session') as mock_get_service, \
             patch('memfuse_core.api.messages.ensure_session_exists') as mock_ensure_session:
            
            mock_db_class.get_instance.return_value = mock_db
            mock_get_service.return_value = None  # No service available
            mock_ensure_session.return_value = {"id": "test_session"}
            
            response = await list_messages(
                session_id="test_session",
                limit="5",
                sort_by="timestamp",
                order="desc",
                buffer_only=None,
                _api_key_data={}
            )
            
            self.validate_api_response_format(response)
            assert len(response.data["messages"]) == 3
            
            # Verify database was called directly
            mock_db.get_messages_by_session.assert_called_once_with(
                session_id="test_session",
                limit=5,
                sort_by="timestamp",
                order="desc"
            )

    @pytest.mark.asyncio
    async def test_empty_results_consistency(self):
        """Test that empty results are handled consistently."""
        # Mock database service returning empty list
        mock_db = MagicMock()
        mock_db.get_messages_by_session.return_value = []
        
        # Mock service returning empty list
        mock_service = AsyncMock()
        mock_service.get_messages_by_session.return_value = []
        
        with patch('memfuse_core.api.messages.DatabaseService') as mock_db_class, \
             patch('memfuse_core.api.messages.get_service_for_session') as mock_get_service, \
             patch('memfuse_core.api.messages.ensure_session_exists') as mock_ensure_session:
            
            mock_db_class.get_instance.return_value = mock_db
            mock_get_service.return_value = mock_service
            mock_ensure_session.return_value = {"id": "empty_session"}
            
            response = await list_messages(
                session_id="empty_session",
                limit="10",
                sort_by="timestamp",
                order="desc",
                buffer_only=None,
                _api_key_data={}
            )
            
            self.validate_api_response_format(response)
            assert len(response.data["messages"]) == 0
            assert isinstance(response.data["messages"], list)

    @pytest.mark.asyncio
    async def test_parameter_validation_consistency(self):
        """Test that parameter validation works consistently."""
        mock_db = MagicMock()
        mock_db.get_messages_by_session.return_value = []
        
        mock_service = AsyncMock()
        mock_service.get_messages_by_session.return_value = []
        
        with patch('memfuse_core.api.messages.DatabaseService') as mock_db_class, \
             patch('memfuse_core.api.messages.get_service_for_session') as mock_get_service, \
             patch('memfuse_core.api.messages.ensure_session_exists') as mock_ensure_session:
            
            mock_db_class.get_instance.return_value = mock_db
            mock_get_service.return_value = mock_service
            mock_ensure_session.return_value = {"id": "test_session"}
            
            # Test various parameter combinations
            test_cases = [
                {"limit": "1", "sort_by": "timestamp", "order": "desc", "buffer_only": None},
                {"limit": "100", "sort_by": "id", "order": "asc", "buffer_only": "false"},
                {"limit": "50", "sort_by": "timestamp", "order": "desc", "buffer_only": "true"},
            ]
            
            for params in test_cases:
                response = await list_messages(
                    session_id="test_session",
                    _api_key_data={},
                    **params
                )
                
                self.validate_api_response_format(response)

    @pytest.mark.asyncio
    async def test_error_handling_consistency(self):
        """Test that error handling maintains response format consistency."""
        mock_db = MagicMock()
        mock_db.get_messages_by_session.return_value = []
        
        # Mock service that raises an exception
        mock_service = AsyncMock()
        mock_service.get_messages_by_session.side_effect = Exception("Service error")
        
        with patch('memfuse_core.api.messages.DatabaseService') as mock_db_class, \
             patch('memfuse_core.api.messages.get_service_for_session') as mock_get_service, \
             patch('memfuse_core.api.messages.ensure_session_exists') as mock_ensure_session:
            
            mock_db_class.get_instance.return_value = mock_db
            mock_get_service.return_value = mock_service
            mock_ensure_session.return_value = {"id": "test_session"}
            
            response = await list_messages(
                session_id="test_session",
                limit="10",
                sort_by="timestamp",
                order="desc",
                buffer_only=None,
                _api_key_data={}
            )
            
            # Even with service error, should fall back to database and maintain format
            self.validate_api_response_format(response)
            assert len(response.data["messages"]) == 0
