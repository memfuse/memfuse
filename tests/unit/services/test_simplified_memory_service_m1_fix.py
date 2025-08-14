"""
Test for SimplifiedMemoryService M1 processing fix.

This test verifies that the M1 processing issue has been resolved:
- Database connection references are correct
- M1 chunks are properly created and stored
- Messages are stored in compatibility tables
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.memfuse_core.services.simplified_memory_service import SimplifiedMemoryService


class TestSimplifiedMemoryServiceM1Fix:
    """Test the M1 processing fix in SimplifiedMemoryService."""

    @pytest.fixture
    def mock_db_manager(self):
        """Create a mock database manager."""
        db_manager = Mock()
        db_manager.conn = Mock()
        db_manager.conn.cursor.return_value.__enter__ = Mock()
        db_manager.conn.cursor.return_value.__exit__ = Mock()
        db_manager.conn.commit = Mock()
        db_manager.conn.rollback = Mock()
        return db_manager

    @pytest.fixture
    def mock_chunk_processor(self):
        """Create a mock chunk processor."""
        processor = Mock()
        processor.create_chunks.return_value = [
            {
                'id': 'chunk_1',
                'content': 'Test chunk content',
                'metadata': {'type': 'episodic'}
            }
        ]
        return processor

    @pytest.fixture
    def service(self, mock_db_manager, mock_chunk_processor):
        """Create a SimplifiedMemoryService instance with mocks."""
        with patch('src.memfuse_core.services.simplified_memory_service.SimplifiedDatabaseManager') as mock_db_class:
            mock_db_class.return_value = mock_db_manager
            
            with patch('src.memfuse_core.services.simplified_memory_service.IntegratedChunkingProcessor') as mock_chunk_class:
                mock_chunk_class.return_value = mock_chunk_processor
                
                service = SimplifiedMemoryService(user_id="test_user")
                return service

    @pytest.mark.asyncio
    async def test_store_to_messages_rounds_tables_uses_correct_connection(self, service):
        """Test that _store_to_messages_rounds_tables uses db_manager.conn correctly."""
        messages = [
            {
                'id': 'msg_1',
                'role': 'user',
                'content': 'Test message',
                'created_at': datetime.now()
            }
        ]
        session_id = 'test_session'
        round_id = 'test_round'

        # Mock the cursor context manager
        mock_cursor = Mock()
        service.db_manager.conn.cursor.return_value.__enter__.return_value = mock_cursor

        # Call the method
        await service._store_to_messages_rounds_tables(messages, session_id, round_id)

        # Verify that db_manager.conn was used (not self.conn)
        service.db_manager.conn.cursor.assert_called()
        service.db_manager.conn.commit.assert_called_once()
        
        # Verify cursor execute was called for round and message insertion
        assert mock_cursor.execute.call_count == 2  # One for round, one for message

    @pytest.mark.asyncio
    async def test_store_to_messages_rounds_tables_handles_errors_correctly(self, service):
        """Test that errors are handled correctly with proper rollback."""
        messages = [{'id': 'msg_1', 'role': 'user', 'content': 'Test'}]
        session_id = 'test_session'
        round_id = 'test_round'

        # Mock cursor to raise an exception
        mock_cursor = Mock()
        mock_cursor.execute.side_effect = Exception("Database error")
        service.db_manager.conn.cursor.return_value.__enter__.return_value = mock_cursor

        # Call should raise the exception
        with pytest.raises(Exception, match="Database error"):
            await service._store_to_messages_rounds_tables(messages, session_id, round_id)

        # Verify rollback was called on db_manager.conn (not self.conn)
        service.db_manager.conn.rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_batch_processes_m1_successfully(self, service):
        """Test that add_batch processes M1 chunks successfully."""
        message_batch_list = [
            [
                {
                    'id': 'msg_1',
                    'role': 'user',
                    'content': 'Test message',
                    'metadata': {'session_id': 'test_session'}
                }
            ]
        ]

        # Mock the internal methods
        service._store_m0_messages = Mock(return_value=['msg_1'])
        service._store_m1_chunks = Mock(return_value=['chunk_1'])
        service._store_to_messages_rounds_tables = Mock()
        service._prepare_session_and_round = Mock(return_value=('session_1', 'round_1'))

        # Call add_batch
        result = await service.add_batch(message_batch_list)

        # Verify success
        assert result['success'] is True
        assert 'message_ids' in result['data']
        assert 'chunk_count' in result['data']

        # Verify all methods were called
        service._prepare_session_and_round.assert_called_once()
        service._store_to_messages_rounds_tables.assert_called_once()
        service._store_m0_messages.assert_called_once()
        service._store_m1_chunks.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
