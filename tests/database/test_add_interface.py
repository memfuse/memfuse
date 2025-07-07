"""Test database add interface after removing insert methods."""

import pytest
import tempfile
import os
from unittest.mock import Mock

from src.memfuse_core.database.sqlite import SQLiteDB
from src.memfuse_core.database.base import Database


class TestDatabaseAddInterface:
    """Test that database backends use add interface exclusively."""

    def setup_method(self):
        """Set up test database."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db_path = self.temp_db.name
        
        # Create SQLite database
        self.sqlite_db = SQLiteDB(self.db_path)
        self.sqlite_db.create_tables()
        
        # Create Database wrapper
        self.database = Database(self.sqlite_db)

    def teardown_method(self):
        """Clean up test database."""
        if hasattr(self, 'sqlite_db'):
            self.sqlite_db.close()
        if hasattr(self, 'db_path') and os.path.exists(self.db_path):
            os.unlink(self.db_path)

    def test_sqlite_has_add_method(self):
        """Test that SQLiteDB has add method."""
        assert hasattr(self.sqlite_db, 'add')
        assert callable(self.sqlite_db.add)

    def test_sqlite_no_insert_method_usage(self):
        """Test that SQLiteDB doesn't expose insert method for external use."""
        # The add method should work without calling insert
        test_data = {
            'id': 'test-user-1',
            'name': 'Test User',
            'description': 'Test Description'
        }
        
        result = self.sqlite_db.add('users', test_data)
        assert result == 'test-user-1'
        
        # Verify data was inserted
        users = self.sqlite_db.select('users', {'id': 'test-user-1'})
        assert len(users) == 1
        assert users[0]['name'] == 'Test User'

    def test_database_wrapper_uses_add(self):
        """Test that Database wrapper uses add method."""
        # Create a user through Database wrapper
        user_id = self.database.create_user(
            user_id='test-user-2',
            name='Test User 2',
            description='Another test user'
        )
        
        assert user_id == 'test-user-2'
        
        # Verify user was created
        user = self.database.get_user('test-user-2')
        assert user is not None
        assert user['name'] == 'Test User 2'

    def test_database_operations_work_with_add(self):
        """Test that all database operations work with add interface."""
        # Test user creation
        user_id = self.database.create_user(name='Test User')
        assert user_id is not None
        
        # Test agent creation
        agent_id = self.database.create_agent(name='Test Agent')
        assert agent_id is not None
        
        # Test session creation
        session_id = self.database.create_session(
            user_id=user_id,
            agent_id=agent_id,
            name='Test Session'
        )
        assert session_id is not None
        
        # Test round creation
        round_id = self.database.create_round(session_id=session_id)
        assert round_id is not None
        
        # Test message creation
        message_id = self.database.add_message(
            round_id=round_id,
            role='user',
            content='Test message'
        )
        assert message_id is not None

    def test_add_method_handles_dict_data(self):
        """Test that add method properly handles dictionary data."""
        test_data = {
            'id': 'test-complex',
            'name': 'Complex Test',
            'metadata': {'key': 'value', 'nested': {'data': 123}}
        }
        
        result = self.sqlite_db.add('users', test_data)
        assert result == 'test-complex'
        
        # Verify complex data was stored correctly
        users = self.sqlite_db.select('users', {'id': 'test-complex'})
        assert len(users) == 1
        # Note: metadata should be JSON string in database
        assert 'metadata' in users[0]

    def test_mock_database_backend_interface(self):
        """Test that Database class works with any backend that has add method."""
        # Create a mock backend with add method
        mock_backend = Mock()
        mock_backend.add.return_value = 'mock-id'
        mock_backend.select.return_value = [{'id': 'mock-id', 'name': 'Mock User'}]
        
        # Create Database with mock backend
        database = Database(mock_backend)
        
        # Test that Database calls add method on backend
        result = database.create_user(user_id='mock-id', name='Mock User')
        
        # Verify add was called
        mock_backend.add.assert_called_once()
        call_args = mock_backend.add.call_args
        assert call_args[0][0] == 'users'  # table name
        assert call_args[0][1]['id'] == 'mock-id'  # data
        assert call_args[0][1]['name'] == 'Mock User'
        
        assert result == 'mock-id'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
