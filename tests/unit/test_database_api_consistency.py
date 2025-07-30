#!/usr/bin/env python3
"""
Test Database API Consistency

This test verifies that the Database class provides a consistent API
that delegates to the backend appropriately.
"""

import pytest
import sys
import os
from pathlib import Path
from unittest.mock import Mock, MagicMock

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from memfuse_core.database.base import Database, DBBase


class MockBackend(DBBase):
    """Mock database backend for testing."""
    
    def __init__(self):
        self.execute_called = False
        self.commit_called = False
        self.close_called = False
        self.create_tables_called = False
        
    def execute(self, query: str, params: tuple = ()):
        self.execute_called = True
        return Mock()
    
    def commit(self):
        self.commit_called = True
    
    def close(self):
        self.close_called = True
    
    def create_tables(self):
        self.create_tables_called = True
    
    def add(self, table: str, data: dict) -> str:
        return "test_id"
    
    def select(self, table: str, conditions: dict = None) -> list:
        return []
    
    def select_one(self, table: str, conditions: dict) -> dict:
        return {}
    
    def update(self, table: str, data: dict, conditions: dict) -> int:
        return 1
    
    def delete(self, table: str, conditions: dict) -> int:
        return 1


class TestDatabaseAPIConsistency:
    """Test suite for Database API consistency."""
    
    def test_database_commit_delegates_to_backend(self):
        """Test that Database.commit() properly delegates to backend.commit()."""
        # Create mock backend
        mock_backend = MockBackend()
        
        # Create Database instance
        db = Database(mock_backend)
        
        # Verify backend.create_tables() was called during initialization
        assert mock_backend.create_tables_called
        
        # Call commit on Database
        db.commit()
        
        # Verify it was delegated to backend
        assert mock_backend.commit_called
    
    def test_database_close_delegates_to_backend(self):
        """Test that Database.close() properly delegates to backend.close()."""
        # Create mock backend
        mock_backend = MockBackend()
        
        # Create Database instance
        db = Database(mock_backend)
        
        # Call close on Database
        db.close()
        
        # Verify it was delegated to backend
        assert mock_backend.close_called
    
    def test_database_api_consistency(self):
        """Test that Database provides consistent API for common operations."""
        # Create mock backend
        mock_backend = MockBackend()
        
        # Create Database instance
        db = Database(mock_backend)
        
        # Test that all expected methods exist
        assert hasattr(db, 'commit'), "Database should have commit() method"
        assert hasattr(db, 'close'), "Database should have close() method"
        assert hasattr(db, 'backend'), "Database should have backend attribute"
        
        # Test that methods are callable
        assert callable(db.commit), "Database.commit should be callable"
        assert callable(db.close), "Database.close should be callable"
        
        # Test that backend methods are still accessible
        assert hasattr(db.backend, 'commit'), "Backend should have commit() method"
        assert hasattr(db.backend, 'close'), "Backend should have close() method"
        assert hasattr(db.backend, 'execute'), "Backend should have execute() method"
    
    def test_database_commit_api_equivalence(self):
        """Test that db.commit() and db.backend.commit() are equivalent."""
        # Create mock backend
        mock_backend = MockBackend()
        
        # Create Database instance
        db = Database(mock_backend)
        
        # Reset the mock state
        mock_backend.commit_called = False
        
        # Call db.commit() (new unified API)
        db.commit()
        assert mock_backend.commit_called
        
        # Reset the mock state
        mock_backend.commit_called = False
        
        # Call db.backend.commit() (old direct API)
        db.backend.commit()
        assert mock_backend.commit_called
        
        # Both approaches should work and have the same effect
    
    def test_database_initialization_calls_create_tables(self):
        """Test that Database initialization calls backend.create_tables()."""
        # Create mock backend
        mock_backend = MockBackend()
        
        # Verify create_tables hasn't been called yet
        assert not mock_backend.create_tables_called
        
        # Create Database instance
        db = Database(mock_backend)
        
        # Verify create_tables was called during initialization
        assert mock_backend.create_tables_called


if __name__ == "__main__":
    # Run tests directly
    test_instance = TestDatabaseAPIConsistency()
    
    print("Running Database API Consistency tests...")
    
    test_instance.test_database_commit_delegates_to_backend()
    print("✅ Database commit delegation test passed")
    
    test_instance.test_database_close_delegates_to_backend()
    print("✅ Database close delegation test passed")
    
    test_instance.test_database_api_consistency()
    print("✅ Database API consistency test passed")
    
    test_instance.test_database_commit_api_equivalence()
    print("✅ Database commit API equivalence test passed")
    
    test_instance.test_database_initialization_calls_create_tables()
    print("✅ Database initialization test passed")
    
    print("✅ All Database API Consistency tests passed!")
