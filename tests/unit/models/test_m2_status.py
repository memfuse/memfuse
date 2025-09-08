"""
Unit tests for M2Status enum.

Tests the M2Status enum values and completeness for M2 fact extraction
processing status tracking.
"""

import pytest
from src.memfuse_core.models.core import M2Status


class TestM2Status:
    """Test cases for M2Status enum."""
    
    def test_m2_status_enum_values(self):
        """Test M2Status enum has correct values."""
        assert M2Status.PENDING == "pending"
        assert M2Status.PROCESSING == "processing" 
        assert M2Status.COMPLETED == "completed"
        assert M2Status.FAILED == "failed"
    
    def test_m2_status_enum_completeness(self):
        """Test M2Status enum has all expected values and no extras."""
        expected_values = {"pending", "processing", "completed", "failed"}
        actual_values = {status.value for status in M2Status}
        assert actual_values == expected_values
    
    def test_m2_status_enum_string_representation(self):
        """Test M2Status enum string representation."""
        assert str(M2Status.PENDING) == "M2Status.PENDING"
        assert str(M2Status.PROCESSING) == "M2Status.PROCESSING"
        assert str(M2Status.COMPLETED) == "M2Status.COMPLETED"
        assert str(M2Status.FAILED) == "M2Status.FAILED"
    
    def test_m2_status_enum_membership(self):
        """Test checking membership in M2Status enum."""
        assert M2Status.PENDING in M2Status
        assert M2Status.PROCESSING in M2Status
        assert M2Status.COMPLETED in M2Status
        assert M2Status.FAILED in M2Status
        
        # Test invalid values are not members
        assert "invalid_status" not in [s.value for s in M2Status]
    
    def test_m2_status_enum_iteration(self):
        """Test iterating over M2Status enum."""
        statuses = list(M2Status)
        assert len(statuses) == 4
        
        values = [status.value for status in statuses]
        assert "pending" in values
        assert "processing" in values
        assert "completed" in values
        assert "failed" in values


if __name__ == '__main__':
    pytest.main([__file__])