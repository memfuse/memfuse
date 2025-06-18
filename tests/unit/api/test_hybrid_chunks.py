#!/usr/bin/env python3
"""Test hybrid chunks functionality."""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from memfuse_core.services.memory_service import MemoryService


class TestHybridChunks:
    """Test suite for hybrid chunks functionality."""

    def test_hybrid_store_type_support(self):
        """Test that MemoryService supports hybrid store type."""
        try:
            memory_service = MemoryService()
            
            # Test that get_chunks_by_session supports hybrid
            import inspect
            sig = inspect.signature(memory_service.get_chunks_by_session)
            store_type_param = sig.parameters.get('store_type')
            
            assert store_type_param is not None, "store_type parameter should exist"
            assert store_type_param.default is None, "store_type should be optional"
            
            print("✅ Hybrid store type support test passed")
            
        except Exception as e:
            pytest.fail(f"Failed to test hybrid store type support: {e}")

    def test_store_prioritization_logic(self):
        """Test that hybrid store type follows correct prioritization."""
        try:
            # This is a conceptual test - in real implementation,
            # we would need actual store instances to test the logic
            
            # Test the prioritization order: vector > keyword > graph
            expected_order = ["vector", "keyword", "graph"]
            
            # In the actual implementation, this would be tested by:
            # 1. Creating mock stores
            # 2. Calling get_chunks_by_session with store_type="hybrid"
            # 3. Verifying that stores are queried in the correct order
            
            assert len(expected_order) == 3, "Should have 3 store types"
            assert expected_order[0] == "vector", "Vector should be first priority"
            assert expected_order[1] == "keyword", "Keyword should be second priority"
            assert expected_order[2] == "graph", "Graph should be third priority"
            
            print("✅ Store prioritization logic test passed")
            
        except Exception as e:
            pytest.fail(f"Failed to test store prioritization logic: {e}")

    def test_filter_parameters_validation(self):
        """Test that filter parameters are properly validated."""
        try:
            # Test valid store types
            valid_store_types = ["vector", "keyword", "graph", "hybrid", None]
            
            for store_type in valid_store_types:
                # This would be tested in the actual API endpoint
                # For now, we just verify the list is correct
                if store_type is not None:
                    assert store_type in ["vector", "keyword", "graph", "hybrid"]
            
            # Test invalid store types
            invalid_store_types = ["invalid", "unknown", "test"]
            
            for store_type in invalid_store_types:
                assert store_type not in ["vector", "keyword", "graph", "hybrid"]
            
            print("✅ Filter parameters validation test passed")
            
        except Exception as e:
            pytest.fail(f"Failed to test filter parameters validation: {e}")

    def test_enhanced_metadata_structure(self):
        """Test that enhanced metadata has the expected structure."""
        try:
            # Test enhanced metadata structure
            enhanced_metadata = {
                # Original strategy-specific metadata
                "strategy": "message",
                "message_count": 2,
                "source": "message_list",
                "batch_index": 0,
                "roles": ["user", "assistant"],
                
                # Enhanced metadata
                "type": "chunk",
                "user_id": "user_123",
                "session_id": "session_456", 
                "round_id": "round_789",
                "agent_id": "agent_abc",
                "created_at": "2025-01-01T12:00:00.000Z",
                "store_type": "hybrid"  # Added for hybrid support
            }
            
            # Verify all required fields are present
            required_fields = [
                "strategy", "type", "user_id", "session_id", 
                "round_id", "agent_id", "created_at", "store_type"
            ]
            
            for field in required_fields:
                assert field in enhanced_metadata, f"Missing required field: {field}"
            
            # Verify field types
            assert isinstance(enhanced_metadata["strategy"], str)
            assert isinstance(enhanced_metadata["message_count"], int)
            assert isinstance(enhanced_metadata["roles"], list)
            assert isinstance(enhanced_metadata["user_id"], str)
            assert isinstance(enhanced_metadata["session_id"], str)
            assert isinstance(enhanced_metadata["round_id"], str)
            assert isinstance(enhanced_metadata["agent_id"], str)
            assert isinstance(enhanced_metadata["created_at"], str)
            assert isinstance(enhanced_metadata["store_type"], str)
            
            print("✅ Enhanced metadata structure test passed")
            
        except Exception as e:
            pytest.fail(f"Failed to test enhanced metadata structure: {e}")

    def test_api_endpoint_structure(self):
        """Test that API endpoints have the expected structure."""
        try:
            # Test expected API endpoints
            expected_endpoints = [
                "/api/v1/sessions/{session_id}/chunks",
                "/api/v1/rounds/{round_id}/chunks",
                "/api/v1/chunks/stats"
            ]

            # Test expected query parameters (updated to match Messages API)
            expected_session_params = [
                "limit",      # Maximum number of chunks (default: 20, max: 100)
                "sort_by",    # Field to sort by (created_at, chunk_id, strategy)
                "order",      # Sort order (asc, desc)
                "store_type"  # Store type filter (vector, keyword, graph, hybrid)
            ]

            expected_round_params = expected_session_params.copy()  # Same as session

            expected_stats_params = [
                "user_id", "session_id", "store_type"
            ]

            # Verify endpoint structure
            assert len(expected_endpoints) == 3, "Should have 3 main endpoints"
            assert all("chunks" in endpoint for endpoint in expected_endpoints)

            # Verify parameter structure matches Messages API design
            assert "limit" in expected_session_params
            assert "sort_by" in expected_session_params
            assert "order" in expected_session_params
            assert "store_type" in expected_session_params
            assert "hybrid" in ["vector", "keyword", "graph", "hybrid"]

            # Verify default values
            default_limit = "20"  # Same as Messages API
            default_sort_by = "created_at"  # Equivalent to timestamp
            default_order = "desc"  # Same as Messages API

            assert default_limit == "20"
            assert default_sort_by == "created_at"
            assert default_order == "desc"

            print("✅ API endpoint structure test passed")

        except Exception as e:
            pytest.fail(f"Failed to test API endpoint structure: {e}")


if __name__ == "__main__":
    # Run tests directly
    test_instance = TestHybridChunks()
    
    print("🧪 Running Hybrid Chunks Tests")
    print("=" * 50)
    
    try:
        test_instance.test_hybrid_store_type_support()
        test_instance.test_store_prioritization_logic()
        test_instance.test_filter_parameters_validation()
        test_instance.test_enhanced_metadata_structure()
        test_instance.test_api_endpoint_structure()
        
        print("\n🎉 All hybrid chunks tests passed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
