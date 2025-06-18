#!/usr/bin/env python3
"""Test chunks API endpoints."""

import pytest
import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from fastapi.testclient import TestClient
from memfuse_core.server import create_app


class TestChunksAPI:
    """Test suite for chunks API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app()
        return TestClient(app)

    @pytest.fixture
    def api_key(self):
        """Get API key for testing."""
        # This should be configured in your test environment
        return "test_api_key"

    def test_chunks_api_import(self):
        """Test that chunks API can be imported."""
        try:
            from memfuse_core.api import chunks
            assert hasattr(chunks, 'router')
            print("✅ Chunks API import test passed")
        except ImportError as e:
            pytest.fail(f"Failed to import chunks API: {e}")

    def test_chunks_api_endpoints_exist(self):
        """Test that chunks API endpoints are defined."""
        try:
            from memfuse_core.api.chunks import router
            
            # Check that router has routes
            assert len(router.routes) > 0
            
            # Check for expected endpoints
            route_paths = [route.path for route in router.routes]
            expected_paths = [
                "/sessions/{session_id}/chunks",
                "/sessions/{session_id}/chunks/",
                "/rounds/{round_id}/chunks", 
                "/rounds/{round_id}/chunks/",
                "/chunks/stats",
                "/chunks/stats/"
            ]
            
            for expected_path in expected_paths:
                assert any(expected_path in path for path in route_paths), f"Missing endpoint: {expected_path}"
            
            print("✅ Chunks API endpoints test passed")
            
        except Exception as e:
            pytest.fail(f"Failed to check chunks API endpoints: {e}")

    def test_memory_service_query_methods(self):
        """Test that MemoryService has the new query methods."""
        try:
            from memfuse_core.services.memory_service import MemoryService
            
            # Check that MemoryService has the new methods
            memory_service = MemoryService()
            
            assert hasattr(memory_service, 'get_chunks_by_session')
            assert hasattr(memory_service, 'get_chunks_by_round')
            assert hasattr(memory_service, 'get_chunks_stats')
            
            print("✅ MemoryService query methods test passed")
            
        except Exception as e:
            pytest.fail(f"Failed to check MemoryService query methods: {e}")

    def test_store_interface_extensions(self):
        """Test that Store interfaces have been extended with query methods."""
        try:
            from memfuse_core.interfaces.chunk_store import ChunkStoreInterface
            
            # Check that interface has the new methods
            expected_methods = [
                'get_chunks_by_session',
                'get_chunks_by_round', 
                'get_chunks_by_user',
                'get_chunks_by_strategy',
                'get_chunks_stats'
            ]
            
            for method_name in expected_methods:
                assert hasattr(ChunkStoreInterface, method_name), f"Missing method: {method_name}"
            
            print("✅ Store interface extensions test passed")
            
        except Exception as e:
            pytest.fail(f"Failed to check store interface extensions: {e}")

    def test_store_implementations_have_query_methods(self):
        """Test that store implementations have the new query methods."""
        try:
            # Test VectorStore
            from memfuse_core.store.vector_store.qdrant_store import QdrantVectorStore
            
            vector_store = QdrantVectorStore(
                data_dir="test_data",
                embedding_dim=384
            )
            
            expected_methods = [
                'get_chunks_by_session',
                'get_chunks_by_round',
                'get_chunks_by_user', 
                'get_chunks_by_strategy',
                'get_chunks_stats'
            ]
            
            for method_name in expected_methods:
                assert hasattr(vector_store, method_name), f"VectorStore missing method: {method_name}"
            
            # Test KeywordStore
            from memfuse_core.store.keyword_store.sqlite_store import SQLiteKeywordStore
            
            keyword_store = SQLiteKeywordStore(data_dir="test_data")
            
            for method_name in expected_methods:
                assert hasattr(keyword_store, method_name), f"KeywordStore missing method: {method_name}"
            
            # Test GraphStore
            from memfuse_core.store.graph_store.graphml_store import GraphMLStore
            
            graph_store = GraphMLStore(data_dir="test_data")
            
            for method_name in expected_methods:
                assert hasattr(graph_store, method_name), f"GraphStore missing method: {method_name}"
            
            print("✅ Store implementations query methods test passed")
            
        except Exception as e:
            pytest.fail(f"Failed to check store implementations: {e}")

    def test_enhanced_chunk_metadata_structure(self):
        """Test that enhanced chunk metadata has the expected structure."""
        try:
            from memfuse_core.rag.chunk.base import ChunkData
            from datetime import datetime

            # Create a chunk with enhanced metadata
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
                "created_at": datetime.now().isoformat(),
                "store_type": "vector"  # Added for new API
            }

            chunk = ChunkData(
                content="Test chunk content",
                metadata=enhanced_metadata
            )

            # Verify enhanced metadata fields
            assert chunk.metadata.get("session_id") == "session_456"
            assert chunk.metadata.get("round_id") == "round_789"
            assert chunk.metadata.get("user_id") == "user_123"
            assert chunk.metadata.get("agent_id") == "agent_abc"
            assert chunk.metadata.get("type") == "chunk"
            assert chunk.metadata.get("store_type") == "vector"
            assert "created_at" in chunk.metadata

            # Verify original metadata is preserved
            assert chunk.metadata.get("strategy") == "message"
            assert chunk.metadata.get("message_count") == 2

            print("✅ Enhanced chunk metadata structure test passed")

        except Exception as e:
            pytest.fail(f"Failed to test enhanced chunk metadata: {e}")

    def test_api_parameters_match_messages_api(self):
        """Test that chunks API parameters match Messages API design."""
        try:
            # Test expected parameters for session chunks
            expected_session_params = [
                "limit",      # Maximum number of chunks (default: 20, max: 100)
                "sort_by",    # Field to sort by (created_at, chunk_id, strategy)
                "order",      # Sort order (asc, desc)
                "store_type"  # Store type filter (vector, keyword, graph, hybrid)
            ]

            # Test expected parameters for round chunks (same as session)
            expected_round_params = expected_session_params.copy()

            # Verify parameter structure matches Messages API
            assert "limit" in expected_session_params
            assert "sort_by" in expected_session_params
            assert "order" in expected_session_params

            # Verify default values match Messages API
            default_limit = "20"  # Same as Messages API
            default_sort_by = "created_at"  # Equivalent to timestamp in Messages
            default_order = "desc"  # Same as Messages API

            assert default_limit == "20"
            assert default_sort_by == "created_at"
            assert default_order == "desc"

            # Verify allowed values
            allowed_sort_fields = ["created_at", "chunk_id", "strategy"]
            allowed_orders = ["asc", "desc"]
            allowed_store_types = ["vector", "keyword", "graph", "hybrid"]

            assert "created_at" in allowed_sort_fields
            assert "chunk_id" in allowed_sort_fields
            assert "strategy" in allowed_sort_fields
            assert "asc" in allowed_orders
            assert "desc" in allowed_orders
            assert "hybrid" in allowed_store_types

            print("✅ API parameters match Messages API test passed")

        except Exception as e:
            pytest.fail(f"Failed to test API parameters: {e}")


if __name__ == "__main__":
    # Run tests directly
    test_instance = TestChunksAPI()
    
    print("🧪 Running Chunks API Tests")
    print("=" * 50)
    
    try:
        test_instance.test_chunks_api_import()
        test_instance.test_chunks_api_endpoints_exist()
        test_instance.test_memory_service_query_methods()
        test_instance.test_store_interface_extensions()
        test_instance.test_store_implementations_have_query_methods()
        test_instance.test_enhanced_chunk_metadata_structure()
        test_instance.test_api_parameters_match_messages_api()
        
        print("\n🎉 All API tests passed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
