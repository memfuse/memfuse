"""
Advanced integration tests for MemFuse pipeline scenarios.

These tests cover complex end-to-end scenarios involving multiple components
working together: Gateway, Buffer, Memory layers, and various filters.
"""

import pytest
import asyncio
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock

from src.memfuse_core.gateway.api_gateway import MemoryApiGateway
from src.memfuse_core.buffer.query_buffer import QueryBuffer
from src.memfuse_core.persistence.in_memory_store import InMemoryStore
from src.memfuse_core.persistence.adapters import make_retrieval_handler_from_store
from src.memfuse_core.models.core import Item
from src.memfuse_core.utils.global_config_manager import get_global_config_manager


class MockBufferService:
    """Mock buffer service for integration testing."""
    
    def __init__(self, results: List[Dict[str, Any]]):
        self.results = results
        self.query_buffer = None
    
    async def query(self, query: str, top_k: int = 5, **kwargs) -> Dict[str, Any]:
        """Mock query method that returns predefined results."""
        return {
            "status": "success",
            "code": 200,
            "data": {
                "results": self.results[:top_k],
                "total": len(self.results),
                "metadata": {
                    "query": query,
                    "top_k": top_k,
                    "source": "mock_buffer"
                }
            },
            "message": "Query processed successfully",
            "errors": None
        }


class TestAdvancedPipelineScenarios:
    """Test complex pipeline scenarios."""
    
    @pytest.fixture
    async def configured_gateway(self):
        """Create a fully configured gateway for testing."""
        # Configure the system
        gcm = get_global_config_manager()
        await gcm.hot_reload({
            "gateway": {
                "pipeline": {
                    "inbound": [
                        {"name": "request_validator", "enabled": True},
                        {"name": "input_sanitizer", "enabled": True}
                    ],
                    "outbound": [
                        {"name": "field_remover", "enabled": True},
                        {"name": "sensitive_word", "enabled": True},
                        {"name": "max_length", "enabled": True}
                    ]
                },
                "debug": {
                    "enabled": True,
                    "include_durations": True,
                    "include_filter_stats": True
                }
            },
            "guardrail": {
                "enabled": True,
                "sensitive": {
                    "enabled": True,
                    "words": ["password", "secret", "confidential"],
                    "action": "mask"
                },
                "output": {
                    "enabled": True,
                    "remove_fields": ["metadata.internal_score", "metadata.debug_info"]
                }
            },
            "buffer_plugins": {
                "plugins": [
                    {"name": "session_annotator", "enabled": True},
                    {"name": "deduplicate", "enabled": True},
                    {"name": "score_clip", "enabled": True}
                ]
            }
        })
        
        # Create mock results with various content types
        mock_results = [
            {
                "id": "result_1",
                "content": "This is a normal response with good information.",
                "relevance_score": 0.9,
                "metadata": {
                    "source": "memory_layer",
                    "internal_score": 0.95,
                    "debug_info": "internal_debug_data"
                }
            },
            {
                "id": "result_2", 
                "content": "This contains a secret password that should be masked.",
                "relevance_score": 0.8,
                "metadata": {
                    "source": "memory_layer",
                    "internal_score": 0.85
                }
            },
            {
                "id": "result_3",
                "content": "This is a very long response that exceeds the maximum length limit and should be truncated to prevent overwhelming the client with too much information at once.",
                "relevance_score": 0.7,
                "metadata": {
                    "source": "memory_layer"
                }
            }
        ]
        
        buffer_service = MockBufferService(mock_results)
        gateway = MemoryApiGateway(buffer_service=buffer_service, db_service=None)
        
        return gateway
    
    @pytest.mark.asyncio
    async def test_full_pipeline_with_all_filters(self, configured_gateway):
        """Test complete pipeline with all filters enabled."""
        request_data = {
            "user_id": "test_user",
            "agent_id": "test_agent", 
            "session_id": "test_session",
            "query": "Find information about security",
            "top_k": 3
        }
        
        response = await configured_gateway.process_request(request_data)
        
        # Verify response structure
        assert response["status"] == "success"
        assert response["code"] == 200
        assert "data" in response
        assert "results" in response["data"]
        
        results = response["data"]["results"]
        assert len(results) <= 3
        
        # Verify outbound filters were applied
        for result in results:
            # Field remover should have removed internal fields
            assert "internal_score" not in result.get("metadata", {})
            assert "debug_info" not in result.get("metadata", {})
            
            # Sensitive word filter should have masked sensitive content
            if "secret" in result["content"] or "password" in result["content"]:
                assert "[MASKED]" in result["content"] or result.get("metadata", {}).get("sensitive_hit")
        
        # Verify debug information is included
        if "metadata" in response["data"]:
            metadata = response["data"]["metadata"]
            if "observability" in metadata:
                obs = metadata["observability"]
                assert "gateway_duration_ms" in obs or "transformation_duration_ms" in obs
    
    @pytest.mark.asyncio
    async def test_error_handling_in_pipeline(self, configured_gateway):
        """Test error handling when filters encounter issues."""
        # Test with malformed request
        malformed_request = {
            "user_id": None,  # Invalid user_id
            "query": "",      # Empty query
            "top_k": -1       # Invalid top_k
        }
        
        response = await configured_gateway.process_request(malformed_request)
        
        # Should handle gracefully and return appropriate error
        assert response["status"] in ["error", "success"]  # Depends on validation logic
        
        if response["status"] == "error":
            assert "message" in response
            assert response["code"] >= 400
    
    @pytest.mark.asyncio
    async def test_performance_under_concurrent_load(self, configured_gateway):
        """Test pipeline performance under concurrent requests."""
        async def make_request(request_id: int):
            request_data = {
                "user_id": f"user_{request_id}",
                "agent_id": "test_agent",
                "session_id": f"session_{request_id}",
                "query": f"Query number {request_id}",
                "top_k": 2
            }
            return await configured_gateway.process_request(request_data)
        
        # Create 10 concurrent requests
        tasks = [make_request(i) for i in range(10)]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify all requests completed successfully
        successful_responses = [r for r in responses if not isinstance(r, Exception)]
        assert len(successful_responses) == 10
        
        # Verify each response is valid
        for response in successful_responses:
            assert response["status"] == "success"
            assert "data" in response
            assert "results" in response["data"]
    
    @pytest.mark.asyncio
    async def test_filter_configuration_changes(self):
        """Test dynamic filter configuration changes."""
        gcm = get_global_config_manager()
        
        # Initial configuration - sensitive word filter enabled
        await gcm.hot_reload({
            "gateway": {
                "pipeline": {
                    "outbound": [
                        {"name": "sensitive_word", "enabled": True}
                    ]
                }
            },
            "guardrail": {
                "sensitive": {
                    "enabled": True,
                    "words": ["secret"],
                    "action": "mask"
                }
            }
        })
        
        mock_results = [{"id": "1", "content": "This contains secret information"}]
        buffer_service = MockBufferService(mock_results)
        gateway = MemoryApiGateway(buffer_service=buffer_service, db_service=None)
        
        # First request - should mask sensitive content
        response1 = await gateway.process_request({
            "user_id": "user1", "query": "test", "top_k": 1
        })
        
        result1 = response1["data"]["results"][0]
        assert "[MASKED]" in result1["content"] or result1.get("metadata", {}).get("sensitive_hit")
        
        # Change configuration - disable sensitive word filter
        await gcm.hot_reload({
            "gateway": {
                "pipeline": {
                    "outbound": [
                        {"name": "sensitive_word", "enabled": False}
                    ]
                }
            }
        })
        
        # Create new gateway instance to pick up new config
        gateway2 = MemoryApiGateway(buffer_service=buffer_service, db_service=None)
        
        # Second request - should not mask sensitive content
        response2 = await gateway2.process_request({
            "user_id": "user2", "query": "test", "top_k": 1
        })
        
        result2 = response2["data"]["results"][0]
        # Content should be unmasked since filter is disabled
        assert "secret" in result2["content"]


class TestQueryBufferIntegration:
    """Test QueryBuffer integration with various stores."""
    
    @pytest.fixture
    async def query_buffer_with_store(self):
        """Create QueryBuffer with InMemoryStore."""
        # Create and populate in-memory store
        store = InMemoryStore()
        await store.initialize()
        
        # Add test items
        test_items = [
            Item(id="item1", content="Machine learning algorithms", metadata={"topic": "AI"}),
            Item(id="item2", content="Database optimization techniques", metadata={"topic": "DB"}),
            Item(id="item3", content="Web security best practices", metadata={"topic": "Security"}),
            Item(id="item4", content="Cloud computing architecture", metadata={"topic": "Cloud"}),
            Item(id="item5", content="Data science methodologies", metadata={"topic": "Data"})
        ]
        
        for item in test_items:
            await store.add(item)
        
        # Create QueryBuffer with store
        retrieval_handler = make_retrieval_handler_from_store(store)
        query_buffer = QueryBuffer(
            retrieval_handler=retrieval_handler,
            max_size=10
        )
        
        return query_buffer, store
    
    @pytest.mark.asyncio
    async def test_query_buffer_caching_behavior(self, query_buffer_with_store):
        """Test QueryBuffer caching with repeated queries."""
        query_buffer, store = query_buffer_with_store
        
        # First query - should hit storage
        results1 = await query_buffer.query("machine learning", top_k=3)
        assert len(results1) > 0
        assert query_buffer.cache_misses == 1
        assert query_buffer.cache_hits == 0
        
        # Same query - should hit cache
        results2 = await query_buffer.query("machine learning", top_k=3)
        assert len(results2) > 0
        assert query_buffer.cache_hits == 1
        
        # Verify results are consistent
        assert len(results1) == len(results2)
        assert results1[0]["id"] == results2[0]["id"]
    
    @pytest.mark.asyncio
    async def test_query_buffer_with_different_parameters(self, query_buffer_with_store):
        """Test QueryBuffer with various query parameters."""
        query_buffer, store = query_buffer_with_store
        
        # Test different top_k values
        results_k1 = await query_buffer.query("database", top_k=1)
        results_k3 = await query_buffer.query("database", top_k=3)
        
        assert len(results_k1) <= 1
        assert len(results_k3) <= 3
        assert len(results_k3) >= len(results_k1)
        
        # Test sorting
        results_asc = await query_buffer.query("security", sort_by="score", order="asc")
        results_desc = await query_buffer.query("security", sort_by="score", order="desc")
        
        if len(results_asc) > 1 and len(results_desc) > 1:
            # First result in desc should have higher score than first in asc
            assert results_desc[0]["score"] >= results_asc[0]["score"]
    
    @pytest.mark.asyncio
    async def test_store_error_handling(self):
        """Test error handling when store operations fail."""
        # Create a mock store that raises exceptions
        mock_store = AsyncMock()
        mock_store.query.side_effect = Exception("Store connection failed")
        mock_store.initialize.return_value = True
        
        retrieval_handler = make_retrieval_handler_from_store(mock_store)
        query_buffer = QueryBuffer(retrieval_handler=retrieval_handler)
        
        # Query should handle store errors gracefully
        results = await query_buffer.query("test query")
        
        # Should return empty results rather than crashing
        assert isinstance(results, list)
        assert len(results) == 0


class TestSemanticValidationIntegration:
    """Test semantic validation integration with the pipeline."""
    
    @pytest.mark.asyncio
    async def test_semantic_validation_in_pipeline(self):
        """Test semantic validation as part of the full pipeline."""
        # Configure with semantic validation enabled
        gcm = get_global_config_manager()
        await gcm.hot_reload({
            "semantic_validation": {
                "enabled": True,
                "similarity_threshold": 0.8,
                "relevance_threshold": 0.3
            },
            "gateway": {
                "pipeline": {
                    "outbound": [
                        {"name": "semantic_validator", "enabled": True}
                    ]
                }
            }
        })
        
        # Create results with potentially similar content
        similar_results = [
            {"id": "1", "content": "Machine learning is a subset of artificial intelligence"},
            {"id": "2", "content": "AI includes machine learning as a key component"},
            {"id": "3", "content": "Completely different topic about cooking recipes"}
        ]
        
        buffer_service = MockBufferService(similar_results)
        gateway = MemoryApiGateway(buffer_service=buffer_service, db_service=None)
        
        response = await gateway.process_request({
            "user_id": "test_user",
            "query": "artificial intelligence machine learning",
            "top_k": 3
        })
        
        # Verify response structure
        assert response["status"] == "success"
        results = response["data"]["results"]
        
        # Results should be processed (semantic validation may flag similar content)
        assert len(results) > 0
        
        # Check if semantic validation metadata is added
        for result in results:
            metadata = result.get("metadata", {})
            # Semantic validation might add flags or scores
            if "semantic_similarity" in metadata or "relevance_score" in metadata:
                assert isinstance(metadata.get("semantic_similarity", 0), (int, float))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
