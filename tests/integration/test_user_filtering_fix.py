#!/usr/bin/env python3
"""
Integration test for user filtering fix in SimplifiedMemoryService.

This test verifies that the user filtering functionality works correctly
after the fix to ensure proper session_id to conversation_id mapping.
"""

import asyncio
import aiohttp
import pytest
from loguru import logger


class TestUserFilteringFix:
    """Test class for user filtering functionality."""
    
    @pytest.mark.asyncio
    async def test_api_user_filtering(self):
        """Test that API correctly filters results by user."""
        # Test with a known user ID that should have data
        test_user_id = "27944703-161c-448f-bcc9-3f122c6997e7"
        test_query = "Hey, remember that time we talked about our jobs and expenses?"
        
        url = f"http://localhost:8000/api/v1/users/{test_user_id}/query"
        payload = {
            "query": test_query,
            "top_k": 5
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                assert response.status == 200
                
                result = await response.json()
                assert result["status"] == "success"
                
                data = result.get("data", {})
                results = data.get("results", [])
                
                # Should return results for valid user
                assert len(results) > 0, "User filtering should return results for valid user"
                
                # All results should have correct user_id in metadata
                for result_item in results:
                    metadata = result_item.get("metadata", {})
                    assert metadata.get("user_id") == test_user_id
    
    @pytest.mark.asyncio
    async def test_api_user_isolation(self):
        """Test that users cannot access other users' data."""
        # Test with a user ID that should not have access to other users' data
        test_user_id = "00000000-0000-0000-0000-000000000000"  # Non-existent user
        test_query = "Hey, remember that time we talked about our jobs and expenses?"
        
        url = f"http://localhost:8000/api/v1/users/{test_user_id}/query"
        payload = {
            "query": test_query,
            "top_k": 5
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                assert response.status == 200
                
                result = await response.json()
                assert result["status"] == "success"
                
                data = result.get("data", {})
                results = data.get("results", [])
                
                # Should return no results for non-existent user
                assert len(results) == 0, "User filtering should return no results for non-existent user"
    
    @pytest.mark.asyncio
    async def test_chunk_type_handling(self):
        """Test that chunk type results are handled correctly."""
        test_user_id = "27944703-161c-448f-bcc9-3f122c6997e7"
        test_query = "save money groceries"
        
        url = f"http://localhost:8000/api/v1/users/{test_user_id}/query"
        payload = {
            "query": test_query,
            "top_k": 3
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                assert response.status == 200
                
                result = await response.json()
                assert result["status"] == "success"
                
                data = result.get("data", {})
                results = data.get("results", [])
                
                if results:
                    # Check that chunk results have proper structure
                    for result_item in results:
                        assert "id" in result_item
                        assert "content" in result_item
                        assert "score" in result_item
                        assert "type" in result_item
                        assert "metadata" in result_item
                        
                        # Chunk results should have type "chunk"
                        if result_item.get("type") == "chunk":
                            assert result_item.get("role") is None  # Chunks don't have roles
                            metadata = result_item.get("metadata", {})
                            assert metadata.get("level") == 1  # M1 layer
                            assert metadata.get("source") == "memory_database"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
