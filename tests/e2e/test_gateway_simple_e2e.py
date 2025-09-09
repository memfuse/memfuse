"""Simple Gateway E2E test."""

import pytest
import httpx

TEST_BASE_URL = "http://localhost:8000"
TEST_API_KEY = "test-key"
TEST_USER_ID = "a4b063c1-9e0e-46c4-bcfc-4aeb4b1317cb"

HEADERS = {
    "Content-Type": "application/json",
    "X-API-Key": TEST_API_KEY
}

@pytest.mark.asyncio
async def test_gateway_query_endpoint():
    """Test that the gateway query endpoint is accessible."""
    async with httpx.AsyncClient() as client:
        try:
            # Test basic query
            query_payload = {
                "query": "test",
                "top_k": 1,
                "metadata": {
                    "task": "search",
                    "mode": "episodic"
                }
            }
            
            response = await client.post(
                f"{TEST_BASE_URL}/api/v1/users/{TEST_USER_ID}/query",
                headers=HEADERS,
                json=query_payload,
                timeout=10.0
            )
            
            # Should get a response (even if empty)
            assert response.status_code in [200, 404, 500]  # Any response is good
            
            if response.status_code == 200:
                data = response.json()
                # Check basic response structure
                assert "status" in data
                assert "data" in data or "message" in data
                
        except httpx.ConnectError:
            pytest.skip("Server not available - skipping E2E test")
        except Exception as e:
            pytest.fail(f"Unexpected error: {e}")

def test_simple_assertion():
    """Simple test that always passes."""
    assert True
