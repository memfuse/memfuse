"""Smoke tests for health check endpoint."""

import pytest
import requests
from typing import Dict, Any


@pytest.mark.smoke
@pytest.mark.api
class TestHealthEndpoint:
    """Smoke tests for the health check endpoint."""
    
    @pytest.fixture
    def api_config(self) -> Dict[str, Any]:
        """Configuration for API testing."""
        return {
            "base_url": "http://localhost:8000",
            "api_key": "test-key",
            "timeout": 30
        }
    
    @pytest.fixture
    def api_headers(self, api_config: Dict[str, Any]) -> Dict[str, str]:
        """Headers for API requests."""
        return {
            "X-API-Key": api_config["api_key"],
            "Content-Type": "application/json"
        }
    
    def test_health_endpoint_success(self, api_config: Dict[str, Any]):
        """Test health endpoint returns success status."""
        response = requests.get(
            f"{api_config['base_url']}/api/v1/health",
            timeout=api_config["timeout"]
        )
        
        # Assert HTTP status code (from the HTTP response)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        # Assert response is JSON
        assert response.headers.get("content-type", "").startswith("application/json")
        
        # Parse JSON response
        data = response.json()
        assert isinstance(data, dict), "Response should be a JSON object"
        
        # Assert API response structure (from the JSON content)
        assert "status" in data, "Health response should include status field"
        assert data["status"] == "success", f"Expected status 'success', got {data['status']}"
        
        assert "code" in data, "Health response should include code field"
        assert data["code"] == 200, f"Expected code 200, got {data['code']}"
        
        # Assert data section exists
        assert "data" in data, "Health response should include data field"
        assert isinstance(data["data"], dict), "Data field should be an object"
        
        # Check nested status in data
        assert "status" in data["data"], "Data should include status field"
        assert data["data"]["status"] == "ok", f"Expected data status 'ok', got {data['data']['status']}"
        
