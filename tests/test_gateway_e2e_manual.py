#!/usr/bin/env python3
"""Manual E2E test for Gateway functionality."""

import asyncio
import httpx
import json

# Test configuration
TEST_BASE_URL = "http://localhost:8000"
TEST_API_KEY = "test-key"

async def test_gateway_e2e():
    """Test Gateway E2E functionality."""
    print("🧪 Gateway E2E Test")
    print("=" * 50)
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        headers = {"X-API-Key": TEST_API_KEY, "Content-Type": "application/json"}
        
        try:
            # 1. Health check
            print("🔍 1. Health check...")
            health_response = await client.get(f"{TEST_BASE_URL}/api/v1/health")
            assert health_response.status_code == 200
            print("✅ Server is healthy")
            
            # 2. Create user
            print("\n🔍 2. Creating test user...")
            user_payload = {
                "id": "gateway_test_user",
                "name": "Gateway Test User",
                "email": "gateway@test.com"
            }
            user_response = await client.post(
                f"{TEST_BASE_URL}/api/v1/users",
                headers=headers,
                json=user_payload
            )
            if user_response.status_code == 201:
                user_data = user_response.json()
                user_id = user_data["data"]["user"]["id"]
                print(f"✅ User created: {user_id}")
            else:
                print(f"⚠️  User creation failed: {user_response.status_code}")
                return False
            
            # 3. Create agent
            print("\n🔍 3. Creating test agent...")
            agent_payload = {
                "name": "Gateway Test Agent",
                "description": "Agent for Gateway testing"
            }
            agent_response = await client.post(
                f"{TEST_BASE_URL}/api/v1/agents",
                headers=headers,
                json=agent_payload
            )
            if agent_response.status_code == 201:
                agent_data = agent_response.json()
                agent_id = agent_data["data"]["agent"]["id"]
                print(f"✅ Agent created: {agent_id}")
            else:
                print(f"⚠️  Agent creation failed: {agent_response.status_code}")
                return False
            
            # 4. Test query (empty results expected)
            print("\n🔍 4. Testing query endpoint...")
            query_payload = {
                "query": "Gateway transformation test",
                "top_k": 3,
                "agent_id": agent_id,
                "metadata": {
                    "task": "search",
                    "mode": "episodic"
                }
            }
            query_response = await client.post(
                f"{TEST_BASE_URL}/api/v1/users/{user_id}/query",
                headers=headers,
                json=query_payload
            )
            
            if query_response.status_code == 200:
                query_data = query_response.json()
                print("✅ Query successful")
                print(f"   Status: {query_data['status']}")
                print(f"   Results: {len(query_data['data']['results'])}")
                print(f"   Message: {query_data['message']}")
                
                # Verify Gateway response structure
                assert "status" in query_data
                assert "data" in query_data
                assert "results" in query_data["data"]
                assert "total" in query_data["data"]
                print("✅ Response structure is correct")
                
                # If there are results, check Gateway transformations
                if query_data["data"]["results"]:
                    result = query_data["data"]["results"][0]
                    
                    # Check field renaming
                    if "relevance_score" in result and "memory_type" in result:
                        print("✅ Field renaming applied")
                    else:
                        print("⚠️  Field renaming not applied")
                    
                    # Check forbidden fields are removed
                    if "score" not in result and "type" not in result:
                        print("✅ Forbidden fields removed")
                    else:
                        print("⚠️  Forbidden fields still present")
                    
                    # Check metadata enrichment
                    if "metadata" in result and "scope" in result["metadata"]:
                        print("✅ Metadata enrichment applied")
                    else:
                        print("⚠️  Metadata enrichment not applied")
                
                return True
            else:
                print(f"❌ Query failed: {query_response.status_code}")
                print(f"   Response: {query_response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            return False

async def main():
    """Main test function."""
    success = await test_gateway_e2e()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 Gateway E2E test PASSED!")
        print("\n💡 Key findings:")
        print("- Server is healthy and accessible")
        print("- Gateway query endpoint is working")
        print("- Response structure is correct")
        print("- Ready for data-driven testing")
        return 0
    else:
        print("❌ Gateway E2E test FAILED!")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))
