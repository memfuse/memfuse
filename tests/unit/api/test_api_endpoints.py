#!/usr/bin/env python3
"""Unit tests for API endpoints."""

import requests
import json
import time

# Test configuration
BASE_URL = "http://localhost:8000"
API_KEY = "test-key"  # Default test API key

def test_health():
    """Test health endpoint."""
    print("1. Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/v1/health")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"   ❌ Health test failed: {e}")
        return False

def test_user_creation():
    """Test user creation."""
    print("2. Testing user creation...")
    try:
        # Check if user exists
        user_name = "test_api_unit_user"
        response = requests.get(
            f"{BASE_URL}/api/v1/users",
            params={"name": user_name},
            headers={"X-API-Key": API_KEY}
        )
        print(f"   User check status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"   Response: {result}")
            users = result.get("data", {}).get("users", [])
            if users:
                print(f"   ✅ User exists: {users[0]['id']}")
                return users[0]
            else:
                print("   User not found, would need to create")
                return None
        else:
            print(f"   ❌ User check failed: {response.text}")
            return None
    except Exception as e:
        print(f"   ❌ User creation test failed: {e}")
        return None

def test_agent_creation():
    """Test agent creation."""
    print("3. Testing agent creation...")
    try:
        agent_name = "test_api_agent"
        response = requests.get(
            f"{BASE_URL}/api/v1/agents",
            params={"name": agent_name},
            headers={"X-API-Key": API_KEY}
        )
        print(f"   Agent check status: {response.status_code}")
        
        if response.status_code == 200:
            agents = response.json().get("data", {}).get("agents", [])
            if agents:
                print(f"   ✅ Agent exists: {agents[0]['id']}")
                return agents[0]
            else:
                print("   Agent not found, would need to create")
                return None
        else:
            print(f"   ❌ Agent check failed: {response.text}")
            return None
    except Exception as e:
        print(f"   ❌ Agent creation test failed: {e}")
        return None

def test_session_creation(user_id):
    """Test session creation."""
    print("4. Testing session creation...")
    try:
        session_name = "test_api_session"
        response = requests.get(
            f"{BASE_URL}/api/v1/sessions",
            params={"name": session_name, "user_id": user_id},
            headers={"X-API-Key": API_KEY}
        )
        print(f"   Session check status: {response.status_code}")
        
        if response.status_code == 200:
            sessions = response.json().get("data", {}).get("sessions", [])
            if sessions:
                print(f"   ✅ Session exists: {sessions[0]['id']}")
                return sessions[0]
            else:
                print("   Session not found, would need to create")
                return None
        else:
            print(f"   ❌ Session check failed: {response.text}")
            return None
    except Exception as e:
        print(f"   ❌ Session creation test failed: {e}")
        return None

def test_message_addition(session_id):
    """Test message addition."""
    print("5. Testing message addition...")
    try:
        messages = [
            {"role": "user", "content": "Hello, I love Taylor Swift's music!"},
            {"role": "assistant", "content": "That's great! Taylor Swift is an amazing artist."},
            {"role": "user", "content": "What's your favorite Taylor Swift song?"},
            {"role": "assistant", "content": "A little bit. I can get into taylor swift."}
        ]
        
        response = requests.post(
            f"{BASE_URL}/api/v1/sessions/{session_id}/messages",
            json={"messages": messages},
            headers={"X-API-Key": API_KEY}
        )
        print(f"   Message addition status: {response.status_code}")
        print(f"   Response: {response.json()}")
        
        if response.status_code == 200:
            result = response.json()
            if result.get("status") == "success":
                print("   ✅ Messages added successfully")
                return True
            else:
                print(f"   ❌ Message addition failed: {result}")
                return False
        else:
            print(f"   ❌ Message addition failed: {response.text}")
            return False
    except Exception as e:
        print(f"   ❌ Message addition test failed: {e}")
        return False

def test_user_query(user_id):
    """Test user query endpoint."""
    print("6. Testing user query...")
    try:
        query_data = {
            "query": "Hey, remember that time we talked about music? What was the artist you mentioned you could get into?",
            "top_k": 15
        }
        
        response = requests.post(
            f"{BASE_URL}/api/v1/users/{user_id}/query",
            json=query_data,
            headers={"X-API-Key": API_KEY}
        )
        print(f"   Query status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"   Query result status: {result.get('status')}")
            
            if result.get("status") == "success":
                data = result.get("data", {})
                results = data.get("results", [])
                total = data.get("total", 0)
                
                print(f"   ✅ Query successful: {total} results found")
                print(f"   Results count: {len(results)}")
                
                if results:
                    first_result = results[0]
                    print(f"   First result content: {first_result.get('content', '')[:50]}...")
                    print(f"   First result score: {first_result.get('score')}")
                    
                    # Check if we found the Taylor Swift answer
                    taylor_swift_found = any("taylor swift" in result.get('content', '').lower() for result in results)
                    if taylor_swift_found:
                        print("   ✅ Found Taylor Swift related content!")
                        return True
                    else:
                        print("   ⚠️  No Taylor Swift content found in results")
                        return False
                else:
                    print("   ❌ No results returned")
                    return False
            else:
                print(f"   ❌ Query failed: {result}")
                return False
        else:
            print(f"   ❌ Query failed: {response.text}")
            return False
    except Exception as e:
        print(f"   ❌ User query test failed: {e}")
        return False

def main():
    """Run API unit tests."""
    print("🧪 API Unit Tests")
    print("=" * 60)
    
    # Wait a moment for server to be ready
    time.sleep(2)
    
    # Test 1: Health check
    if not test_health():
        print("❌ Health check failed - server may not be running")
        return False
    
    # Test 2: User creation/check
    user = test_user_creation()
    if not user:
        print("❌ User test failed")
        return False
    
    # Test 3: Agent creation/check
    agent = test_agent_creation()
    if not agent:
        print("❌ Agent test failed")
        return False
    
    # Test 4: Session creation/check
    session = test_session_creation(user["id"])
    if not session:
        print("❌ Session test failed")
        return False
    
    # Test 5: Message addition
    if not test_message_addition(session["id"]):
        print("❌ Message addition test failed")
        return False
    
    # Wait a moment for processing
    time.sleep(3)
    
    # Test 6: User query
    if not test_user_query(user["id"]):
        print("❌ User query test failed")
        return False
    
    print("\n" + "=" * 60)
    print("✅ All API unit tests passed!")
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 API unit tests completed successfully!")
    else:
        print("\n❌ Some API unit tests failed")
    
    exit(0 if success else 1)
