#!/usr/bin/env python3
"""
Simple end-to-end test for MemFuse API.

This script tests the basic API functionality without complex dependencies.
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint."""
    try:
        response = requests.get(f"{BASE_URL}/api/v1/health", timeout=5)
        print(f"✅ Health check: {response.status_code}")
        if response.status_code == 200:
            print(f"   Response: {response.json()}")
            return True
        else:
            print(f"   Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_user_creation():
    """Test user creation."""
    try:
        # Check if user exists
        response = requests.get(f"{BASE_URL}/api/v1/users?name=test_user", timeout=5)
        if response.status_code == 200:
            users = response.json().get("data", [])
            if users:
                user_id = users[0]["id"]
                print(f"✅ User exists: test_user (ID: {user_id})")
                return user_id
        
        # Create user if not exists
        user_data = {
            "name": "test_user",
            "email": "test@example.com"
        }
        response = requests.post(f"{BASE_URL}/api/v1/users", json=user_data, timeout=5)
        if response.status_code == 200:
            user_id = response.json()["data"]["user"]["id"]
            print(f"✅ User created: test_user (ID: {user_id})")
            return user_id
        else:
            print(f"❌ User creation failed: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"❌ User creation error: {e}")
        return None

def test_agent_creation():
    """Test agent creation."""
    try:
        # Check if agent exists
        response = requests.get(f"{BASE_URL}/api/v1/agents?name=test_assistant", timeout=5)
        if response.status_code == 200:
            agents = response.json().get("data", [])
            if agents:
                agent_id = agents[0]["id"]
                print(f"✅ Agent exists: test_assistant (ID: {agent_id})")
                return agent_id
        
        # Create agent if not exists
        agent_data = {
            "name": "test_assistant",
            "description": "Test assistant for E2E testing"
        }
        response = requests.post(f"{BASE_URL}/api/v1/agents", json=agent_data, timeout=5)
        if response.status_code == 200:
            agent_id = response.json()["data"]["agent"]["id"]
            print(f"✅ Agent created: test_assistant (ID: {agent_id})")
            return agent_id
        else:
            print(f"❌ Agent creation failed: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"❌ Agent creation error: {e}")
        return None

def test_session_creation(user_id, agent_id):
    """Test session creation."""
    try:
        session_data = {
            "user_id": user_id,
            "agent_id": agent_id,
            "name": "test_session"
        }
        response = requests.post(f"{BASE_URL}/api/v1/sessions", json=session_data, timeout=30)
        if response.status_code == 200:
            session_id = response.json()["data"]["session"]["id"]
            print(f"✅ Session created: test_session (ID: {session_id})")
            return session_id
        else:
            print(f"❌ Session creation failed: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"❌ Session creation error: {e}")
        return None

def test_message_storage(session_id):
    """Test message storage."""
    try:
        # Store a user message
        message_data = {
            "session_id": session_id,
            "role": "user",
            "content": "Hello, this is a test message about space exploration."
        }
        response = requests.post(f"{BASE_URL}/api/v1/messages", json=message_data, timeout=5)
        if response.status_code == 200:
            message_id = response.json()["data"]["message"]["id"]
            print(f"✅ User message stored (ID: {message_id})")
            
            # Store an assistant response
            response_data = {
                "session_id": session_id,
                "role": "assistant",
                "content": "Hello! I'd be happy to help you with space exploration topics. What would you like to know?"
            }
            response = requests.post(f"{BASE_URL}/api/v1/messages", json=response_data, timeout=5)
            if response.status_code == 200:
                response_id = response.json()["data"]["message"]["id"]
                print(f"✅ Assistant message stored (ID: {response_id})")
                return True
            else:
                print(f"❌ Assistant message storage failed: {response.status_code} - {response.text}")
                return False
        else:
            print(f"❌ User message storage failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Message storage error: {e}")
        return False

def test_message_retrieval(session_id):
    """Test message retrieval."""
    try:
        response = requests.get(f"{BASE_URL}/api/v1/sessions/{session_id}/messages", timeout=5)
        if response.status_code == 200:
            messages = response.json()["data"]["messages"]
            print(f"✅ Retrieved {len(messages)} messages")
            for msg in messages:
                print(f"   {msg['role']}: {msg['content'][:50]}...")
            return True
        else:
            print(f"❌ Message retrieval failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Message retrieval error: {e}")
        return False

def test_memory_search(session_id):
    """Test memory search functionality."""
    try:
        search_data = {
            "query": "space exploration",
            "session_id": session_id,
            "limit": 5
        }
        response = requests.post(f"{BASE_URL}/api/v1/memory/search", json=search_data, timeout=10)
        if response.status_code == 200:
            results = response.json()["data"]["results"]
            print(f"✅ Memory search returned {len(results)} results")
            for result in results[:2]:  # Show first 2 results
                print(f"   Score: {result.get('score', 'N/A'):.3f} - {result.get('content', 'No content')[:50]}...")
            return True
        else:
            print(f"❌ Memory search failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Memory search error: {e}")
        return False

def main():
    """Run all end-to-end tests."""
    print("🚀 Starting MemFuse End-to-End Tests...")
    
    # Test 1: Health check
    if not test_health():
        print("❌ Health check failed. Make sure server is running.")
        return False
    
    # Test 2: User creation
    user_id = test_user_creation()
    if not user_id:
        print("❌ User creation failed.")
        return False
    
    # Test 3: Agent creation
    agent_id = test_agent_creation()
    if not agent_id:
        print("❌ Agent creation failed.")
        return False
    
    # Test 4: Session creation
    session_id = test_session_creation(user_id, agent_id)
    if not session_id:
        print("❌ Session creation failed.")
        return False
    
    # Test 5: Message storage
    if not test_message_storage(session_id):
        print("❌ Message storage failed.")
        return False
    
    # Wait a bit for processing
    print("⏳ Waiting for message processing...")
    time.sleep(2)
    
    # Test 6: Message retrieval
    if not test_message_retrieval(session_id):
        print("❌ Message retrieval failed.")
        return False
    
    # Test 7: Memory search
    if not test_memory_search(session_id):
        print("❌ Memory search failed.")
        return False
    
    print("\n🎉 All end-to-end tests passed!")
    print("✅ MemFuse API is working correctly")
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
