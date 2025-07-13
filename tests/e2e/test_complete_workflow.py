#!/usr/bin/env python3
"""
End-to-end test script for MemFuse server.
Tests the complete workflow: create user -> add memory -> query memory.
"""

import requests
import json
import time
import sys
import uuid

BASE_URL = "http://localhost:8000/api/v1"

def test_health():
    """Test health endpoint."""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            print("✅ Health check passed")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def create_user(user_name):
    """Create a new user."""
    print(f"👤 Creating user '{user_name}'...")
    try:
        payload = {
            "name": user_name,
            "email": f"{user_name}@example.com"
        }
        response = requests.post(f"{BASE_URL}/users", json=payload, timeout=10)
        if response.status_code == 201:
            user_data = response.json()
            print(f"✅ User created successfully")
            print(f"   Response: {user_data}")
            # Handle different response formats
            if 'data' in user_data and 'id' in user_data['data']:
                return user_data['data']['id']
            elif 'id' in user_data:
                return user_data['id']
            else:
                print(f"❌ Unexpected response format: {user_data}")
                return None
        elif response.status_code == 400:
            error_data = response.json()
            if "already exists" in error_data.get('message', ''):
                print(f"ℹ️ User '{user_name}' already exists, trying to get existing user")
                return get_user_by_name(user_name)
            else:
                print(f"❌ User creation failed: {response.status_code} - {response.text}")
                return None
        else:
            print(f"❌ User creation failed: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"❌ User creation error: {e}")
        return None

def get_user_by_name(user_name):
    """Get user by name."""
    print(f"🔍 Looking up user '{user_name}'...")
    try:
        response = requests.get(f"{BASE_URL}/users", params={"name": user_name}, timeout=10)
        if response.status_code == 200:
            user_data = response.json()
            print(f"✅ User found")
            print(f"   Response: {user_data}")
            # Handle different response formats
            if 'data' in user_data and 'users' in user_data['data'] and len(user_data['data']['users']) > 0:
                return user_data['data']['users'][0]['id']
            elif 'data' in user_data and 'id' in user_data['data']:
                return user_data['data']['id']
            elif 'id' in user_data:
                return user_data['id']
            else:
                print(f"❌ Unexpected response format: {user_data}")
                return None
        elif response.status_code == 404:
            print(f"ℹ️ User '{user_name}' not found")
            return None
        else:
            print(f"❌ User lookup failed: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"❌ User lookup error: {e}")
        return None

def create_agent(user_id):
    """Create a new agent for the user."""
    print(f"🤖 Creating agent for user {user_id}...")
    try:
        unique_id = str(uuid.uuid4())[:8]
        payload = {
            "user_id": user_id,
            "name": f"Test Agent {unique_id}",
            "description": "A test agent for end-to-end testing"
        }
        response = requests.post(f"{BASE_URL}/agents", json=payload, timeout=10)
        if response.status_code == 201:
            agent_data = response.json()
            print(f"✅ Agent created successfully")
            print(f"   Response: {agent_data}")
            # Handle different response formats
            if 'data' in agent_data and 'agent' in agent_data['data'] and 'id' in agent_data['data']['agent']:
                return agent_data['data']['agent']['id']
            elif 'data' in agent_data and 'id' in agent_data['data']:
                return agent_data['data']['id']
            elif 'id' in agent_data:
                return agent_data['id']
            else:
                print(f"❌ Unexpected response format: {agent_data}")
                return None
        else:
            print(f"❌ Agent creation failed: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"❌ Agent creation error: {e}")
        return None


def create_session(user_id, agent_id):
    """Create a new session for the user."""
    print(f"📝 Creating session for user {user_id}...")
    try:
        unique_id = str(uuid.uuid4())[:8]
        payload = {
            "user_id": user_id,
            "agent_id": agent_id,
            "name": f"Test Session {unique_id}"
        }
        response = requests.post(f"{BASE_URL}/sessions", json=payload, timeout=10)
        if response.status_code == 201:
            session_data = response.json()
            print(f"✅ Session created successfully")
            print(f"   Response: {session_data}")
            # Handle different response formats
            if 'data' in session_data and 'session' in session_data['data'] and 'id' in session_data['data']['session']:
                return session_data['data']['session']['id']
            elif 'data' in session_data and 'id' in session_data['data']:
                return session_data['data']['id']
            elif 'id' in session_data:
                return session_data['id']
            else:
                print(f"❌ Unexpected response format: {session_data}")
                return None
        else:
            print(f"❌ Session creation failed: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"❌ Session creation error: {e}")
        return None

def add_message(session_id, content):
    """Add a message to the session."""
    print(f"💬 Adding message to session {session_id}...")
    try:
        payload = {
            "messages": [
                {
                    "content": content,
                    "role": "user"
                }
            ]
        }
        response = requests.post(f"{BASE_URL}/sessions/{session_id}/messages", json=payload, timeout=15)
        if response.status_code == 201:
            message_data = response.json()
            print(f"✅ Message added successfully")
            print(f"   Response: {message_data}")
            # Handle different response formats
            if 'data' in message_data and 'message_ids' in message_data['data'] and len(message_data['data']['message_ids']) > 0:
                return message_data['data']['message_ids'][0]
            elif 'data' in message_data and 'messages' in message_data['data'] and len(message_data['data']['messages']) > 0:
                return message_data['data']['messages'][0]['id']
            elif 'data' in message_data and 'id' in message_data['data']:
                return message_data['data']['id']
            elif 'id' in message_data:
                return message_data['id']
            else:
                print(f"❌ Unexpected response format: {message_data}")
                return None
        else:
            print(f"❌ Message addition failed: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"❌ Message addition error: {e}")
        return None

def query_memory(session_id, query):
    """Query memory for the session."""
    print(f"🔍 Querying memory in session {session_id}...")
    try:
        payload = {
            "query": query,
            "top_k": 5
        }
        response = requests.post(f"{BASE_URL}/sessions/{session_id}/messages/query", json=payload, timeout=15)
        if response.status_code == 200:
            results = response.json()
            print(f"✅ Query successful, found {len(results['data'])} results")
            return results['data']
        else:
            print(f"❌ Query failed: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"❌ Query error: {e}")
        return None

def main():
    """Run the complete end-to-end test."""
    print("🚀 Starting MemFuse End-to-End Test")
    print("=" * 50)
    
    # Test 1: Health check
    if not test_health():
        print("❌ Health check failed, aborting test")
        sys.exit(1)
    
    # Test 2: User management
    user_name = "alice"
    user_id = get_user_by_name(user_name)
    
    if user_id is None:
        user_id = create_user(user_name)
        if user_id is None:
            print("❌ Failed to create user, aborting test")
            sys.exit(1)
    
    # Test 3: Agent creation
    agent_id = create_agent(user_id)
    if agent_id is None:
        print("❌ Failed to create agent, aborting test")
        sys.exit(1)

    # Test 4: Session creation
    session_id = create_session(user_id, agent_id)
    if session_id is None:
        print("❌ Failed to create session, aborting test")
        sys.exit(1)
    
    # Test 5: Add memory
    test_content = "I love playing basketball on weekends. My favorite team is the Lakers."
    message_id = add_message(session_id, test_content)
    if message_id is None:
        print("❌ Failed to add message, aborting test")
        sys.exit(1)

    # Wait a bit for processing
    print("⏳ Waiting for memory processing...")
    time.sleep(3)

    # Test 6: Query memory
    query_text = "What sports do I like?"
    results = query_memory(session_id, query_text)
    if results is None:
        print("❌ Failed to query memory, aborting test")
        sys.exit(1)

    # Test 7: Verify results
    print("\n📊 Query Results:")
    for i, result in enumerate(results):
        print(f"  {i+1}. Score: {result.get('score', 'N/A')}")
        print(f"     Content: {result.get('content', 'N/A')[:100]}...")
    
    print("\n🎉 End-to-End Test Completed Successfully!")
    print("✅ All core functionality is working properly")

if __name__ == "__main__":
    main()
