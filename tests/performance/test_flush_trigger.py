#!/usr/bin/env python3
"""
Performance test script to trigger HybridBuffer flush mechanism and test M0/M1/M2 parallel processing.

This script simulates high-volume message writing to test:
1. HybridBuffer flush behavior under load
2. M0/M1/M2 parallel processing performance
3. Memory layer scalability
4. End-to-end pipeline performance

Usage:
    python tests/performance/test_flush_trigger.py

Requirements:
    - MemFuse server running on localhost:8000
    - poetry run memfuse-core
"""

import requests
import time
import uuid
import sys
from typing import Dict, Any, List

BASE_URL = "http://localhost:8000/api/v1"


def get_or_create_user(name: str) -> str:
    """Get existing user or create new one. Returns user_id."""
    # Try to find existing user
    try:
        response = requests.get(f"{BASE_URL}/users", params={"name": name})
        if response.status_code == 200:
            users = response.json().get("data", {}).get("users", [])
            if users:
                user_id = users[0]["id"]
                print(f"✓ Found existing user: {name} (ID: {user_id})")
                return user_id
    except Exception:
        pass  # User doesn't exist, will create below

    # Create new user
    user_data = {"name": name, "description": "User created by performance test script"}
    response = requests.post(f"{BASE_URL}/users", json=user_data)

    if response.status_code == 200:
        user_id = response.json()["data"]["user"]["id"]
        print(f"✓ Created new user: {name} (ID: {user_id})")
        return user_id
    else:
        raise Exception(f"Failed to create user: {response.text}")


def get_or_create_agent(name: str) -> str:
    """Get existing agent or create new one. Returns agent_id."""
    # Try to find existing agent
    try:
        response = requests.get(f"{BASE_URL}/agents", params={"name": name})
        if response.status_code == 200:
            agents = response.json().get("data", {}).get("agents", [])
            if agents:
                agent_id = agents[0]["id"]
                print(f"✓ Found existing agent: {name} (ID: {agent_id})")
                return agent_id
    except Exception:
        pass  # Agent doesn't exist, will create below

    # Create new agent
    agent_data = {"name": name, "description": "Agent created by performance test script"}
    response = requests.post(f"{BASE_URL}/agents", json=agent_data)

    if response.status_code == 200:
        agent_id = response.json()["data"]["agent"]["id"]
        print(f"✓ Created new agent: {name} (ID: {agent_id})")
        return agent_id
    else:
        raise Exception(f"Failed to create agent: {response.text}")


def create_session(user_id: str, agent_id: str) -> str:
    """Create a new session. Returns session_id."""
    session_data = {
        "user_id": user_id,
        "agent_id": agent_id,
        "description": "Performance test session for M0/M1/M2 parallel processing"
    }

    response = requests.post(f"{BASE_URL}/sessions", json=session_data)

    if response.status_code == 200:
        session_id = response.json()["data"]["session"]["id"]
        print(f"✓ Created session: {session_id}")
        return session_id
    else:
        raise Exception(f"Failed to create session: {response.text}")


def send_message(session_id: str, content: str, role: str = "user") -> Dict[str, Any]:
    """Send a message to the session."""
    message_data = {
        "messages": [{
            "content": content,
            "role": role,
            "metadata": {
                "test_type": "performance_flush_trigger",
                "timestamp": time.time()
            }
        }]
    }

    response = requests.post(f"{BASE_URL}/sessions/{session_id}/messages", json=message_data)

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to send message: {response.text}")


def generate_test_messages(count: int) -> List[str]:
    """Generate test messages with varying content to trigger different processing patterns."""
    messages = []

    # Technical discussion messages (good for M1 fact extraction)
    tech_topics = [
        "Machine learning algorithms like neural networks require large datasets for training.",
        "Distributed systems face challenges with consistency, availability, and partition tolerance.",
        "Database indexing improves query performance but increases storage overhead.",
        "Microservices architecture provides scalability but introduces complexity.",
        "Cloud computing offers elasticity and cost efficiency for modern applications."
    ]

    # Conversational messages (good for M0 episodic memory)
    conversations = [
        "I'm working on a new project that involves data processing.",
        "The weather has been quite unpredictable lately.",
        "I enjoyed the book you recommended last week.",
        "Let's schedule a meeting to discuss the quarterly results.",
        "The new restaurant downtown has excellent reviews."
    ]

    # Complex analytical content (good for M2 relational processing)
    analytical = [
        "The correlation between user engagement and feature adoption shows a strong positive relationship.",
        "Market analysis indicates three key trends: automation, personalization, and sustainability.",
        "Performance metrics demonstrate 40% improvement after optimization implementation.",
        "Customer feedback reveals pain points in onboarding, support, and feature discovery.",
        "Revenue growth patterns suggest seasonal variations with Q4 peaks."
    ]

    all_templates = tech_topics + conversations + analytical

    for i in range(count):
        template = all_templates[i % len(all_templates)]
        # Add variation to make each message unique
        message = f"[Message {i + 1}] {template} Additional context: {uuid.uuid4().hex[:8]}"
        messages.append(message)

    return messages


def run_performance_test():
    """Run the performance test for M0/M1/M2 parallel processing."""
    print("🚀 MemFuse M0/M1/M2 Parallel Processing Performance Test")
    print("=" * 70)

    try:
        # Setup test entities
        user_id = get_or_create_user("perf_test_user")
        agent_id = get_or_create_agent("perf_test_assistant")
        session_id = create_session(user_id, agent_id)

        print("\n📊 Test Configuration:")
        print(f"   User ID: {user_id}")
        print(f"   Agent ID: {agent_id}")
        print(f"   Session ID: {session_id}")

        # Generate test messages
        message_count = 15  # This should trigger multiple flush cycles
        test_messages = generate_test_messages(message_count)

        print(f"\n📝 Generated {len(test_messages)} test messages")
        print(f"   Expected flush triggers: ~{message_count // 5} (based on max_size=5)")

        # Send messages and measure performance
        print("\n🔄 Sending messages to trigger M0/M1/M2 parallel processing...")

        start_time = time.time()
        responses = []

        for i, message in enumerate(test_messages, 1):
            print(f"   [{i:2d}/{message_count}] Sending message...")

            message_start = time.time()
            response = send_message(session_id, message)
            message_time = time.time() - message_start

            responses.append({
                "message_id": response.get("data", {}).get("message", {}).get("id"),
                "processing_time": message_time,
                "message_number": i
            })

            print(f"   [{i:2d}/{message_count}] ✓ Processed in {message_time:.3f}s")

            # Small delay to allow buffer processing
            time.sleep(0.1)

        total_time = time.time() - start_time

        # Performance analysis
        print("\n📈 Performance Results:")
        print(f"   Total messages: {message_count}")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Average time per message: {total_time / message_count:.3f}s")
        print(f"   Messages per second: {message_count / total_time:.2f}")

        processing_times = [r["processing_time"] for r in responses]
        print(f"   Min processing time: {min(processing_times):.3f}s")
        print(f"   Max processing time: {max(processing_times):.3f}s")
        print(f"   Avg processing time: {sum(processing_times) / len(processing_times):.3f}s")

        print("\n✅ Performance test completed successfully!")
        print("   Check server logs for M0/M1/M2 parallel processing details")
        print("   Look for 'ParallelMemoryLayerManager' and 'MemoryLayerImpl' log entries")

        return {
            "success": True,
            "total_messages": message_count,
            "total_time": total_time,
            "responses": responses,
            "session_id": session_id
        }

    except Exception as e:
        print(f"\n❌ Performance test failed: {e}")
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print("❌ MemFuse server is not responding properly")
            print("   Please start the server with: poetry run memfuse-core")
            sys.exit(1)
    except requests.exceptions.RequestException:
        print("❌ Cannot connect to MemFuse server at localhost:8000")
        print("   Please start the server with: poetry run memfuse-core")
        sys.exit(1)

    # Run the performance test
    result = run_performance_test()

    if result["success"]:
        print("\n🎯 Test Summary:")
        print(f"   Session ID: {result.get('session_id', 'N/A')}")
        print(f"   Total Messages: {result.get('total_messages', 0)}")
        print(f"   Total Time: {result.get('total_time', 0):.2f}s")
        print(f"   Performance: {result.get('total_messages', 0) / result.get('total_time', 1):.2f} msg/s")
        sys.exit(0)
    else:
        print(f"\n💥 Test failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)
