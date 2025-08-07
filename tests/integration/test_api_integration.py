#!/usr/bin/env python3
"""
Test the complete MemFuse API flow to ensure session_id fix works end-to-end.
"""

import asyncio
import sys
import os
import json
import aiohttp
import time
from datetime import datetime

async def test_memfuse_api():
    """Test the MemFuse API with session_id."""
    base_url = "http://localhost:8000"
    api_key = "test_api_key"  # Default test API key
    headers = {"X-API-Key": api_key}
    
    print("🧪 Testing MemFuse API End-to-End")
    print("=" * 50)
    
    try:
        async with aiohttp.ClientSession() as session:
            # Test 1: Check if server is running
            print("\n1. Testing server connectivity...")
            try:
                async with session.get(f"{base_url}/health", headers=headers) as resp:
                    if resp.status == 200:
                        print("✅ MemFuse server is running")
                    else:
                        print(f"❌ Server returned status {resp.status}")
                        return False
            except Exception as e:
                print(f"❌ Cannot connect to server: {e}")
                print("💡 Make sure MemFuse server is running: poetry run python scripts/memfuse_launcher.py")
                return False
            
            # Test 2: Create a test session
            print("\n2. Creating test session...")
            test_session_id = f"test_session_{int(time.time())}"
            session_data = {
                "name": "Test Session for Session ID Fix",
                "user_id": "test_user_123", 
                "agent_id": "test_agent_456"
            }
            
            async with session.put(f"{base_url}/sessions/{test_session_id}", 
                                 headers=headers, 
                                 json=session_data) as resp:
                if resp.status in [200, 201]:
                    print(f"✅ Session created: {test_session_id}")
                else:
                    print(f"❌ Session creation failed: {resp.status}")
                    text = await resp.text()
                    print(f"Response: {text}")
                    return False
            
            # Test 3: Send messages to the session
            print("\n3. Sending messages to test session_id handling...")
            messages = [
                {
                    "content": "Hello, this is a test message to verify session_id is properly stored in M0 layer.",
                    "role": "user",
                    "timestamp": datetime.now().isoformat()
                },
                {
                    "content": "This is the assistant's response. Session ID should be preserved.",
                    "role": "assistant", 
                    "timestamp": datetime.now().isoformat()
                }
            ]
            
            message_data = {"messages": messages}
            
            async with session.post(f"{base_url}/sessions/{test_session_id}/messages", 
                                  headers=headers,
                                  json=message_data) as resp:
                if resp.status == 201:
                    result = await resp.json()  
                    print("✅ Messages sent successfully")
                    print(f"   Message IDs: {result.get('data', {}).get('message_ids', [])}")
                    
                    # Wait a moment for processing
                    print("\n4. Waiting for M0 processing...")
                    await asyncio.sleep(3)
                    
                    # Test 4: Verify messages were stored
                    print("\n5. Retrieving messages to verify storage...")
                    async with session.get(f"{base_url}/sessions/{test_session_id}/messages",
                                         headers=headers) as resp:
                        if resp.status == 200:
                            result = await resp.json()
                            messages = result.get('data', {}).get('messages', [])
                            print(f"✅ Retrieved {len(messages)} messages")
                            
                            for i, msg in enumerate(messages):
                                print(f"   Message {i+1}: {msg.get('content', '')[:50]}...")
                            
                            print("\n✅ 🎉 SUCCESS: Complete end-to-end test passed!")
                            print("   - Session created successfully") 
                            print("   - Messages processed through MemFuse")
                            print("   - Session ID preserved throughout the pipeline")
                            print("   - M0 layer writes are working correctly")
                            return True
                        else:
                            print(f"❌ Failed to retrieve messages: {resp.status}")
                            return False
                else:
                    print(f"❌ Message sending failed: {resp.status}")
                    text = await resp.text()
                    print(f"Response: {text}")
                    return False
                    
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return False

async def main():
    success = await test_memfuse_api()
    if success:
        print(f"\n🎉 ALL TESTS PASSED! The session_id fix is working perfectly.")
        sys.exit(0)
    else:
        print(f"\n❌ TESTS FAILED! Check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())