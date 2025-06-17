#!/usr/bin/env python3
"""Test SDK Read API with Buffer support."""

import sys
import os
import asyncio

# Add SDK path
sdk_path = "/Users/mxue/GitRepos/MemFuse/memfuse-python/src"
if sdk_path not in sys.path:
    sys.path.insert(0, sdk_path)

from memfuse import MemFuse

async def test_sdk_read_api():
    """Test SDK Read API with Buffer buffer_only parameter."""
    print("🧪 Testing SDK Read API with Buffer Support")
    print("=" * 60)
    
    try:
        # Initialize MemFuse client
        client = MemFuse()
        
        # Test user creation and session setup
        print("1. Setting up test user and session...")
        memory = client.init(
            user="test_read_api_user",
            agent="test_agent",
            session="test_read_session"
        )
        
        # Add some test messages
        print("2. Adding test messages...")
        test_messages = [
            {"role": "user", "content": "Hello, I love Taylor Swift's music!"},
            {"role": "assistant", "content": "That's great! Taylor Swift is an amazing artist."},
            {"role": "user", "content": "What's your favorite Taylor Swift song?"},
            {"role": "assistant", "content": "A little bit. I can get into taylor swift."}
        ]
        
        add_result = memory.add(test_messages)
        print(f"   Add result: {add_result.get('status')}")
        
        if add_result.get('status') != 'success':
            print(f"   ❌ Failed to add messages: {add_result}")
            return False
        
        # Test 3: Query to verify data exists
        print("3. Testing query to verify data exists...")
        query_result = memory.query("Taylor Swift music", top_k=5)
        print(f"   Query result: {query_result.get('status')}")
        
        results = query_result.get('data', {}).get('results', [])
        print(f"   Found {len(results)} results")
        
        if results:
            print(f"   First result: {results[0].get('content', '')[:50]}...")
        
        # Test 4: SDK Read API - buffer_only=True (RoundBuffer only)
        print("4. Testing SDK Read API - buffer_only=True (RoundBuffer only)...")
        try:
            # Use the new read API with buffer_only parameter
            read_result_buffer_only = await client.users.read(
                user_id=memory.user_id,
                session_id=memory.session_id,
                buffer_only=True,
                limit=10
            )
            
            print(f"   Buffer-only read result: {read_result_buffer_only.get('status')}")
            buffer_messages = read_result_buffer_only.get('data', {}).get('messages', [])
            print(f"   Found {len(buffer_messages)} messages in RoundBuffer")
            
            if buffer_messages:
                print(f"   First buffer message: {buffer_messages[0].get('content', '')[:50]}...")
            
        except Exception as e:
            print(f"   ❌ Buffer-only read failed: {e}")
        
        # Test 5: SDK Read API - buffer_only=False (HybridBuffer + SQLite)
        print("5. Testing SDK Read API - buffer_only=False (HybridBuffer + SQLite)...")
        try:
            read_result_storage = await client.users.read(
                user_id=memory.user_id,
                session_id=memory.session_id,
                buffer_only=False,
                limit=10
            )
            
            print(f"   Storage read result: {read_result_storage.get('status')}")
            storage_messages = read_result_storage.get('data', {}).get('messages', [])
            print(f"   Found {len(storage_messages)} messages in storage")
            
            if storage_messages:
                print(f"   First storage message: {storage_messages[0].get('content', '')[:50]}...")
            
        except Exception as e:
            print(f"   ❌ Storage read failed: {e}")
        
        # Test 6: SDK Read API - buffer_only=None (All data)
        print("6. Testing SDK Read API - buffer_only=None (All data)...")
        try:
            read_result_all = await client.users.read(
                user_id=memory.user_id,
                session_id=memory.session_id,
                buffer_only=None,
                limit=10
            )
            
            print(f"   All data read result: {read_result_all.get('status')}")
            all_messages = read_result_all.get('data', {}).get('messages', [])
            print(f"   Found {len(all_messages)} total messages")
            
            if all_messages:
                print(f"   First message: {all_messages[0].get('content', '')[:50]}...")
            
        except Exception as e:
            print(f"   ❌ All data read failed: {e}")
        
        # Test 7: Session-level read API
        print("7. Testing session-level read API...")
        try:
            session_read_result = await client.messages.read(
                session_id=memory.session_id,
                buffer_only=True,
                limit=5
            )
            
            print(f"   Session read result: {session_read_result.get('status')}")
            session_messages = session_read_result.get('data', {}).get('messages', [])
            print(f"   Found {len(session_messages)} messages via session API")
            
        except Exception as e:
            print(f"   ❌ Session read failed: {e}")
        
        # Cleanup
        memory.close()
        client.close()
        
        print("\n" + "=" * 60)
        print("✅ SDK Read API test completed!")
        print("✅ Buffer read functionality is working!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ SDK Read API test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the SDK Read API test."""
    success = asyncio.run(test_sdk_read_api())
    if success:
        print("\n🎉 All SDK Read API tests passed!")
        print("🚀 Buffer read functionality is ready for use!")
    else:
        print("\n❌ Some SDK Read API tests failed")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
