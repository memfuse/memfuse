#!/usr/bin/env python3
"""Unit tests using MemFuse SDK."""

import sys
import os
import asyncio

# Add SDK path
sdk_path = "/Users/mxue/GitRepos/MemFuse/memfuse-python/src"
if sdk_path not in sys.path:
    sys.path.insert(0, sdk_path)

from memfuse import MemFuse

async def test_sdk_basic_functionality():
    """Test basic SDK functionality."""
    print("🧪 SDK Unit Test: Basic Functionality")
    print("=" * 60)
    
    try:
        # Test 1: Initialize MemFuse client
        print("1. Initializing MemFuse client...")
        client = MemFuse()
        print("   ✅ MemFuse client created")
        
        # Test 2: Initialize memory
        print("2. Initializing memory...")
        memory = client.init(
            user="test_sdk_unit_user",
            agent="test_sdk_agent",
            session="test_sdk_session"
        )
        print(f"   ✅ Memory initialized")
        print(f"   User ID: {memory.user_id}")
        print(f"   Session ID: {memory.session_id}")
        
        # Test 3: Add messages
        print("3. Adding test messages...")
        test_messages = [
            {"role": "user", "content": "Hello, I love Taylor Swift's music!"},
            {"role": "assistant", "content": "That's great! Taylor Swift is an amazing artist."},
            {"role": "user", "content": "What's your favorite Taylor Swift song?"},
            {"role": "assistant", "content": "A little bit. I can get into taylor swift."}
        ]
        
        add_result = memory.add(test_messages)
        print(f"   Add result status: {add_result.get('status')}")
        
        if add_result.get('status') != 'success':
            print(f"   ❌ Failed to add messages: {add_result}")
            return False
        
        print("   ✅ Messages added successfully")
        
        # Test 4: Query memory
        print("4. Testing memory query...")
        query_result = memory.query("Taylor Swift music", top_k=5)
        print(f"   Query result status: {query_result.get('status')}")
        
        if query_result.get('status') == 'success':
            data = query_result.get('data', {})
            results = data.get('results', [])
            total = data.get('total', 0)
            
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
                    success = True
                else:
                    print("   ⚠️  No Taylor Swift content found in results")
                    print("   Available results:")
                    for i, result in enumerate(results[:3]):
                        print(f"     {i+1}. {result.get('content', '')[:50]}...")
                    success = False
            else:
                print("   ❌ No results returned")
                success = False
        else:
            print(f"   ❌ Query failed: {query_result}")
            success = False
        
        # Test 5: Test messages list API with buffer_only parameter
        print("5. Testing messages list API...")
        try:
            # Standard list
            list_result = await client.messages.list(
                session_id=memory.session_id,
                limit=10
            )
            print(f"   Standard list status: {list_result.get('status')}")
            
            if list_result.get('status') == 'success':
                messages = list_result.get('data', {}).get('messages', [])
                print(f"   ✅ Standard list: {len(messages)} messages")
            
            # List with buffer_only=True
            buffer_list_result = await client.messages.list(
                session_id=memory.session_id,
                limit=10,
                buffer_only=True
            )
            print(f"   Buffer-only list status: {buffer_list_result.get('status')}")
            
            if buffer_list_result.get('status') == 'success':
                buffer_messages = buffer_list_result.get('data', {}).get('messages', [])
                print(f"   ✅ Buffer-only list: {len(buffer_messages)} messages")
            
        except Exception as e:
            print(f"   ⚠️  Messages list API test failed: {e}")
        
        # Cleanup
        memory.close()
        client.close()
        
        print("\n" + "=" * 60)
        if success:
            print("✅ SDK unit test completed successfully!")
        else:
            print("⚠️  SDK unit test completed with warnings!")
        
        return success
        
    except Exception as e:
        print(f"\n❌ SDK unit test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sdk_sync():
    """Test SDK sync functionality."""
    print("\n🧪 SDK Unit Test: Sync Functionality")
    print("=" * 60)
    
    try:
        # Test sync version
        print("1. Testing sync SDK...")
        client = MemFuse()
        memory = client.init(
            user="test_sdk_sync_user",
            agent="test_sdk_agent",
            session="test_sdk_sync_session"
        )
        
        # Add messages
        test_messages = [
            {"role": "user", "content": "I enjoy listening to music"},
            {"role": "assistant", "content": "Music is wonderful! What genre do you like?"}
        ]
        
        add_result = memory.add(test_messages)
        print(f"   Add result: {add_result.get('status')}")
        
        # Test sync list
        try:
            sync_list_result = client.messages.list_sync(
                session_id=memory.session_id,
                limit=5,
                buffer_only=True
            )
            print(f"   Sync list status: {sync_list_result.get('status')}")
            
            if sync_list_result.get('status') == 'success':
                sync_messages = sync_list_result.get('data', {}).get('messages', [])
                print(f"   ✅ Sync list: {len(sync_messages)} messages")
            
        except Exception as e:
            print(f"   ⚠️  Sync list test failed: {e}")
        
        memory.close()
        client.close()
        
        print("✅ Sync SDK test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Sync SDK test failed: {e}")
        return False

def main():
    """Run SDK unit tests."""
    print("🧪 MemFuse SDK Unit Tests")
    print("=" * 80)
    
    # Test async functionality
    success1 = asyncio.run(test_sdk_basic_functionality())
    
    # Test sync functionality
    success2 = test_sdk_sync()
    
    overall_success = success1 and success2
    
    print("\n" + "=" * 80)
    if overall_success:
        print("🎉 All SDK unit tests passed!")
    else:
        print("⚠️  Some SDK unit tests had issues")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
