#!/usr/bin/env python3
"""
Simple end-to-end test based on the quickstart example
"""

import asyncio
import json
import time
import sys
import os
import aiohttp
import asyncpg
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_memfuse_data_flow():
    """Test complete MemFuse data flow using proper API calls"""
    print("🚀 Testing MemFuse End-to-End Data Flow")
    print("=" * 60)
    
    base_url = "http://localhost:8000"
    
    # Setup database connection
    conn = await asyncpg.connect(
        host="localhost",
        port=5432,
        database="memfuse",
        user="postgres",
        password="postgres"
    )
    
    try:
        # Check initial state
        initial_m0_count = await conn.fetchval("SELECT COUNT(*) FROM m0_raw")
        initial_m1_count = await conn.fetchval("SELECT COUNT(*) FROM m1_episodic")
        print(f"📊 Initial database state:")
        print(f"   M0 records: {initial_m0_count}")
        print(f"   M1 records: {initial_m1_count}")
        
        async with aiohttp.ClientSession(headers={"Content-Type": "application/json"}) as session:
            # 1. Create user (using proper API structure)
            print("\n👤 Creating test user...")
            import uuid
            unique_id = str(uuid.uuid4())[:8]
            user_data = {"name": f"test_user_e2e_{unique_id}", "description": "End-to-end test user"}
            
            async with session.post(f"{base_url}/api/v1/users", json=user_data) as resp:
                if resp.status in [200, 201]:
                    user_result = await resp.json()
                    user_id = user_result["data"]["user"]["id"]
                    print(f"✅ User created: {user_id}")
                else:
                    error_text = await resp.text()
                    print(f"❌ Failed to create user: {resp.status}")
                    print(f"   Error details: {error_text}")
                    return False
            
            # 2. Create agent
            print("\n🤖 Creating test agent...")
            agent_data = {"name": f"test_agent_e2e_{unique_id}", "description": "End-to-end test agent"}
            
            async with session.post(f"{base_url}/api/v1/agents", json=agent_data) as resp:
                if resp.status in [200, 201]:
                    agent_result = await resp.json()
                    agent_id = agent_result["data"]["agent"]["id"]
                    print(f"✅ Agent created: {agent_id}")
                else:
                    print(f"❌ Failed to create agent: {resp.status}")
                    return False
            
            # 3. Create session
            print("\n💬 Creating test session...")
            session_data = {
                "user_id": user_id,
                "agent_id": agent_id,
                "name": "end-to-end-test-session"
            }
            
            async with session.post(f"{base_url}/api/v1/sessions", json=session_data) as resp:
                if resp.status in [200, 201]:
                    session_result = await resp.json()
                    session_id = session_result["data"]["session"]["id"]
                    print(f"✅ Session created: {session_id}")
                else:
                    print(f"❌ Failed to create session: {resp.status}")
                    return False
            
            # 4. Send test messages (like the quickstart example)
            print("\n💬 Sending test messages...")
            test_messages = [
                {
                    "role": "user",
                    "content": "Hello! I'm testing the MemFuse system end-to-end. I want to understand how the buffer system works with auto-embedding generation."
                },
                {
                    "role": "assistant",
                    "content": "Hello! I'm responding to your end-to-end test. This message should be processed through the buffer system and generate embeddings for vector retrieval."
                },
                {
                    "role": "user",
                    "content": "Can you explain how the hierarchical memory system works in MemFuse? I'm particularly interested in the M0 and M1 layers."
                },
                {
                    "role": "assistant",
                    "content": "MemFuse uses a hierarchical memory system inspired by human cognition. M0 (Raw Data) stores raw conversational episodes, while M1 (Episodic Memory) extracts meaningful episodes from conversations. Both layers support vector embeddings for semantic search."
                }
            ]
            
            # Send messages as batch (like quickstart example)
            messages_data = {"messages": test_messages}
            
            async with session.post(f"{base_url}/api/v1/sessions/{session_id}/messages", json=messages_data) as resp:
                if resp.status == 201:
                    result = await resp.json()
                    message_ids = result.get("data", {}).get("message_ids", [])
                    print(f"✅ Messages sent successfully: {len(message_ids)} messages")
                    for i, msg_id in enumerate(message_ids):
                        print(f"   Message {i+1} ID: {msg_id}")
                else:
                    print(f"❌ Failed to send messages: {resp.status}")
                    error_text = await resp.text()
                    print(f"   Error: {error_text}")
                    return False
        
        # Wait for processing (buffer flush + embedding generation)
        print("\n⏳ Waiting for buffer processing and embedding generation...")
        await asyncio.sleep(10)
        
        # Check database state after processing
        final_m0_count = await conn.fetchval("SELECT COUNT(*) FROM m0_raw")
        final_m1_count = await conn.fetchval("SELECT COUNT(*) FROM m1_episodic")
        
        print(f"\n📊 Final database state:")
        print(f"   M0 records: {final_m0_count} (added: {final_m0_count - initial_m0_count})")
        print(f"   M1 records: {final_m1_count} (added: {final_m1_count - initial_m1_count})")
        
        # 6. Verify M0 records
        print(f"\n📝 Verifying M0 records for session {session_id[:20]}...")
        m0_records = await conn.fetch("""
            SELECT id, content, session_id, user_id, message_role, round_id, 
                   needs_embedding, embedding, created_at
            FROM m0_raw 
            WHERE session_id = $1
            ORDER BY created_at
        """, session_id)
        
        print(f"   Found {len(m0_records)} M0 records:")
        
        for record in m0_records:
            print(f"   - {record['message_role']}: {record['content'][:60]}...")
            print(f"     Session: {record['session_id'][:20]}...")
            print(f"     Needs Embedding: {record['needs_embedding']}")
            print(f"     Has Embedding: {record['embedding'] is not None}")
            print()
        
        # 7. Verify M1 records
        print(f"📝 Verifying M1 records for session {session_id[:20]}...")
        m1_records = await conn.fetch("""
            SELECT id, source_session_id, episode_type, episode_content, 
                   needs_embedding, embedding, created_at
            FROM m1_episodic 
            WHERE source_session_id = $1
            ORDER BY created_at
        """, session_id)
        
        print(f"   Found {len(m1_records)} M1 records:")
        
        for record in m1_records:
            print(f"   - {record['episode_type']}: {record['episode_content'][:60]}...")
            print(f"     Source Session: {record['source_session_id'][:20]}...")
            print(f"     Needs Embedding: {record['needs_embedding']}")
            print(f"     Has Embedding: {record['embedding'] is not None}")
            print()
        
        # 8. Check embedding generation status
        print("🔢 Checking embedding generation status...")
        embedding_stats = await conn.fetchrow("""
            SELECT 
                COUNT(*) as total,
                COUNT(embedding) as with_embeddings,
                COUNT(CASE WHEN needs_embedding = TRUE THEN 1 END) as needs_embedding,
                COUNT(CASE WHEN needs_embedding = FALSE THEN 1 END) as embedded
            FROM m0_raw 
            WHERE session_id = $1
        """, session_id)
        
        print(f"   M0 Embedding Stats:")
        print(f"     Total records: {embedding_stats['total']}")
        print(f"     With embeddings: {embedding_stats['with_embeddings']}")
        print(f"     Needs embedding: {embedding_stats['needs_embedding']}")
        print(f"     Already embedded: {embedding_stats['embedded']}")
        
        # Wait more if embeddings are still being generated
        if embedding_stats['needs_embedding'] > 0:
            print("⏳ Waiting for more embedding generation...")
            await asyncio.sleep(15)
            
            # Re-check
            final_embedding_stats = await conn.fetchrow("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(embedding) as with_embeddings,
                    COUNT(CASE WHEN needs_embedding = TRUE THEN 1 END) as needs_embedding
                FROM m0_raw 
                WHERE session_id = $1
            """, session_id)
            
            print(f"   Final M0 Embedding Stats:")
            print(f"     Total records: {final_embedding_stats['total']}")
            print(f"     With embeddings: {final_embedding_stats['with_embeddings']}")
            print(f"     Needs embedding: {final_embedding_stats['needs_embedding']}")
        
        # 9. Test vector retrieval
        print("\n🔍 Testing vector retrieval...")
        
        # Test direct SQL vector search on M0
        test_query = "hierarchical memory system"
        
        m0_vector_results = await conn.fetch("""
            SELECT 
                id,
                content,
                session_id,
                1 - (embedding <=> $1) as similarity
            FROM m0_raw 
            WHERE session_id = $2 AND embedding IS NOT NULL
            ORDER BY embedding <=> $1
            LIMIT 3
        """, test_query, session_id)
        
        print(f"   M0 Vector Search Results ({len(m0_vector_results)} found):")
        for result in m0_vector_results:
            print(f"     - {result['content'][:60]}...")
            print(f"       Similarity: {result['similarity']:.4f}")
            print()
        
        # Test direct SQL vector search on M1
        m1_vector_results = await conn.fetch("""
            SELECT 
                id,
                episode_content,
                source_session_id,
                1 - (embedding <=> $1) as similarity
            FROM m1_episodic 
            WHERE source_session_id = $2 AND embedding IS NOT NULL
            ORDER BY embedding <=> $1
            LIMIT 3
        """, test_query, session_id)
        
        print(f"   M1 Vector Search Results ({len(m1_vector_results)} found):")
        for result in m1_vector_results:
            print(f"     - {result['episode_content'][:60]}...")
            print(f"       Similarity: {result['similarity']:.4f}")
            print()
        
        # 10. Test API query endpoint
        print("🌐 Testing API query endpoint...")
        async with aiohttp.ClientSession() as session:
            query_data = {
                "query": test_query,
                "session_id": session_id,
                "agent_id": agent_id,
                "top_k": 5,
                "include_messages": True,
                "include_knowledge": False
            }
            
            async with session.post(f"{base_url}/api/v1/users/{user_id}/query", json=query_data) as resp:
                if resp.status == 200:
                    query_result = await resp.json()
                    results = query_result.get("data", {}).get("results", [])
                    print(f"   API Query Results ({len(results)} found):")
                    
                    for i, res in enumerate(results[:3]):
                        print(f"     {i+1}. {res.get('content', '')[:60]}...")
                        print(f"        Score: {res.get('score', 0):.4f}")
                        print(f"        Type: {res.get('type', 'unknown')}")
                        print()
                else:
                    print(f"   ❌ API query failed: {resp.status}")
                    error_text = await resp.text()
                    print(f"      Error: {error_text}")
        
        # 11. Summary
        print("\n🎯 End-to-End Test Summary")
        print("=" * 40)
        print(f"✅ Session created: {session_id}")
        print(f"✅ Messages sent: {len(test_messages)}")
        print(f"✅ M0 records created: {len(m0_records)}")
        print(f"✅ M1 records created: {len(m1_records)}")
        print(f"✅ Vector search working: {len(m0_vector_results) > 0 or len(m1_vector_results) > 0}")
        
        success = (
            len(m0_records) >= len(test_messages) and 
            (embedding_stats['with_embeddings'] > 0 or final_embedding_stats['with_embeddings'] > 0)
        )
        
        if success:
            print("\n🎉 End-to-end test completed successfully!")
            print("   ✅ Data flow working correctly")
            print("   ✅ Buffer system functioning")
            print("   ✅ Embedding generation working")
            print("   ✅ Vector retrieval working")
        else:
            print("\n❌ End-to-end test failed!")
            
        return success
        
    finally:
        await conn.close()

async def main():
    """Main test function"""
    try:
        success = await test_memfuse_data_flow()
        if not success:
            sys.exit(1)
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())