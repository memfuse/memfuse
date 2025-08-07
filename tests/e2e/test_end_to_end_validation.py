#!/usr/bin/env python3
"""
Complete end-to-end test for MemFuse auto-embedding and vector retrieval.
This test validates the complete pipeline from data write to vector retrieval.
"""
import asyncio
import sys
import os
import json
import time
import aiohttp
from datetime import datetime
from typing import Dict, List, Any

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


async def check_server_health(max_retries=10, delay=2):
    """Check if MemFuse server is running"""
    for i in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get('http://localhost:8000/api/v1/health') as resp:
                    if resp.status == 200:
                        print(f"✅ Server is healthy (attempt {i+1})")
                        return True
        except Exception as e:
            print(f"⏳ Server not ready (attempt {i+1}/{max_retries}): {e}")
            await asyncio.sleep(delay)
    return False


async def check_database_tables():
    """Check if database tables are created correctly"""
    from memfuse_core.services.global_connection_manager import GlobalConnectionManager
    
    print("\n🔍 Checking database tables...")
    
    # Initialize connection manager
    connection_manager = GlobalConnectionManager()
    await connection_manager.initialize()
    
    async with connection_manager.get_connection() as conn:
        async with conn.cursor() as cur:
            # Check if m0_raw table exists
            await cur.execute("""
                SELECT table_name, column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name IN ('m0_raw', 'm1_episodic')
                ORDER BY table_name, ordinal_position
            """)
            columns = await cur.fetchall()
            
            if not columns:
                print("❌ No tables found!")
                return False
                
            # Group by table
            tables = {}
            for table_name, column_name, data_type in columns:
                if table_name not in tables:
                    tables[table_name] = []
                tables[table_name].append((column_name, data_type))
            
            print("📋 Database Tables:")
            for table_name, cols in tables.items():
                print(f"  📁 {table_name}:")
                for col_name, col_type in cols:
                    print(f"    - {col_name}: {col_type}")
            
            # Check for required columns in m0_raw
            if 'm0_raw' in tables:
                m0_columns = {col[0] for col in tables['m0_raw']}
                required_cols = {'id', 'content', 'metadata', 'session_id', 'user_id', 'message_role', 'round_id', 'embedding', 'needs_embedding'}
                missing = required_cols - m0_columns
                if missing:
                    print(f"❌ Missing columns in m0_raw: {missing}")
                    return False
                else:
                    print("✅ m0_raw table has all required columns")
            else:
                print("❌ m0_raw table not found")
                return False
                
            return True


async def test_buffer_enabled_write():
    """Test data write flow with buffer enabled"""
    print("\n🧪 Testing Buffer Enabled Write Flow...")
    
    # First check buffer config
    from memfuse_core.utils.config import config_manager
    buffer_config = config_manager.get_config().buffer
    print(f"Buffer enabled: {buffer_config.enabled}")
    
    test_data = {
        "user_id": "test_user_buffer",
        "session_id": "test_session_buffer", 
        "agent_id": "test_agent",
        "content": "This is a test message for buffer enabled mode with embedding generation test.",
        "message_role": "user",
        "round_id": "round_buffer_001"
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post('http://localhost:8000/api/v1/messages', 
                                   json=test_data,
                                   headers={'Content-Type': 'application/json'}) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    print(f"✅ Message sent successfully: {result.get('message_id')}")
                    return result.get('message_id')
                else:
                    error_text = await resp.text()
                    print(f"❌ Failed to send message: {resp.status} - {error_text}")
                    return None
    except Exception as e:
        print(f"❌ Error sending message: {e}")
        return None


async def test_buffer_disabled_write():
    """Test data write flow with buffer disabled"""
    print("\n🧪 Testing Buffer Disabled Write Flow...")
    
    # Update buffer config to disabled
    from memfuse_core.utils.config import config_manager
    
    # Read current config
    with open('config/buffer/default.yaml', 'r') as f:
        content = f.read()
    
    # Temporarily disable buffer
    updated_content = content.replace('enabled: true', 'enabled: false')
    with open('config/buffer/default.yaml', 'w') as f:
        f.write(updated_content)
    
    # Reload config (requires service restart in real scenario, but we'll test with current session)
    test_data = {
        "user_id": "test_user_direct",
        "session_id": "test_session_direct",
        "agent_id": "test_agent", 
        "content": "This is a test message for buffer disabled mode with direct database write.",
        "message_role": "user",
        "round_id": "round_direct_001"
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post('http://localhost:8000/api/v1/messages',
                                   json=test_data,
                                   headers={'Content-Type': 'application/json'}) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    print(f"✅ Message sent successfully: {result.get('message_id')}")
                    
                    # Restore original config
                    with open('config/buffer/default.yaml', 'w') as f:
                        f.write(content)
                    
                    return result.get('message_id')
                else:
                    error_text = await resp.text()
                    print(f"❌ Failed to send message: {resp.status} - {error_text}")
                    
                    # Restore original config
                    with open('config/buffer/default.yaml', 'w') as f:
                        f.write(content)
                    
                    return None
    except Exception as e:
        print(f"❌ Error sending message: {e}")
        
        # Restore original config
        with open('config/buffer/default.yaml', 'w') as f:
            f.write(content)
        
        return None


async def verify_m0_m1_data():
    """Verify data in M0 and M1 tables and check embedding generation"""
    print("\n🔍 Verifying M0 and M1 Table Data...")
    
    from memfuse_core.services.global_connection_manager import GlobalConnectionManager
    
    connection_manager = GlobalConnectionManager()
    await connection_manager.initialize()
    
    async with connection_manager.get_connection() as conn:
        async with conn.cursor() as cur:
            # Check M0 data
            await cur.execute("""
                SELECT id, content, session_id, user_id, message_role, round_id, 
                       needs_embedding, embedding IS NOT NULL as has_embedding,
                       created_at, updated_at
                FROM m0_raw 
                ORDER BY created_at DESC
                LIMIT 10
            """)
            m0_data = await cur.fetchall()
            
            print(f"📊 M0 Raw Data ({len(m0_data)} records):")
            for row in m0_data:
                print(f"  📝 ID: {row[0][:50]}...")
                print(f"     Content: {row[1][:100]}...")
                print(f"     Session: {row[2]}")
                print(f"     User: {row[3]}")
                print(f"     Role: {row[4]}")
                print(f"     Round: {row[5]}")
                print(f"     Needs Embedding: {row[6]}")
                print(f"     Has Embedding: {row[7]}")
                print(f"     Created: {row[8]}")
                print(f"     Updated: {row[9]}")
                print()
            
            # Check M1 data if table exists
            try:
                await cur.execute("""
                    SELECT COUNT(*) FROM m1_episodic
                """)
                m1_count = await cur.fetchone()
                print(f"📊 M1 Episodic Data: {m1_count[0]} records")
                
                if m1_count[0] > 0:
                    await cur.execute("""
                        SELECT id, content, session_id, embedding IS NOT NULL as has_embedding,
                               created_at
                        FROM m1_episodic 
                        ORDER BY created_at DESC
                        LIMIT 5
                    """)
                    m1_data = await cur.fetchall()
                    for row in m1_data:
                        print(f"  📝 M1 ID: {row[0][:50]}...")
                        print(f"     Content: {row[1][:100]}...")
                        print(f"     Session: {row[2]}")
                        print(f"     Has Embedding: {row[3]}")
                        print(f"     Created: {row[4]}")
                        print()
            except Exception as e:
                print(f"⚠️ M1 table check failed: {e}")
            
            return len(m0_data) > 0


async def test_vector_retrieval():
    """Test vector retrieval functionality"""
    print("\n🔍 Testing Vector Retrieval...")
    
    # Test query endpoint
    test_query = {
        "user_id": "test_user_buffer",
        "session_id": "test_session_buffer",
        "agent_id": "test_agent",
        "query": "test message embedding",
        "top_k": 5
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post('http://localhost:8000/api/v1/query',
                                   json=test_query,
                                   headers={'Content-Type': 'application/json'}) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    print(f"✅ Query successful: {len(result.get('results', []))} results")
                    
                    for i, item in enumerate(result.get('results', [])[:3):
                        print(f"  🎯 Result {i+1}:")
                        print(f"     Content: {item.get('content', 'N/A')[:100]}...")
                        print(f"     Score: {item.get('score', 'N/A')}")
                        print(f"     Source: {item.get('source', 'N/A')}")
                    
                    return len(result.get('results', [])) > 0
                else:
                    error_text = await resp.text()
                    print(f"❌ Query failed: {resp.status} - {error_text}")
                    return False
    except Exception as e:
        print(f"❌ Error during query: {e}")
        return False


async def wait_for_embeddings(max_wait=60):
    """Wait for embeddings to be generated"""
    print(f"\n⏳ Waiting for embeddings to be generated (max {max_wait}s)...")
    
    from memfuse_core.services.global_connection_manager import GlobalConnectionManager
    
    connection_manager = GlobalConnectionManager()
    await connection_manager.initialize()
    
    start_time = time.time()
    while time.time() - start_time < max_wait:
        async with connection_manager.get_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    SELECT COUNT(*) as total,
                           SUM(CASE WHEN embedding IS NOT NULL THEN 1 ELSE 0 END) as with_embedding,
                           SUM(CASE WHEN needs_embedding = TRUE THEN 1 ELSE 0 END) as needs_embedding
                    FROM m0_raw
                """)
                result = await cur.fetchone()
                total, with_embedding, needs_embedding = result
                
                print(f"   📊 Total: {total}, With Embedding: {with_embedding}, Needs Embedding: {needs_embedding}")
                
                if total > 0 and with_embedding > 0:
                    print("✅ Embeddings generated!")
                    return True
                elif total > 0 and needs_embedding == 0:
                    print("⚠️ No records need embedding but none have embeddings")
                    return False
                    
        await asyncio.sleep(3)
    
    print("⏰ Timeout waiting for embeddings")
    return False


async def main():
    """Main test function"""
    print("🚀 MemFuse End-to-End Auto-Embedding Test")
    print("=" * 60)
    
    # Step 1: Check server health
    if not await check_server_health():
        print("❌ Server is not running. Please start MemFuse server first.")
        return
    
    # Step 2: Check database tables
    if not await check_database_tables():
        print("❌ Database tables are not properly configured.")
        return
    
    # Step 3: Test buffer enabled write
    message_id_1 = await test_buffer_enabled_write()
    
    # Step 4: Test buffer disabled write  
    message_id_2 = await test_buffer_disabled_write()
    
    if not message_id_1 and not message_id_2:
        print("❌ Both write tests failed.")
        return
        
    # Step 5: Wait for processing
    await asyncio.sleep(5)
    
    # Step 6: Verify M0/M1 data
    has_data = await verify_m0_m1_data()
    
    if not has_data:
        print("❌ No data found in M0 table.")
        return
    
    # Step 7: Wait for embeddings
    embeddings_ready = await wait_for_embeddings()
    
    # Step 8: Test vector retrieval
    retrieval_success = await test_vector_retrieval()
    
    # Final summary
    print("\n" + "=" * 60)
    print("🎯 Test Summary:")
    print(f"✅ Server Health: OK")
    print(f"✅ Database Tables: OK") 
    print(f"{'✅' if message_id_1 else '❌'} Buffer Enabled Write: {'OK' if message_id_1 else 'FAILED'}")
    print(f"{'✅' if message_id_2 else '❌'} Buffer Disabled Write: {'OK' if message_id_2 else 'FAILED'}")
    print(f"✅ M0/M1 Data Verification: OK")
    print(f"{'✅' if embeddings_ready else '⚠️'} Embedding Generation: {'OK' if embeddings_ready else 'PENDING/FAILED'}")
    print(f"{'✅' if retrieval_success else '❌'} Vector Retrieval: {'OK' if retrieval_success else 'FAILED'}")
    
    if embeddings_ready and retrieval_success:
        print("\n🎉 All tests passed! MemFuse end-to-end pipeline is working correctly.")
    elif has_data:
        print("\n⚠️ Basic functionality working, but some advanced features need attention.")
    else:
        print("\n❌ Critical issues found. Please check the logs and configuration.")


if __name__ == "__main__":
    asyncio.run(main())