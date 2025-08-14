#!/usr/bin/env python3
"""
Test M1 chunk creation and storage in SimplifiedMemoryService.
"""

import asyncio
import sys
import os
import uuid

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from memfuse_core.services.simplified_memory_service import SimplifiedMemoryService
from memfuse_core.utils.config import config_manager

async def test_m1_chunk_creation():
    """Test M1 chunk creation and storage."""
    print("🧪 Testing M1 chunk creation and storage...")
    
    # Initialize config
    config_dict = config_manager.get_config()
    
    # Create SimplifiedMemoryService
    service = SimplifiedMemoryService(cfg=config_dict, user="test_m1_user")
    await service.initialize()
    
    print("✅ SimplifiedMemoryService initialized")
    
    # Create test messages that should create multiple M1 chunks
    test_messages = []
    for i in range(5):
        # Create messages with enough content to exceed token limit
        long_content = f"This is a very long message number {i}. " * 20  # ~100 tokens each
        test_messages.append({
            "id": str(uuid.uuid4()),
            "content": long_content,
            "metadata": {
                "conversation_id": str(uuid.uuid4()),
                "timestamp": "2025-08-14T19:00:00Z"
            }
        })
    
    print(f"📤 Testing add_batch with {len(test_messages)} messages")
    print(f"   Each message has ~{len(test_messages[0]['content']) // 4} tokens")
    print(f"   Total estimated tokens: ~{sum(len(msg['content']) // 4 for msg in test_messages)}")
    
    # Test add_batch
    result = await service.add_batch([test_messages])  # Wrap in list for MessageBatchList
    
    print(f"📥 Response received:")
    print(f"  Status: {result.get('status')}")
    print(f"  Message: {result.get('message')}")
    print(f"  Data: {result.get('data')}")
    
    if result.get("status") == "success":
        data = result.get("data", {})
        message_ids = data.get("message_ids", [])
        chunk_count = data.get("chunk_count", 0)
        
        print(f"  ✅ Successfully stored {len(message_ids)} M0 messages")
        print(f"  ✅ Created {chunk_count} M1 chunks")
        
        # Verify in database
        print("\n🔍 Verifying in database...")
        
        # Check M0 count
        with service.db_manager.conn.cursor() as cur:
            conversation_ids = [msg['metadata']['conversation_id'] for msg in test_messages]
            cur.execute("SELECT COUNT(*) FROM m0_raw WHERE conversation_id::text = ANY(%s)",
                       (conversation_ids,))
            m0_count = cur.fetchone()[0]
            print(f"  M0 messages in DB: {m0_count}")

            # Check M1 count
            cur.execute("SELECT COUNT(*) FROM m1_episodic WHERE conversation_id::text = ANY(%s)",
                       (conversation_ids,))
            m1_count = cur.fetchone()[0]
            print(f"  M1 chunks in DB: {m1_count}")
            
            if m1_count > 0:
                # Show M1 chunk details
                cur.execute("""
                    SELECT chunk_id, LEFT(content, 100) as content_preview, token_count,
                           embedding IS NOT NULL as has_embedding
                    FROM m1_episodic
                    WHERE conversation_id::text = ANY(%s)
                    LIMIT 3
                """, (conversation_ids,))
                
                chunks = cur.fetchall()
                print(f"  M1 chunk samples:")
                for chunk in chunks:
                    print(f"    - ID: {chunk[0]}")
                    print(f"      Content: {chunk[1]}...")
                    print(f"      Tokens: {chunk[2]}")
                    print(f"      Has embedding: {chunk[3]}")
    else:
        print(f"  ❌ Error: {result.get('message')}")
    
    print("✅ M1 chunk creation test completed")

if __name__ == "__main__":
    asyncio.run(test_m1_chunk_creation())
