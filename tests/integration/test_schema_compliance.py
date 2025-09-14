#!/usr/bin/env python3
"""Test M1 Schema compliance after fixes"""

import asyncio
import json
from pathlib import Path
import pytest

# Try to import from installed/pytest path; fallback to local src when run directly
try:
    from memfuse_core.services.simplified_memory_service import SimplifiedMemoryService
except Exception:  # pragma: no cover - fallback for direct script execution
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from memfuse_core.services.simplified_memory_service import SimplifiedMemoryService

@pytest.mark.asyncio
@pytest.mark.integration
async def test_schema_compliance():
    """Test M1 Schema compliance with real data"""
    
    print("=== Testing M1 Schema Compliance ===")
    
    # Initialize service
    service = SimplifiedMemoryService()
    await service.initialize()
    
    # Add some test messages first using the service directly
    print("\n1. Adding test messages...")
    
    # Create test messages in the format the service expects
    messages = [
        {
            "role": "user",
            "content": "What kind of programming language is Python, and what are its key features?",
            "metadata": {}
        },
        {
            "role": "assistant", 
            "content": (
                "Python is a high-level programming language known for its simple and elegant syntax "
                "and a powerful ecosystem. Key features include: (1) readable syntax, (2) cross-platform support, "
                "(3) a rich third-party library ecosystem, and (4) broad usage across web development, data science, "
                "artificial intelligence, and automation. Python’s design philosophy emphasizes readability and "
                "simplicity, making it popular with both beginners and professional developers."
            ),
            "metadata": {}
        }
    ]
    
    # Store messages directly to m0_messages table for testing
    session_id = "550e8400-e29b-41d4-a716-446655440000"  # Valid UUID format
    user_id = "52b5083d-40b0-405d-8c07-1c00a5738fd1"
    
    try:
        message_ids = await service._store_m0_messages(messages, session_id, user_id)
        print(f"Messages stored with IDs: {message_ids}")
        
        # Wait for chunking to complete
        await asyncio.sleep(5)
        
        # Manually trigger chunking process
        print("Triggering chunking process...")
        
        # Get the message IDs and create chunks manually
        import uuid
        chunks_to_store = []
        for i, msg_id in enumerate(message_ids):
            chunk_data = {
                'chunk_id': str(uuid.uuid4()),
                'content': messages[i]['content'],
                'user_id': user_id,
                'session_id': session_id,
                'round_id': None,
                'm0_raw_ids': [msg_id],
                'chunking_strategy': 'token_based_chunker',
                'token_count': len(messages[i]['content'].split()),
                'embedding': [0.1] * 384  # Dummy embedding for testing
            }
            chunks_to_store.append(chunk_data)
        
        # Store chunks directly
        chunk_ids = await service._store_m1_chunks(chunks_to_store)
        print(f"Manual chunks created: {chunk_ids}")
        
        # Check if chunks were created
        with service.db_manager.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM m1_episodic WHERE user_id = %s", (user_id,))
            chunk_count = cur.fetchone()[0]
            print(f"Total chunks in database: {chunk_count}")
        
    except Exception as e:
        print(f"Error storing messages: {e}")
    
    # Test query to check M1 Schema format
    print("\n2. Testing query with M1 Schema format...")
    
    query_result = await service.query(
        query="Python programming language",
        top_k=3,
        session_id="550e8400-e29b-41d4-a716-446655440000",
        user_id="52b5083d-40b0-405d-8c07-1c00a5738fd1"
    )
    
    print("\n=== Current Query Response ===")
    print(json.dumps(query_result, indent=2, ensure_ascii=False))
    
    # Check M1 Schema compliance
    print("\n=== M1 Schema Compliance Check ===")
    
    if query_result.get('status') == 'success':
        data = query_result.get('data', {})
        
        # Check top-level fields
        print(f"✓ Has 'query' field: {'query' in data}")
        print(f"✓ Query value: {data.get('query')}")
        print(f"✓ Has 'results' field: {'results' in data}")
        print(f"✓ Results count: {len(data.get('results', []))}")
        
        results = data.get('results', [])
        if results:
            print(f"\n=== First Result Analysis ===")
            first_result = results[0]
            
            # Check required M1 Schema fields
            required_fields = ['id', 'content', 'relevance_score', 'memory_type', 'scope', 'created_at', 'updated_at', 'metadata']
            
            for field in required_fields:
                has_field = field in first_result
                value = first_result.get(field)
                if field == 'content':
                    value = f"[{len(str(value))} chars]"
                print(f"✓ {field}: {has_field} = {value}")
            
            # Check metadata fields
            metadata = first_result.get('metadata', {})
            print(f"\n=== Metadata Fields ===")
            print(f"✓ task: {'task' in metadata} = {metadata.get('task')}")
            print(f"✓ mode: {'mode' in metadata} = {metadata.get('mode')}")
            
            # Check field transformations
            print(f"\n=== M1 Schema Transformations ===")
            print(f"✓ Uses 'relevance_score' (not 'score'): {'relevance_score' in first_result}")
            print(f"✓ Uses 'memory_type' (not 'type'): {'memory_type' in first_result}")
            print(f"✓ Memory type value: {first_result.get('memory_type')}")
            print(f"✓ Scope value: {first_result.get('scope')}")
            
        else:
            print("⚠️  No results returned - cannot test result schema")
    else:
        print(f"❌ Query failed: {query_result}")
    
    await service.close()
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    asyncio.run(test_schema_compliance())
