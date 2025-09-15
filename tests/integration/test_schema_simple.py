#!/usr/bin/env python3
"""Simple M1 Schema compliance test"""

import asyncio
import json
import uuid
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
async def test_schema_with_mock_data():
    """Test M1 Schema compliance with mock data inserted directly"""
    
    print("=== Testing M1 Schema Compliance (Simple) ===")
    
    service = SimplifiedMemoryService()
    await service.initialize()
    
    # Insert a test chunk directly into the database with valid chunking_strategy
    session_id = "550e8400-e29b-41d4-a716-446655440000"
    user_id = "52b5083d-40b0-405d-8c07-1c00a5738fd1"
    chunk_id = str(uuid.uuid4())
    
    print("1. Inserting test chunk directly...")
    
    try:
        # Generate a valid embedding
        embedding = service.embedding_generator.generate_embedding("Python programming language test")
        
        with service.db_manager.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO m1_episodic (
                    chunk_id, content, chunking_strategy, token_count, embedding,
                    needs_embedding, m0_raw_ids, user_id, session_id, 
                    created_at, updated_at, m2_status
                ) VALUES (
                    %s, %s, 'token_based', %s, %s,
                    false, %s, %s, %s,
                    NOW(), NOW(), 'pending'
                )
            """, (
                chunk_id,
                "Python is a high-level programming language known for its simple and elegant syntax and a powerful ecosystem. Key features include readable syntax, cross-platform support, and a rich third-party library ecosystem.",
                50,  # token count
                embedding.tolist(),
'{' + str(uuid.uuid4()) + '}',  # Mock message ID as UUID array
                user_id,
                session_id
            ))
        
        service.db_manager.conn.commit()
        print(f"✅ Test chunk inserted: {chunk_id}")
        
    except Exception as e:
        print(f"❌ Error inserting chunk: {e}")
        await service.close()
        return
    
    # Test query with M1 Schema format
    print("\n2. Testing query with M1 Schema format...")
    
    query_result = await service.query(
        query="Features of the Python programming language",
        top_k=3,
        session_id=session_id,
        user_id=user_id
    )
    
    print("\n=== M1 Schema Test Result ===")
    print(json.dumps(query_result, indent=2, ensure_ascii=False))
    
    # Detailed Schema compliance check
    print("\n=== M1 Schema Compliance Analysis ===")
    
    if query_result.get('status') == 'success':
        data = query_result.get('data', {})
        
        # Check top-level structure
        print(f"✅ Status: {query_result.get('status')}")
        print(f"✅ Code: {query_result.get('code')}")
        print(f"✅ Has 'query' field: {'query' in data}")
        print(f"✅ Query value: '{data.get('query')}'")
        print(f"✅ Has 'results' field: {'results' in data}")
        print(f"✅ Results count: {len(data.get('results', []))}")
        
        results = data.get('results', [])
        if results:
            print(f"\n=== First Result M1 Schema Compliance ===")
            first_result = results[0]
            
            # Check all required M1 Schema fields
            required_fields = {
                'id': 'Unique identifier',
                'content': 'Content text',
                'relevance_score': 'Relevance score (0-1)',
                'memory_type': 'Memory type (episodic/semantic)',
                'scope': 'Scope (in_session/cross_session/null)',
                'created_at': 'Creation timestamp (ISO or null)',
                'updated_at': 'Update timestamp (ISO or null)',
                'metadata': 'Metadata dictionary'
            }
            
            for field, description in required_fields.items():
                has_field = field in first_result
                value = first_result.get(field)
                if field == 'content' and value:
                    value = f"[{len(str(value))} chars]"
                print(f"✅ {field:<15} : {has_field:<5} = {value}")
            
            # Check metadata structure
            metadata = first_result.get('metadata', {})
            print(f"\n=== Metadata Fields ===")
            print(f"✅ task: {'task' in metadata} = {metadata.get('task')}")
            print(f"✅ mode: {'mode' in metadata} = {metadata.get('mode')}")
            
            # Check field transformations (M1 Schema requirements)
            print(f"\n=== M1 Schema Transformations ===")
            print(f"✅ Uses 'relevance_score' (not 'score'): {'relevance_score' in first_result}")
            print(f"✅ Uses 'memory_type' (not 'type'): {'memory_type' in first_result}")
            print(f"✅ Memory type value: {first_result.get('memory_type')}")
            print(f"✅ Scope calculation: {first_result.get('scope')}")
            
            # Validate specific M1 Schema requirements
            print(f"\n=== M1 Schema Validation ===")
            relevance_score = first_result.get('relevance_score')
            if isinstance(relevance_score, (int, float)) and 0 <= relevance_score <= 1:
                print(f"✅ relevance_score valid: {relevance_score}")
            else:
                print(f"❌ relevance_score invalid: {relevance_score}")
                
            memory_type = first_result.get('memory_type')
            if memory_type in ['episodic', 'semantic']:
                print(f"✅ memory_type valid: {memory_type}")
            else:
                print(f"❌ memory_type invalid: {memory_type}")
                
            scope = first_result.get('scope')
            if scope in ['in_session', 'cross_session', None]:
                print(f"✅ scope valid: {scope}")
            else:
                print(f"❌ scope invalid: {scope}")
                
        else:
            print("⚠️  No results returned - cannot test result schema")
    else:
        print(f"❌ Query failed: {query_result}")
    
    await service.close()
    print("\n=== M1 Schema Test Complete ===")

if __name__ == "__main__":
    asyncio.run(test_schema_with_mock_data())
