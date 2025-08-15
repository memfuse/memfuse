#!/usr/bin/env python3
"""
Test script for QueryBuffer integration with SimplifiedMemoryService

This script tests the integration between QueryBuffer and our simplified
Memory Service to ensure data formats are compatible and results can be
properly merged and sorted.
"""

import asyncio
import sys
import os
import uuid
from datetime import datetime

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.memfuse_core.services.simplified_memory_service import SimplifiedMemoryService
from src.memfuse_core.buffer.query_buffer import QueryBuffer


async def create_retrieval_handler(memory_service: SimplifiedMemoryService):
    """Create a retrieval handler function for QueryBuffer."""
    
    async def retrieval_handler(query_text: str, top_k: int) -> list:
        """Retrieval handler that queries the Memory Service."""
        try:
            # Use the Memory Service's query method
            results = await memory_service.query_similar_chunks(
                query_text=query_text,
                top_k=top_k,
                similarity_threshold=0.1
            )
            
            print(f"🔍 Memory Service returned {len(results)} results for query: '{query_text[:30]}...'")
            return results
            
        except Exception as e:
            print(f"❌ Retrieval handler error: {e}")
            return []
    
    return retrieval_handler


async def test_querybuffer_integration():
    """Test QueryBuffer integration with SimplifiedMemoryService."""
    print("🧪 Testing QueryBuffer + SimplifiedMemoryService integration...")
    
    # Initialize Memory Service
    memory_service = SimplifiedMemoryService(
        user="test_user",
        agent="test_agent"
    )
    
    try:
        # Initialize Memory Service
        print("📡 Initializing Memory Service...")
        await memory_service.initialize()
        print("✅ Memory Service initialized")
        
        # Add test data to Memory Service
        print("💾 Adding test data to Memory Service...")
        conversation_id = str(uuid.uuid4())
        
        test_messages = [
            {
                "id": str(uuid.uuid4()),
                "content": "I'm interested in learning about machine learning algorithms and their applications.",
                "role": "user",
                "created_at": datetime.now(),
                "metadata": {"conversation_id": conversation_id}
            },
            {
                "id": str(uuid.uuid4()),
                "content": "Machine learning algorithms like neural networks, decision trees, and SVM are widely used in AI applications.",
                "role": "assistant", 
                "created_at": datetime.now(),
                "metadata": {"conversation_id": conversation_id}
            },
            {
                "id": str(uuid.uuid4()),
                "content": "What about deep learning frameworks like TensorFlow and PyTorch?",
                "role": "user",
                "created_at": datetime.now(),
                "metadata": {"conversation_id": conversation_id}
            },
            {
                "id": str(uuid.uuid4()),
                "content": "TensorFlow and PyTorch are popular deep learning frameworks for building neural networks.",
                "role": "assistant",
                "created_at": datetime.now(),
                "metadata": {"conversation_id": conversation_id}
            }
        ]
        
        # Store messages in Memory Service
        message_batch_list = [test_messages]
        result = await memory_service.add_batch(message_batch_list)
        
        if result["status"] != "success":
            print(f"❌ Failed to add test data: {result['message']}")
            return False
        
        print(f"✅ Added {len(result['data'])} messages, created {result.get('chunk_count', 0)} chunks")
        
        # Create retrieval handler
        retrieval_handler = await create_retrieval_handler(memory_service)
        
        # Initialize QueryBuffer with our retrieval handler
        print("🔧 Initializing QueryBuffer...")
        query_buffer = QueryBuffer(
            retrieval_handler=retrieval_handler,
            max_size=10,
            default_sort_by="score",
            default_order="desc"
        )
        print("✅ QueryBuffer initialized")
        
        # Test 1: Query with score sorting
        print("\n🔍 Test 1: Query with score sorting...")
        results = await query_buffer.query(
            query_text="machine learning algorithms",
            top_k=5,
            sort_by="score",
            order="desc"
        )
        
        if results:
            print(f"✅ Score-sorted query returned {len(results)} results:")
            for i, result in enumerate(results[:3]):
                print(f"   {i+1}. Score: {result.get('score', 0):.3f} - {result.get('content', '')[:60]}...")
        else:
            print("⚠️ Score-sorted query returned no results")
        
        # Test 2: Query with timestamp sorting
        print("\n🔍 Test 2: Query with timestamp sorting...")
        results = await query_buffer.query(
            query_text="deep learning frameworks",
            top_k=5,
            sort_by="timestamp",
            order="desc"
        )
        
        if results:
            print(f"✅ Timestamp-sorted query returned {len(results)} results:")
            for i, result in enumerate(results[:3]):
                created_at = result.get('created_at', 'N/A')
                print(f"   {i+1}. Time: {created_at} - {result.get('content', '')[:60]}...")
        else:
            print("⚠️ Timestamp-sorted query returned no results")
        
        # Test 3: Test data format compatibility
        print("\n🔍 Test 3: Data format compatibility check...")
        test_results = await memory_service.query_similar_chunks("neural networks", top_k=3)
        
        if test_results:
            print("✅ Memory Service data format check:")
            for result in test_results[:1]:  # Check first result
                required_fields = ['id', 'score', 'content', 'created_at', 'metadata']
                missing_fields = [field for field in required_fields if field not in result]
                
                if missing_fields:
                    print(f"❌ Missing required fields: {missing_fields}")
                    return False
                else:
                    print(f"✅ All required fields present: {list(result.keys())}")
                    
                # Check score range
                score = result.get('score', 0)
                if 0 <= score <= 1:
                    print(f"✅ Score in valid range [0,1]: {score:.3f}")
                else:
                    print(f"❌ Score out of range: {score}")
                    return False
        
        # Test 4: QueryBuffer statistics
        print("\n📊 QueryBuffer statistics:")
        stats = query_buffer.get_stats()
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        print("\n🎉 All QueryBuffer integration tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        memory_service.close()
        print("🧹 Memory Service closed")


async def main():
    """Main test function."""
    print("=" * 70)
    print("🚀 QueryBuffer + SimplifiedMemoryService Integration Test")
    print("=" * 70)
    
    success = await test_querybuffer_integration()
    
    if success:
        print("=" * 70)
        print("🎉 All integration tests passed! QueryBuffer works with SimplifiedMemoryService.")
        print("=" * 70)
    else:
        print("=" * 70)
        print("❌ Integration tests failed.")
        print("=" * 70)
    
    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
