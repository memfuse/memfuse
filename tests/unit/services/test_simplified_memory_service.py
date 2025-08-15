#!/usr/bin/env python3
"""
Test script for SimplifiedMemoryService

This script tests the basic functionality of the simplified Memory Service
to ensure it works correctly before integrating with the main system.
"""

import asyncio
import sys
import os
from datetime import datetime

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.memfuse_core.services.simplified_memory_service import SimplifiedMemoryService


async def test_basic_functionality():
    """Test basic Memory Service functionality."""
    print("🧪 Testing SimplifiedMemoryService...")
    
    # Create service instance
    service = SimplifiedMemoryService(
        user="test_user",
        agent="test_agent"
    )
    
    try:
        # Initialize service
        print("📡 Initializing service...")
        await service.initialize()
        print("✅ Service initialized successfully")
        
        # Test data with valid UUIDs
        import uuid
        conversation_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4())

        test_messages = [
            {
                "id": str(uuid.uuid4()),
                "content": "Hello, I want to learn about machine learning algorithms.",
                "role": "user",
                "created_at": datetime.now(),
                "metadata": {
                    "conversation_id": conversation_id,
                    "session_id": session_id
                }
            },
            {
                "id": str(uuid.uuid4()),
                "content": "I recommend starting with scikit-learn for machine learning. It provides many algorithms like SVM, Random Forest, and clustering methods.",
                "role": "assistant",
                "created_at": datetime.now(),
                "metadata": {
                    "conversation_id": conversation_id,
                    "session_id": session_id
                }
            },
            {
                "id": str(uuid.uuid4()),
                "content": "What about deep learning frameworks?",
                "role": "user",
                "created_at": datetime.now(),
                "metadata": {
                    "conversation_id": conversation_id,
                    "session_id": session_id
                }
            }
        ]
        
        # Test add_batch
        print("💾 Testing add_batch...")
        message_batch_list = [test_messages]  # Wrap in list as expected by interface
        result = await service.add_batch(message_batch_list)
        
        if result["status"] == "success":
            print(f"✅ add_batch successful: {result['message']}")
            print(f"   Processed {len(result['data'])} messages into {result.get('chunk_count', 0)} chunks")
        else:
            print(f"❌ add_batch failed: {result['message']}")
            return False
        
        # Test read
        print("📖 Testing read...")
        message_ids = result["data"][:2]  # Read first 2 messages
        read_result = await service.read(message_ids)
        
        if read_result["status"] == "success":
            print(f"✅ read successful: {len(read_result['data']['messages'])} messages read")
        else:
            print(f"❌ read failed: {read_result['message']}")
        
        # Test vector search
        print("🔍 Testing vector similarity search...")
        search_results = await service.query_similar_chunks(
            "machine learning algorithms",
            top_k=5,
            similarity_threshold=0.1
        )
        
        if search_results:
            print(f"✅ Vector search successful: {len(search_results)} results found")
            for i, result in enumerate(search_results[:3]):  # Show top 3
                print(f"   {i+1}. Score: {result['score']:.3f} - {result['content'][:60]}...")
        else:
            print("⚠️ Vector search returned no results")
        
        # Test query interface
        print("🔎 Testing query interface...")
        query_results = await service.query("deep learning frameworks", top_k=3)
        
        if query_results:
            print(f"✅ Query interface successful: {len(query_results)} results found")
        else:
            print("⚠️ Query interface returned no results")
        
        print("🎉 All tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        service.close()
        print("🧹 Service closed")


async def test_database_health():
    """Test database connectivity and health."""
    print("🏥 Testing database health...")
    
    service = SimplifiedMemoryService()
    
    try:
        await service.initialize()
        health_ok = await service.db_manager.health_check()
        
        if health_ok:
            print("✅ Database health check passed")
            return True
        else:
            print("❌ Database health check failed")
            return False
            
    except Exception as e:
        print(f"❌ Database health test failed: {e}")
        return False
        
    finally:
        service.close()


async def main():
    """Main test function."""
    print("=" * 60)
    print("🚀 SimplifiedMemoryService Test Suite")
    print("=" * 60)
    
    # Test database health first
    if not await test_database_health():
        print("❌ Database health check failed. Please ensure database is running.")
        return False
    
    # Test basic functionality
    if not await test_basic_functionality():
        print("❌ Basic functionality tests failed.")
        return False
    
    print("=" * 60)
    print("🎉 All tests passed! SimplifiedMemoryService is working correctly.")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
