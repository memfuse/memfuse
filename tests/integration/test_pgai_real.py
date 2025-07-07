#!/usr/bin/env python3
"""
Real integration test for pgai store with actual PostgreSQL database.
This test requires a running PostgreSQL instance with pgai extensions.
"""

import asyncio
import os
import sys
from typing import List

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from memfuse_core.store.pgai_store import PgaiStore
from memfuse_core.store.pgai_vector_wrapper import PgaiVectorWrapper
from memfuse_core.rag.chunk.base import ChunkData
from memfuse_core.rag.encode.base import EncoderBase
from memfuse_core.rag.encode.MiniLM import MiniLMEncoder


async def test_pgai_store_real():
    """Test PgaiStore with real PostgreSQL database."""
    print("🚀 Starting real pgai store test...")
    
    # Database configuration
    config = {
        'host': 'localhost',
        'port': 5432,
        'database': 'memfuse',
        'user': 'postgres',
        'password': 'postgres',
        'table': 'test_chunks'
    }
    
    try:
        # Initialize store
        print("📦 Initializing PgaiStore...")
        store = PgaiStore(config)
        success = await store.initialize()
        if not success:
            raise Exception("Failed to initialize PgaiStore")
        print("✅ PgaiStore initialized successfully")
        
        # Clear any existing data
        print("🧹 Clearing existing data...")
        await store.clear()
        
        # Test data
        test_chunks = [
            ChunkData(
                chunk_id="chunk_1",
                content="This is the first test chunk for pgai integration.",
                metadata={"source": "test", "type": "integration"}
            ),
            ChunkData(
                chunk_id="chunk_2",
                content="This is the second test chunk with different content.",
                metadata={"source": "test", "type": "integration", "priority": "high"}
            ),
            ChunkData(
                chunk_id="chunk_3",
                content="Third chunk contains some technical information about databases.",
                metadata={"source": "docs", "type": "technical"}
            )
        ]
        
        # Test add operation
        print("➕ Testing add operation...")
        await store.add(test_chunks)
        print(f"✅ Added {len(test_chunks)} chunks successfully")
        
        # Test count
        print("🔢 Testing count operation...")
        count = await store.count()
        print(f"✅ Count: {count} chunks in database")
        assert count == len(test_chunks), f"Expected {len(test_chunks)}, got {count}"
        
        # Test read operation
        print("📖 Testing read operation...")
        chunk_ids = [chunk.chunk_id for chunk in test_chunks]
        retrieved_chunks = await store.read(chunk_ids)
        print(f"✅ Retrieved {len(retrieved_chunks)} chunks")
        assert len(retrieved_chunks) == len(test_chunks), "Retrieved chunk count mismatch"
        
        # Verify content
        for original, retrieved in zip(test_chunks, retrieved_chunks):
            assert original.chunk_id == retrieved.chunk_id, f"ID mismatch: {original.chunk_id} != {retrieved.chunk_id}"
            assert original.content == retrieved.content, f"Content mismatch for {original.chunk_id}"
            print(f"✅ Chunk {original.chunk_id} content verified")
        
        # Test query operation (text search)
        print("🔍 Testing query operation...")
        query_results = await store.query("technical information", top_k=2)
        print(f"✅ Query returned {len(query_results)} results")
        assert len(query_results) > 0, "Query should return at least one result"
        
        # Print query results
        for i, chunk in enumerate(query_results):
            print(f"  Result {i+1}: {chunk.chunk_id}")
            print(f"    Content: {chunk.content[:50]}...")
        
        # Test delete operation
        print("🗑️ Testing delete operation...")
        await store.delete([test_chunks[0].chunk_id])
        remaining_count = await store.count()
        print(f"✅ After deletion, count: {remaining_count}")
        assert remaining_count == len(test_chunks) - 1, "Delete operation failed"
        
        # Test clear operation
        print("🧹 Testing clear operation...")
        await store.clear()
        final_count = await store.count()
        print(f"✅ After clear, count: {final_count}")
        assert final_count == 0, "Clear operation failed"
        
        print("🎉 All PgaiStore tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise
    finally:
        if 'store' in locals():
            await store.close()


async def test_pgai_vector_wrapper_real():
    """Test PgaiVectorWrapper with real PostgreSQL database."""
    print("\n🚀 Starting real pgai vector wrapper test...")

    # Database configuration
    config = {
        'host': 'localhost',
        'port': 5432,
        'database': 'memfuse',
        'user': 'postgres',
        'password': 'postgres',
        'table': 'test_vector_chunks'
    }

    try:
        # Initialize encoder
        print("🔧 Initializing encoder...")
        encoder = MiniLMEncoder(model_name="all-MiniLM-L6-v2")

        # Initialize pgai store
        print("📦 Initializing PgaiStore...")
        pgai_store = PgaiStore(config, table_name="test_vector_chunks")
        success = await pgai_store.initialize()
        if not success:
            raise Exception("Failed to initialize PgaiStore for vector wrapper")

        # Initialize wrapper
        print("📦 Initializing PgaiVectorWrapper...")
        wrapper = PgaiVectorWrapper(pgai_store, encoder)
        success = await wrapper.initialize()
        if not success:
            raise Exception("Failed to initialize PgaiVectorWrapper")
        print("✅ PgaiVectorWrapper initialized successfully")
        
        # Clear any existing data
        print("🧹 Clearing existing data...")
        await wrapper.clear()
        
        # Test data
        test_chunks = [
            ChunkData(
                chunk_id="vec_chunk_1",
                content="Machine learning algorithms are powerful tools for data analysis.",
                metadata={"topic": "ml", "difficulty": "intermediate"}
            ),
            ChunkData(
                chunk_id="vec_chunk_2",
                content="Deep learning neural networks can process complex patterns.",
                metadata={"topic": "dl", "difficulty": "advanced"}
            )
        ]
        
        # Test add operation
        print("➕ Testing vector wrapper add operation...")
        await wrapper.add(test_chunks)
        print(f"✅ Added {len(test_chunks)} chunks via wrapper")
        
        # Test query operation
        print("🔍 Testing vector wrapper query operation...")
        query_results = await wrapper.query("machine learning neural networks", top_k=5)
        print(f"✅ Vector query returned {len(query_results)} results")
        
        # Print results
        for i, chunk in enumerate(query_results):
            print(f"  Result {i+1}: {chunk.chunk_id}")
            print(f"    Content: {chunk.content[:60]}...")
        
        # Test count
        print("🔢 Testing vector wrapper count...")
        count = await wrapper.count()
        print(f"✅ Vector wrapper count: {count}")
        assert count == len(test_chunks), f"Expected {len(test_chunks)}, got {count}"
        
        print("🎉 All PgaiVectorWrapper tests passed!")
        
    except Exception as e:
        print(f"❌ Vector wrapper test failed: {e}")
        raise
    finally:
        if 'wrapper' in locals():
            await wrapper.close()


async def main():
    """Run all real integration tests."""
    print("🧪 Running pgai real integration tests...")
    print("=" * 60)
    
    try:
        await test_pgai_store_real()
        await test_pgai_vector_wrapper_real()
        
        print("\n" + "=" * 60)
        print("🎉 All real integration tests passed successfully!")
        print("✅ pgai integration is working correctly with real PostgreSQL database")
        
    except Exception as e:
        print(f"\n❌ Integration tests failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
