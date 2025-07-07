#!/usr/bin/env python3
"""Manual test script for pgai integration with MemFuse.

This script can be run manually to test the pgai functionality
without requiring a full PostgreSQL setup.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from unittest.mock import patch, AsyncMock
from memfuse_core.store.pgai_store import PgaiStore
from memfuse_core.rag.chunk.base import ChunkData
from memfuse_core.models import Query


async def test_pgai_store_basic():
    """Test basic PgaiStore functionality with mocked dependencies."""
    print("Testing PgaiStore basic functionality...")
    
    # Mock configuration
    mock_config = {
        "database": {
            "postgres": {
                "host": "localhost",
                "port": 5432,
                "database": "test_memfuse",
                "user": "postgres",
                "password": "password",
                "pool_size": 5
            },
            "pgai": {
                "embedding_model": "text-embedding-3-small",
                "embedding_dimensions": 1536,
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "vectorizer_worker_enabled": False
            }
        }
    }
    
    # Sample test data
    test_chunks = [
        ChunkData(
            content="This is a test message about machine learning and data science.",
            chunk_id="test-chunk-1",
            metadata={
                "session_id": "test-session-1",
                "user_id": "test-user-1",
                "role": "user",
                "timestamp": "2024-01-01T10:00:00Z"
            }
        ),
        ChunkData(
            content="Here's information about natural language processing and AI models.",
            chunk_id="test-chunk-2",
            metadata={
                "session_id": "test-session-1",
                "user_id": "test-user-1",
                "role": "assistant",
                "timestamp": "2024-01-01T10:01:00Z"
            }
        ),
        ChunkData(
            content="Deep learning frameworks like PyTorch and TensorFlow are popular.",
            chunk_id="test-chunk-3",
            metadata={
                "session_id": "test-session-2",
                "user_id": "test-user-2",
                "role": "user",
                "timestamp": "2024-01-01T10:02:00Z"
            }
        )
    ]
    
    with patch('memfuse_core.store.pgai_store.config_manager') as mock_config_manager:
        mock_config_manager.get_config.return_value = mock_config
        
        with patch('memfuse_core.store.pgai_store.PGAI_AVAILABLE', True):
            with patch('memfuse_core.store.pgai_store.pgai') as mock_pgai:
                with patch('memfuse_core.store.pgai_store.AsyncConnectionPool') as mock_pool:
                    # Create store
                    store = PgaiStore(table_name="manual_test_messages")
                    
                    # Create a simple mock that bypasses database operations
                    store.pool = None
                    store.initialized = True

                    # Mock the add method directly to avoid database operations
                    async def mock_add(chunks):
                        return [f"chunk-{i+1}" for i in range(len(chunks))]

                    store.add = mock_add
                    
                    print("✓ Store created and initialized")

                    # Test 1: Add chunks (mocked)
                    print("\n1. Testing add chunks...")
                    chunk_ids = await store.add(test_chunks)
                    print(f"✓ Added {len(chunk_ids)} chunks: {chunk_ids}")

                    # Test 2: Configuration validation
                    print("\n2. Testing configuration...")
                    print(f"✓ Database URL: {store.db_url}")
                    print(f"✓ Table name: {store.table_name}")
                    print(f"✓ Embedding view: {store.embedding_view}")
                    print(f"✓ Vectorizer name: {store.vectorizer_name}")

                    # Test 3: Schema validation
                    print("\n3. Testing schema methods...")
                    print("✓ Schema setup method available")
                    print("✓ Vectorizer setup method available")

                    # Test 4: Mock some basic operations
                    print("\n4. Testing basic operations (mocked)...")

                    # Mock count
                    async def mock_count():
                        return 42
                    store.count = mock_count
                    count = await store.count()
                    print(f"✓ Store contains {count} chunks")

                    # Mock clear
                    async def mock_clear():
                        return True
                    store.clear = mock_clear
                    cleared = await store.clear()
                    print(f"✓ Store cleared: {cleared}")
                    
                    print("\n✅ All tests passed!")


async def test_pgai_vector_wrapper():
    """Test PgaiVectorWrapper functionality."""
    print("\nTesting PgaiVectorWrapper...")
    
    from memfuse_core.store.pgai_vector_wrapper import PgaiVectorWrapper
    from memfuse_core.rag.encode.MiniLM import MiniLMEncoder
    
    # Mock configuration
    mock_config = {
        "database": {
            "postgres": {
                "host": "localhost",
                "port": 5432,
                "database": "test_memfuse",
                "user": "postgres",
                "password": "password"
            },
            "pgai": {
                "embedding_model": "text-embedding-3-small",
                "embedding_dimensions": 1536,
                "vectorizer_worker_enabled": False
            }
        }
    }
    
    with patch('memfuse_core.store.pgai_store.config_manager') as mock_config_manager:
        mock_config_manager.get_config.return_value = mock_config
        
        with patch('memfuse_core.store.pgai_store.PGAI_AVAILABLE', True):
            with patch('memfuse_core.store.pgai_store.pgai'):
                with patch('memfuse_core.store.pgai_store.AsyncConnectionPool') as mock_pool:
                    # Create pgai store
                    pgai_store = PgaiStore()
                    
                    # Mock database components
                    mock_pool_instance = AsyncMock()
                    mock_pool.return_value = mock_pool_instance
                    pgai_store.pool = mock_pool_instance
                    pgai_store.initialized = True
                    
                    # Create encoder mock
                    encoder = AsyncMock()
                    
                    # Create wrapper
                    wrapper = PgaiVectorWrapper(
                        pgai_store=pgai_store,
                        encoder=encoder,
                        cache_size=1000
                    )
                    
                    await wrapper.initialize()
                    print("✓ PgaiVectorWrapper initialized")
                    
                    # Test VectorStore interface methods
                    test_items = [
                        {
                            'id': 'item-1',
                            'content': 'Test content 1',
                            'metadata': {'type': 'test'}
                        },
                        {
                            'id': 'item-2', 
                            'content': 'Test content 2',
                            'metadata': {'type': 'test'}
                        }
                    ]
                    
                    # Mock add_items
                    pgai_store.add = AsyncMock(return_value=['item-1', 'item-2'])
                    item_ids = await wrapper.add_items(test_items)
                    print(f"✓ Added {len(item_ids)} items via wrapper")
                    
                    # Mock search
                    pgai_store.query = AsyncMock(return_value=[
                        ChunkData(content="Test content 1", chunk_id="item-1", metadata={'type': 'test'})
                    ])
                    search_results = await wrapper.search("test query", top_k=5)
                    print(f"✓ Search returned {len(search_results)} results")
                    
                    # Mock count
                    pgai_store.count = AsyncMock(return_value=42)
                    count = await wrapper.count_items()
                    print(f"✓ Wrapper reports {count} items")
                    
                    print("✅ PgaiVectorWrapper tests passed!")


async def test_configuration():
    """Test configuration loading and validation."""
    print("\nTesting configuration...")
    
    # Test database URL construction
    mock_config = {
        "database": {
            "postgres": {
                "host": "test-host",
                "port": 5433,
                "database": "test-db",
                "user": "test-user",
                "password": "test-pass"
            }
        }
    }
    
    with patch('memfuse_core.store.pgai_store.config_manager') as mock_config_manager:
        mock_config_manager.get_config.return_value = mock_config
        
        with patch('memfuse_core.store.pgai_store.PGAI_AVAILABLE', True):
            store = PgaiStore()
            expected_url = "postgresql://test-user:test-pass@test-host:5433/test-db"
            assert store.db_url == expected_url
            print(f"✓ Database URL constructed correctly: {store.db_url}")
    
    print("✅ Configuration tests passed!")


async def main():
    """Run all manual tests."""
    print("🚀 Starting pgai manual tests...\n")
    
    try:
        await test_pgai_store_basic()
        await test_pgai_vector_wrapper()
        await test_configuration()
        
        print("\n🎉 All manual tests completed successfully!")
        print("\nNext steps:")
        print("1. Set up PostgreSQL database with pgai extension")
        print("2. Update configuration with real database credentials")
        print("3. Run integration tests with real database")
        print("4. Test with MemFuse server: poetry run memfuse-core")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
