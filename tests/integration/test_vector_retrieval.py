#!/usr/bin/env python3
"""
Test script to verify M1 embedding generation and vector retrieval functionality.
"""

import asyncio
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from memfuse_core.services.simplified_memory_service import SimplifiedMemoryService
from memfuse_core.utils.config import config_manager

async def test_vector_retrieval():
    """Test vector retrieval functionality."""
    print("🧪 Testing SimplifiedMemoryService vector retrieval...")
    
    # Initialize config
    config_dict = config_manager.get_config()
    
    # Create SimplifiedMemoryService
    service = SimplifiedMemoryService(cfg=config_dict, user="test_user")
    await service.initialize()
    
    print("✅ SimplifiedMemoryService initialized")
    
    # Test query
    test_query = "dogs and pets"
    print(f"🔍 Testing query: '{test_query}'")
    
    # Perform vector search
    results = await service.query_similar_chunks(test_query, top_k=3)
    
    print(f"📊 Found {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"  {i}. Score: {result.get('score', 'N/A'):.4f}")
        print(f"     Content: {result.get('content', 'N/A')[:100]}...")
        print(f"     Chunk ID: {result.get('chunk_id', 'N/A')}")
        print()
    
    # Test another query
    test_query2 = "music and artists"
    print(f"🔍 Testing query: '{test_query2}'")
    
    results2 = await service.query_similar_chunks(test_query2, top_k=3)
    
    print(f"📊 Found {len(results2)} results:")
    for i, result in enumerate(results2, 1):
        print(f"  {i}. Score: {result.get('score', 'N/A'):.4f}")
        print(f"     Content: {result.get('content', 'N/A')[:100]}...")
        print(f"     Chunk ID: {result.get('chunk_id', 'N/A')}")
        print()
    
    print("✅ Vector retrieval test completed successfully!")

if __name__ == "__main__":
    asyncio.run(test_vector_retrieval())
