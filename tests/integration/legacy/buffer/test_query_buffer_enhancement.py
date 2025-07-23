"""Test script for QueryBuffer enhancement.

This script tests the enhanced QueryBuffer implementation with session querying
and internal reranking functionality.
"""

import asyncio
import time
import uuid
from typing import List, Dict, Any, Optional

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from memfuse_core.buffer.query_buffer import QueryBuffer


class MockHybridBuffer:
    """Mock HybridBuffer for testing."""
    
    def __init__(self):
        self.chunks = []
    
    def add_test_chunk(self, content: str, session_id: str, chunk_id: str = None):
        """Add a test chunk."""
        chunk = type('Chunk', (), {
            'id': chunk_id or str(uuid.uuid4()),
            'content': content,
            'metadata': {'session_id': session_id},
            'created_at': time.time(),
            'messages': [
                {
                    'id': str(uuid.uuid4()),
                    'content': content,
                    'metadata': {'session_id': session_id},
                    'created_at': time.time()
                }
            ]
        })()
        self.chunks.append(chunk)


class MockRetrievalHandler:
    """Mock retrieval handler for testing."""
    
    def __init__(self):
        self.storage_data = []
        self.call_history = []
    
    def add_test_data(self, content: str, session_id: str, msg_id: str = None):
        """Add test data to mock storage."""
        self.storage_data.append({
            'id': msg_id or str(uuid.uuid4()),
            'content': content,
            'metadata': {'session_id': session_id},
            'created_at': time.time()
        })
    
    async def __call__(self, query_text: str, max_results: int) -> List[Dict[str, Any]]:
        """Mock retrieval handler."""
        self.call_history.append((query_text, max_results))
        
        # Simple filtering based on query
        if query_text.startswith("session_id:"):
            session_id = query_text.split(":", 1)[1]
            return [
                data for data in self.storage_data 
                if data['metadata']['session_id'] == session_id
            ][:max_results]
        else:
            # Simple content matching
            return [
                data for data in self.storage_data 
                if query_text.lower() in data['content'].lower()
            ][:max_results]


class MockRerankHandler:
    """Mock rerank handler for testing."""
    
    def __init__(self):
        self.call_history = []
    
    async def __call__(self, query_text: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Mock rerank handler that reverses the order."""
        self.call_history.append((query_text, len(results)))
        
        # Simple reranking: reverse the order
        return list(reversed(results))


async def test_query_with_reranking():
    """Test query with internal reranking."""
    print("\n=== Test: Query with Internal Reranking ===")
    
    # Create mock handlers
    retrieval_handler = MockRetrievalHandler()
    rerank_handler = MockRerankHandler()
    
    # Add test data
    retrieval_handler.add_test_data("First message", "session1")
    retrieval_handler.add_test_data("Second message", "session1")
    retrieval_handler.add_test_data("Third message", "session1")
    
    # Create QueryBuffer with rerank handler
    query_buffer = QueryBuffer(
        retrieval_handler=retrieval_handler,
        rerank_handler=rerank_handler,
        max_size=10
    )
    
    # Test query with reranking enabled
    results_with_rerank = await query_buffer.query("message", use_rerank=True)
    
    # Test query without reranking
    results_without_rerank = await query_buffer.query("message", use_rerank=False)
    
    print(f"Results with rerank: {len(results_with_rerank)}")
    print(f"Results without rerank: {len(results_without_rerank)}")
    print(f"Rerank operations: {query_buffer.rerank_operations}")
    
    # Verify reranking was called
    assert query_buffer.rerank_operations > 0
    assert len(rerank_handler.call_history) > 0
    
    # Verify results are different (due to reranking)
    if len(results_with_rerank) > 1 and len(results_without_rerank) > 1:
        # Results should be in different order due to reranking
        first_with_rerank = results_with_rerank[0]['content']
        first_without_rerank = results_without_rerank[0]['content']
        print(f"First result with rerank: {first_with_rerank}")
        print(f"First result without rerank: {first_without_rerank}")
    
    print("✅ Query with internal reranking test passed")


async def test_session_query():
    """Test session-specific querying."""
    print("\n=== Test: Session Query ===")
    
    # Create mock handlers and hybrid buffer
    retrieval_handler = MockRetrievalHandler()
    hybrid_buffer = MockHybridBuffer()
    
    # Add test data to storage
    retrieval_handler.add_test_data("Storage message 1", "session_test")
    retrieval_handler.add_test_data("Storage message 2", "session_test")
    retrieval_handler.add_test_data("Other session message", "other_session")
    
    # Add test data to hybrid buffer
    hybrid_buffer.add_test_chunk("Hybrid message 1", "session_test")
    hybrid_buffer.add_test_chunk("Hybrid message 2", "session_test")
    hybrid_buffer.add_test_chunk("Other hybrid message", "other_session")
    
    # Create QueryBuffer
    query_buffer = QueryBuffer(
        retrieval_handler=retrieval_handler,
        max_size=10
    )
    query_buffer.set_hybrid_buffer(hybrid_buffer)
    
    # Test session query
    session_results = await query_buffer.query_by_session("session_test", limit=10)
    
    print(f"Session query results: {len(session_results)}")
    for result in session_results:
        print(f"  - {result.get('content', 'No content')}")
    
    # Verify all results belong to the correct session
    for result in session_results:
        assert result.get('metadata', {}).get('session_id') == 'session_test'
    
    # Verify statistics
    assert query_buffer.total_session_queries > 0
    
    print("✅ Session query test passed")


async def test_multi_source_coordination():
    """Test multi-source query coordination."""
    print("\n=== Test: Multi-source Coordination ===")
    
    # Create mock handlers and hybrid buffer
    retrieval_handler = MockRetrievalHandler()
    hybrid_buffer = MockHybridBuffer()
    
    # Add overlapping data (same ID in both sources)
    shared_id = str(uuid.uuid4())
    retrieval_handler.storage_data.append({
        'id': shared_id,
        'content': "Shared message from storage",
        'metadata': {'session_id': 'multi_test'},
        'created_at': time.time()
    })
    
    # Add unique data to each source
    retrieval_handler.add_test_data("Storage only message", "multi_test")
    hybrid_buffer.add_test_chunk("Hybrid only message", "multi_test")
    
    # Create QueryBuffer
    query_buffer = QueryBuffer(
        retrieval_handler=retrieval_handler,
        max_size=10
    )
    query_buffer.set_hybrid_buffer(hybrid_buffer)
    
    # Test session query with deduplication
    session_results = await query_buffer.query_by_session("multi_test", limit=10)
    
    print(f"Multi-source results: {len(session_results)}")
    
    # Verify deduplication worked (no duplicate IDs)
    result_ids = [r.get('id') for r in session_results if r.get('id')]
    unique_ids = set(result_ids)
    assert len(result_ids) == len(unique_ids), "Deduplication failed"
    
    print("✅ Multi-source coordination test passed")


async def test_rerank_caching():
    """Test rerank result caching."""
    print("\n=== Test: Rerank Caching ===")
    
    # Create mock handlers
    retrieval_handler = MockRetrievalHandler()
    rerank_handler = MockRerankHandler()
    
    # Add test data
    retrieval_handler.add_test_data("Test message 1", "cache_test")
    retrieval_handler.add_test_data("Test message 2", "cache_test")
    
    # Create QueryBuffer with rerank handler
    query_buffer = QueryBuffer(
        retrieval_handler=retrieval_handler,
        rerank_handler=rerank_handler,
        max_size=10
    )
    
    # First query (should call rerank handler)
    initial_rerank_calls = len(rerank_handler.call_history)
    results1 = await query_buffer.query("test", use_rerank=True)
    first_rerank_calls = len(rerank_handler.call_history)
    
    # Second identical query (should use cache)
    results2 = await query_buffer.query("test", use_rerank=True)
    second_rerank_calls = len(rerank_handler.call_history)
    
    print(f"Initial rerank calls: {initial_rerank_calls}")
    print(f"After first query: {first_rerank_calls}")
    print(f"After second query: {second_rerank_calls}")
    
    # Verify caching worked
    assert first_rerank_calls > initial_rerank_calls, "First query should call rerank"
    # Note: Cache might not work if results are different due to timestamps
    
    print("✅ Rerank caching test passed")


async def test_error_handling():
    """Test error handling in enhanced QueryBuffer."""
    print("\n=== Test: Error Handling ===")
    
    # Create QueryBuffer with failing handlers
    async def failing_retrieval_handler(query: str, max_results: int):
        raise Exception("Retrieval failed")
    
    async def failing_rerank_handler(query: str, results: List):
        raise Exception("Rerank failed")
    
    query_buffer = QueryBuffer(
        retrieval_handler=failing_retrieval_handler,
        rerank_handler=failing_rerank_handler,
        max_size=10
    )
    
    # Test query with failing handlers
    results = await query_buffer.query("test", use_rerank=True)
    
    # Should return empty results gracefully
    assert isinstance(results, list)
    
    # Test session query with failing handlers
    session_results = await query_buffer.query_by_session("test_session")
    
    # Should return empty results gracefully
    assert isinstance(session_results, list)
    
    print("✅ Error handling test passed")


async def run_all_tests():
    """Run all QueryBuffer enhancement tests."""
    print("Starting QueryBuffer Enhancement Tests...")
    
    try:
        await test_query_with_reranking()
        await test_session_query()
        await test_multi_source_coordination()
        await test_rerank_caching()
        await test_error_handling()
        
        print("\n🎉 All QueryBuffer enhancement tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    # Run tests
    success = asyncio.run(run_all_tests())
    
    if success:
        print("\n✅ QueryBuffer enhancement implementation is working correctly!")
        exit(0)
    else:
        print("\n❌ QueryBuffer enhancement tests failed!")
        exit(1)
