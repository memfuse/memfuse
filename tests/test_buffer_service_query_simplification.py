"""Test script for BufferService query simplification.

This script tests the simplified BufferService query implementation that delegates
all query logic to QueryBuffer while maintaining service layer responsibilities.
"""

import asyncio
import time
import uuid
from typing import List, Dict, Any

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from memfuse_core.services.buffer_service import BufferService
from memfuse_core.interfaces import MessageList, MessageBatchList


class MockMemoryService:
    """Mock memory service for testing."""
    
    def __init__(self, user_id: str = "test_user"):
        self._user_id = user_id
        self.query_calls = []
        self.storage_data = []
    
    def add_test_data(self, content: str, session_id: str = "default", msg_id: str = None):
        """Add test data to mock storage."""
        self.storage_data.append({
            'id': msg_id or str(uuid.uuid4()),
            'content': content,
            'metadata': {'session_id': session_id},
            'created_at': time.time()
        })
    
    async def query(self, query: str, **kwargs) -> Dict[str, Any]:
        """Mock query method."""
        self.query_calls.append((query, kwargs))
        
        # Simple content matching
        results = [
            data for data in self.storage_data 
            if query.lower() in data['content'].lower()
        ]
        
        return {
            "status": "success",
            "data": {"results": results, "total": len(results)}
        }
    
    async def add_batch(self, rounds: List[MessageList]) -> Dict[str, Any]:
        """Mock add_batch method."""
        return {
            "status": "success",
            "message": f"Added {len(rounds)} rounds"
        }
    
    async def get_messages_by_session(self, session_id: str, **kwargs) -> List[Dict[str, Any]]:
        """Mock get_messages_by_session method."""
        return [
            data for data in self.storage_data 
            if data['metadata']['session_id'] == session_id
        ]


def create_test_message(content: str, session_id: str = "default") -> Dict[str, Any]:
    """Create a test message."""
    return {
        "content": content,
        "role": "user",
        "metadata": {
            "session_id": session_id
        }
    }


async def test_simplified_query():
    """Test simplified query method delegation."""
    print("\n=== Test: Simplified Query Delegation ===")
    
    # Create BufferService with mock memory service
    mock_memory_service = MockMemoryService("test_user_query")
    mock_memory_service.add_test_data("Test message about music", "session1")
    mock_memory_service.add_test_data("Another test message", "session1")
    mock_memory_service.add_test_data("Different content", "session2")
    
    config = {
        'buffer': {
            'round_buffer': {'max_tokens': 800, 'max_size': 5, 'token_model': 'gpt-4o-mini'},
            'hybrid_buffer': {'max_size': 5, 'chunk_strategy': 'message', 'embedding_model': 'all-MiniLM-L6-v2'},
            'performance': {'max_flush_workers': 2, 'max_flush_queue_size': 100, 'flush_timeout': 30.0}
        },
        'retrieval': {'use_rerank': False}  # Disable rerank for simpler testing
    }
    
    buffer_service = BufferService(
        memory_service=mock_memory_service,
        user="test_user_query",
        config=config
    )
    
    # Test query delegation
    result = await buffer_service.query("music", top_k=5)
    
    print(f"Query result: {result}")
    
    # Verify response structure
    assert result["status"] == "success"
    assert result["code"] == 200
    assert "data" in result
    assert "results" in result["data"]
    assert "total" in result["data"]
    
    # Verify QueryBuffer was used (should have results from storage)
    results = result["data"]["results"]
    assert len(results) > 0
    
    print("✅ Simplified query delegation test passed")


async def test_query_with_reranking():
    """Test query with reranking enabled."""
    print("\n=== Test: Query with Reranking ===")
    
    # Create BufferService with reranking enabled
    mock_memory_service = MockMemoryService("test_user_rerank")
    mock_memory_service.add_test_data("First music message", "session1")
    mock_memory_service.add_test_data("Second music message", "session1")
    mock_memory_service.add_test_data("Third music message", "session1")
    
    config = {
        'buffer': {
            'round_buffer': {'max_tokens': 800, 'max_size': 5, 'token_model': 'gpt-4o-mini'},
            'hybrid_buffer': {'max_size': 5, 'chunk_strategy': 'message', 'embedding_model': 'all-MiniLM-L6-v2'},
            'performance': {'max_flush_workers': 2, 'max_flush_queue_size': 100, 'flush_timeout': 30.0}
        },
        'retrieval': {'use_rerank': True}  # Enable rerank
    }
    
    buffer_service = BufferService(
        memory_service=mock_memory_service,
        user="test_user_rerank",
        config=config
    )
    
    # Test query with reranking
    result = await buffer_service.query("music", top_k=3)
    
    print(f"Query with rerank result: {result}")
    
    # Verify response structure
    assert result["status"] == "success"
    assert "data" in result
    assert "results" in result["data"]
    
    # Check QueryBuffer statistics for rerank operations
    query_stats = buffer_service.query_buffer.get_stats()
    print(f"QueryBuffer stats: {query_stats}")
    
    print("✅ Query with reranking test passed")


async def test_session_query_delegation():
    """Test session query delegation to QueryBuffer."""
    print("\n=== Test: Session Query Delegation ===")
    
    # Create BufferService with mock memory service
    mock_memory_service = MockMemoryService("test_user_session")
    mock_memory_service.add_test_data("Session 1 message 1", "session_test")
    mock_memory_service.add_test_data("Session 1 message 2", "session_test")
    mock_memory_service.add_test_data("Session 2 message", "other_session")
    
    config = {
        'buffer': {
            'round_buffer': {'max_tokens': 800, 'max_size': 5, 'token_model': 'gpt-4o-mini'},
            'hybrid_buffer': {'max_size': 5, 'chunk_strategy': 'message', 'embedding_model': 'all-MiniLM-L6-v2'},
            'performance': {'max_flush_workers': 2, 'max_flush_queue_size': 100, 'flush_timeout': 30.0}
        }
    }
    
    buffer_service = BufferService(
        memory_service=mock_memory_service,
        user="test_user_session",
        config=config
    )
    
    # Test session query delegation
    session_messages = await buffer_service.get_messages_by_session(
        session_id="session_test",
        limit=10
    )
    
    print(f"Session messages: {len(session_messages)}")
    for msg in session_messages:
        print(f"  - {msg.get('content', 'No content')}")
    
    # Verify all messages belong to the correct session
    for msg in session_messages:
        assert msg.get('metadata', {}).get('session_id') == 'session_test'
    
    # Verify QueryBuffer session query statistics
    query_stats = buffer_service.query_buffer.get_stats()
    assert query_stats.get('total_session_queries', 0) > 0
    
    print("✅ Session query delegation test passed")


async def test_buffer_only_session_query():
    """Test buffer-only session query (direct access)."""
    print("\n=== Test: Buffer-only Session Query ===")
    
    # Create BufferService
    mock_memory_service = MockMemoryService("test_user_buffer_only")
    
    config = {
        'buffer': {
            'round_buffer': {'max_tokens': 800, 'max_size': 5, 'token_model': 'gpt-4o-mini'},
            'hybrid_buffer': {'max_size': 5, 'chunk_strategy': 'message', 'embedding_model': 'all-MiniLM-L6-v2'},
            'performance': {'max_flush_workers': 2, 'max_flush_queue_size': 100, 'flush_timeout': 30.0}
        }
    }
    
    buffer_service = BufferService(
        memory_service=mock_memory_service,
        user="test_user_buffer_only",
        config=config
    )
    
    # Add some messages to RoundBuffer first
    test_messages = [create_test_message("Buffer message 1", "buffer_session")]
    await buffer_service.add(test_messages, session_id="buffer_session")
    
    # Test buffer-only session query
    buffer_messages = await buffer_service.get_messages_by_session(
        session_id="buffer_session",
        limit=10,
        buffer_only=True
    )
    
    print(f"Buffer-only messages: {len(buffer_messages)}")
    
    # Should return messages from RoundBuffer
    assert isinstance(buffer_messages, list)
    
    print("✅ Buffer-only session query test passed")


async def test_service_statistics():
    """Test service-level statistics tracking."""
    print("\n=== Test: Service Statistics ===")
    
    # Create BufferService
    mock_memory_service = MockMemoryService("test_user_stats")
    mock_memory_service.add_test_data("Stats test message", "stats_session")
    
    config = {
        'buffer': {
            'round_buffer': {'max_tokens': 800, 'max_size': 5, 'token_model': 'gpt-4o-mini'},
            'hybrid_buffer': {'max_size': 5, 'chunk_strategy': 'message', 'embedding_model': 'all-MiniLM-L6-v2'},
            'performance': {'max_flush_workers': 2, 'max_flush_queue_size': 100, 'flush_timeout': 30.0}
        }
    }
    
    buffer_service = BufferService(
        memory_service=mock_memory_service,
        user="test_user_stats",
        config=config
    )
    
    # Record initial statistics
    initial_queries = buffer_service.total_queries
    
    # Perform some queries
    await buffer_service.query("test", top_k=5)
    await buffer_service.query("message", top_k=3)
    
    # Verify statistics were updated
    assert buffer_service.total_queries == initial_queries + 2
    
    # Get comprehensive buffer statistics
    buffer_stats = await buffer_service.get_buffer_stats()
    
    print(f"Buffer statistics: {buffer_stats}")
    
    # Verify statistics structure
    assert "version" in buffer_stats
    assert "write_buffer" in buffer_stats
    assert "query_buffer" in buffer_stats
    
    print("✅ Service statistics test passed")


async def run_all_tests():
    """Run all BufferService query simplification tests."""
    print("Starting BufferService Query Simplification Tests...")
    
    try:
        await test_simplified_query()
        await test_query_with_reranking()
        await test_session_query_delegation()
        await test_buffer_only_session_query()
        await test_service_statistics()
        
        print("\n🎉 All BufferService query simplification tests passed!")
        
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
        print("\n✅ BufferService query simplification implementation is working correctly!")
        exit(0)
    else:
        print("\n❌ BufferService query simplification tests failed!")
        exit(1)
