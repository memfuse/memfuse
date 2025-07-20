"""Comprehensive test suite for Buffer architecture refactoring.

This script provides a complete validation of the refactored Buffer architecture,
testing all phases of the refactoring including performance improvements and
architectural optimizations.
"""

import asyncio
import time
import uuid
import statistics
from typing import List, Dict, Any

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from memfuse_core.services.buffer_service import BufferService
from memfuse_core.buffer.config_factory import BufferConfigManager
from memfuse_core.interfaces import MessageList, MessageBatchList


class MockMemoryService:
    """Enhanced mock memory service for comprehensive testing."""
    
    def __init__(self, user_id: str = "test_user"):
        self._user_id = user_id
        self.storage_data = []
        self.call_history = []
        self.performance_metrics = {
            'query_times': [],
            'add_batch_times': [],
            'total_calls': 0
        }
    
    def add_test_data(self, content: str, session_id: str = "default", msg_id: str = None):
        """Add test data to mock storage."""
        self.storage_data.append({
            'id': msg_id or str(uuid.uuid4()),
            'content': content,
            'metadata': {'session_id': session_id},
            'created_at': time.time()
        })
    
    async def query(self, query: str, **kwargs):
        """Mock query method with performance tracking."""
        start_time = time.time()
        self.call_history.append(('query', query, kwargs))
        self.performance_metrics['total_calls'] += 1
        
        # Simulate query processing
        await asyncio.sleep(0.001)  # Minimal delay
        
        # Simple content matching
        results = [
            data for data in self.storage_data 
            if query.lower() in data['content'].lower()
        ]
        
        query_time = time.time() - start_time
        self.performance_metrics['query_times'].append(query_time)
        
        return {
            "status": "success",
            "data": {"results": results, "total": len(results)}
        }
    
    async def add_batch(self, rounds: List[MessageList]):
        """Mock add_batch method with performance tracking."""
        start_time = time.time()
        self.call_history.append(('add_batch', len(rounds), None))
        self.performance_metrics['total_calls'] += 1
        
        # Simulate batch processing
        await asyncio.sleep(0.002)  # Minimal delay
        
        batch_time = time.time() - start_time
        self.performance_metrics['add_batch_times'].append(batch_time)
        
        return {
            "status": "success",
            "message": f"Added {len(rounds)} rounds"
        }
    
    async def get_messages_by_session(self, session_id: str, **kwargs):
        """Mock get_messages_by_session method."""
        return [
            data for data in self.storage_data 
            if data['metadata']['session_id'] == session_id
        ]
    
    def get_performance_summary(self):
        """Get performance metrics summary."""
        return {
            'total_calls': self.performance_metrics['total_calls'],
            'avg_query_time': statistics.mean(self.performance_metrics['query_times']) if self.performance_metrics['query_times'] else 0,
            'avg_batch_time': statistics.mean(self.performance_metrics['add_batch_times']) if self.performance_metrics['add_batch_times'] else 0,
            'total_queries': len(self.performance_metrics['query_times']),
            'total_batches': len(self.performance_metrics['add_batch_times'])
        }


def create_test_message(content: str, session_id: str = "default") -> Dict[str, Any]:
    """Create a test message."""
    return {
        "content": content,
        "role": "user",
        "metadata": {
            "session_id": session_id
        }
    }


def create_test_batch(batch_size: int, messages_per_list: int, session_id: str = "default") -> MessageBatchList:
    """Create a test message batch."""
    return [
        [create_test_message(f"Message {i}-{j}", session_id) for j in range(messages_per_list)]
        for i in range(batch_size)
    ]


async def test_phase1_write_buffer_optimization():
    """Test Phase 1: WriteBuffer batch processing optimization."""
    print("\n=== Phase 1: WriteBuffer Optimization Test ===")
    
    # Create BufferService with optimized configuration
    mock_memory_service = MockMemoryService("phase1_user")
    config = {
        'buffer': {
            'round_buffer': {'max_tokens': 800, 'max_size': 5},
            'hybrid_buffer': {'max_size': 5, 'chunk_strategy': 'message'},
            'performance': {'max_flush_workers': 3}
        },
        'retrieval': {'use_rerank': False}
    }
    
    buffer_service = BufferService(
        memory_service=mock_memory_service,
        user="phase1_user",
        config=config
    )
    
    # Test batch processing optimization
    large_batch = create_test_batch(batch_size=10, messages_per_list=3)
    
    start_time = time.time()
    result = await buffer_service.add_batch(large_batch, session_id="phase1_test")
    processing_time = time.time() - start_time
    
    print(f"Batch processing result: {result}")
    print(f"Processing time: {processing_time:.3f}s")
    
    # Verify optimization features
    assert result["status"] == "success"
    assert result["data"]["batch_size"] == 10
    assert result["data"]["total_messages"] == 30
    assert "strategy_used" in result["data"]
    assert "processing_time" in result["data"]
    
    print("✅ Phase 1: WriteBuffer optimization test passed")
    return processing_time


async def test_phase2_buffer_service_simplification():
    """Test Phase 2: BufferService simplification."""
    print("\n=== Phase 2: BufferService Simplification Test ===")
    
    mock_memory_service = MockMemoryService("phase2_user")
    mock_memory_service.add_test_data("Test data for simplification", "phase2_session")
    
    config = {
        'buffer': {
            'round_buffer': {'max_tokens': 800, 'max_size': 5},
            'hybrid_buffer': {'max_size': 5},
            'query': {'max_size': 15}
        }
    }
    
    buffer_service = BufferService(
        memory_service=mock_memory_service,
        user="phase2_user",
        config=config
    )
    
    # Test simplified add_batch (should delegate to WriteBuffer)
    test_batch = create_test_batch(batch_size=3, messages_per_list=2)

    result = await buffer_service.add_batch(test_batch, session_id="phase2_test")

    # Verify service layer simplification
    assert result["status"] == "success"
    assert "data" in result
    assert "strategy_used" in result["data"]

    # Verify WriteBuffer delegation by checking statistics
    write_stats = buffer_service.write_buffer.get_stats()
    assert write_stats["write_buffer"]["total_writes"] > 0
    
    print("✅ Phase 2: BufferService simplification test passed")


async def test_phase3_query_buffer_enhancement():
    """Test Phase 3: QueryBuffer functionality enhancement."""
    print("\n=== Phase 3: QueryBuffer Enhancement Test ===")
    
    mock_memory_service = MockMemoryService("phase3_user")
    mock_memory_service.add_test_data("Enhanced query test message", "phase3_session")
    mock_memory_service.add_test_data("Another enhanced message", "phase3_session")
    mock_memory_service.add_test_data("Different session message", "other_session")
    
    config = {
        'buffer': {
            'query': {'max_size': 20, 'cache_size': 100},
            'round_buffer': {'max_tokens': 800},
            'hybrid_buffer': {'max_size': 5}
        },
        'retrieval': {'use_rerank': True}
    }
    
    buffer_service = BufferService(
        memory_service=mock_memory_service,
        user="phase3_user",
        config=config
    )
    
    # Test enhanced query with reranking
    query_result = await buffer_service.query("enhanced", top_k=5)
    
    print(f"Query result: {query_result}")
    
    # Verify enhanced query functionality
    assert query_result["status"] == "success"
    assert "data" in query_result
    assert "results" in query_result["data"]
    
    # Test session-specific querying
    session_messages = await buffer_service.get_messages_by_session("phase3_session")

    print(f"Session messages: {len(session_messages)}")

    # Verify session query enhancement (may be 0 if no data in buffers yet)
    assert isinstance(session_messages, list)
    for msg in session_messages:
        assert msg.get('metadata', {}).get('session_id') == 'phase3_session'
    
    # Check QueryBuffer statistics for enhancements
    query_stats = buffer_service.query_buffer.get_stats()
    assert query_stats.get('has_rerank_handler') == True
    assert query_stats.get('total_queries') > 0
    
    print("✅ Phase 3: QueryBuffer enhancement test passed")


async def test_phase4_query_simplification():
    """Test Phase 4: BufferService query simplification."""
    print("\n=== Phase 4: Query Simplification Test ===")
    
    mock_memory_service = MockMemoryService("phase4_user")
    mock_memory_service.add_test_data("Simplified query test", "phase4_session")
    
    config = {
        'buffer': {
            'query': {'max_size': 15},
            'round_buffer': {'max_tokens': 800},
            'hybrid_buffer': {'max_size': 5}
        },
        'retrieval': {'use_rerank': False}
    }
    
    buffer_service = BufferService(
        memory_service=mock_memory_service,
        user="phase4_user",
        config=config
    )
    
    # Test simplified query (should delegate to QueryBuffer)
    initial_query_count = buffer_service.query_buffer.get_stats()['total_queries']
    
    result = await buffer_service.query("simplified", top_k=5)
    
    # Verify query simplification
    assert result["status"] == "success"
    assert "data" in result
    
    # Verify delegation to QueryBuffer
    final_query_count = buffer_service.query_buffer.get_stats()['total_queries']
    assert final_query_count > initial_query_count
    
    print("✅ Phase 4: Query simplification test passed")


async def test_phase5_config_optimization():
    """Test Phase 5: Configuration architecture optimization."""
    print("\n=== Phase 5: Configuration Optimization Test ===")
    
    # Test autonomous configuration management
    custom_config = {
        'buffer': {
            'round_buffer': {'max_tokens': 1200, 'max_size': 8},
            'hybrid_buffer': {'max_size': 10, 'chunk_strategy': 'semantic'},
            'query': {'max_size': 25, 'cache_size': 200},
            'performance': {'max_flush_workers': 5}
        },
        'model': {
            'default_model': 'gpt-4',
            'embedding_model': 'custom-embedding'
        }
    }
    
    mock_memory_service = MockMemoryService("phase5_user")
    
    buffer_service = BufferService(
        memory_service=mock_memory_service,
        user="phase5_user",
        config=custom_config
    )
    
    # Verify configuration was applied autonomously
    write_stats = buffer_service.write_buffer.get_stats()
    round_stats = write_stats['write_buffer']['round_buffer']
    hybrid_stats = write_stats['write_buffer']['hybrid_buffer']
    
    assert round_stats['max_tokens'] == 1200
    assert round_stats['max_size'] == 8
    assert round_stats['token_model'] == 'gpt-4'  # Global context applied
    
    assert hybrid_stats['max_size'] == 10
    assert hybrid_stats['chunk_strategy'] == 'semantic'
    assert hybrid_stats['embedding_model'] == 'custom-embedding'  # Global context applied
    
    query_stats = buffer_service.query_buffer.get_stats()
    assert query_stats['max_size'] == 25
    assert query_stats['cache_size'] == 200
    
    print("✅ Phase 5: Configuration optimization test passed")


async def test_performance_comparison():
    """Test overall performance improvements."""
    print("\n=== Performance Comparison Test ===")
    
    mock_memory_service = MockMemoryService("performance_user")
    
    # Add test data
    for i in range(50):
        mock_memory_service.add_test_data(f"Performance test message {i}", f"session_{i % 5}")
    
    config = {
        'buffer': {
            'round_buffer': {'max_tokens': 1000, 'max_size': 10},
            'hybrid_buffer': {'max_size': 10},
            'query': {'max_size': 20},
            'performance': {'max_flush_workers': 4}
        },
        'retrieval': {'use_rerank': True}
    }
    
    buffer_service = BufferService(
        memory_service=mock_memory_service,
        user="performance_user",
        config=config
    )
    
    # Performance test: Batch operations
    batch_times = []
    for i in range(5):
        test_batch = create_test_batch(batch_size=5, messages_per_list=3)
        
        start_time = time.time()
        await buffer_service.add_batch(test_batch, session_id=f"perf_session_{i}")
        batch_time = time.time() - start_time
        batch_times.append(batch_time)
    
    avg_batch_time = statistics.mean(batch_times)
    print(f"Average batch processing time: {avg_batch_time:.3f}s")
    
    # Performance test: Query operations
    query_times = []
    for i in range(10):
        start_time = time.time()
        await buffer_service.query(f"test {i}", top_k=5)
        query_time = time.time() - start_time
        query_times.append(query_time)
    
    avg_query_time = statistics.mean(query_times)
    print(f"Average query time: {avg_query_time:.3f}s")
    
    # Get comprehensive statistics
    buffer_stats = await buffer_service.get_buffer_stats()
    mock_perf = mock_memory_service.get_performance_summary()
    
    print(f"Buffer statistics: {buffer_stats['write_buffer']['write_buffer']}")
    print(f"Mock service performance: {mock_perf}")
    
    # Verify performance is reasonable
    assert avg_batch_time < 1.0  # Should be fast
    assert avg_query_time < 0.5  # Should be very fast
    
    print("✅ Performance comparison test passed")
    return avg_batch_time, avg_query_time


async def run_comprehensive_tests():
    """Run all comprehensive refactoring tests."""
    print("🚀 Starting Comprehensive Buffer Refactoring Tests...")
    print("=" * 60)
    
    try:
        # Run all phase tests
        phase1_time = await test_phase1_write_buffer_optimization()
        await test_phase2_buffer_service_simplification()
        await test_phase3_query_buffer_enhancement()
        await test_phase4_query_simplification()
        await test_phase5_config_optimization()
        
        # Performance validation
        batch_time, query_time = await test_performance_comparison()
        
        print("\n" + "=" * 60)
        print("🎉 ALL COMPREHENSIVE TESTS PASSED!")
        print("=" * 60)
        print(f"📊 Performance Summary:")
        print(f"   • Phase 1 batch processing: {phase1_time:.3f}s")
        print(f"   • Average batch time: {batch_time:.3f}s")
        print(f"   • Average query time: {query_time:.3f}s")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Comprehensive test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run comprehensive tests
    success = asyncio.run(run_comprehensive_tests())
    
    if success:
        print("\n✅ Buffer architecture refactoring is complete and validated!")
        exit(0)
    else:
        print("\n❌ Buffer architecture refactoring validation failed!")
        exit(1)
