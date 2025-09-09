"""
Integration tests for M2 semantic fact extraction pipeline.

Tests end-to-end M2 processing workflows, database transaction integrity,
and multi-component interactions for the complete fact extraction system.
"""

import asyncio
import json
import pytest
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List

from src.memfuse_core.services.simplified_memory_service import SimplifiedMemoryService
from src.memfuse_core.models.core import M2Status, M2FactStatus, Fact


@pytest.mark.unit
@pytest.mark.services
@pytest.mark.integration
@pytest.mark.asyncio
class TestM2IntegrationPipeline:
    """Integration tests for complete M2 processing pipeline."""
    
    @pytest.fixture
    async def integration_service(self):
        """Create and initialize a SimplifiedMemoryService for integration testing."""
        service = SimplifiedMemoryService(
            user="integration_test_user",
            agent="integration_test_agent",
            cfg={
                'chunk_token_limit': 500,
                'min_chunk_tokens': 300,
                'max_chunk_tokens': 700
            }
        )
        await service.initialize()
        yield service
        await service.close()
    
    @pytest.fixture
    async def integration_test_data(self, integration_service):
        """Create comprehensive test data for integration testing."""
        session_id = str(uuid.uuid4())
        round_id = str(uuid.uuid4())
        
        # Create session in database
        from src.memfuse_core.services.sync_connection_pool import sync_connection_pool
        
        with sync_connection_pool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO sessions (id, user_id, agent_id, name, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO NOTHING
                """, (session_id, integration_service._user_id, 'integration_test_agent', 
                      'Integration Test Session', datetime.now(), datetime.now()))
                conn.commit()
        
        # Create rich test messages for M2 processing
        test_messages = [
            {
                "id": str(uuid.uuid4()),
                "content": "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data without being explicitly programmed.",
                "role": "user",
                "created_at": datetime.now() - timedelta(hours=2),
                "metadata": {
                    "session_id": session_id,
                    "user_id": integration_service._user_id,
                    "message_type": "educational"
                }
            },
            {
                "id": str(uuid.uuid4()),
                "content": "Deep learning uses neural networks with multiple hidden layers to process complex data patterns. It's particularly effective for image recognition and natural language processing.",
                "role": "assistant", 
                "created_at": datetime.now() - timedelta(hours=1, minutes=30),
                "metadata": {
                    "session_id": session_id,
                    "user_id": integration_service._user_id,
                    "message_type": "informational"
                }
            },
            {
                "id": str(uuid.uuid4()),
                "content": "Supervised learning algorithms require labeled training data to learn the mapping between inputs and desired outputs. Common examples include classification and regression tasks.",
                "role": "user",
                "created_at": datetime.now() - timedelta(hours=1),
                "metadata": {
                    "session_id": session_id,
                    "user_id": integration_service._user_id,
                    "message_type": "educational"
                }
            },
            {
                "id": str(uuid.uuid4()),
                "content": "Unsupervised learning, in contrast, works with unlabeled data to discover hidden patterns. Clustering and dimensionality reduction are typical unsupervised learning tasks.",
                "role": "assistant",
                "created_at": datetime.now() - timedelta(minutes=30),
                "metadata": {
                    "session_id": session_id,
                    "user_id": integration_service._user_id,
                    "message_type": "informational"
                }
            },
            {
                "id": str(uuid.uuid4()),
                "content": "Reinforcement learning is another major paradigm where agents learn through trial and error by receiving rewards or penalties for their actions in an environment.",
                "role": "user",
                "created_at": datetime.now(),
                "metadata": {
                    "session_id": session_id,
                    "user_id": integration_service._user_id,
                    "message_type": "educational"
                }
            }
        ]
        
        # Create M1 chunks from messages
        message_batch_list = [test_messages]
        result = await integration_service.add_batch(message_batch_list, session_id=session_id)
        
        if result['status'] != 'success':
            raise Exception(f"Failed to create integration test data: {result['message']}")
        
        return {
            'session_id': session_id,
            'user_id': integration_service._user_id,
            'messages': test_messages,
            'service': integration_service
        }
    
    async def test_m2_pipeline_error_recovery(self, integration_service, integration_test_data):
        """Test M2 pipeline handles errors gracefully and maintains data integrity."""
        user_id = integration_test_data['user_id']
        
        # Get a pending chunk for error testing
        pending_chunks = await integration_service._get_pending_m2_chunks(
            batch_size=1,
            user_id=user_id
        )
        
        assert len(pending_chunks) >= 1
        chunk_id = pending_chunks[0]
        
        # Lock chunk for processing
        lock_result = await integration_service._lock_chunk_for_m2_processing(chunk_id, user_id)
        assert lock_result is True
        
        # Simulate processing error
        error_message = "Simulated LLM processing error for integration testing"
        failure_result = await integration_service._mark_chunk_m2_failed(
            chunk_id=chunk_id,
            error=error_message,
            user_id=user_id
        )
        
        assert failure_result is True
        
        # Verify error state
        failed_chunk_data = await integration_service._get_m1_chunk(chunk_id, user_id)
        assert failed_chunk_data['m2_status'] == M2Status.FAILED.value
        assert failed_chunk_data['m2_processing_ended_at'] is not None
        
        # Verify chunk is not in pending list
        remaining_pending = await integration_service._get_pending_m2_chunks(
            batch_size=10,
            user_id=user_id
        )
        assert chunk_id not in remaining_pending
        
        # Test recovery: manually reset chunk to pending for retry
        from src.memfuse_core.services.sync_connection_pool import sync_connection_pool
        
        with sync_connection_pool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE m1_episodic 
                    SET m2_status = %s, m2_processing_started_at = NULL, m2_processing_ended_at = NULL
                    WHERE chunk_id = %s AND user_id = %s
                """, (M2Status.PENDING.value, chunk_id, user_id))
                conn.commit()
        
        # Verify chunk is back in pending list
        retry_pending = await integration_service._get_pending_m2_chunks(
            batch_size=10,
            user_id=user_id
        )
        assert chunk_id in retry_pending
        
        # Test successful retry
        retry_lock_result = await integration_service._lock_chunk_for_m2_processing(chunk_id, user_id)
        assert retry_lock_result is True
        
        retry_facts = [
            {
                'text': 'Retry extraction successful after initial failure.',
                'confidence': 0.85,
                'chunk_ids': [chunk_id],
                'metadata': {'extraction_attempt': 'retry'},
                'policy_version': 'v1.0'
            }
        ]
        
        retry_completion = await integration_service._mark_chunk_m2_completed(
            chunk_id=chunk_id,
            facts=retry_facts,
            user_id=user_id
        )
        
        assert retry_completion is True
        
        # Verify final successful state
        final_chunk_data = await integration_service._get_m1_chunk(chunk_id, user_id)
        assert final_chunk_data['m2_status'] == M2Status.COMPLETED.value
    
    async def test_m2_database_transaction_integrity(self, integration_service, integration_test_data):
        """Test M2 operations maintain database transaction integrity."""
        user_id = integration_test_data['user_id']
        
        # Get a pending chunk
        pending_chunks = await integration_service._get_pending_m2_chunks(
            batch_size=1,
            user_id=user_id
        )
        
        assert len(pending_chunks) >= 1
        chunk_id = pending_chunks[0]
        
        # Test transaction integrity during completion
        lock_result = await integration_service._lock_chunk_for_m2_processing(chunk_id, user_id)
        assert lock_result is True
        
        # Create test facts
        test_facts = [
            {
                'text': 'Transaction integrity test fact one.',
                'confidence': 0.90,
                'chunk_ids': [chunk_id],
                'metadata': {'test': 'transaction_integrity'},
                'policy_version': 'v1.0'
            },
            {
                'text': 'Transaction integrity test fact two.',
                'confidence': 0.88,
                'chunk_ids': [chunk_id],
                'metadata': {'test': 'transaction_integrity'},
                'policy_version': 'v1.0'
            }
        ]
        
        # Complete processing with multiple facts
        completion_result = await integration_service._mark_chunk_m2_completed(
            chunk_id=chunk_id,
            facts=test_facts,
            user_id=user_id
        )
        
        assert completion_result is True
        
        # Verify both chunk status update and fact storage happened atomically
        from src.memfuse_core.services.sync_connection_pool import sync_connection_pool
        
        with sync_connection_pool.get_connection() as conn:
            with conn.cursor() as cur:
                # Verify chunk status
                cur.execute("""
                    SELECT m2_status, m2_processing_ended_at FROM m1_episodic 
                    WHERE chunk_id = %s AND user_id = %s
                """, (chunk_id, user_id))
                
                chunk_row = cur.fetchone()
                assert chunk_row is not None
                assert chunk_row[0] == M2Status.COMPLETED.value
                assert chunk_row[1] is not None
                
                # Verify facts were stored
                cur.execute("""
                    SELECT COUNT(*) FROM m2_semantic 
                    WHERE %s = ANY(chunk_ids) AND user_id = %s
                """, (chunk_id, user_id))
                
                fact_count = cur.fetchone()[0]
                assert fact_count == 2  # Both facts should be stored
    
    async def test_m2_concurrent_processing_safety(self, integration_service, integration_test_data):
        """Test M2 pipeline handles concurrent processing attempts safely."""
        user_id = integration_test_data['user_id']
        
        # Get a pending chunk
        pending_chunks = await integration_service._get_pending_m2_chunks(
            batch_size=1,
            user_id=user_id
        )
        
        assert len(pending_chunks) >= 1
        chunk_id = pending_chunks[0]
        
        # First lock should succeed
        first_lock = await integration_service._lock_chunk_for_m2_processing(chunk_id, user_id)
        assert first_lock is True
        
        # Second concurrent lock attempt should fail
        second_lock = await integration_service._lock_chunk_for_m2_processing(chunk_id, user_id)
        assert second_lock is False
        
        # First process completes successfully
        test_facts = [
            {
                'text': 'Concurrent processing safety test fact.',
                'confidence': 0.85,
                'chunk_ids': [chunk_id],
                'metadata': {'test': 'concurrent_safety'},
                'policy_version': 'v1.0'
            }
        ]
        
        completion_result = await integration_service._mark_chunk_m2_completed(
            chunk_id=chunk_id,
            facts=test_facts,
            user_id=user_id
        )
        
        assert completion_result is True
        
        # Attempt to lock completed chunk should fail
        post_completion_lock = await integration_service._lock_chunk_for_m2_processing(chunk_id, user_id)
        assert post_completion_lock is False
        
        # Verify final state is correct
        final_chunk_data = await integration_service._get_m1_chunk(chunk_id, user_id)
        assert final_chunk_data['m2_status'] == M2Status.COMPLETED.value


@pytest.mark.unit
@pytest.mark.services
@pytest.mark.integration
@pytest.mark.asyncio  
class TestM2PerformanceAndScaling:
    """Integration tests for M2 pipeline performance and scaling characteristics."""
    
    @pytest.fixture
    async def performance_service(self):
        """Create a service optimized for performance testing."""
        service = SimplifiedMemoryService(
            user="performance_test_user",
            agent="performance_test_agent",
            cfg={
                'chunk_token_limit': 400,  # Smaller chunks for performance testing
                'min_chunk_tokens': 200,
                'max_chunk_tokens': 500
            }
        )
        await service.initialize()
        yield service
        await service.close()
        

if __name__ == '__main__':
    pytest.main([__file__, '-v'])