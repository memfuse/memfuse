#!/usr/bin/env python3
"""Test enhanced chunks functionality with session_id and round_id metadata."""

import pytest
import asyncio
import uuid
from datetime import datetime
from typing import List, Dict, Any

# Import test utilities
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from memfuse_core.rag.chunk.base import ChunkData
from memfuse_core.rag.chunk.message import MessageChunkStrategy
from memfuse_core.services.memory_service import MemoryService


class TestEnhancedChunks:
    """Test suite for enhanced chunks functionality."""

    @pytest.fixture
    def sample_message_batch_list(self):
        """Create sample message batch list for testing."""
        session_id = f"test_session_{uuid.uuid4()}"
        return [
            [
                {
                    'role': 'user',
                    'content': 'Hello, how are you today?',
                    'metadata': {'session_id': session_id}
                },
                {
                    'role': 'assistant', 
                    'content': 'I am doing well, thank you for asking!',
                    'metadata': {'session_id': session_id}
                }
            ],
            [
                {
                    'role': 'user',
                    'content': 'What is the weather like?',
                    'metadata': {'session_id': session_id}
                },
                {
                    'role': 'assistant',
                    'content': 'I do not have access to current weather information.',
                    'metadata': {'session_id': session_id}
                }
            ]
        ]

    @pytest.fixture
    def memory_service(self):
        """Create a MemoryService instance for testing."""
        return MemoryService(
            user="test_user",
            agent="test_agent",
            session_id="test_session"
        )

    def test_session_and_round_preparation(self, sample_message_batch_list):
        """Test the _prepare_session_and_round method."""
        async def run_test():
            memory_service = MemoryService()
            
            # Test session_id and round_id preparation
            session_id, round_id = await memory_service._prepare_session_and_round(sample_message_batch_list)
            
            # Verify session_id is extracted correctly
            expected_session_id = sample_message_batch_list[0][0]['metadata']['session_id']
            assert session_id == expected_session_id
            
            # Verify round_id is a valid UUID
            assert isinstance(round_id, str)
            assert len(round_id) == 36  # UUID length
            
            # Verify round_id is unique on multiple calls
            session_id2, round_id2 = await memory_service._prepare_session_and_round(sample_message_batch_list)
            assert session_id == session_id2  # Same session
            assert round_id != round_id2      # Different rounds

        asyncio.run(run_test())

    def test_chunk_creation_with_strategy(self, sample_message_batch_list):
        """Test chunk creation using MessageChunkStrategy."""
        async def run_test():
            strategy = MessageChunkStrategy()
            chunks = await strategy.create_chunks(sample_message_batch_list)

            # Verify chunks are created (should match number of MessageLists in batch)
            expected_chunks = len(sample_message_batch_list)
            assert len(chunks) == expected_chunks, f"Expected {expected_chunks} chunks, got {len(chunks)}"

            # Verify chunk structure
            for chunk in chunks:
                assert isinstance(chunk, ChunkData)
                assert chunk.chunk_id is not None
                assert chunk.content is not None
                assert isinstance(chunk.metadata, dict)

                # Verify strategy metadata
                assert chunk.metadata.get("strategy") == "message"
                assert "message_count" in chunk.metadata
                assert "roles" in chunk.metadata

        asyncio.run(run_test())

    def test_enhanced_chunk_metadata(self, sample_message_batch_list):
        """Test enhanced chunk metadata with session_id and round_id."""
        async def run_test():
            memory_service = MemoryService()
            
            # Prepare session and round
            session_id, round_id = await memory_service._prepare_session_and_round(sample_message_batch_list)
            
            # Create chunks
            strategy = MessageChunkStrategy()
            chunks = await strategy.create_chunks(sample_message_batch_list)
            
            # Enhance chunks with metadata
            enhanced_chunks = []
            for chunk in chunks:
                enhanced_chunk = ChunkData(
                    content=chunk.content,
                    chunk_id=chunk.chunk_id,
                    metadata={
                        **chunk.metadata,  # Original metadata
                        "type": "chunk",
                        "session_id": session_id,
                        "round_id": round_id,
                        "user_id": "test_user_123",
                        "agent_id": "test_agent_456",
                        "created_at": datetime.now().isoformat(),
                    }
                )
                enhanced_chunks.append(enhanced_chunk)
            
            # Verify enhanced metadata
            for chunk in enhanced_chunks:
                assert chunk.metadata.get("session_id") == session_id
                assert chunk.metadata.get("round_id") == round_id
                assert chunk.metadata.get("user_id") == "test_user_123"
                assert chunk.metadata.get("agent_id") == "test_agent_456"
                assert chunk.metadata.get("type") == "chunk"
                assert "created_at" in chunk.metadata
                
                # Verify original strategy metadata is preserved
                assert chunk.metadata.get("strategy") == "message"
                assert "message_count" in chunk.metadata

        asyncio.run(run_test())

    def test_parallel_processing_simulation(self, sample_message_batch_list):
        """Test simulation of parallel processing (messages + chunks)."""
        async def run_test():
            memory_service = MemoryService()
            
            # Prepare session and round
            session_id, round_id = await memory_service._prepare_session_and_round(sample_message_batch_list)
            
            # Simulate parallel tasks
            async def simulate_message_storage():
                """Simulate message storage task."""
                await asyncio.sleep(0.1)  # Simulate DB write time
                return [f"msg_{i}" for i in range(len(sample_message_batch_list))]
            
            async def simulate_chunk_processing():
                """Simulate chunk processing task."""
                strategy = MessageChunkStrategy()
                chunks = await strategy.create_chunks(sample_message_batch_list)
                
                # Enhance chunks
                enhanced_chunks = []
                for chunk in chunks:
                    enhanced_chunk = ChunkData(
                        content=chunk.content,
                        chunk_id=chunk.chunk_id,
                        metadata={
                            **chunk.metadata,
                            "session_id": session_id,
                            "round_id": round_id,
                            "type": "chunk",
                        }
                    )
                    enhanced_chunks.append(enhanced_chunk)
                
                return enhanced_chunks
            
            # Execute tasks in parallel
            start_time = asyncio.get_event_loop().time()
            message_ids, chunks = await asyncio.gather(
                simulate_message_storage(),
                simulate_chunk_processing()
            )
            end_time = asyncio.get_event_loop().time()
            
            # Verify results
            expected_count = len(sample_message_batch_list)
            assert len(message_ids) == expected_count
            assert len(chunks) == expected_count
            
            # Verify parallel execution was faster than sequential
            # (This is a simple check - in real scenarios the benefit would be more significant)
            assert end_time - start_time < 0.2  # Should complete in less than 200ms
            
            # Verify chunks have correct metadata
            for chunk in chunks:
                assert chunk.metadata.get("session_id") == session_id
                assert chunk.metadata.get("round_id") == round_id

        asyncio.run(run_test())

    def test_error_handling(self):
        """Test error handling in session/round preparation."""
        async def run_test():
            memory_service = MemoryService()
            
            # Test with empty message batch list
            with pytest.raises(ValueError, match="No messages found"):
                await memory_service._prepare_session_and_round([])
            
            # Test with messages without session_id
            invalid_batch = [
                [
                    {'role': 'user', 'content': 'Hello'},  # No metadata
                    {'role': 'assistant', 'content': 'Hi'}
                ]
            ]
            
            with pytest.raises(ValueError, match="No session_id found"):
                await memory_service._prepare_session_and_round(invalid_batch)

        asyncio.run(run_test())

    def test_chunk_metadata_consistency(self, sample_message_batch_list):
        """Test that chunk metadata is consistent across multiple creations."""
        async def run_test():
            memory_service = MemoryService()
            
            # Create chunks multiple times with same session/round
            session_id, round_id = await memory_service._prepare_session_and_round(sample_message_batch_list)
            
            strategy = MessageChunkStrategy()
            chunks1 = await strategy.create_chunks(sample_message_batch_list)
            chunks2 = await strategy.create_chunks(sample_message_batch_list)
            
            # Verify chunks have consistent structure
            assert len(chunks1) == len(chunks2)
            
            for chunk1, chunk2 in zip(chunks1, chunks2):
                # Content should be the same
                assert chunk1.content == chunk2.content
                
                # Metadata structure should be the same
                assert chunk1.metadata.get("strategy") == chunk2.metadata.get("strategy")
                assert chunk1.metadata.get("message_count") == chunk2.metadata.get("message_count")
                
                # Chunk IDs should be different (unique)
                assert chunk1.chunk_id != chunk2.chunk_id

        asyncio.run(run_test())


if __name__ == "__main__":
    # Run tests directly
    test_instance = TestEnhancedChunks()
    
    # Create sample data
    sample_data = [
        [
            {
                'role': 'user',
                'content': 'Hello, how are you today?',
                'metadata': {'session_id': 'test_session_123'}
            },
            {
                'role': 'assistant', 
                'content': 'I am doing well, thank you for asking!',
                'metadata': {'session_id': 'test_session_123'}
            }
        ]
    ]
    
    print("🧪 Running Enhanced Chunks Tests")
    print("=" * 50)
    
    try:
        test_instance.test_session_and_round_preparation(sample_data)
        print("✅ Session and round preparation test passed")
        
        test_instance.test_chunk_creation_with_strategy(sample_data)
        print("✅ Chunk creation test passed")
        
        test_instance.test_enhanced_chunk_metadata(sample_data)
        print("✅ Enhanced metadata test passed")
        
        test_instance.test_parallel_processing_simulation(sample_data)
        print("✅ Parallel processing test passed")
        
        test_instance.test_error_handling()
        print("✅ Error handling test passed")
        
        test_instance.test_chunk_metadata_consistency(sample_data)
        print("✅ Metadata consistency test passed")
        
        print("\n🎉 All tests passed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
