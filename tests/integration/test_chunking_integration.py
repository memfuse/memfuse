"""Integration tests for chunking system components."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from memfuse_core.rag.chunk import MessageChunkStrategy, ContextualChunkStrategy, CharacterChunkStrategy
from memfuse_core.rag.chunk.base import ChunkData


class TestChunkingIntegration:
    """Integration tests for chunking system."""
    
    @pytest.mark.integration
    @pytest.mark.chunking
    @pytest.mark.asyncio
    async def test_memory_service_with_message_chunk_strategy(self, mock_memory_service, sample_message_batch):
        """Test MemoryService integration with MessageChunkStrategy."""
        
        # Set up the strategy
        mock_memory_service.chunk_strategy = MessageChunkStrategy()
        
        # Test add_batch method
        result = await mock_memory_service.add_batch(sample_message_batch)
        
        # Verify the result structure
        assert "status" in result
        assert "data" in result
        
        # Verify that chunk strategy was called
        # (This would be verified through the actual chunking process)
    
    @pytest.mark.integration
    @pytest.mark.chunking
    @pytest.mark.asyncio
    async def test_memory_service_with_contextual_chunk_strategy(self, mock_memory_service, sample_message_batch):
        """Test MemoryService integration with ContextualChunkStrategy."""
        
        # Set up the strategy with short max length to force splitting
        mock_memory_service.chunk_strategy = ContextualChunkStrategy(max_chunk_length=50)
        
        # Create longer content to test splitting
        long_message_batch = [
            [
                {
                    "role": "user", 
                    "content": "This is a very long message that should definitely exceed the maximum chunk length and be split into multiple chunks for proper processing and retrieval."
                },
                {
                    "role": "assistant",
                    "content": "This is also a long response that provides detailed information and should also be split appropriately."
                }
            ]
        ]
        
        # Test add_batch method
        result = await mock_memory_service.add_batch(long_message_batch)
        
        # Verify the result structure
        assert "status" in result
        assert "data" in result
    
    @pytest.mark.integration
    @pytest.mark.chunking
    @pytest.mark.asyncio
    async def test_memory_service_with_character_chunk_strategy(self, mock_memory_service, sample_message_batch):
        """Test MemoryService integration with CharacterChunkStrategy."""
        
        # Set up the strategy
        mock_memory_service.chunk_strategy = CharacterChunkStrategy(
            max_chunk_length=100, 
            overlap_length=20
        )
        
        # Test add_batch method
        result = await mock_memory_service.add_batch(sample_message_batch)
        
        # Verify the result structure
        assert "status" in result
        assert "data" in result
    
    @pytest.mark.integration
    @pytest.mark.chunking
    @pytest.mark.asyncio
    async def test_buffer_service_chunking_integration(self, mock_buffer_service, sample_messages):
        """Test BufferService integration with chunking."""
        
        # Test adding messages through BufferService
        result = await mock_buffer_service.add(sample_messages)
        
        # Verify the result
        assert "status" in result
        
        # Test querying through BufferService
        query_result = await mock_buffer_service.query("space exploration", top_k=5)
        
        # Verify query result structure
        assert "status" in query_result
        assert "data" in query_result
    
    @pytest.mark.integration
    @pytest.mark.chunking
    @pytest.mark.asyncio
    async def test_chunk_strategy_with_stores(self, mock_vector_store, mock_keyword_store, mock_graph_store):
        """Test chunk strategy integration with different stores."""
        
        # Create chunks using different strategies
        message_batch = [
            [
                {"role": "user", "content": "Test message for store integration"},
                {"role": "assistant", "content": "Response for store integration testing"}
            ]
        ]
        
        strategies = [
            MessageChunkStrategy(),
            ContextualChunkStrategy(max_chunk_length=100),
            CharacterChunkStrategy(max_chunk_length=50, overlap_length=10)
        ]
        
        for strategy in strategies:
            chunks = await strategy.create_chunks(message_batch)
            
            # Verify chunks were created
            assert len(chunks) > 0
            assert all(isinstance(chunk, ChunkData) for chunk in chunks)
            
            # Simulate storing chunks (would normally be done by MemoryService)
            # This tests that chunks have the right structure for stores
            for chunk in chunks:
                assert hasattr(chunk, 'content')
                assert hasattr(chunk, 'chunk_id')
                assert hasattr(chunk, 'metadata')
                assert isinstance(chunk.metadata, dict)
    
    @pytest.mark.integration
    @pytest.mark.chunking
    @pytest.mark.asyncio
    async def test_chunking_with_different_message_formats(self):
        """Test chunking with various message formats."""
        
        # Test different message formats
        test_cases = [
            # Standard format
            [
                [
                    {"role": "user", "content": "Standard message"},
                    {"role": "assistant", "content": "Standard response"}
                ]
            ],
            # With metadata
            [
                [
                    {
                        "role": "user", 
                        "content": "Message with metadata",
                        "metadata": {"session_id": "test", "timestamp": "2024-01-01"}
                    }
                ]
            ],
            # Dict content
            [
                [
                    {
                        "role": "user",
                        "content": {"text": "Dict content format"}
                    }
                ]
            ],
            # Missing role
            [
                [
                    {"content": "Message without role"}
                ]
            ]
        ]
        
        strategy = MessageChunkStrategy()
        
        for message_batch in test_cases:
            chunks = await strategy.create_chunks(message_batch)
            
            # Should handle all formats gracefully
            assert isinstance(chunks, list)
            
            # If chunks were created, they should be valid
            for chunk in chunks:
                assert isinstance(chunk, ChunkData)
                assert chunk.content is not None
                assert chunk.chunk_id is not None
                assert isinstance(chunk.metadata, dict)
    
    @pytest.mark.integration
    @pytest.mark.chunking
    @pytest.mark.asyncio
    async def test_chunking_performance_with_large_batches(self):
        """Test chunking performance with large message batches."""
        
        # Create a large batch of messages
        large_batch = []
        for i in range(50):  # 50 MessageLists
            message_list = [
                {"role": "user", "content": f"User message {i} with some content"},
                {"role": "assistant", "content": f"Assistant response {i} with detailed information"}
            ]
            large_batch.append(message_list)
        
        strategies = [
            MessageChunkStrategy(),
            ContextualChunkStrategy(max_chunk_length=200),
            CharacterChunkStrategy(max_chunk_length=150, overlap_length=30)
        ]
        
        for strategy in strategies:
            chunks = await strategy.create_chunks(large_batch)
            
            # Verify chunks were created
            assert len(chunks) > 0
            
            # Verify all chunks are valid
            for chunk in chunks:
                assert isinstance(chunk, ChunkData)
                assert len(chunk.content) > 0
                assert chunk.chunk_id is not None
                assert "strategy" in chunk.metadata
            
            # Verify chunk IDs are unique
            chunk_ids = [chunk.chunk_id for chunk in chunks]
            assert len(chunk_ids) == len(set(chunk_ids))
    
    @pytest.mark.integration
    @pytest.mark.chunking
    @pytest.mark.asyncio
    async def test_chunking_error_handling(self):
        """Test chunking error handling with invalid inputs."""
        
        strategy = MessageChunkStrategy()
        
        # Test with empty input
        chunks = await strategy.create_chunks([])
        assert chunks == []
        
        # Test with None content
        message_batch_with_none = [
            [
                {"role": "user", "content": None},
                {"role": "assistant", "content": "Valid content"}
            ]
        ]
        
        chunks = await strategy.create_chunks(message_batch_with_none)
        assert len(chunks) == 1
        assert "Valid content" in chunks[0].content
        
        # Test with empty strings
        message_batch_with_empty = [
            [
                {"role": "user", "content": ""},
                {"role": "assistant", "content": "Non-empty content"}
            ]
        ]
        
        chunks = await strategy.create_chunks(message_batch_with_empty)
        assert len(chunks) == 1
        assert "Non-empty content" in chunks[0].content
