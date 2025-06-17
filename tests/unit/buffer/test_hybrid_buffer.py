"""Tests for HybridBuffer in Buffer."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from memfuse_core.buffer.hybrid_buffer import HybridBuffer
from memfuse_core.rag.chunk.base import ChunkData


@pytest.fixture
def sample_rounds():
    """Fixture providing sample rounds for testing."""
    return [
        [
            {
                "id": "msg_1",
                "role": "user", 
                "content": "Hello, how are you?",
                "metadata": {"session_id": "session_1"}
            },
            {
                "id": "msg_2",
                "role": "assistant",
                "content": "I'm doing well, thank you!",
                "metadata": {"session_id": "session_1"}
            }
        ],
        [
            {
                "id": "msg_3",
                "role": "user",
                "content": "What's the weather like?",
                "metadata": {"session_id": "session_1"}
            }
        ]
    ]


@pytest.fixture
def mock_chunk_strategy():
    """Fixture providing a mock chunk strategy."""
    strategy = MagicMock()
    strategy.create_chunks = AsyncMock(return_value=[
        ChunkData(content="Test chunk 1", metadata={"strategy": "test"}),
        ChunkData(content="Test chunk 2", metadata={"strategy": "test"})
    ])
    return strategy


@pytest.fixture
def mock_embedding_model():
    """Fixture providing a mock embedding model."""
    async def mock_create_embedding(text, model=None):
        # Return a simple embedding based on text length
        return [0.1] * min(384, len(text))
    
    return mock_create_embedding


class TestHybridBufferInitialization:
    """Test cases for HybridBuffer initialization."""
    
    def test_default_initialization(self):
        """Test HybridBuffer initialization with default parameters."""
        buffer = HybridBuffer()
        
        assert buffer.max_size == 5
        assert buffer.chunk_strategy_name == "message"
        assert buffer.embedding_model_name == "all-MiniLM-L6-v2"
        assert buffer.chunks == []
        assert buffer.embeddings == []
        assert buffer.original_rounds == []
        assert buffer.chunk_strategy is None
        assert buffer.embedding_model is None
        assert buffer.sqlite_handler is None
        assert buffer.qdrant_handler is None
    
    def test_custom_initialization(self):
        """Test HybridBuffer initialization with custom parameters."""
        buffer = HybridBuffer(
            max_size=10,
            chunk_strategy="contextual",
            embedding_model="custom-model"
        )
        
        assert buffer.max_size == 10
        assert buffer.chunk_strategy_name == "contextual"
        assert buffer.embedding_model_name == "custom-model"
    
    def test_set_storage_handlers(self):
        """Test setting storage handlers."""
        buffer = HybridBuffer()
        sqlite_handler = AsyncMock()
        qdrant_handler = AsyncMock()
        
        buffer.set_storage_handlers(sqlite_handler, qdrant_handler)
        
        assert buffer.sqlite_handler == sqlite_handler
        assert buffer.qdrant_handler == qdrant_handler


class TestHybridBufferChunkStrategy:
    """Test cases for chunk strategy loading and usage."""
    
    @pytest.mark.asyncio
    async def test_load_message_chunk_strategy(self):
        """Test loading message chunk strategy."""
        buffer = HybridBuffer(chunk_strategy="message")
        
        with patch('memfuse_core.buffer.hybrid_buffer.MessageChunkStrategy') as mock_strategy_class:
            mock_strategy = MagicMock()
            mock_strategy_class.return_value = mock_strategy
            
            await buffer._load_chunk_strategy()
            
            assert buffer.chunk_strategy == mock_strategy
            mock_strategy_class.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_load_contextual_chunk_strategy(self):
        """Test loading contextual chunk strategy."""
        buffer = HybridBuffer(chunk_strategy="contextual")
        
        with patch('memfuse_core.buffer.hybrid_buffer.ContextualChunkStrategy') as mock_strategy_class:
            mock_strategy = MagicMock()
            mock_strategy_class.return_value = mock_strategy
            
            await buffer._load_chunk_strategy()
            
            assert buffer.chunk_strategy == mock_strategy
            mock_strategy_class.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_load_unknown_chunk_strategy(self):
        """Test loading unknown chunk strategy falls back to message."""
        buffer = HybridBuffer(chunk_strategy="unknown")
        
        with patch('memfuse_core.buffer.hybrid_buffer.MessageChunkStrategy') as mock_strategy_class:
            mock_strategy = MagicMock()
            mock_strategy_class.return_value = mock_strategy
            
            await buffer._load_chunk_strategy()
            
            assert buffer.chunk_strategy == mock_strategy
            mock_strategy_class.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_load_chunk_strategy_error_fallback(self):
        """Test fallback when chunk strategy loading fails."""
        buffer = HybridBuffer()
        
        with patch('memfuse_core.buffer.hybrid_buffer.MessageChunkStrategy', side_effect=ImportError("Module not found")):
            await buffer._load_chunk_strategy()
            
            # Should create fallback strategy
            assert buffer.chunk_strategy is not None
            assert hasattr(buffer.chunk_strategy, 'create_chunks')


class TestHybridBufferEmbeddingModel:
    """Test cases for embedding model loading and usage."""
    
    @pytest.mark.asyncio
    async def test_load_embedding_model(self):
        """Test loading embedding model."""
        buffer = HybridBuffer()
        
        with patch('memfuse_core.buffer.hybrid_buffer.create_embedding') as mock_create_embedding:
            await buffer._load_embedding_model()
            
            assert buffer.embedding_model == mock_create_embedding
    
    @pytest.mark.asyncio
    async def test_load_embedding_model_error_fallback(self):
        """Test fallback when embedding model loading fails."""
        buffer = HybridBuffer()
        
        with patch('memfuse_core.buffer.hybrid_buffer.create_embedding', side_effect=ImportError("Module not found")):
            await buffer._load_embedding_model()
            
            # Should use fallback embedding
            assert buffer.embedding_model == buffer._create_fallback_embedding
    
    @pytest.mark.asyncio
    async def test_generate_embedding_success(self, mock_embedding_model):
        """Test successful embedding generation."""
        buffer = HybridBuffer()
        buffer.embedding_model = mock_embedding_model
        
        result = await buffer._generate_embedding("test text")
        
        assert isinstance(result, list)
        assert len(result) > 0
    
    @pytest.mark.asyncio
    async def test_generate_embedding_error_fallback(self):
        """Test embedding generation error fallback."""
        buffer = HybridBuffer()
        buffer.embedding_model = AsyncMock(side_effect=Exception("Embedding failed"))
        
        result = await buffer._generate_embedding("test text")
        
        # Should return fallback embedding
        assert isinstance(result, list)
        assert len(result) == 384  # Default embedding dimension
    
    @pytest.mark.asyncio
    async def test_create_fallback_embedding(self):
        """Test fallback embedding creation."""
        buffer = HybridBuffer()
        
        result = await buffer._create_fallback_embedding("test text")
        
        assert isinstance(result, list)
        assert len(result) == 384
        assert all(isinstance(x, float) for x in result)


class TestHybridBufferAddRounds:
    """Test cases for adding rounds from RoundBuffer."""
    
    @pytest.mark.asyncio
    async def test_add_from_rounds_basic(self, sample_rounds, mock_chunk_strategy, mock_embedding_model):
        """Test basic round addition."""
        buffer = HybridBuffer(max_size=10)
        buffer.chunk_strategy = mock_chunk_strategy
        buffer.embedding_model = mock_embedding_model
        
        await buffer.add_from_rounds(sample_rounds)
        
        assert len(buffer.original_rounds) == 2  # Two rounds added
        assert len(buffer.chunks) == 4  # 2 chunks per round
        assert len(buffer.embeddings) == 4  # One embedding per chunk
        assert buffer.total_rounds_received == 2
        assert buffer.total_chunks_created == 4
    
    @pytest.mark.asyncio
    async def test_add_from_rounds_empty(self):
        """Test adding empty rounds list."""
        buffer = HybridBuffer()
        
        await buffer.add_from_rounds([])
        
        assert len(buffer.original_rounds) == 0
        assert len(buffer.chunks) == 0
        assert len(buffer.embeddings) == 0
        assert buffer.total_rounds_received == 0
    
    @pytest.mark.asyncio
    async def test_add_from_rounds_fifo_removal(self, sample_rounds, mock_chunk_strategy, mock_embedding_model):
        """Test FIFO removal when buffer exceeds max_size."""
        buffer = HybridBuffer(max_size=2)
        buffer.chunk_strategy = mock_chunk_strategy
        buffer.embedding_model = mock_embedding_model
        
        # Add rounds that will create more chunks than max_size
        await buffer.add_from_rounds(sample_rounds)
        
        # Should have exactly max_size chunks (FIFO removal)
        assert len(buffer.chunks) == 2
        assert len(buffer.embeddings) == 2
        assert len(buffer.original_rounds) == 2  # One round per chunk
        assert buffer.total_fifo_removals == 2  # Two oldest items removed
    
    @pytest.mark.asyncio
    async def test_add_from_rounds_chunk_creation_error(self, sample_rounds, mock_embedding_model):
        """Test handling chunk creation errors."""
        buffer = HybridBuffer()
        
        # Mock chunk strategy that fails
        mock_strategy = MagicMock()
        mock_strategy.create_chunks = AsyncMock(side_effect=Exception("Chunk creation failed"))
        buffer.chunk_strategy = mock_strategy
        buffer.embedding_model = mock_embedding_model
        
        await buffer.add_from_rounds(sample_rounds)
        
        # Should still add original rounds even if chunking fails
        assert len(buffer.original_rounds) == 2
        assert len(buffer.chunks) == 0  # No chunks created due to error
        assert buffer.total_rounds_received == 2
    
    @pytest.mark.asyncio
    async def test_add_from_rounds_lazy_loading(self, sample_rounds):
        """Test lazy loading of chunk strategy and embedding model."""
        buffer = HybridBuffer()
        
        with patch.object(buffer, '_load_chunk_strategy', new_callable=AsyncMock) as mock_load_strategy:
            with patch.object(buffer, '_load_embedding_model', new_callable=AsyncMock) as mock_load_embedding:
                # Mock the loaded components
                buffer.chunk_strategy = mock_chunk_strategy
                buffer.embedding_model = mock_embedding_model
                
                await buffer.add_from_rounds(sample_rounds)
                
                mock_load_strategy.assert_called_once()
                mock_load_embedding.assert_called_once()


class TestHybridBufferStorage:
    """Test cases for storage operations."""
    
    @pytest.mark.asyncio
    async def test_flush_to_storage_success(self, sample_rounds, mock_chunk_strategy, mock_embedding_model):
        """Test successful flush to storage."""
        buffer = HybridBuffer()
        buffer.chunk_strategy = mock_chunk_strategy
        buffer.embedding_model = mock_embedding_model
        
        sqlite_handler = AsyncMock()
        qdrant_handler = AsyncMock()
        buffer.set_storage_handlers(sqlite_handler, qdrant_handler)
        
        await buffer.add_from_rounds(sample_rounds)
        
        result = await buffer.flush_to_storage()
        
        assert result is True
        sqlite_handler.assert_called_once()
        qdrant_handler.assert_called_once()
        assert len(buffer.original_rounds) == 0  # Cleared after flush
        assert buffer.total_flushes == 1
    
    @pytest.mark.asyncio
    async def test_flush_to_storage_empty_buffer(self):
        """Test flushing empty buffer."""
        buffer = HybridBuffer()
        
        result = await buffer.flush_to_storage()
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_flush_to_storage_no_handlers(self, sample_rounds, mock_chunk_strategy, mock_embedding_model):
        """Test flushing without storage handlers."""
        buffer = HybridBuffer()
        buffer.chunk_strategy = mock_chunk_strategy
        buffer.embedding_model = mock_embedding_model
        
        await buffer.add_from_rounds(sample_rounds)
        
        result = await buffer.flush_to_storage()
        
        assert result is True  # Should succeed even without handlers
        assert len(buffer.original_rounds) == 0  # Still cleared
    
    @pytest.mark.asyncio
    async def test_flush_to_storage_error(self, sample_rounds, mock_chunk_strategy, mock_embedding_model):
        """Test handling storage errors during flush."""
        buffer = HybridBuffer()
        buffer.chunk_strategy = mock_chunk_strategy
        buffer.embedding_model = mock_embedding_model
        
        sqlite_handler = AsyncMock(side_effect=Exception("SQLite error"))
        buffer.set_storage_handlers(sqlite_handler, None)
        
        await buffer.add_from_rounds(sample_rounds)
        
        result = await buffer.flush_to_storage()
        
        assert result is False
        assert len(buffer.original_rounds) > 0  # Not cleared on error
        assert buffer.total_flushes == 0
    
    @pytest.mark.asyncio
    async def test_write_to_sqlite(self, sample_rounds):
        """Test writing to SQLite."""
        buffer = HybridBuffer()
        sqlite_handler = AsyncMock()
        buffer.set_storage_handlers(sqlite_handler, None)
        
        await buffer._write_to_sqlite(sample_rounds)
        
        sqlite_handler.assert_called_once_with(sample_rounds)
    
    @pytest.mark.asyncio
    async def test_write_to_qdrant(self, mock_chunk_strategy, mock_embedding_model):
        """Test writing to Qdrant."""
        buffer = HybridBuffer()
        buffer.chunk_strategy = mock_chunk_strategy
        buffer.embedding_model = mock_embedding_model
        
        qdrant_handler = AsyncMock()
        buffer.set_storage_handlers(None, qdrant_handler)
        
        # Add some chunks and embeddings
        chunks = [ChunkData(content="test", metadata={})]
        buffer.chunks = chunks
        buffer.embeddings = [[0.1, 0.2, 0.3]]
        
        await buffer._write_to_qdrant(chunks)
        
        qdrant_handler.assert_called_once()
        # Check that points were formatted correctly
        call_args = qdrant_handler.call_args[0][0]
        assert len(call_args) == 1
        assert "id" in call_args[0]
        assert "vector" in call_args[0]
        assert "payload" in call_args[0]


class TestHybridBufferReadAPI:
    """Test cases for Read API functionality."""
    
    @pytest.mark.asyncio
    async def test_get_all_messages_for_read_api(self, sample_rounds, mock_chunk_strategy, mock_embedding_model):
        """Test getting all messages for Read API."""
        buffer = HybridBuffer()
        buffer.chunk_strategy = mock_chunk_strategy
        buffer.embedding_model = mock_embedding_model
        
        await buffer.add_from_rounds(sample_rounds)
        
        result = await buffer.get_all_messages_for_read_api()
        
        assert len(result) == 3  # Total messages across all rounds
        assert all("source" in msg["metadata"] for msg in result)
        assert all(msg["metadata"]["source"] == "hybrid_buffer" for msg in result)
    
    @pytest.mark.asyncio
    async def test_get_messages_with_limit(self, sample_rounds, mock_chunk_strategy, mock_embedding_model):
        """Test getting messages with limit."""
        buffer = HybridBuffer()
        buffer.chunk_strategy = mock_chunk_strategy
        buffer.embedding_model = mock_embedding_model
        
        await buffer.add_from_rounds(sample_rounds)
        
        result = await buffer.get_all_messages_for_read_api(limit=2)
        
        assert len(result) == 2
    
    @pytest.mark.asyncio
    async def test_get_messages_with_sorting(self, mock_chunk_strategy, mock_embedding_model):
        """Test getting messages with sorting."""
        buffer = HybridBuffer()
        buffer.chunk_strategy = mock_chunk_strategy
        buffer.embedding_model = mock_embedding_model
        
        # Create rounds with different timestamps
        rounds = [
            [{"id": "1", "role": "user", "content": "First", "created_at": "2024-01-01T10:00:00Z"}],
            [{"id": "2", "role": "user", "content": "Second", "created_at": "2024-01-01T09:00:00Z"}]
        ]
        
        await buffer.add_from_rounds(rounds)
        
        # Test descending order (default)
        result_desc = await buffer.get_all_messages_for_read_api(sort_by="timestamp", order="desc")
        assert result_desc[0]["id"] == "1"  # Later timestamp first
        
        # Test ascending order
        result_asc = await buffer.get_all_messages_for_read_api(sort_by="timestamp", order="asc")
        assert result_asc[0]["id"] == "2"  # Earlier timestamp first


class TestHybridBufferStats:
    """Test cases for statistics functionality."""
    
    def test_get_stats_empty(self):
        """Test getting stats for empty buffer."""
        buffer = HybridBuffer(max_size=10, chunk_strategy="contextual", embedding_model="custom")
        
        stats = buffer.get_stats()
        
        assert stats["chunks_count"] == 0
        assert stats["rounds_count"] == 0
        assert stats["embeddings_count"] == 0
        assert stats["max_size"] == 10
        assert stats["chunk_strategy"] == "contextual"
        assert stats["embedding_model"] == "custom"
        assert stats["total_rounds_received"] == 0
        assert stats["total_chunks_created"] == 0
        assert stats["total_flushes"] == 0
        assert stats["total_fifo_removals"] == 0
        assert stats["has_sqlite_handler"] is False
        assert stats["has_qdrant_handler"] is False
    
    @pytest.mark.asyncio
    async def test_get_stats_with_data(self, sample_rounds, mock_chunk_strategy, mock_embedding_model):
        """Test getting stats with data."""
        buffer = HybridBuffer()
        buffer.chunk_strategy = mock_chunk_strategy
        buffer.embedding_model = mock_embedding_model
        
        sqlite_handler = AsyncMock()
        qdrant_handler = AsyncMock()
        buffer.set_storage_handlers(sqlite_handler, qdrant_handler)
        
        await buffer.add_from_rounds(sample_rounds)
        
        stats = buffer.get_stats()
        
        assert stats["chunks_count"] > 0
        assert stats["rounds_count"] > 0
        assert stats["embeddings_count"] > 0
        assert stats["total_rounds_received"] == 2
        assert stats["total_chunks_created"] > 0
        assert stats["has_sqlite_handler"] is True
        assert stats["has_qdrant_handler"] is True


class TestHybridBufferConcurrency:
    """Test cases for concurrent operations."""
    
    @pytest.mark.asyncio
    async def test_concurrent_add_rounds(self, mock_chunk_strategy, mock_embedding_model):
        """Test concurrent round additions."""
        buffer = HybridBuffer(max_size=20)
        buffer.chunk_strategy = mock_chunk_strategy
        buffer.embedding_model = mock_embedding_model
        
        # Create multiple rounds
        rounds_list = []
        for i in range(10):
            rounds = [[{"id": f"msg_{i}", "role": "user", "content": f"Message {i}"}]]
            rounds_list.append(rounds)
        
        # Add rounds concurrently
        tasks = [buffer.add_from_rounds(rounds) for rounds in rounds_list]
        await asyncio.gather(*tasks)
        
        # Should handle concurrent access gracefully
        assert buffer.total_rounds_received >= 1
        assert isinstance(buffer.get_stats(), dict)
    
    @pytest.mark.asyncio
    async def test_concurrent_flush_operations(self, sample_rounds, mock_chunk_strategy, mock_embedding_model):
        """Test concurrent flush operations."""
        buffer = HybridBuffer()
        buffer.chunk_strategy = mock_chunk_strategy
        buffer.embedding_model = mock_embedding_model
        
        sqlite_handler = AsyncMock()
        qdrant_handler = AsyncMock()
        buffer.set_storage_handlers(sqlite_handler, qdrant_handler)
        
        await buffer.add_from_rounds(sample_rounds)
        
        # Try multiple concurrent flushes
        tasks = [buffer.flush_to_storage() for _ in range(5)]
        results = await asyncio.gather(*tasks)
        
        # At least one should succeed
        assert any(results)
        # Handlers should be called at least once
        assert sqlite_handler.call_count >= 1
