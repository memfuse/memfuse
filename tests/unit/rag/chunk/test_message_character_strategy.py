"""Unit tests for MessageCharacterChunkStrategy."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from typing import List, Dict, Any

from src.memfuse_core.rag.chunk.message_character import MessageCharacterChunkStrategy
from src.memfuse_core.rag.chunk.base import ChunkData


class TestMessageCharacterChunkStrategy:
    """Test cases for MessageCharacterChunkStrategy."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.strategy = MessageCharacterChunkStrategy(
            max_words_per_group=50,  # Small for testing
            max_words_per_chunk=50,
            enable_contextual=False  # Disable for basic tests
        )
    
    def test_init(self):
        """Test strategy initialization."""
        assert self.strategy.max_words_per_group == 50
        assert self.strategy.max_words_per_chunk == 50
        assert self.strategy.role_format == "[{role}]"
        assert self.strategy.chunk_separator == "\n\n"
        assert not self.strategy.enable_contextual
        assert self.strategy.context_window_size == 2
    
    def test_count_words_english(self):
        """Test word counting for English text."""
        text = "Hello world this is a test"
        count = self.strategy._count_words(text)
        assert count == 6
    
    def test_count_words_cjk(self):
        """Test word counting for CJK characters."""
        text = "你好世界"  # 4 Chinese characters
        count = self.strategy._count_words(text)
        assert count == 4
    
    def test_count_words_mixed(self):
        """Test word counting for mixed text."""
        text = "Hello 你好 world 世界"  # 2 English words + 4 CJK characters
        count = self.strategy._count_words(text)
        assert count == 6
    
    def test_is_cjk_character(self):
        """Test CJK character detection."""
        assert self.strategy._is_cjk_character('你')  # Chinese
        assert self.strategy._is_cjk_character('あ')  # Hiragana
        assert self.strategy._is_cjk_character('ア')  # Katakana
        assert self.strategy._is_cjk_character('한')  # Korean
        assert not self.strategy._is_cjk_character('A')  # English
        assert not self.strategy._is_cjk_character('1')  # Number
    
    def test_extract_session_id(self):
        """Test session ID extraction from messages."""
        # Test with metadata
        message1 = {
            "role": "user",
            "content": "Hello",
            "metadata": {"session_id": "session_123"}
        }
        assert self.strategy._extract_session_id(message1) == "session_123"
        
        # Test with direct session_id
        message2 = {
            "role": "user", 
            "content": "Hello",
            "session_id": "session_456"
        }
        assert self.strategy._extract_session_id(message2) == "session_456"
        
        # Test with no session_id
        message3 = {"role": "user", "content": "Hello"}
        assert self.strategy._extract_session_id(message3) is None
    
    def test_group_messages_by_word_count(self):
        """Test message grouping by word count."""
        messages = [
            {"role": "user", "content": "Hello world"},  # 2 words
            {"role": "assistant", "content": "Hi there"},  # 2 words
            {"role": "user", "content": "How are you today?"},  # 4 words
        ]

        # Total: 2 + 2 + 4 = 8 words, with max_words_per_group=50, should fit in 1 group
        groups = self.strategy._group_messages_by_word_count(messages)
        assert len(groups) == 1
        assert len(groups[0]) == 3  # All three messages fit in one group

        # Test with smaller limit to force splitting
        strategy_small = MessageCharacterChunkStrategy(max_words_per_group=5)
        groups_small = strategy_small._group_messages_by_word_count(messages)
        assert len(groups_small) == 2
        assert len(groups_small[0]) == 2  # First two messages (4 words total)
        assert len(groups_small[1]) == 1  # Last message (4 words)
    
    def test_format_message_group(self):
        """Test message group formatting."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"}
        ]
        
        formatted = self.strategy._format_message_group(messages)
        expected = "[USER]: Hello\n\n[ASSISTANT]: Hi there"
        assert formatted == expected
    
    @pytest.mark.asyncio
    async def test_create_chunks_basic(self):
        """Test basic chunk creation without contextual enhancement."""
        message_batch_list = [
            [
                {"role": "user", "content": "Hello world"},
                {"role": "assistant", "content": "Hi there"}
            ]
        ]
        
        chunks = await self.strategy.create_chunks(message_batch_list)
        
        assert len(chunks) == 1
        assert isinstance(chunks[0], ChunkData)
        assert chunks[0].metadata["strategy"] == "message_character"
        assert not chunks[0].metadata["has_context"]
        assert "[USER]: Hello world" in chunks[0].content
        assert "[ASSISTANT]: Hi there" in chunks[0].content
    
    @pytest.mark.asyncio
    async def test_create_chunks_empty_input(self):
        """Test chunk creation with empty input."""
        chunks = await self.strategy.create_chunks([])
        assert chunks == []
        
        chunks = await self.strategy.create_chunks([[]])
        assert chunks == []
    
    @pytest.mark.asyncio
    async def test_create_chunks_with_contextual_disabled(self):
        """Test chunk creation with contextual enhancement disabled."""
        # Create strategy with vector store but contextual disabled
        mock_vector_store = AsyncMock()
        strategy = MessageCharacterChunkStrategy(
            enable_contextual=False,
            vector_store=mock_vector_store
        )
        
        message_batch_list = [
            [{"role": "user", "content": "Test message", "metadata": {"session_id": "test_session"}}]
        ]
        
        chunks = await strategy.create_chunks(message_batch_list)
        
        assert len(chunks) == 1
        assert not chunks[0].metadata["has_context"]
        # Vector store should not be called
        mock_vector_store.get_chunks_by_session.assert_not_called()
    
    def test_split_oversized_group(self):
        """Test splitting of oversized message groups."""
        # Create a message that exceeds the word limit
        long_content = " ".join(["word"] * 60)  # 60 words, exceeds limit of 50
        messages = [{"role": "user", "content": long_content}]
        
        chunks = self.strategy._split_oversized_group(messages)
        
        # Should be split into multiple chunks
        assert len(chunks) > 1
        for chunk in chunks:
            # Each chunk should have headers
            assert "=== Chunk" in chunk
            assert "===" in chunk
    
    def test_format_chunk_with_header(self):
        """Test chunk formatting with headers."""
        parts = ["[USER]: Hello", "[ASSISTANT]: Hi"]
        chunk_number = 1
        
        formatted = self.strategy._format_chunk_with_header(parts, chunk_number)
        
        assert "=== Chunk 1 ===" in formatted
        assert "[USER]: Hello" in formatted
        assert "[ASSISTANT]: Hi" in formatted
        assert formatted.count("===") >= 2  # Header and footer

    @pytest.mark.asyncio
    async def test_create_chunks_with_contextual_enabled(self):
        """Test chunk creation with contextual enhancement enabled."""
        # Mock vector store with previous chunks
        mock_vector_store = AsyncMock()
        previous_chunks = [
            ChunkData(
                content="Previous chunk 1",
                metadata={"created_at": "2024-01-01T10:00:00Z"}
            ),
            ChunkData(
                content="Previous chunk 2",
                metadata={"created_at": "2024-01-01T11:00:00Z"}
            )
        ]
        mock_vector_store.get_chunks_by_session.return_value = previous_chunks

        strategy = MessageCharacterChunkStrategy(
            enable_contextual=True,
            context_window_size=2,
            vector_store=mock_vector_store
        )

        message_batch_list = [
            [{"role": "user", "content": "New message", "metadata": {"session_id": "test_session"}}]
        ]

        chunks = await strategy.create_chunks(message_batch_list)

        assert len(chunks) == 1
        assert chunks[0].metadata["has_context"] is True
        assert chunks[0].metadata["context_window_size"] == 2
        assert len(chunks[0].metadata["context_chunk_ids"]) == 2

        # Verify vector store was called
        mock_vector_store.get_chunks_by_session.assert_called_once_with("test_session")

    @pytest.mark.asyncio
    async def test_get_previous_chunks(self):
        """Test retrieval of previous chunks from vector store."""
        mock_vector_store = AsyncMock()
        session_chunks = [
            ChunkData(content="Chunk 1", metadata={"created_at": "2024-01-01T10:00:00Z"}),
            ChunkData(content="Chunk 2", metadata={"created_at": "2024-01-01T11:00:00Z"}),
            ChunkData(content="Chunk 3", metadata={"created_at": "2024-01-01T12:00:00Z"}),
        ]
        mock_vector_store.get_chunks_by_session.return_value = session_chunks

        strategy = MessageCharacterChunkStrategy(
            context_window_size=2,
            vector_store=mock_vector_store
        )

        previous_chunks = await strategy._get_previous_chunks("test_session")

        # Should return the last 2 chunks
        assert len(previous_chunks) == 2
        assert previous_chunks[0].content == "Chunk 2"
        assert previous_chunks[1].content == "Chunk 3"

    def test_build_context_window(self):
        """Test context window building logic."""
        previous_chunks = [
            ChunkData(content="Chunk 1", metadata={}),
            ChunkData(content="Chunk 2", metadata={}),
        ]

        strategy = MessageCharacterChunkStrategy(context_window_size=2)

        # Test first chunk in batch
        context = strategy._build_context_window(previous_chunks, 0)
        assert len(context) == 2
        assert context[0].content == "Chunk 1"
        assert context[1].content == "Chunk 2"

        # Test subsequent chunk in batch
        context = strategy._build_context_window(previous_chunks, 1)
        assert len(context) == 2  # Still uses previous chunks for now

    @pytest.mark.asyncio
    async def test_generate_contextual_description(self):
        """Test contextual description generation."""
        strategy = MessageCharacterChunkStrategy()

        current_chunk = "This is the current chunk content"
        context_chunks = [
            ChunkData(content="Previous chunk 1", metadata={}),
            ChunkData(content="Previous chunk 2", metadata={}),
        ]

        description = await strategy._generate_contextual_description(current_chunk, context_chunks)

        assert "Context from 2 previous chunks" in description
        assert "This is the current chunk content"[:20] in description
