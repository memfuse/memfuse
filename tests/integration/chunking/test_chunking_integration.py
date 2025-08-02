"""Integration tests for chunking strategies and workflows."""

import asyncio
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from memfuse_core.rag.chunk.contextual import ContextualChunkStrategy
from memfuse_core.rag.chunk.base import ChunkData
from tests.mocks.llm import MockProvider


class TestChunkingIntegration:
    """Integration tests for chunking strategies and complete workflows."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_llm = MockProvider({
            "custom_responses": {
                "contextual": "This chunk discusses machine learning and AI concepts.",
                "conversation": "This conversation covers technical topics and explanations.",
                "neural": "This section explains neural networks and deep learning."
            }
        })
    
    @pytest.mark.asyncio
    async def test_contextual_chunking_integration(self):
        """Test complete ContextualChunkStrategy integration."""
        strategy = ContextualChunkStrategy(
            max_words_per_group=50,
            enable_contextual=True,
            llm_provider=self.mock_llm
        )
        
        # Test with realistic conversation data
        message_batch_list = [
            [
                {"role": "user", "content": "What is machine learning and how does it work?"},
                {"role": "assistant", "content": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It works by using algorithms to analyze data, identify patterns, and make predictions or decisions."}
            ],
            [
                {"role": "user", "content": "Can you give me some examples?"},
                {"role": "assistant", "content": "Sure! Common examples include email spam detection, recommendation systems like Netflix or Amazon, image recognition in photos, voice assistants, and predictive text on your phone."}
            ],
            [
                {"role": "user", "content": "What about neural networks?"},
                {"role": "assistant", "content": "Neural networks are a specific type of machine learning model inspired by the human brain. They consist of interconnected nodes (neurons) that process information in layers."}
            ]
        ]
        
        chunks = await strategy.create_chunks(message_batch_list)
        
        # Verify chunks were created
        assert len(chunks) >= 3
        
        # Verify chunk structure
        for chunk in chunks:
            assert isinstance(chunk, ChunkData)
            assert chunk.content
            assert chunk.chunk_id
            assert chunk.metadata
            
            # Verify contextual enhancement
            if chunk.metadata.get("has_context"):
                assert "contextual_description" in chunk.metadata
                assert chunk.metadata["gpt_enhanced"] is True
        
        # Verify content preservation
        all_content = " ".join(chunk.content for chunk in chunks)
        assert "machine learning" in all_content.lower()
        assert "neural networks" in all_content.lower()
    
    @pytest.mark.asyncio
    async def test_cjk_language_support_integration(self):
        """Test CJK (Chinese, Japanese, Korean) language support integration."""
        strategy = ContextualChunkStrategy(
            max_words_per_group=30,
            enable_contextual=False  # Focus on CJK handling
        )
        
        # Test with mixed CJK and English content
        message_batch_list = [
            [
                {"role": "user", "content": "什么是机器学习？What is machine learning?"},
                {"role": "assistant", "content": "机器学习是人工智能的一个分支。Machine learning is a branch of AI that enables computers to learn from data."}
            ],
            [
                {"role": "user", "content": "機械学習とは何ですか？"},
                {"role": "assistant", "content": "機械学習は、コンピュータがデータから学習する技術です。"}
            ],
            [
                {"role": "user", "content": "머신러닝이 뭔가요?"},
                {"role": "assistant", "content": "머신러닝은 컴퓨터가 데이터로부터 학습하는 기술입니다."}
            ]
        ]
        
        chunks = await strategy.create_chunks(message_batch_list)

        # Debug: Print chunks information
        print(f"DEBUG: Created {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            print(f"DEBUG: Chunk {i}: {chunk.content[:100]}...")

        # Verify CJK content is properly handled
        assert len(chunks) > 0, f"Expected at least 1 chunk, got {len(chunks)}"
        
        # Verify at least some chunks contain CJK characters
        cjk_chunks = [chunk for chunk in chunks if any(ord(char) > 127 for char in chunk.content)]
        assert len(cjk_chunks) > 0, "Expected at least one chunk with CJK characters"

        for chunk in chunks:
            # Verify word counting works with CJK
            word_count = chunk.metadata.get("word_count", 0)
            assert word_count > 0, f"Expected word_count > 0, got {word_count}"

            # Verify proper formatting (each chunk should have at least USER or ASSISTANT)
            content = chunk.content
            assert "[USER]:" in content or "[ASSISTANT]:" in content, f"Expected proper formatting in: {content[:100]}..."


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
