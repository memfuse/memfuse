"""Integration tests for chunking strategies and workflows."""

import asyncio
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from memfuse_core.rag.chunk.message_character import MessageCharacterChunkStrategy
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
    async def test_message_character_chunking_integration(self):
        """Test complete MessageCharacterChunkStrategy integration."""
        strategy = MessageCharacterChunkStrategy(
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
        
        # Verify chunking results
        assert len(chunks) >= 3  # Should create multiple chunks
        
        # Verify chunk structure and metadata
        for i, chunk in enumerate(chunks):
            assert isinstance(chunk, ChunkData)
            assert chunk.content
            assert chunk.chunk_id
            
            # Verify metadata
            metadata = chunk.metadata
            assert metadata.get("strategy") == "message_character"
            assert metadata.get("chunk_index") == i
            assert "session_id" in metadata
            assert "gpt_enhanced" in metadata
            
            # Verify content formatting
            assert "[USER]:" in chunk.content or "[ASSISTANT]:" in chunk.content
            
            # Check for LLM enhancement
            if metadata.get("gpt_enhanced"):
                assert metadata.get("contextual_description")
                assert isinstance(metadata["contextual_description"], str)
                assert len(metadata["contextual_description"]) > 0
    
    @pytest.mark.asyncio
    async def test_cjk_language_support_integration(self):
        """Test CJK (Chinese, Japanese, Korean) language support integration."""
        strategy = MessageCharacterChunkStrategy(
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
        
        # Verify CJK content is properly handled
        assert len(chunks) >= 3
        
        for chunk in chunks:
            # Verify CJK characters are preserved
            content = chunk.content
            assert any(ord(char) > 127 for char in content)  # Contains non-ASCII (CJK) characters
            
            # Verify word counting works with CJK
            word_count = chunk.metadata.get("word_count", 0)
            assert word_count > 0
            
            # Verify proper formatting
            assert "[USER]:" in content and "[ASSISTANT]:" in content
    
    @pytest.mark.asyncio
    async def test_large_conversation_chunking_integration(self):
        """Test chunking of large conversations with multiple rounds."""
        strategy = MessageCharacterChunkStrategy(
            max_words_per_group=100,
            enable_contextual=True,
            llm_provider=self.mock_llm
        )
        
        # Create a large conversation
        message_batch_list = []
        for i in range(10):
            message_batch_list.append([
                {
                    "role": "user", 
                    "content": f"This is question number {i+1}. Can you explain topic {i+1} in detail? I want to understand the concepts thoroughly and get comprehensive information about this subject matter."
                },
                {
                    "role": "assistant", 
                    "content": f"Certainly! Topic {i+1} is quite interesting and involves multiple aspects. Let me break it down for you step by step. First, we need to understand the fundamental principles. Then we can explore the practical applications and real-world examples. Finally, I'll discuss the implications and future developments in this area."
                }
            ])
        
        chunks = await strategy.create_chunks(message_batch_list)
        
        # Verify large conversation handling
        assert len(chunks) >= 5  # Should create multiple chunks due to size
        
        # Verify chunk size management
        for chunk in chunks:
            word_count = chunk.metadata.get("word_count", 0)
            assert word_count <= 150  # Should respect max_words_per_group with some tolerance
            
            # Verify sequential indexing
            chunk_index = chunk.metadata.get("chunk_index")
            assert chunk_index is not None
            assert chunk_index >= 0
        
        # Verify chunks are in correct order
        chunk_indices = [chunk.metadata.get("chunk_index") for chunk in chunks]
        assert chunk_indices == sorted(chunk_indices)
    
    @pytest.mark.asyncio
    async def test_contextual_enhancement_integration(self):
        """Test contextual enhancement with previous chunk context."""
        strategy = MessageCharacterChunkStrategy(
            enable_contextual=True,
            context_window_size=2,
            llm_provider=self.mock_llm
        )
        
        # Create conversation that builds on previous context
        message_batch_list = [
            [
                {"role": "user", "content": "Let's talk about artificial intelligence."},
                {"role": "assistant", "content": "AI is a broad field of computer science."}
            ],
            [
                {"role": "user", "content": "What about machine learning specifically?"},
                {"role": "assistant", "content": "Machine learning is a subset of AI that focuses on learning from data."}
            ],
            [
                {"role": "user", "content": "And deep learning?"},
                {"role": "assistant", "content": "Deep learning uses neural networks with multiple layers."}
            ]
        ]
        
        chunks = await strategy.create_chunks(message_batch_list)
        
        # Verify contextual enhancement
        assert len(chunks) >= 3
        
        # Later chunks should have contextual descriptions that reference earlier context
        for i, chunk in enumerate(chunks):
            if i > 0 and chunk.metadata.get("contextual_description"):
                description = chunk.metadata["contextual_description"]
                
                # Should be meaningful description
                assert len(description) > 10
                assert isinstance(description, str)
                
                # Should indicate contextual awareness
                # (Mock provider should return contextual responses)
                assert any(word in description.lower() for word in 
                          ["chunk", "discusses", "covers", "explains", "section"])
    
    @pytest.mark.asyncio
    async def test_error_resilience_integration(self):
        """Test chunking resilience to various error conditions."""
        # Test with failing LLM provider
        failing_llm = MockProvider({
            "should_fail": True,
            "failure_message": "LLM service unavailable"
        })
        
        strategy = MessageCharacterChunkStrategy(
            enable_contextual=True,
            llm_provider=failing_llm
        )
        
        message_batch_list = [
            [
                {"role": "user", "content": "Test message"},
                {"role": "assistant", "content": "Test response"}
            ]
        ]
        
        # Should not raise exception
        chunks = await strategy.create_chunks(message_batch_list)
        
        # Should still create chunks without LLM enhancement
        assert len(chunks) > 0
        chunk = chunks[0]
        assert chunk.content
        assert chunk.metadata.get("strategy") == "message_character"
        
        # Should gracefully handle LLM failure
        assert chunk.metadata.get("gpt_enhanced") is not True
    
    @pytest.mark.asyncio
    async def test_empty_and_edge_cases_integration(self):
        """Test chunking with empty and edge case inputs."""
        strategy = MessageCharacterChunkStrategy()
        
        # Test empty input
        empty_chunks = await strategy.create_chunks([])
        assert len(empty_chunks) == 0
        
        # Test with empty messages
        empty_message_batch = [[]]
        empty_chunks = await strategy.create_chunks(empty_message_batch)
        assert len(empty_chunks) == 0
        
        # Test with very short content
        short_message_batch = [
            [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"}
            ]
        ]
        
        short_chunks = await strategy.create_chunks(short_message_batch)
        assert len(short_chunks) == 1
        
        chunk = short_chunks[0]
        assert "[USER]: Hi" in chunk.content
        assert "[ASSISTANT]: Hello" in chunk.content
        assert chunk.metadata.get("word_count") > 0
    
    @pytest.mark.asyncio
    async def test_metadata_consistency_integration(self):
        """Test metadata consistency across different chunking scenarios."""
        strategy = MessageCharacterChunkStrategy(
            enable_contextual=True,
            llm_provider=self.mock_llm
        )
        
        message_batch_list = [
            [
                {"role": "user", "content": "Question about AI and machine learning systems"},
                {"role": "assistant", "content": "AI and ML are related but distinct fields with different applications"}
            ],
            [
                {"role": "user", "content": "How do they differ?"},
                {"role": "assistant", "content": "AI is broader, ML is a specific approach within AI"}
            ]
        ]
        
        chunks = await strategy.create_chunks(message_batch_list)
        
        # Verify metadata consistency
        required_fields = ["strategy", "chunk_index", "word_count", "message_count"]
        
        for i, chunk in enumerate(chunks):
            metadata = chunk.metadata
            
            # Check required fields
            for field in required_fields:
                assert field in metadata, f"Missing {field} in chunk {i}"
            
            # Verify field types and values
            assert metadata["strategy"] == "message_character"
            assert isinstance(metadata["chunk_index"], int)
            assert metadata["chunk_index"] == i
            assert isinstance(metadata["word_count"], int)
            assert metadata["word_count"] > 0
            assert isinstance(metadata["message_count"], int)
            assert metadata["message_count"] > 0
            
            # Check optional LLM fields
            if metadata.get("gpt_enhanced"):
                assert "contextual_description" in metadata
                assert isinstance(metadata["contextual_description"], str)


if __name__ == "__main__":
    # Allow running as standalone script for manual testing
    async def run_chunking_integration_tests():
        test_instance = TestChunkingIntegration()
        test_instance.setup_method()
        
        print("Running chunking integration tests...")
        
        await test_instance.test_message_character_chunking_integration()
        print("✅ Message character chunking integration test passed")
        
        await test_instance.test_cjk_language_support_integration()
        print("✅ CJK language support integration test passed")
        
        await test_instance.test_large_conversation_chunking_integration()
        print("✅ Large conversation chunking integration test passed")
        
        await test_instance.test_contextual_enhancement_integration()
        print("✅ Contextual enhancement integration test passed")
        
        await test_instance.test_error_resilience_integration()
        print("✅ Error resilience integration test passed")
        
        await test_instance.test_empty_and_edge_cases_integration()
        print("✅ Empty and edge cases integration test passed")
        
        await test_instance.test_metadata_consistency_integration()
        print("✅ Metadata consistency integration test passed")
        
        print("\n🎉 All chunking integration tests passed!")
    
    asyncio.run(run_chunking_integration_tests())
