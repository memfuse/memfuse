"""Tests for TokenCounter utility in Buffer."""

import pytest
from unittest.mock import patch, MagicMock

from memfuse_core.utils.token_counter import (
    TokenCounter,
    get_token_counter,
    count_tokens,
    count_message_tokens
)


class TestTokenCounter:
    """Test cases for TokenCounter class."""
    
    def test_initialization_with_tiktoken(self):
        """Test TokenCounter initialization when tiktoken is available."""
        with patch('memfuse_core.utils.token_counter.TIKTOKEN_AVAILABLE', True):
            with patch('memfuse_core.utils.token_counter.tiktoken') as mock_tiktoken:
                mock_encoding = MagicMock()
                mock_tiktoken.encoding_for_model.return_value = mock_encoding
                
                counter = TokenCounter("gpt-4")
                
                assert counter.model == "gpt-4"
                assert counter.tiktoken_available is True
                assert counter.encoding == mock_encoding
                mock_tiktoken.encoding_for_model.assert_called_once_with("gpt-4")
    
    def test_initialization_without_tiktoken(self):
        """Test TokenCounter initialization when tiktoken is not available."""
        with patch('memfuse_core.utils.token_counter.TIKTOKEN_AVAILABLE', False):
            counter = TokenCounter()
            
            assert counter.model == "gpt-4o-mini"
            assert counter.tiktoken_available is False
            assert counter.encoding is None
    
    def test_initialization_with_tiktoken_error(self):
        """Test TokenCounter initialization when tiktoken fails to load model."""
        with patch('memfuse_core.utils.token_counter.TIKTOKEN_AVAILABLE', True):
            with patch('memfuse_core.utils.token_counter.tiktoken') as mock_tiktoken:
                mock_tiktoken.encoding_for_model.side_effect = Exception("Model not found")
                mock_encoding = MagicMock()
                mock_tiktoken.get_encoding.return_value = mock_encoding
                
                counter = TokenCounter("unknown-model")
                
                assert counter.encoding == mock_encoding
                mock_tiktoken.get_encoding.assert_called_once_with("cl100k_base")
    
    def test_count_tokens_with_tiktoken(self):
        """Test token counting with tiktoken."""
        with patch('memfuse_core.utils.token_counter.TIKTOKEN_AVAILABLE', True):
            with patch('memfuse_core.utils.token_counter.tiktoken') as mock_tiktoken:
                mock_encoding = MagicMock()
                mock_encoding.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens
                mock_tiktoken.encoding_for_model.return_value = mock_encoding
                
                counter = TokenCounter()
                result = counter.count_tokens("Hello world")
                
                assert result == 5
                mock_encoding.encode.assert_called_once_with("Hello world")
    
    def test_count_tokens_with_fallback(self):
        """Test token counting with fallback mechanism."""
        with patch('memfuse_core.utils.token_counter.TIKTOKEN_AVAILABLE', False):
            counter = TokenCounter()
            result = counter.count_tokens("Hello world")
            
            # "Hello world" = 2 words * 1.3 = 2.6 -> 2 tokens
            assert result == 2
    
    def test_count_tokens_empty_string(self):
        """Test token counting with empty string."""
        counter = TokenCounter()
        result = counter.count_tokens("")
        
        assert result == 0
    
    def test_count_tokens_cjk_characters(self):
        """Test token counting with CJK characters."""
        with patch('memfuse_core.utils.token_counter.TIKTOKEN_AVAILABLE', False):
            counter = TokenCounter()
            result = counter.count_tokens("你好世界")
            
            # 4 CJK characters * 1.3 = 5.2 -> 5 tokens
            assert result == 5
    
    def test_count_tokens_mixed_text(self):
        """Test token counting with mixed English and CJK text."""
        with patch('memfuse_core.utils.token_counter.TIKTOKEN_AVAILABLE', False):
            counter = TokenCounter()
            result = counter.count_tokens("Hello 你好 world 世界")
            
            # "Hello" (1) + "你" (1) + "好" (1) + "world" (1) + "世" (1) + "界" (1) = 6 * 1.3 = 7.8 -> 7
            assert result == 7
    
    def test_count_message_tokens_with_tiktoken(self):
        """Test message token counting with tiktoken."""
        with patch('memfuse_core.utils.token_counter.TIKTOKEN_AVAILABLE', True):
            with patch('memfuse_core.utils.token_counter.tiktoken') as mock_tiktoken:
                mock_encoding = MagicMock()
                mock_encoding.encode.return_value = [1, 2, 3, 4, 5, 6, 7, 8]  # 8 tokens per message
                mock_tiktoken.encoding_for_model.return_value = mock_encoding
                
                counter = TokenCounter()
                messages = [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"}
                ]
                result = counter.count_message_tokens(messages)
                
                assert result == 16  # 8 tokens per message * 2 messages
                assert mock_encoding.encode.call_count == 2
    
    def test_count_message_tokens_with_fallback(self):
        """Test message token counting with fallback."""
        with patch('memfuse_core.utils.token_counter.TIKTOKEN_AVAILABLE', False):
            counter = TokenCounter()
            messages = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"}
            ]
            result = counter.count_message_tokens(messages)
            
            # Message 1: "user" (1) + "Hello" (1) + 4 (formatting) = 6
            # Message 2: "assistant" (1) + "Hi there" (2) + 4 (formatting) = 7
            # Total: 13 tokens (after applying 1.3 multiplier and formatting)
            assert result > 0  # Exact value depends on fallback calculation
    
    def test_count_message_tokens_empty_list(self):
        """Test message token counting with empty list."""
        counter = TokenCounter()
        result = counter.count_message_tokens([])
        
        assert result == 0
    
    def test_count_message_tokens_missing_fields(self):
        """Test message token counting with missing role/content fields."""
        with patch('memfuse_core.utils.token_counter.TIKTOKEN_AVAILABLE', False):
            counter = TokenCounter()
            messages = [
                {"role": "user"},  # Missing content
                {"content": "Hello"}  # Missing role
            ]
            result = counter.count_message_tokens(messages)
            
            assert result > 0  # Should handle missing fields gracefully
    
    def test_fallback_count_words_with_cjk(self):
        """Test CJK word counting in fallback mechanism."""
        counter = TokenCounter()
        
        # Test pure English
        result = counter._count_words_with_cjk("Hello world")
        assert result == 2
        
        # Test pure CJK
        result = counter._count_words_with_cjk("你好世界")
        assert result == 4
        
        # Test mixed
        result = counter._count_words_with_cjk("Hello你好world")
        assert result == 4  # "Hello" + "你" + "好" + "world"
        
        # Test empty string
        result = counter._count_words_with_cjk("")
        assert result == 1  # Minimum 1 for non-empty text handling


class TestGlobalFunctions:
    """Test cases for global convenience functions."""
    
    def test_get_token_counter_singleton(self):
        """Test that get_token_counter returns singleton instance."""
        counter1 = get_token_counter()
        counter2 = get_token_counter()
        
        assert counter1 is counter2
    
    def test_get_token_counter_different_models(self):
        """Test that different models create different instances."""
        counter1 = get_token_counter("gpt-4o-mini")
        counter2 = get_token_counter("gpt-3.5-turbo")
        
        assert counter1 is not counter2
        assert counter1.model == "gpt-4o-mini"
        assert counter2.model == "gpt-3.5-turbo"
    
    def test_count_tokens_convenience_function(self):
        """Test count_tokens convenience function."""
        with patch('memfuse_core.utils.token_counter.TIKTOKEN_AVAILABLE', False):
            result = count_tokens("Hello world")
            assert result == 2
    
    def test_count_message_tokens_convenience_function(self):
        """Test count_message_tokens convenience function."""
        with patch('memfuse_core.utils.token_counter.TIKTOKEN_AVAILABLE', False):
            messages = [{"role": "user", "content": "Hello"}]
            result = count_message_tokens(messages)
            assert result > 0


class TestErrorHandling:
    """Test cases for error handling scenarios."""
    
    def test_tiktoken_encoding_error_fallback(self):
        """Test fallback when tiktoken encoding fails."""
        with patch('memfuse_core.utils.token_counter.TIKTOKEN_AVAILABLE', True):
            with patch('memfuse_core.utils.token_counter.tiktoken') as mock_tiktoken:
                mock_encoding = MagicMock()
                mock_encoding.encode.side_effect = Exception("Encoding error")
                mock_tiktoken.encoding_for_model.return_value = mock_encoding
                
                counter = TokenCounter()
                result = counter.count_tokens("Hello world")
                
                # Should fallback to word-based counting
                assert result == 2
    
    def test_message_token_counting_encoding_error(self):
        """Test fallback in message token counting when encoding fails."""
        with patch('memfuse_core.utils.token_counter.TIKTOKEN_AVAILABLE', True):
            with patch('memfuse_core.utils.token_counter.tiktoken') as mock_tiktoken:
                mock_encoding = MagicMock()
                mock_encoding.encode.side_effect = Exception("Encoding error")
                mock_tiktoken.encoding_for_model.return_value = mock_encoding
                
                counter = TokenCounter()
                messages = [{"role": "user", "content": "Hello"}]
                result = counter.count_message_tokens(messages)
                
                # Should fallback to word-based counting
                assert result > 0


@pytest.fixture
def sample_messages():
    """Fixture providing sample messages for testing."""
    return [
        {
            "role": "user",
            "content": "What is the weather like today?",
            "metadata": {"session_id": "test_session"}
        },
        {
            "role": "assistant", 
            "content": "I don't have access to real-time weather data.",
            "metadata": {"session_id": "test_session"}
        }
    ]


class TestIntegration:
    """Integration tests for TokenCounter."""
    
    def test_realistic_message_counting(self, sample_messages):
        """Test token counting with realistic message data."""
        counter = TokenCounter()
        result = counter.count_message_tokens(sample_messages)
        
        # Should return reasonable token count
        assert result > 10  # At least some tokens
        assert result < 100  # Not too many for this simple example
    
    def test_performance_with_large_messages(self):
        """Test performance with large message lists."""
        counter = TokenCounter()
        
        # Create large message list
        large_messages = []
        for i in range(100):
            large_messages.append({
                "role": "user" if i % 2 == 0 else "assistant",
                "content": f"This is message number {i} with some content to count tokens for."
            })
        
        result = counter.count_message_tokens(large_messages)
        
        # Should handle large lists efficiently
        assert result > 1000  # Should have substantial token count
        assert isinstance(result, int)  # Should return integer
