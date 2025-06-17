"""Token counting utility for MemFuse Buffer.

This module provides accurate token counting functionality with tiktoken support
and fallback mechanisms for MessageList processing in the Buffer architecture.
"""

import re
from typing import List, Dict, Any, Optional

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

from loguru import logger


class TokenCounter:
    """Token counting utility with tiktoken support and fallback mechanism.
    
    This class provides accurate token counting for text and MessageList objects,
    prioritizing tiktoken when available and falling back to word-based estimation.
    """
    
    def __init__(self, model: str = "gpt-4o-mini"):
        """Initialize the token counter.
        
        Args:
            model: Model name for tiktoken encoding (default: gpt-4o-mini)
        """
        self.model = model
        self.tiktoken_available = TIKTOKEN_AVAILABLE
        self.encoding = None
        
        if self.tiktoken_available:
            try:
                self.encoding = tiktoken.encoding_for_model(model)
                logger.debug(f"TokenCounter: Initialized with tiktoken for model {model}")
            except Exception as e:
                logger.warning(f"TokenCounter: Failed to load model {model}, using cl100k_base: {e}")
                try:
                    # Fallback to cl100k_base encoding
                    self.encoding = tiktoken.get_encoding("cl100k_base")
                except Exception as e2:
                    logger.error(f"TokenCounter: Failed to load cl100k_base encoding: {e2}")
                    self.tiktoken_available = False
        else:
            logger.info("TokenCounter: tiktoken not available, using fallback mechanism")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens in the text
        """
        if not text:
            return 0
            
        if self.tiktoken_available and self.encoding:
            try:
                return len(self.encoding.encode(text))
            except Exception as e:
                logger.warning(f"TokenCounter: tiktoken encoding failed, using fallback: {e}")
                return self._fallback_count_tokens(text)
        else:
            return self._fallback_count_tokens(text)
    
    def count_message_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """Count total tokens in a MessageList.
        
        This method accounts for message formatting overhead and role tokens.
        
        Args:
            messages: List of message dictionaries (MessageList)
            
        Returns:
            Total number of tokens including formatting overhead
        """
        if not messages:
            return 0
            
        total_tokens = 0
        
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            
            if self.tiktoken_available and self.encoding:
                try:
                    # Format: <|start|>{role}<|message|>{content}<|end|>
                    # This approximates the actual token format used by chat models
                    formatted = f"<|start|>{role}<|message|>{content}<|end|>"
                    total_tokens += len(self.encoding.encode(formatted))
                except Exception as e:
                    logger.warning(f"TokenCounter: tiktoken message encoding failed, using fallback: {e}")
                    # Fallback calculation
                    role_tokens = self._fallback_count_tokens(role)
                    content_tokens = self._fallback_count_tokens(content)
                    # Add formatting overhead (4 tokens for start/message/end markers)
                    total_tokens += role_tokens + content_tokens + 4
            else:
                # Fallback calculation
                role_tokens = self._fallback_count_tokens(role)
                content_tokens = self._fallback_count_tokens(content)
                # Add formatting overhead (4 tokens for start/message/end markers)
                total_tokens += role_tokens + content_tokens + 4
        
        return total_tokens
    
    def _fallback_count_tokens(self, text: str) -> int:
        """Fallback token counting using word-based estimation.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Estimated number of tokens
        """
        if not text:
            return 0
        
        # Count words with CJK character support
        word_count = self._count_words_with_cjk(text)
        
        # Apply multiplier (1.3 is empirical value for English/Chinese mixed text)
        return int(word_count * 1.3)
    
    def _count_words_with_cjk(self, text: str) -> int:
        """Count words with support for CJK characters.
        
        Args:
            text: Text to count words for
            
        Returns:
            Number of words/characters
        """
        # Split by whitespace for regular words
        words = re.findall(r'\S+', text)
        
        total_count = 0
        for word in words:
            # Count CJK characters individually
            cjk_chars = re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u309f\u30a0-\u30ff]', word)
            # Count non-CJK parts as single words
            non_cjk_parts = re.split(r'[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u309f\u30a0-\u30ff]', word)
            non_cjk_words = [part for part in non_cjk_parts if part.strip()]
            
            total_count += len(cjk_chars) + len(non_cjk_words)
        
        return max(total_count, 1)  # Ensure at least 1 for non-empty text


# Global instance management
_token_counter: Optional[TokenCounter] = None


def get_token_counter(model: str = "gpt-4o-mini") -> TokenCounter:
    """Get global token counter instance.
    
    Args:
        model: Model name for tiktoken encoding
        
    Returns:
        Global TokenCounter instance
    """
    global _token_counter
    if _token_counter is None or _token_counter.model != model:
        _token_counter = TokenCounter(model)
    return _token_counter


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Convenience function to count tokens in text.
    
    Args:
        text: Text to count tokens for
        model: Model name for tiktoken encoding
        
    Returns:
        Number of tokens in the text
    """
    counter = get_token_counter(model)
    return counter.count_tokens(text)


def count_message_tokens(messages: List[Dict[str, Any]], model: str = "gpt-4o-mini") -> int:
    """Convenience function to count tokens in MessageList.
    
    Args:
        messages: List of message dictionaries
        model: Model name for tiktoken encoding
        
    Returns:
        Total number of tokens including formatting overhead
    """
    counter = get_token_counter(model)
    return counter.count_message_tokens(messages)
