"""Message-character based chunking strategy implementation.

This strategy combines message-based grouping with character-aware splitting,
implementing advanced chunking logic with intelligent text processing and contextual enhancement.
"""

import asyncio
import logging
import re
from typing import Any, Dict, List, Optional
from .base import ChunkData, ChunkStrategy
from ...llm.base import LLMProvider, LLMRequest
from ...llm.prompts.manager import get_prompt_manager

logger = logging.getLogger(__name__)


class ContextualChunkStrategy(ChunkStrategy):
    """Contextual chunking strategy with LLM enhancement.

    This strategy implements a comprehensive contextual approach:
    1. Message Grouping: Group messages by word count limits
    2. Chunk Formatting: Convert groups to formatted text chunks
    3. Contextual Enhancement: Enhance chunks with sliding window context and LLM descriptions

    The strategy supports both basic chunking and advanced contextual chunking
    with LLM-powered enhancement for better semantic understanding and retrieval.
    """
    
    def __init__(
        self,
        max_words_per_group: int = 800,
        max_words_per_chunk: int = 800,
        role_format: str = "[{role}]",
        chunk_separator: str = "\n\n",
        enable_contextual: bool = True,
        context_window_size: int = 2,
        gpt_model: Optional[str] = None,
        vector_store = None,  # Will be injected by MemoryService
        llm_provider: Optional[LLMProvider] = None  # LLM provider for contextual descriptions
    ):
        """Initialize the contextual chunking strategy.

        Args:
            max_words_per_group: Maximum words per message group
            max_words_per_chunk: Maximum words per chunk
            role_format: Format string for role labels (e.g., "[{role}]")
            chunk_separator: Separator between messages in a chunk
            enable_contextual: Whether to enable contextual enhancement
            context_window_size: Number of previous chunks to use as context
            gpt_model: LLM model for contextual descriptions (default: grok-3-mini)
            vector_store: Vector store instance for context retrieval
            llm_provider: LLM provider for generating contextual descriptions
        """
        self.max_words_per_group = max_words_per_group
        self.max_words_per_chunk = max_words_per_chunk
        self.role_format = role_format
        self.chunk_separator = chunk_separator
        self.enable_contextual = enable_contextual
        self.context_window_size = context_window_size
        self.gpt_model = gpt_model or "grok-3-mini"
        self.vector_store = vector_store
        self.llm_provider = llm_provider

        # Initialize prompt manager
        self.prompt_manager = get_prompt_manager()
    
    async def create_chunks(self, message_batch_list: List[List[Dict[str, Any]]]) -> List[ChunkData]:
        """Create chunks from message batch list with contextual enhancement.
        
        Args:
            message_batch_list: List of lists of messages (MessageBatchList)
            
        Returns:
            List of ChunkData objects with contextual information
        """
        if not message_batch_list:
            return []
        
        logger.info(f"Creating chunks from {len(message_batch_list)} message batches")
        
        # Phase 1: Message Grouping - combine all messages and group by word count
        all_messages = []
        for batch in message_batch_list:
            all_messages.extend(batch)
        
        if not all_messages:
            return []
        
        # Extract session_id from first message for context retrieval
        session_id = self._extract_session_id(all_messages[0])
        
        # Group messages by word count
        message_groups = self._group_messages_by_word_count(all_messages)
        logger.info(f"Grouped {len(all_messages)} messages into {len(message_groups)} groups")
        
        # Phase 2: Chunk Formatting - convert groups to formatted chunks
        formatted_chunks = self._groups_to_chunks(message_groups)
        logger.info(f"Created {len(formatted_chunks)} formatted chunks")
        
        # Phase 3: Contextual Enhancement
        if self.enable_contextual and session_id and self.vector_store:
            enhanced_chunks = await self._add_contextual_information(
                formatted_chunks, session_id
            )
        else:
            # Create basic ChunkData without contextual enhancement
            enhanced_chunks = []
            for i, chunk_content in enumerate(formatted_chunks):
                metadata = {
                    "strategy": "contextual",
                    "chunk_index": i,
                    "has_context": False,
                    "session_id": session_id,
                    "gpt_enhanced": False,
                    "context_window_size": 0,
                    "context_chunk_ids": []
                }
                enhanced_chunks.append(ChunkData(
                    content=chunk_content,
                    metadata=metadata
                ))
        
        logger.info(f"Successfully created {len(enhanced_chunks)} enhanced chunks")
        return enhanced_chunks
    
    def _extract_session_id(self, message: Dict[str, Any]) -> Optional[str]:
        """Extract session_id from message metadata."""
        if isinstance(message, dict):
            # Check metadata first
            if 'metadata' in message and isinstance(message['metadata'], dict):
                if 'session_id' in message['metadata']:
                    return message['metadata']['session_id']
            # Check message itself for backward compatibility
            if 'session_id' in message:
                return message['session_id']
        return None
    
    def _group_messages_by_word_count(
        self, 
        messages: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """Group messages by word count limits.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            List of message groups
        """
        if not messages:
            return []
        
        groups = []
        current_group = []
        current_word_count = 0
        
        for message in messages:
            content = message.get('content', '')
            message_word_count = self._count_words(content)
            
            # If this single message exceeds the limit, put it in its own group
            if message_word_count > self.max_words_per_group:
                # First, save the current group if it has messages
                if current_group:
                    groups.append(current_group)
                    current_group = []
                    current_word_count = 0
                
                # Add the oversized message as its own group
                groups.append([message])
                logger.debug(f"Message with {message_word_count} words exceeds limit, creating separate group")
                
            # If adding this message would exceed the limit, start a new group
            elif current_word_count + message_word_count > self.max_words_per_group:
                # Save the current group
                if current_group:
                    groups.append(current_group)
                
                # Start new group with this message
                current_group = [message]
                current_word_count = message_word_count
                
            # Otherwise, add to current group
            else:
                current_group.append(message)
                current_word_count += message_word_count
        
        # Don't forget the last group
        if current_group:
            groups.append(current_group)
        
        logger.debug(f"Grouped {len(messages)} messages into {len(groups)} groups")
        return groups

    def _count_words(self, text: str) -> int:
        """Count words in text with CJK character support.

        Args:
            text: Text to count words in

        Returns:
            Number of words in the text
        """
        if not text or not isinstance(text, str):
            return 0

        # Count CJK characters (each character is considered one word)
        cjk_count = self._count_cjk_characters(text)

        # Remove CJK characters to process non-CJK content
        non_cjk_text = self._remove_cjk_characters(text)

        # Count non-CJK words by splitting on whitespace
        if non_cjk_text.strip():
            # Remove punctuation and split by whitespace
            cleaned_text = re.sub(r'[^\w\s]', ' ', non_cjk_text)
            non_cjk_words = [word for word in cleaned_text.split() if word.strip()]
        else:
            non_cjk_words = []

        # Total word count = CJK characters + non-CJK words
        total_words = cjk_count + len(non_cjk_words)
        return total_words

    def _count_cjk_characters(self, text: str) -> int:
        """Count CJK (Chinese, Japanese, Korean) characters in text."""
        cjk_count = 0
        for char in text:
            if self._is_cjk_character(char):
                cjk_count += 1
        return cjk_count

    def _is_cjk_character(self, char: str) -> bool:
        """Check if a character is a CJK character."""
        code_point = ord(char)

        # CJK Unicode ranges
        cjk_ranges = [
            (0x4E00, 0x9FFF),   # CJK Unified Ideographs
            (0x3400, 0x4DBF),   # CJK Extension A
            (0x20000, 0x2A6DF), # CJK Extension B
            (0x2A700, 0x2B73F), # CJK Extension C
            (0x2B740, 0x2B81F), # CJK Extension D
            (0x2B820, 0x2CEAF), # CJK Extension E
            (0x2CEB0, 0x2EBEF), # CJK Extension F
            (0x3040, 0x309F),   # Hiragana
            (0x30A0, 0x30FF),   # Katakana
            (0xAC00, 0xD7AF),   # Hangul Syllables
            (0x1100, 0x11FF),   # Hangul Jamo
            (0x3130, 0x318F),   # Hangul Compatibility Jamo
            (0xA960, 0xA97F),   # Hangul Jamo Extended-A
            (0xD7B0, 0xD7FF),   # Hangul Jamo Extended-B
        ]

        return any(start <= code_point <= end for start, end in cjk_ranges)

    def _remove_cjk_characters(self, text: str) -> str:
        """Remove CJK characters from text."""
        return ''.join(char for char in text if not self._is_cjk_character(char))

    def _groups_to_chunks(self, message_groups: List[List[Dict[str, Any]]]) -> List[str]:
        """Convert message groups into formatted text chunks.

        Args:
            message_groups: List of message groups

        Returns:
            List of formatted text chunks
        """
        chunks = []

        for group_idx, group in enumerate(message_groups):
            if not group:
                continue

            # First, convert the group to a single formatted string
            group_text = self._format_message_group(group)
            group_word_count = self._count_words(group_text)

            # If the group fits within the limit, create a single chunk
            if group_word_count <= self.max_words_per_chunk:
                chunks.append(group_text)
                logger.debug(f"Group {group_idx + 1}: Single chunk ({group_word_count} words)")
            else:
                # Split the oversized group into multiple chunks
                group_chunks = self._split_oversized_group(group)
                chunks.extend(group_chunks)
                logger.info(f"Group {group_idx + 1}: Split into {len(group_chunks)} chunks (total {group_word_count} words)")

        logger.debug(f"Created {len(chunks)} total chunks from {len(message_groups)} groups")
        return chunks

    def _format_message_group(self, messages: List[Dict[str, Any]]) -> str:
        """Format a group of messages into a single text string.

        Args:
            messages: List of message dictionaries

        Returns:
            Formatted text string
        """
        formatted_messages = []

        for message in messages:
            role = message.get('role', 'unknown').upper()
            content = message.get('content', '').strip()

            if content:
                formatted_role = self.role_format.format(role=role)
                # Add colon and space only if not already present in the format
                if self.role_format.endswith(":"):
                    formatted_messages.append(f"{formatted_role} {content}")
                else:
                    formatted_messages.append(f"{formatted_role}: {content}")

        return self.chunk_separator.join(formatted_messages)

    def _split_oversized_group(self, messages: List[Dict[str, Any]]) -> List[str]:
        """Split an oversized message group into multiple chunks.

        Args:
            messages: List of message dictionaries

        Returns:
            List of formatted chunk strings
        """
        chunks = []
        current_chunk_parts = []
        current_word_count = 0
        chunk_number = 1

        for message in messages:
            role = message.get('role', 'unknown').upper()
            content = message.get('content', '').strip()

            if not content:
                continue

            formatted_role = self.role_format.format(role=role)
            # Add colon and space only if not already present in the format
            if self.role_format.endswith(":"):
                message_text = f"{formatted_role} {content}"
            else:
                message_text = f"{formatted_role}: {content}"
            message_word_count = self._count_words(message_text)

            # If this single message exceeds the chunk limit, split it
            if message_word_count > self.max_words_per_chunk:
                # First, save current chunk if it has content
                if current_chunk_parts:
                    chunk_text = self._format_chunk_with_header(current_chunk_parts, chunk_number)
                    chunks.append(chunk_text)
                    chunk_number += 1
                    current_chunk_parts = []
                    current_word_count = 0

                # Split the oversized message into multiple chunks
                message_chunks = self._split_single_message(message, chunk_number)
                chunks.extend(message_chunks)
                chunk_number += len(message_chunks)

            # If adding this message would exceed the limit, start a new chunk
            elif current_word_count + message_word_count > self.max_words_per_chunk:
                # Save current chunk
                if current_chunk_parts:
                    chunk_text = self._format_chunk_with_header(current_chunk_parts, chunk_number)
                    chunks.append(chunk_text)
                    chunk_number += 1

                # Start new chunk with this message
                current_chunk_parts = [message_text]
                current_word_count = message_word_count

            # Otherwise, add to current chunk
            else:
                current_chunk_parts.append(message_text)
                current_word_count += message_word_count

        # Don't forget the last chunk
        if current_chunk_parts:
            chunk_text = self._format_chunk_with_header(current_chunk_parts, chunk_number)
            chunks.append(chunk_text)

        return chunks

    def _split_single_message(self, message: Dict[str, Any], starting_chunk_number: int) -> List[str]:
        """Split a single oversized message into multiple chunks.

        Args:
            message: Message dictionary
            starting_chunk_number: Starting chunk number for headers

        Returns:
            List of formatted chunk strings
        """
        role = message.get('role', 'unknown').upper()
        content = message.get('content', '').strip()
        formatted_role = self.role_format.format(role=role)

        # Split content into words while preserving spaces
        words = content.split()
        chunks = []
        chunk_number = starting_chunk_number

        current_words = []

        for word in words:
            # Calculate word count if we add this word
            if self.role_format.endswith(":"):
                test_text = f"{formatted_role} {' '.join(current_words + [word])}"
            else:
                test_text = f"{formatted_role}: {' '.join(current_words + [word])}"
            test_word_count = self._count_words(test_text)

            if test_word_count > self.max_words_per_chunk and current_words:
                # Create chunk with current words
                if self.role_format.endswith(":"):
                    chunk_content = f"{formatted_role} {' '.join(current_words)}"
                else:
                    chunk_content = f"{formatted_role}: {' '.join(current_words)}"

                if chunk_number == starting_chunk_number:
                    # First chunk - no continuation marker
                    chunk_text = self._format_split_chunk(chunk_content, chunk_number, is_first=True, is_last=False)
                else:
                    # Middle chunk - has continuation markers
                    if self.role_format.endswith(":"):
                        chunk_content = f"{formatted_role} [CONTINUES] {' '.join(current_words)}"
                    else:
                        chunk_content = f"{formatted_role}: [CONTINUES] {' '.join(current_words)}"
                    chunk_text = self._format_split_chunk(chunk_content, chunk_number, is_first=False, is_last=False)

                chunks.append(chunk_text)
                chunk_number += 1
                current_words = [word]
            else:
                current_words.append(word)

        # Handle the last chunk
        if current_words:
            if chunks:  # This is a continuation chunk
                if self.role_format.endswith(":"):
                    chunk_content = f"{formatted_role} [CONTINUES] {' '.join(current_words)}"
                else:
                    chunk_content = f"{formatted_role}: [CONTINUES] {' '.join(current_words)}"
                chunk_text = self._format_split_chunk(chunk_content, chunk_number, is_first=False, is_last=True)
            else:  # This is the only chunk (shouldn't happen, but handle gracefully)
                if self.role_format.endswith(":"):
                    chunk_content = f"{formatted_role} {' '.join(current_words)}"
                else:
                    chunk_content = f"{formatted_role}: {' '.join(current_words)}"
                chunk_text = self._format_split_chunk(chunk_content, chunk_number, is_first=True, is_last=True)

            chunks.append(chunk_text)

        return chunks

    def _format_chunk_with_header(self, parts: List[str], chunk_number: int) -> str:
        """Format a chunk with header for multi-chunk groups."""
        header = f"=== Chunk {chunk_number} ==="
        footer = "=" * len(header)
        content = self.chunk_separator.join(parts)
        return f"{header}\n{content}\n{footer}"

    def _format_split_chunk(self, content: str, chunk_number: int, is_first: bool, is_last: bool) -> str:
        """Format a chunk that's part of a split message."""
        header = f"=== Chunk {chunk_number} ==="
        footer = "=" * len(header)

        if not is_last:
            content += "\n[CONTINUES...]"

        return f"{header}\n{content}\n{footer}"

    async def _add_contextual_information(
        self,
        formatted_chunks: List[str],
        session_id: str
    ) -> List[ChunkData]:
        """Add contextual information to chunks using previous chunks from the same session.

        Args:
            formatted_chunks: List of formatted chunk strings
            session_id: Session ID for context retrieval

        Returns:
            List of enhanced ChunkData objects with contextual information
        """
        if not formatted_chunks:
            return []

        # Step 1: Get previous chunks from vector store (one-time retrieval)
        previous_chunks = await self._get_previous_chunks(session_id)
        logger.info(f"Retrieved {len(previous_chunks)} previous chunks for context")

        # Step 2: Create enhanced chunks with sliding window context
        enhanced_chunks = []

        # Create tasks for parallel processing
        tasks = []
        for i, chunk_content in enumerate(formatted_chunks):
            task = self._create_enhanced_chunk_async(
                chunk_content, i, previous_chunks, session_id
            )
            tasks.append(task)

        # Execute all tasks in parallel
        enhanced_chunks = await asyncio.gather(*tasks)

        logger.info(f"Created {len(enhanced_chunks)} enhanced chunks with contextual information")
        return enhanced_chunks

    async def _get_previous_chunks(self, session_id: str) -> List[ChunkData]:
        """Get previous chunks from the same session for context.

        Args:
            session_id: Session ID to filter chunks

        Returns:
            List of previous ChunkData objects, sorted by creation time
        """
        if not self.vector_store:
            logger.warning("No vector store available for context retrieval")
            return []

        try:
            # Get chunks by session from vector store
            session_chunks = await self.vector_store.get_chunks_by_session(session_id)

            # Sort by created_at timestamp to ensure proper order
            sorted_chunks = sorted(
                session_chunks,
                key=lambda x: x.metadata.get('created_at', ''),
                reverse=False  # Oldest first
            )

            # Return the last N chunks for context
            return sorted_chunks[-self.context_window_size:] if sorted_chunks else []

        except Exception as e:
            logger.error(f"Error retrieving previous chunks for session {session_id}: {e}")
            return []

    async def _create_enhanced_chunk_async(
        self,
        chunk_content: str,
        chunk_index: int,
        previous_chunks: List[ChunkData],
        session_id: str
    ) -> ChunkData:
        """Create an enhanced chunk with contextual information (async version).

        Args:
            chunk_content: The formatted chunk content
            chunk_index: Index of this chunk in the current batch
            previous_chunks: List of previous chunks for context
            session_id: Session ID

        Returns:
            Enhanced ChunkData with contextual information
        """
        # Build context window for this chunk
        context_chunks = self._build_context_window(previous_chunks, chunk_index)

        # Create base metadata
        metadata = {
            "strategy": "contextual",
            "chunk_index": chunk_index,
            "session_id": session_id,
            "has_context": len(context_chunks) > 0,
            "context_window_size": len(context_chunks),
            "context_chunk_ids": [chunk.chunk_id for chunk in context_chunks]
        }

        # Add contextual description if LLM provider is available
        if self.llm_provider and context_chunks:
            # Generate contextual description - let errors bubble up
            contextual_description = await self._generate_contextual_description(
                chunk_content, context_chunks
            )
            metadata["contextual_description"] = contextual_description
            metadata["gpt_enhanced"] = True
        else:
            metadata["gpt_enhanced"] = False

        return ChunkData(
            content=chunk_content,
            metadata=metadata
        )

    def _build_context_window(
        self,
        previous_chunks: List[ChunkData],
        current_chunk_index: int
    ) -> List[ChunkData]:
        """Build context window for the current chunk using sliding window approach.

        Args:
            previous_chunks: List of previous chunks from the session
            current_chunk_index: Index of the current chunk being processed

        Returns:
            List of chunks to use as context for the current chunk
        """
        if not previous_chunks:
            return []

        # For the first chunk in current batch, use the last N previous chunks
        if current_chunk_index == 0:
            return previous_chunks[-self.context_window_size:] if previous_chunks else []

        # For subsequent chunks, combine previous chunks with already processed chunks
        # This simulates the sliding window effect within the current batch
        available_context = previous_chunks[-self.context_window_size:]

        # Note: In a full implementation, we would also include the already processed
        # chunks from the current batch, but for simplicity in this basic version,
        # we'll just use the previous session chunks
        return available_context

    async def _generate_contextual_description(
        self,
        current_chunk: str,
        context_chunks: List[ChunkData]
    ) -> str:
        """Generate contextual description using LLM provider.

        Args:
            current_chunk: The current chunk content
            context_chunks: List of context chunks

        Returns:
            Contextual description string

        Raises:
            ValueError: If no LLM provider is configured
            Exception: If LLM generation fails
        """
        # Require LLM provider - no fallback
        if not self.llm_provider:
            raise ValueError("LLM provider is required for contextual descriptions")

        # Build context from previous chunks
        past_messages = "\n\n--- Previous Chunk ---\n\n".join([
            chunk.content for chunk in context_chunks
        ]) if context_chunks else ""

        # Use optimized prompt template
        prompt = self.prompt_manager.get_prompt(
            "contextual_chunking",
            past_messages=past_messages,
            current_messages=current_chunk,
            chunk_content=current_chunk
        )

        # Create LLM request
        request = LLMRequest(
            messages=[{"role": "user", "content": prompt}],
            model=self.gpt_model,
            max_tokens=150,
            temperature=0.3
        )

        # Generate response - let errors bubble up
        response = await self.llm_provider.generate(request)

        if not response.success:
            raise Exception(f"LLM generation failed: {response.error}")

        return response.content.strip()

    def _generate_template_based_description(
        self,
        current_chunk: str,
        context_chunks: List[ChunkData]
    ) -> str:
        """Generate template-based contextual description as fallback.

        Args:
            current_chunk: The current chunk content
            context_chunks: List of context chunks

        Returns:
            Template-based contextual description
        """
        context_summary = f"Context from {len(context_chunks)} previous chunks" if context_chunks else "No previous context"

        # Extract key information from current chunk
        preview = current_chunk[:100] + "..." if len(current_chunk) > 100 else current_chunk

        return f"Contextual chunk: {context_summary}. Content: {preview}"
