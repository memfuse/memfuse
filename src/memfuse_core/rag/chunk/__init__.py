"""Chunk module for MemFuse RAG system.

This module provides chunking functionality for processing message lists
into retrievable chunks with various strategies.
"""

from .base import ChunkData, ChunkStrategy
from .message import MessageChunkStrategy
from .contextual import ContextualChunkStrategy
from .character import CharacterChunkStrategy
from .message_character import MessageCharacterChunkStrategy

__all__ = [
    "ChunkData",
    "ChunkStrategy",
    "MessageChunkStrategy",
    "ContextualChunkStrategy",
    "CharacterChunkStrategy",
    "MessageCharacterChunkStrategy",
]
