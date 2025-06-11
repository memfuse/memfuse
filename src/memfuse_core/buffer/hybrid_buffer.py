"""HybridBuffer implementation for MemFuse Buffer.

The HybridBuffer manages both chunks (for vector search) and original rounds (for storage).
It implements FIFO logic and never fully clears, only removing the oldest data when
the buffer reaches capacity.
"""

import asyncio
from typing import Any, Dict, List, Optional, Callable
from loguru import logger

from ..interfaces import MessageList
from ..rag.chunk.base import ChunkData


class HybridBuffer:
    """Hybrid buffer managing both chunks and original rounds.
    
    This buffer maintains two parallel data structures:
    1. chunks: For vector search and retrieval
    2. original_rounds: For storage and Read API
    
    It implements FIFO logic and never fully clears.
    """
    
    def __init__(
        self,
        max_size: int = 5,
        chunk_strategy: str = "message",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """Initialize the HybridBuffer.
        
        Args:
            max_size: Maximum number of items to keep in buffer
            chunk_strategy: Chunking strategy to use
            embedding_model: Embedding model for vector generation
        """
        self.max_size = max_size
        self.chunk_strategy_name = chunk_strategy
        self.embedding_model_name = embedding_model
        
        # Parallel data structures
        self.chunks: List[ChunkData] = []
        self.embeddings: List[List[float]] = []
        self.original_rounds: List[MessageList] = []
        
        # Strategy and model instances (lazy loaded)
        self.chunk_strategy: Optional[Any] = None
        self.embedding_model: Optional[Any] = None
        
        # Storage handlers (set by BufferService)
        self.sqlite_handler: Optional[Callable] = None
        self.qdrant_handler: Optional[Callable] = None
        
        # Async lock for thread safety
        self._lock = asyncio.Lock()
        
        # Statistics
        self.total_rounds_received = 0
        self.total_chunks_created = 0
        self.total_flushes = 0
        self.total_fifo_removals = 0
        
        logger.info(f"HybridBuffer: Initialized with max_size={max_size}, chunk_strategy={chunk_strategy}")
    
    def set_storage_handlers(
        self,
        sqlite_handler: Optional[Callable] = None,
        qdrant_handler: Optional[Callable] = None
    ) -> None:
        """Set storage handlers for batch writing.
        
        Args:
            sqlite_handler: Handler for SQLite operations
            qdrant_handler: Handler for Qdrant operations
        """
        self.sqlite_handler = sqlite_handler
        self.qdrant_handler = qdrant_handler
        logger.debug("HybridBuffer: Storage handlers set")
    
    async def add_from_rounds(self, rounds: List[MessageList]) -> None:
        """Add rounds from RoundBuffer transfer.
        
        Args:
            rounds: List of MessageList objects from RoundBuffer
        """
        if not rounds:
            return
        
        async with self._lock:
            logger.info(f"HybridBuffer: Receiving {len(rounds)} rounds from RoundBuffer")
            
            # Lazy load chunk strategy
            if self.chunk_strategy is None:
                await self._load_chunk_strategy()
            
            # Lazy load embedding model
            if self.embedding_model is None:
                await self._load_embedding_model()
            
            for round_messages in rounds:
                # Create chunks from the round
                try:
                    chunks = await self.chunk_strategy.create_chunks([round_messages])
                    logger.debug(f"HybridBuffer: Created {len(chunks)} chunks from round")
                    
                    for chunk in chunks:
                        # Generate embedding
                        embedding = await self._generate_embedding(chunk.content)
                        
                        # Add to parallel structures
                        self.chunks.append(chunk)
                        self.embeddings.append(embedding)
                        self.original_rounds.append(round_messages)
                        self.total_chunks_created += 1
                        
                        # FIFO removal if over capacity
                        if len(self.chunks) > self.max_size:
                            self.chunks.pop(0)
                            self.embeddings.pop(0)
                            self.original_rounds.pop(0)
                            self.total_fifo_removals += 1
                            logger.debug("HybridBuffer: FIFO removal of oldest item")
                
                except Exception as e:
                    logger.error(f"HybridBuffer: Error processing round: {e}")
                    # Still add the original round for storage
                    self.original_rounds.append(round_messages)
                    if len(self.original_rounds) > self.max_size:
                        self.original_rounds.pop(0)
                        self.total_fifo_removals += 1
            
            self.total_rounds_received += len(rounds)
            logger.info(f"HybridBuffer: Now contains {len(self.chunks)} chunks, {len(self.original_rounds)} rounds")

            # TODO: Auto-flush disabled temporarily due to performance issues
            # Will be re-enabled after optimizing the flush process
            logger.info("HybridBuffer: Auto-flush disabled - data stored in memory buffer only")
            # flush_success = await self.flush_to_storage()
            # if flush_success:
            #     logger.info("HybridBuffer: Auto-flush completed successfully")
            # else:
            #     logger.error("HybridBuffer: Auto-flush failed")
    
    async def _load_chunk_strategy(self) -> None:
        """Lazy load the chunk strategy."""
        try:
            if self.chunk_strategy_name == "message":
                from ..rag.chunk.message import MessageChunkStrategy
                self.chunk_strategy = MessageChunkStrategy()
            elif self.chunk_strategy_name == "contextual":
                from ..rag.chunk.contextual import ContextualChunkStrategy
                self.chunk_strategy = ContextualChunkStrategy()
            else:
                # Default to message strategy
                from ..rag.chunk.message import MessageChunkStrategy
                self.chunk_strategy = MessageChunkStrategy()
                logger.warning(f"HybridBuffer: Unknown chunk strategy '{self.chunk_strategy_name}', using message")
            
            logger.debug(f"HybridBuffer: Loaded chunk strategy: {self.chunk_strategy_name}")
        except Exception as e:
            logger.error(f"HybridBuffer: Failed to load chunk strategy: {e}")
            # Create a minimal fallback strategy
            self.chunk_strategy = self._create_fallback_strategy()
    
    def _create_fallback_strategy(self):
        """Create a minimal fallback chunk strategy."""
        class FallbackStrategy:
            async def create_chunks(self, message_batch_list):
                chunks = []
                for batch_index, message_list in enumerate(message_batch_list):
                    content = " ".join(msg.get("content", "") for msg in message_list)
                    chunk = ChunkData(
                        content=content,
                        metadata={"strategy": "fallback", "batch_index": batch_index}
                    )
                    chunks.append(chunk)
                return chunks
        
        return FallbackStrategy()
    
    async def _load_embedding_model(self) -> None:
        """Lazy load the embedding model."""
        try:
            from ..utils.embeddings import create_embedding
            self.embedding_model = create_embedding
            logger.debug(f"HybridBuffer: Loaded embedding model: {self.embedding_model_name}")
        except Exception as e:
            logger.error(f"HybridBuffer: Failed to load embedding model: {e}")
            self.embedding_model = self._create_fallback_embedding
    
    async def _create_fallback_embedding(self, text: str) -> List[float]:
        """Create a fallback embedding."""
        # Simple hash-based embedding
        import hashlib
        hash_obj = hashlib.md5(text.encode())
        hash_bytes = hash_obj.digest()
        # Convert to 384-dimensional vector (matching all-MiniLM-L6-v2)
        embedding = []
        for i in range(384):
            byte_index = i % len(hash_bytes)
            value = (hash_bytes[byte_index] / 128.0) - 1.0
            embedding.append(value)
        return embedding
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            if callable(self.embedding_model):
                if asyncio.iscoroutinefunction(self.embedding_model):
                    return await self.embedding_model(text, model=self.embedding_model_name)
                else:
                    return self.embedding_model(text, model=self.embedding_model_name)
            else:
                return await self._create_fallback_embedding(text)
        except Exception as e:
            logger.error(f"HybridBuffer: Embedding generation failed: {e}")
            return await self._create_fallback_embedding(text)
    
    async def flush_to_storage(self) -> bool:
        """Flush all data to persistent storage.
        
        Returns:
            True if flush was successful, False otherwise
        """
        async with self._lock:
            if not self.original_rounds:
                logger.debug("HybridBuffer: No data to flush")
                return True
            
            try:
                # Prepare data for storage
                rounds_to_write = self.original_rounds.copy()
                chunks_to_write = self.chunks.copy()
                
                logger.info(f"HybridBuffer: Flushing {len(rounds_to_write)} rounds and {len(chunks_to_write)} chunks")

                # Parallel write to SQLite and Qdrant
                write_tasks = []

                if self.sqlite_handler and rounds_to_write:
                    logger.info("HybridBuffer: Adding SQLite write task")
                    write_tasks.append(self._write_to_sqlite(rounds_to_write))

                if self.qdrant_handler and chunks_to_write:
                    logger.info("HybridBuffer: Adding Qdrant write task")
                    write_tasks.append(self._write_to_qdrant(chunks_to_write))

                if write_tasks:
                    logger.info(f"HybridBuffer: Executing {len(write_tasks)} write tasks")
                    results = await asyncio.gather(*write_tasks, return_exceptions=True)
                    logger.info("HybridBuffer: Write tasks completed")
                    
                    # Check for errors
                    for i, result in enumerate(results):
                        if isinstance(result, Exception):
                            logger.error(f"HybridBuffer: Storage write {i} failed: {result}")
                            return False
                
                # Clear original rounds after successful write
                self.original_rounds.clear()
                self.total_flushes += 1
                
                logger.info("HybridBuffer: Flush completed successfully")
                return True
                
            except Exception as e:
                logger.error(f"HybridBuffer: Flush failed: {e}")
                return False
    
    async def _write_to_sqlite(self, rounds: List[MessageList]) -> None:
        """Write rounds to SQLite storage.
        
        Args:
            rounds: List of MessageList objects to write
        """
        if self.sqlite_handler:
            logger.info(f"HybridBuffer: Starting SQLite write for {len(rounds)} rounds")
            await self.sqlite_handler(rounds)
            logger.info(f"HybridBuffer: Completed SQLite write for {len(rounds)} rounds")
    
    async def _write_to_qdrant(self, chunks: List[ChunkData]) -> None:
        """Write chunks to Qdrant storage.
        
        Args:
            chunks: List of ChunkData objects to write
        """
        if self.qdrant_handler:
            # Convert chunks to points format expected by Qdrant
            points = []
            for i, chunk in enumerate(chunks):
                if i < len(self.embeddings):
                    point = {
                        "id": f"chunk_{hash(chunk.content)}_{i}",
                        "vector": self.embeddings[i],
                        "payload": {
                            "content": chunk.content,
                            "metadata": chunk.metadata
                        }
                    }
                    points.append(point)
            
            if points:
                await self.qdrant_handler(points)
                logger.debug(f"HybridBuffer: Wrote {len(points)} chunks to Qdrant")
    
    async def get_all_messages_for_read_api(
        self,
        limit: Optional[int] = None,
        sort_by: str = "timestamp",
        order: str = "desc"
    ) -> List[Dict[str, Any]]:
        """Get all messages in buffer for Read API.
        
        Args:
            limit: Maximum number of messages to return
            sort_by: Field to sort by ('timestamp' or 'id')
            order: Sort order ('asc' or 'desc')
            
        Returns:
            List of message dictionaries formatted for API response
        """
        async with self._lock:
            all_messages = []
            
            for round_messages in self.original_rounds:
                for message in round_messages:
                    # Convert to API format
                    api_message = {
                        "id": message.get("id", ""),
                        "role": message.get("role", "user"),
                        "content": message.get("content", ""),
                        "created_at": message.get("created_at", ""),
                        "updated_at": message.get("updated_at", ""),
                        "metadata": message.get("metadata", {}).copy()
                    }
                    # Add buffer source metadata
                    api_message["metadata"]["source"] = "hybrid_buffer"
                    all_messages.append(api_message)
            
            # Sort messages
            if sort_by == "timestamp":
                all_messages.sort(
                    key=lambda x: x.get("created_at", ""),
                    reverse=(order == "desc")
                )
            elif sort_by == "id":
                all_messages.sort(
                    key=lambda x: x.get("id", ""),
                    reverse=(order == "desc")
                )
            
            # Apply limit
            if limit is not None and limit > 0:
                all_messages = all_messages[:limit]
            
            return all_messages
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics.
        
        Returns:
            Dictionary with buffer statistics
        """
        return {
            "chunks_count": len(self.chunks),
            "rounds_count": len(self.original_rounds),
            "embeddings_count": len(self.embeddings),
            "max_size": self.max_size,
            "chunk_strategy": self.chunk_strategy_name,
            "embedding_model": self.embedding_model_name,
            "total_rounds_received": self.total_rounds_received,
            "total_chunks_created": self.total_chunks_created,
            "total_flushes": self.total_flushes,
            "total_fifo_removals": self.total_fifo_removals,
            "has_sqlite_handler": self.sqlite_handler is not None,
            "has_qdrant_handler": self.qdrant_handler is not None
        }
