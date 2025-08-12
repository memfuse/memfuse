"""
PgVectorScale Memory Layer Implementation.

This module provides a Memory Layer implementation that uses pgvectorscale
for high-performance vector similarity search with M0/M1 architecture.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional
from datetime import datetime
from loguru import logger

from ..interfaces.memory_layer import (
    MemoryLayer,
    MemoryLayerConfig,
    WriteResult,
    QueryResult,
    LayerStatus
)
from ..interfaces.message_interface import MessageBatchList
from ..store.vector_store.pgvectorscale_store import PgVectorScaleStore
from ..rag.chunk.base import ChunkData


class PgVectorScaleMemoryLayer(MemoryLayer):
    """
    Memory Layer implementation using pgvectorscale for high-performance vector search.
    
    This implementation provides:
    - M0 Layer: Raw streaming messages with metadata
    - M1 Layer: Intelligent chunks with embeddings and StreamingDiskANN indexing
    - Normalized similarity scores (0-1 range) for cross-system compatibility
    - Integration with existing chunking strategies and global embedding models
    """
    
    def __init__(
        self,
        user_id: str,
        config: Optional[MemoryLayerConfig] = None,
        db_config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the PgVectorScale Memory Layer.

        Args:
            user_id: User identifier
            config: Memory layer configuration
            db_config: Database connection configuration
        """
        self.user_id = user_id
        self.config = config or MemoryLayerConfig()
        self.initialized = False
        
        # Database configuration
        self.db_config = db_config or {
            'host': 'localhost',
            'port': 5432,
            'database': 'memfuse',
            'user': 'postgres',
            'password': 'postgres'
        }
        
        # Initialize pgvectorscale store
        self.store = PgVectorScaleStore(
            db_config=self.db_config,
            model_name="all-MiniLM-L6-v2",
            cache_size=1000,
            buffer_size=50
        )
        
        # Layer status
        self.layer_status = {
            "M0": LayerStatus.INACTIVE,
            "M1": LayerStatus.INACTIVE
        }
        
        # Performance metrics
        self.total_operations = 0
        self.total_write_time = 0.0
        self.total_query_time = 0.0
        
        logger.info(f"PgVectorScaleMemoryLayer: Initialized for user {user_id}")
    
    async def initialize(self) -> bool:
        """Initialize the memory layer and underlying store."""
        try:
            logger.info("PgVectorScaleMemoryLayer: Initializing...")
            
            # Initialize the pgvectorscale store
            await self.store.initialize()
            
            # Update layer status
            self.layer_status["M0"] = LayerStatus.ACTIVE
            self.layer_status["M1"] = LayerStatus.ACTIVE
            self.initialized = True
            
            logger.info("PgVectorScaleMemoryLayer: Initialization complete")
            return True
            
        except Exception as e:
            logger.error(f"PgVectorScaleMemoryLayer: Initialization failed: {e}")
            self.layer_status["M0"] = LayerStatus.ERROR
            self.layer_status["M1"] = LayerStatus.ERROR
            return False
    
    async def write(
        self,
        message_batch_list: MessageBatchList,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> WriteResult:
        """Write message batches to the memory layer using M0 -> M1 pipeline.
        
        Args:
            message_batch_list: List of message batches to write
            session_id: Optional session identifier
            metadata: Optional metadata dictionary
            
        Returns:
            WriteResult with operation details
        """
        if not self.initialized:
            await self.initialize()
        
        start_time = time.time()
        self.total_operations += 1
        
        try:
            # Convert message batches to ChunkData objects
            chunks = await self._convert_messages_to_chunks(
                message_batch_list, session_id, metadata
            )
            
            if not chunks:
                logger.warning("PgVectorScaleMemoryLayer: No valid chunks to write")
                return WriteResult(
                    success=False,
                    message="No valid chunks to write",
                    items_written=0,
                    metadata={"error": "no_valid_chunks"}
                )
            
            # Write chunks using the pgvectorscale store (M0 -> M1 pipeline)
            chunk_ids = await self.store.add(chunks)
            
            # Calculate metrics
            write_time = time.time() - start_time
            self.total_write_time += write_time
            
            logger.info(f"PgVectorScaleMemoryLayer: Wrote {len(chunks)} chunks in {write_time:.3f}s")
            
            return WriteResult(
                success=True,
                message=f"Successfully wrote {len(chunks)} chunks to M0/M1 layers",
                layer_results={
                    "M0": {"success": True, "items_processed": len(chunks)},
                    "M1": {"success": True, "items_processed": len(chunks)}
                },
                metadata={
                    "chunk_ids": chunk_ids,
                    "session_id": session_id,
                    "write_time": write_time,
                    "layers_written": ["M0", "M1"]
                }
            )
            
        except Exception as e:
            write_time = time.time() - start_time
            logger.error(f"PgVectorScaleMemoryLayer: Write failed after {write_time:.3f}s: {e}")
            
            return WriteResult(
                success=False,
                message=f"Write operation failed: {str(e)}",
                layer_results={
                    "M0": {"success": False, "error": str(e)},
                    "M1": {"success": False, "error": str(e)}
                },
                metadata={"error": str(e), "write_time": write_time}
            )
    
    async def query(
        self,
        query: str,
        top_k: int = 10,
        **kwargs
    ) -> QueryResult:
        """Query the memory layer using high-performance vector similarity search.
        
        Args:
            query: Query string
            top_k: Number of top results to return
            **kwargs: Additional query parameters
            
        Returns:
            QueryResult with search results
        """
        if not self.initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # Perform vector similarity search using pgvectorscale
            similar_chunks = await self.store.query(query, top_k, **kwargs)
            
            # Convert ChunkData objects to result format
            results = []
            for chunk in similar_chunks:
                result_item = {
                    "content": chunk.content,
                    "chunk_id": chunk.chunk_id,
                    "similarity_score": chunk.metadata.get("similarity_score", 0.0),
                    "distance": chunk.metadata.get("distance", 1.0),
                    "source": chunk.metadata.get("source", "pgvectorscale_m1"),
                    "metadata": chunk.metadata
                }
                results.append(result_item)
            
            # Calculate metrics
            query_time = time.time() - start_time
            self.total_query_time += query_time
            
            logger.info(f"PgVectorScaleMemoryLayer: Query returned {len(results)} results in {query_time:.3f}s")
            
            return QueryResult(
                results=results,
                layer_sources={"M0": [], "M1": results, "M2": []},
                total_count=len(results),
                metadata={
                    "query": query,
                    "top_k": top_k,
                    "query_time": query_time,
                    "search_strategy": "pgvectorscale_streamingdiskann",
                    "active_layers": ["M0", "M1"]
                }
            )
            
        except Exception as e:
            query_time = time.time() - start_time
            logger.error(f"PgVectorScaleMemoryLayer: Query failed after {query_time:.3f}s: {e}")
            
            return QueryResult(
                results=[],
                layer_sources={"M0": [], "M1": [], "M2": []},
                total_count=0,
                metadata={
                    "error": str(e),
                    "query_time": query_time,
                    "query": query
                }
            )
    
    async def _convert_messages_to_chunks(
        self,
        message_batch_list: MessageBatchList,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[ChunkData]:
        """Convert message batches to ChunkData objects."""
        chunks = []
        global_sequence_number = 1  # Global sequence number across all batches

        for batch_idx, message_batch in enumerate(message_batch_list):
            for msg_idx, message in enumerate(message_batch):
                # Extract message content and metadata
                content = message.get("content", "")
                if not content or not content.strip():
                    continue

                # Create chunk metadata
                chunk_metadata = {
                    "role": message.get("role", "user"),
                    "conversation_id": session_id or f"batch_{batch_idx}",
                    "sequence_number": global_sequence_number,
                    "batch_index": batch_idx,
                    "message_index": msg_idx,
                    "timestamp": datetime.now().isoformat(),
                    **(metadata or {}),
                    **(message.get("metadata", {}))
                }

                # Create ChunkData object
                chunk = ChunkData(
                    content=content,
                    metadata=chunk_metadata
                )
                chunks.append(chunk)
                global_sequence_number += 1  # Increment global sequence number
        
        logger.info(f"PgVectorScaleMemoryLayer: Converted {len(message_batch_list)} batches to {len(chunks)} chunks")
        return chunks
    
    async def get_layer_status(self) -> Dict[str, LayerStatus]:
        """Get the current status of all memory layers."""
        return self.layer_status.copy()
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get memory layer statistics."""
        try:
            # Get store statistics
            store_stats = await self.store.get_stats()
            
            # Calculate performance metrics
            avg_write_time = self.total_write_time / max(1, self.total_operations)
            avg_query_time = self.total_query_time / max(1, self.total_operations)
            
            return {
                "layer_status": self.layer_status,
                "total_operations": self.total_operations,
                "average_write_time": avg_write_time,
                "average_query_time": avg_query_time,
                "total_write_time": self.total_write_time,
                "total_query_time": self.total_query_time,
                "store_stats": store_stats,
                "initialized": self.initialized
            }
            
        except Exception as e:
            logger.error(f"PgVectorScaleMemoryLayer: Stats retrieval failed: {e}")
            return {
                "error": str(e),
                "layer_status": self.layer_status,
                "initialized": self.initialized
            }
    
    async def close(self) -> None:
        """Close the memory layer and cleanup resources."""
        try:
            if self.store:
                await self.store.close()
            
            self.layer_status = {
                "M0": LayerStatus.INACTIVE,
                "M1": LayerStatus.INACTIVE
            }
            self.initialized = False
            
            logger.info("PgVectorScaleMemoryLayer: Closed successfully")
            
        except Exception as e:
            logger.error(f"PgVectorScaleMemoryLayer: Close failed: {e}")
    
    async def cleanup(self) -> bool:
        """Cleanup resources and connections."""
        try:
            await self.close()
            return True
        except Exception as e:
            logger.error(f"PgVectorScaleMemoryLayer: Cleanup failed: {e}")
            return False

    async def reset_layer(self) -> bool:
        """Reset the memory layer to initial state."""
        try:
            # For now, we don't implement full reset functionality
            # This would require clearing all data from M0/M1 tables
            logger.warning("PgVectorScaleMemoryLayer: Reset not implemented")
            return False
        except Exception as e:
            logger.error(f"PgVectorScaleMemoryLayer: Reset failed: {e}")
            return False

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the memory layer."""
        try:
            health_status = {
                "status": "healthy",
                "initialized": self.initialized,
                "layer_status": self.layer_status,
                "database_connection": False,
                "store_health": {}
            }

            # Check database connection
            if self.store and self.store.conn:
                try:
                    with self.store.conn.cursor() as cur:
                        cur.execute("SELECT 1")
                        health_status["database_connection"] = True
                except Exception:
                    health_status["database_connection"] = False
                    health_status["status"] = "unhealthy"

            # Get store statistics for health info
            if self.initialized:
                store_stats = await self.store.get_stats()
                health_status["store_health"] = store_stats

            return health_status

        except Exception as e:
            logger.error(f"PgVectorScaleMemoryLayer: Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "initialized": self.initialized
            }

    async def get_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics about the memory layer."""
        return await self.get_stats()

    async def write_parallel(
        self,
        message_batch_list: MessageBatchList,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> WriteResult:
        """Write message batches in parallel (fallback to sequential for now)."""
        # For now, fallback to sequential write
        # In the future, this could implement parallel processing across multiple workers
        logger.info("PgVectorScaleMemoryLayer: Parallel write not implemented, using sequential")
        return await self.write(message_batch_list, session_id, metadata)

    def __del__(self):
        """Cleanup on object destruction."""
        if hasattr(self, 'store') and self.store:
            try:
                # Note: Can't use await in __del__, so we just close the connection
                if hasattr(self.store, 'conn') and self.store.conn:
                    self.store.conn.close()
            except Exception:
                pass
