"""Buffer-specific retrieval implementation for MemFuse server.

This module implements retrieval specifically for buffer layers (HybridBuffer, RoundBuffer)
using the existing RAG infrastructure for encoding and similarity calculation.
"""

import asyncio
from typing import Dict, List, Optional, Any, Union
from loguru import logger
import numpy as np

from ..base import BaseRetrieval
from ..encode.base import EncoderRegistry
from ..fusion.strategies import ScoreFusionStrategy, SimpleWeightedSum


class BufferRetrieval(BaseRetrieval):
    """Buffer-specific retrieval implementation.
    
    This class provides unified vector and text-based retrieval for all buffer types
    (HybridBuffer, RoundBuffer) using the existing RAG infrastructure.
    """

    def __init__(
        self,
        encoder_name: str = "minilm",
        encoder_config: Optional[Dict[str, Any]] = None,
        fusion_strategy: Optional[ScoreFusionStrategy] = None,
        similarity_threshold: float = 0.1,
        **kwargs
    ):
        """Initialize the buffer retrieval system.

        Args:
            encoder_name: Name of the encoder to use (registered in EncoderRegistry)
            encoder_config: Configuration for the encoder
            fusion_strategy: Strategy for fusing scores from different sources
            similarity_threshold: Minimum similarity threshold for results
            **kwargs: Additional arguments
        """
        self.encoder_name = encoder_name
        self.encoder_config = encoder_config or {}
        self.fusion_strategy = fusion_strategy or SimpleWeightedSum()
        self.similarity_threshold = similarity_threshold
        
        # Initialize encoder
        self.encoder = None
        self._encoder_lock = asyncio.Lock()
        
        logger.info(f"BufferRetrieval: Initialized with encoder={encoder_name}, threshold={similarity_threshold}")

    async def _get_encoder(self):
        """Get or initialize the encoder."""
        if self.encoder is None:
            async with self._encoder_lock:
                if self.encoder is None:
                    try:
                        self.encoder = EncoderRegistry.create(self.encoder_name, **self.encoder_config)
                        logger.info(f"BufferRetrieval: Initialized {self.encoder_name} encoder")
                    except Exception as e:
                        logger.error(f"BufferRetrieval: Failed to initialize encoder {self.encoder_name}: {e}")
                        raise
        return self.encoder

    async def retrieve(
        self,
        query: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        top_k: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant items from buffers.

        Args:
            query: Query string
            user_id: User ID (optional)
            session_id: Session ID (optional)
            top_k: Number of results to return
            **kwargs: Additional arguments including buffer references

        Returns:
            List of retrieved items with scores
        """
        try:
            # Extract buffer references from kwargs
            hybrid_buffer = kwargs.get('hybrid_buffer')
            round_buffer = kwargs.get('round_buffer')
            
            all_results = []
            
            # Retrieve from HybridBuffer using vector similarity
            if hybrid_buffer:
                hybrid_results = await self._retrieve_from_hybrid_buffer(
                    query, hybrid_buffer, top_k
                )
                all_results.extend(hybrid_results)
                logger.info(f"BufferRetrieval: Got {len(hybrid_results)} results from HybridBuffer")
            
            # Retrieve from RoundBuffer using text matching
            if round_buffer:
                round_results = await self._retrieve_from_round_buffer(
                    query, round_buffer, top_k
                )
                all_results.extend(round_results)
                logger.info(f"BufferRetrieval: Got {len(round_results)} results from RoundBuffer")
            
            # Deduplicate and sort results
            final_results = await self._deduplicate_and_sort(all_results, top_k)
            
            logger.info(f"BufferRetrieval: Returning {len(final_results)} total results")
            return final_results
            
        except Exception as e:
            logger.error(f"BufferRetrieval: Retrieval failed: {e}")
            return []

    async def _retrieve_from_hybrid_buffer(
        self,
        query: str,
        hybrid_buffer,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Retrieve from HybridBuffer using vector similarity."""
        try:
            if not hasattr(hybrid_buffer, 'chunks') or not hybrid_buffer.chunks:
                logger.debug("BufferRetrieval: HybridBuffer has no chunks")
                return []
            
            if not hasattr(hybrid_buffer, 'embeddings') or not hybrid_buffer.embeddings:
                logger.debug("BufferRetrieval: HybridBuffer has no embeddings")
                return []
            
            # Get encoder and generate query embedding
            encoder = await self._get_encoder()
            query_embedding = await encoder.encode_text(query)
            
            # Calculate similarities
            similarities = []
            async with hybrid_buffer._data_lock:
                for i, (chunk, chunk_embedding) in enumerate(zip(hybrid_buffer.chunks, hybrid_buffer.embeddings)):
                    if chunk_embedding is not None:
                        similarity = self._cosine_similarity(query_embedding, np.array(chunk_embedding))
                        if similarity >= self.similarity_threshold:
                            similarities.append({
                                'index': i,
                                'chunk': chunk,
                                'similarity': similarity
                            })
            
            # Sort by similarity and take top_k
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            top_results = similarities[:top_k]
            
            # Format results
            results = []
            for result in top_results:
                chunk = result['chunk']
                results.append({
                    'id': f"hybrid_vector_{result['index']}",
                    'content': chunk.content,
                    'score': result['similarity'],
                    'type': 'message',
                    'role': 'assistant',
                    'created_at': None,
                    'updated_at': None,
                    'metadata': {
                        **chunk.metadata,
                        'source': 'hybrid_buffer_vector',
                        'retrieval': {
                            'source': 'hybrid_buffer',
                            'method': 'vector_similarity',
                            'similarity_score': result['similarity'],
                            'encoder': self.encoder_name
                        }
                    }
                })
            
            return results
            
        except Exception as e:
            logger.error(f"BufferRetrieval: HybridBuffer retrieval failed: {e}")
            return []

    async def _retrieve_from_round_buffer(
        self,
        query: str,
        round_buffer,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Retrieve from RoundBuffer using enhanced text matching."""
        try:
            if not hasattr(round_buffer, 'rounds') or not round_buffer.rounds:
                logger.debug("BufferRetrieval: RoundBuffer has no rounds")
                return []
            
            query_lower = query.lower()
            results = []
            
            # Enhanced keyword matching
            query_words = set(query_lower.split())
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'user:', 'assistant:'}
            query_words = query_words - stop_words
            
            # Access RoundBuffer data safely
            if hasattr(round_buffer, '_lock'):
                async with round_buffer._lock:
                    results = await self._process_round_buffer_data(
                        round_buffer.rounds, query_words, query_lower, top_k
                    )
            else:
                results = await self._process_round_buffer_data(
                    round_buffer.rounds, query_words, query_lower, top_k
                )
            
            return results
            
        except Exception as e:
            logger.error(f"BufferRetrieval: RoundBuffer retrieval failed: {e}")
            return []

    async def _process_round_buffer_data(
        self,
        rounds: List[List[Dict]],
        query_words: set,
        query_lower: str,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Process RoundBuffer data for text matching."""
        results = []
        
        for round_idx, message_list in enumerate(rounds):
            if len(results) >= top_k:
                break
                
            for msg_idx, message in enumerate(message_list):
                if len(results) >= top_k:
                    break
                    
                content = message.get('content', '').lower()
                content_words = set(content.split()) - {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'user:', 'assistant:'}
                
                # Calculate overlap score
                if query_words and content_words:
                    overlap = len(query_words.intersection(content_words))
                    if overlap > 0:
                        score = overlap / len(query_words)
                        
                        if score >= self.similarity_threshold:
                            result = {
                                "id": f"round_{round_idx}_{msg_idx}",
                                "content": message.get('content', ''),
                                "score": score,
                                "type": "message",
                                "role": message.get('role', 'unknown'),
                                "created_at": message.get('created_at'),
                                "updated_at": message.get('updated_at'),
                                "metadata": {
                                    **message.get('metadata', {}),
                                    "source": "round_buffer",
                                    "round_index": round_idx,
                                    "message_index": msg_idx,
                                    "retrieval": {
                                        "source": "round_buffer",
                                        "method": "keyword_overlap",
                                        "overlap_score": overlap,
                                        "total_query_words": len(query_words)
                                    }
                                }
                            }
                            results.append(result)
        
        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            # Normalize vectors
            vec1_norm = vec1 / np.linalg.norm(vec1)
            vec2_norm = vec2 / np.linalg.norm(vec2)
            
            # Calculate cosine similarity
            similarity = np.dot(vec1_norm, vec2_norm)
            
            # Ensure result is between 0 and 1
            return max(0.0, min(1.0, (similarity + 1) / 2))
            
        except Exception as e:
            logger.error(f"BufferRetrieval: Cosine similarity calculation failed: {e}")
            return 0.0

    async def _deduplicate_and_sort(
        self,
        results: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Deduplicate and sort results by score."""
        try:
            # Deduplicate by content hash
            seen_content = set()
            unique_results = []
            
            for result in results:
                content_hash = hash(result.get('content', ''))
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    unique_results.append(result)
            
            # Sort by score (descending)
            unique_results.sort(key=lambda x: x.get('score', 0), reverse=True)
            
            return unique_results[:top_k]
            
        except Exception as e:
            logger.error(f"BufferRetrieval: Deduplication failed: {e}")
            return results[:top_k]
