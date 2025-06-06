"""Hybrid retrieval implementation for MemFuse server.

This module implements a hybrid retrieval system that combines results from
vector, graph, and keyword stores to provide more comprehensive and accurate
retrieval results.
"""

import asyncio
from loguru import logger
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from ..base import BaseRetrieval
from ...models import StoreType, Query, QueryResult
from ..chunk.base import ChunkData

# Import storage types
from ...store.vector_store.base import VectorStore
from ...store.graph_store.base import GraphStore
from ...store.keyword_store.base import KeywordStore

# Import score fusion strategies
from ..fusion import (
    SimpleWeightedSum,
    NormalizedWeightedSum,
    ReciprocalRankFusion
)


@dataclass
class ContextualRetrievalResult:
    """Contextual retrieval result."""
    similar_chunks: List[ChunkData]
    connected_contextual: List[ChunkData]
    similar_contextual: List[ChunkData]
    total_pieces: int
    formatted_context: str
    retrieval_stats: Dict[str, Any]


class HybridRetrieval(BaseRetrieval):
    """Hybrid retrieval implementation.

    This class combines results from vector, graph, and keyword stores
    to provide more comprehensive and accurate retrieval results.
    """

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        graph_store: Optional[GraphStore] = None,
        keyword_store: Optional[KeywordStore] = None,
        cache_size: int = 100,
        vector_weight: float = 0.5,
        graph_weight: float = 0.3,
        keyword_weight: float = 0.2,
        fusion_strategy: str = "rrf"
    ):
        """Initialize the hybrid retrieval.

        Args:
            vector_store: Vector store instance
            graph_store: Graph store instance
            keyword_store: Keyword store instance
            cache_size: Size of the query cache
            vector_weight: Weight for vector store results
            graph_weight: Weight for graph store results
            keyword_weight: Weight for keyword store results
            fusion_strategy: Score fusion strategy to use ('simple', 'normalized', or 'rrf')
        """
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.keyword_store = keyword_store

        # Initialize query cache
        self.query_cache = {}  # Simple dictionary cache for now
        self.cache_size = cache_size

        # Set weights
        self.weights = {
            StoreType.VECTOR: vector_weight,
            StoreType.GRAPH: graph_weight,
            StoreType.KEYWORD: keyword_weight
        }

        # Set fusion strategy
        self.fusion_strategy_name = fusion_strategy
        if fusion_strategy == "simple":
            self.fusion_strategy = SimpleWeightedSum()
        elif fusion_strategy == "normalized":
            self.fusion_strategy = NormalizedWeightedSum()
        elif fusion_strategy == "rrf":
            # Use a smaller k value (0.2) to give more weight to top results
            self.fusion_strategy = ReciprocalRankFusion(k=0.2)
        else:
            # Default to RRF with smaller k value
            self.fusion_strategy = ReciprocalRankFusion(k=0.2)

    async def retrieve(
        self,
        query: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        top_k: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant items based on the query.

        Args:
            query: Query string
            user_id: User ID (optional)
            session_id: Session ID (optional)
            top_k: Number of results to return
            **kwargs: Additional arguments

        Returns:
            List of retrieved items
        """
        # Create query object
        query_obj = Query(
            text=query,
            metadata={
                "user_id": user_id,
                "session_id": session_id,
                **kwargs
            }
        )

        # Get use flags from kwargs
        use_vector = kwargs.get("use_vector", True)
        use_graph = kwargs.get("use_graph", True)
        use_keyword = kwargs.get("use_keyword", True)

        # Query all stores
        results = await self._query(
            query_obj,
            top_k=top_k,
            use_vector=use_vector,
            use_graph=use_graph,
            use_keyword=use_keyword
        )

        # Check if we have any results
        if not results:
            logger.warning(f"No results found for query: {query}")
            return []

        # Merge results
        merged_results = self._merge_results(results, top_k, query_obj)

        # Convert to dictionaries
        result_dicts = []
        for result in merged_results:
            # Convert to dictionary
            result_dict = {
                "id": result.id,
                "content": result.content,
                "score": result.score,
                "metadata": result.metadata
            }
            result_dicts.append(result_dict)

        return result_dicts

    async def contextual_retrieve(
        self,
        query: str,
        session_id: str,
        top_chunks: int = 10,
        top_contextual: int = 10,
        similarity_threshold: float = 0.0
    ) -> ContextualRetrievalResult:
        """Three-layer contextual retrieval.

        Args:
            query: Search query text
            session_id: Session ID to filter chunks
            top_chunks: Number of top similar chunks to retrieve
            top_contextual: Number of top similar contextual chunks
            similarity_threshold: Minimum similarity score

        Returns:
            ContextualRetrievalResult with all retrieved content
        """
        logger.info(f"Contextual retrieval for query: {query[:50]}...")

        try:
            # Execute three-layer retrieval strategy
            similar_chunks, connected_contextual, similar_contextual = await asyncio.gather(
                self._find_similar_chunks(query, session_id, top_chunks, similarity_threshold),
                self._find_connected_contextual(query, session_id, top_chunks),
                self._find_similar_contextual(query, session_id, top_contextual, similarity_threshold),
                return_exceptions=True
            )

            # Handle exceptions
            if isinstance(similar_chunks, Exception):
                logger.error(f"Error finding similar chunks: {similar_chunks}")
                similar_chunks = []
            if isinstance(connected_contextual, Exception):
                logger.error(f"Error finding connected contextual: {connected_contextual}")
                connected_contextual = []
            if isinstance(similar_contextual, Exception):
                logger.error(f"Error finding similar contextual: {similar_contextual}")
                similar_contextual = []

            # Calculate statistics
            total_pieces = len(similar_chunks) + len(connected_contextual) + len(similar_contextual)

            # Format contextual context
            formatted_context = self._format_contextual_results(
                similar_chunks, connected_contextual, similar_contextual
            )

            retrieval_stats = {
                "similar_chunks_count": len(similar_chunks),
                "connected_contextual_count": len(connected_contextual),
                "similar_contextual_count": len(similar_contextual),
                "total_pieces": total_pieces,
                "query_length": len(query),
                "session_id": session_id
            }

            logger.info(f"Contextual retrieval completed:")
            logger.info(f"  - Similar chunks: {len(similar_chunks)}")
            logger.info(f"  - Connected contextual: {len(connected_contextual)}")
            logger.info(f"  - Similar contextual: {len(similar_contextual)}")
            logger.info(f"  - Total pieces: {total_pieces}")

            return ContextualRetrievalResult(
                similar_chunks=similar_chunks,
                connected_contextual=connected_contextual,
                similar_contextual=similar_contextual,
                total_pieces=total_pieces,
                formatted_context=formatted_context,
                retrieval_stats=retrieval_stats
            )

        except Exception as e:
            logger.error(f"Error in contextual retrieval: {e}")
            return ContextualRetrievalResult(
                similar_chunks=[],
                connected_contextual=[],
                similar_contextual=[],
                total_pieces=0,
                formatted_context="",
                retrieval_stats={"error": str(e)}
            )

    async def _query(
        self,
        query: Query,
        top_k: int = 5,
        use_vector: bool = True,
        use_graph: bool = True,
        use_keyword: bool = True
    ) -> List[QueryResult]:
        """Query all stores and combine results.

        Args:
            query: Query to execute
            top_k: Number of results to return
            use_vector: Whether to use vector store
            use_graph: Whether to use graph store
            use_keyword: Whether to use keyword store

        Returns:
            List of query results
        """
        all_results = []

        # Check cache first
        cache_key = f"{query.text}_{top_k}_{use_vector}_{use_graph}_{use_keyword}"
        if cache_key in self.query_cache:
            logger.debug(f"Cache hit for query: {query.text}")
            return self.query_cache[cache_key]

        # Query vector store
        if use_vector and self.vector_store:
            try:
                logger.info(f"Querying vector store with query: {query.text[:50]}...")
                vector_results = await self._query_store(
                    self.vector_store, query, top_k)
                all_results.extend(vector_results)
                logger.info(
                    f"Retrieved {len(vector_results)} results from vector store")
            except Exception as e:
                logger.error(f"Error querying vector store: {e}")

        # Query graph store
        if use_graph and self.graph_store:
            try:
                graph_results = await self._query_store(
                    self.graph_store, query, top_k)
                all_results.extend(graph_results)
                logger.debug(
                    f"Retrieved {len(graph_results)} results from graph store")
            except Exception as e:
                logger.error(f"Error querying graph store: {e}")

        # Query keyword store
        if use_keyword and self.keyword_store:
            try:
                logger.info(f"Querying keyword store with query: {query.text[:50]}...")
                keyword_results = await self._query_store(
                    self.keyword_store, query, top_k)
                all_results.extend(keyword_results)
                logger.info(
                    f"Retrieved {len(keyword_results)} results from keyword store")
            except Exception as e:
                logger.error(f"Error querying keyword store: {e}")

        # Update cache
        if len(self.query_cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.query_cache))
            del self.query_cache[oldest_key]

        self.query_cache[cache_key] = all_results

        return all_results

    async def _query_store(
        self,
        store: Any,
        query: Query,
        top_k: int
    ) -> List[QueryResult]:
        """Query a store.

        Args:
            store: Store to query
            query: Query to execute
            top_k: Number of results to return

        Returns:
            List of query results
        """
        try:
            store_type = getattr(store, 'store_type', 'unknown')
            # Create a preview of the query text
            if len(query.text) > 40:
                query_preview = query.text[:40] + "..."
            else:
                query_preview = query.text

            # Log user_id for debugging
            user_id = query.metadata.get("user_id", "none")
            logger.debug(
                f"Querying {store_type} store with query: {query_preview}, user_id: {user_id}")

            # The query object contains user_id which will be used for filtering at the database level
            results = await store.query(query, top_k)
            logger.debug(
                f"Retrieved {len(results)} results from {store_type} store with user_id filter: {user_id}")

            # Apply user_id filter as a post-processing step
            if user_id != "none" and results:
                filtered_results = []
                for result in results:
                    result_user_id = result.metadata.get("user_id")
                    if result_user_id == user_id:
                        filtered_results.append(result)
                    else:
                        logger.debug(
                            f"Store filtering: Removing result with user_id={result_user_id}, expected {user_id}")

                if len(filtered_results) != len(results):
                    logger.debug(
                        f"Filtered {len(results) - len(filtered_results)} results from {store_type} store")

                results = filtered_results

            logger.debug(f"Got {len(results)} results from {store_type} store")
            return results
        except Exception as e:
            logger.error(f"Error querying store: {e}", exc_info=True)
            return []

    def _merge_results(
        self,
        results: List[QueryResult],
        top_k: int,
        query: Optional[Query] = None
    ) -> List[QueryResult]:
        """Merge and deduplicate results.

        Args:
            results: List of query results
            top_k: Number of results to return
            query: Original query object for filtering (optional)

        Returns:
            List of merged query results
        """
        # Group results by ID
        result_map: Dict[str, List[QueryResult]] = {}

        for result in results:
            if result.id not in result_map:
                result_map[result.id] = []

            result_map[result.id].append(result)

        # Use the selected fusion strategy to merge results
        merged_results = self.fusion_strategy.fuse_scores(
            result_map, self.weights)

        # Add fusion strategy to metadata
        for result in merged_results:
            if "retrieval" in result.metadata:
                # Store the fusion strategy name in metadata
                retrieval_meta = result.metadata["retrieval"]
                retrieval_meta["fusion_strategy"] = self.fusion_strategy_name

        # Apply user_id filter if provided
        if query and query.metadata and "user_id" in query.metadata:
            user_id = query.metadata["user_id"]
            filtered_results = []
            for result in merged_results:
                result_user_id = result.metadata.get("user_id")
                if result_user_id == user_id:
                    filtered_results.append(result)
                else:
                    logger.debug(
                        f"Merge filtering: Removing result with user_id={result_user_id}, expected {user_id}")

            merged_results = filtered_results

        # Sort by score and limit to top_k
        merged_results.sort(key=lambda x: x.score, reverse=True)

        return merged_results[:top_k]

    async def _find_similar_chunks(
        self,
        query: str,
        session_id: str,
        top_k: int,
        similarity_threshold: float
    ) -> List[ChunkData]:
        """Layer 1: Find chunks similar to query based on original content."""
        if not self.vector_store:
            logger.warning("No vector store available for similar chunk search")
            return []

        try:
            # Create query object with session filter
            query_obj = Query(
                text=query,
                metadata={"session_id": session_id}
            )

            # Query vector store for similar chunks
            chunks = await self.vector_store.query(query_obj, top_k * 2)  # Get more for filtering

            # Filter by similarity threshold if needed
            if similarity_threshold > 0.0:
                filtered_chunks = []
                for chunk in chunks:
                    # Assume chunks have similarity score in metadata
                    similarity = chunk.metadata.get("similarity", 1.0)
                    if similarity >= similarity_threshold:
                        filtered_chunks.append(chunk)
                chunks = filtered_chunks

            return chunks[:top_k]

        except Exception as e:
            logger.error(f"Error finding similar chunks: {e}")
            return []

    async def _find_connected_contextual(
        self,
        query: str,
        session_id: str,
        top_k: int
    ) -> List[ChunkData]:
        """Layer 2: Find contextual descriptions connected to similar chunks."""
        try:
            # First get similar chunks
            similar_chunks = await self._find_similar_chunks(query, session_id, top_k, 0.0)
            if not similar_chunks:
                return []

            # Extract chunk IDs
            chunk_ids = [chunk.chunk_id for chunk in similar_chunks if hasattr(chunk, 'chunk_id')]
            if not chunk_ids:
                return []

            # Get all chunks for session
            query_obj = Query(
                text="",  # Empty query for session-based retrieval
                metadata={"session_id": session_id}
            )

            session_chunks = await self.vector_store.query(query_obj, 1000)  # Get all session chunks

            # Find chunks with contextual descriptions that are connected to similar chunks
            connected_contextual = []
            for chunk in session_chunks:
                # Check if this chunk has contextual description and is connected
                if (hasattr(chunk, 'metadata') and
                    chunk.metadata.get('contextual_description') and
                    hasattr(chunk, 'chunk_id') and
                    chunk.chunk_id in chunk_ids):
                    connected_contextual.append(chunk)

            return connected_contextual

        except Exception as e:
            logger.error(f"Error finding connected contextual chunks: {e}")
            return []

    async def _find_similar_contextual(
        self,
        query: str,
        session_id: str,
        top_k: int,
        similarity_threshold: float
    ) -> List[ChunkData]:
        """Layer 3: Find chunks with contextual descriptions similar to query."""
        if not self.vector_store:
            return []

        try:
            # Get all chunks for session
            query_obj = Query(
                text="",  # Empty query for session-based retrieval
                metadata={"session_id": session_id}
            )

            session_chunks = await self.vector_store.query(query_obj, 1000)  # Get all session chunks

            # Filter chunks that have contextual descriptions
            contextual_chunks = []
            for chunk in session_chunks:
                if (hasattr(chunk, 'metadata') and
                    chunk.metadata.get('contextual_description')):
                    contextual_chunks.append(chunk)

            if not contextual_chunks:
                return []

            # For now, use keyword matching on contextual descriptions
            # TODO: Implement proper embedding similarity for contextual descriptions
            query_words = set(query.lower().split())
            scored_chunks = []

            for chunk in contextual_chunks:
                description = chunk.metadata.get('contextual_description', '').lower()
                description_words = set(description.split())

                # Simple word overlap scoring
                overlap = len(query_words.intersection(description_words))
                if overlap > 0:
                    # Add similarity score to metadata
                    chunk.metadata['similarity'] = overlap / len(query_words)
                    if chunk.metadata['similarity'] >= similarity_threshold:
                        scored_chunks.append(chunk)

            # Sort by similarity and return top_k
            scored_chunks.sort(key=lambda x: x.metadata.get('similarity', 0), reverse=True)
            return scored_chunks[:top_k]

        except Exception as e:
            logger.error(f"Error finding similar contextual chunks: {e}")
            return []

    def _format_contextual_results(
        self,
        similar_chunks: List[ChunkData],
        connected_contextual: List[ChunkData],
        similar_contextual: List[ChunkData]
    ) -> str:
        """Format retrieval results into contextual context string."""
        context_parts = []

        # Add similar chunks (original content)
        if similar_chunks:
            context_parts.append("=== SIMILAR CHUNKS (Original Content) ===")
            for i, chunk in enumerate(similar_chunks):
                similarity_info = ""
                if hasattr(chunk, 'metadata') and chunk.metadata.get('similarity'):
                    similarity_info = f" (similarity: {chunk.metadata['similarity']:.3f})"
                context_parts.append(f"Chunk {i+1}{similarity_info}:\n{chunk.content}")

        # Add connected contextual chunks
        if connected_contextual:
            context_parts.append("\n=== CONNECTED CONTEXTUAL CHUNKS ===")
            for i, chunk in enumerate(connected_contextual):
                description = chunk.metadata.get('contextual_description', 'No description')
                context_parts.append(f"Contextual {i+1}: {description}")

        # Add similar contextual chunks
        if similar_contextual:
            context_parts.append("\n=== SIMILAR CONTEXTUAL CHUNKS ===")
            for i, chunk in enumerate(similar_contextual):
                description = chunk.metadata.get('contextual_description', 'No description')
                similarity_info = ""
                if chunk.metadata.get('similarity'):
                    similarity_info = f" (similarity: {chunk.metadata['similarity']:.3f})"
                context_parts.append(f"Similar Contextual {i+1}{similarity_info}:\n{description}")

        return "\n\n".join(context_parts)

    async def answer_with_context(
        self,
        question: str,
        session_id: str,
        llm_provider=None,
        response_format: str = "text",
        top_chunks: int = 10,
        top_contextual: int = 10
    ) -> str:
        """Answer question using contextual contextual retrieval.

        Args:
            question: Question to answer
            session_id: Session ID for context retrieval
            llm_provider: LLM provider for generating answers
            response_format: "text" for open-ended, "structured" for specific format
            top_chunks: Number of similar chunks to retrieve
            top_contextual: Number of contextual chunks to retrieve

        Returns:
            Generated answer string
        """
        try:
            # Step 1: Retrieve context using advanced contextual retrieval
            context_result = await self.contextual_retrieve(
                query=question,
                session_id=session_id,
                top_chunks=top_chunks,
                top_contextual=top_contextual
            )

            if context_result.total_pieces == 0:
                return "No relevant context found for this question."

            # Step 2: Create prompt
            prompt = f"""Based on the following comprehensive context, answer the question.

CONTEXT:
{context_result.formatted_context}

QUESTION:
{question}

Please provide a comprehensive answer based on the context above."""

            # Step 3: Generate answer
            if llm_provider:
                from ...llm.base import LLMRequest

                request = LLMRequest(
                    messages=[{"role": "user", "content": prompt}],
                    model="grok-3-mini",
                    temperature=0.1,
                    max_tokens=500
                )

                response = await llm_provider.generate(request)

                if response.success:
                    return response.content.strip()
                else:
                    return f"Error generating answer: {response.error}"
            else:
                # Fallback: return formatted context
                return f"Context retrieved ({context_result.total_pieces} pieces):\n\n{context_result.formatted_context}"

        except Exception as e:
            logger.error(f"Error in answer_with_context: {e}")
            return f"Error processing question: {str(e)}"
