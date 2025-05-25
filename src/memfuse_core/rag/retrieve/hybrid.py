"""Hybrid retrieval implementation for MemFuse server.

This module implements a hybrid retrieval system that combines results from
vector, graph, and keyword stores to provide more comprehensive and accurate
retrieval results.
"""

import asyncio
from loguru import logger
from typing import Dict, List, Optional, Any

from ..base import BaseRetrieval
from ...models import StoreType, Query, QueryResult

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
