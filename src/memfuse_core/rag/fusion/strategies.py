"""Score fusion strategies for multi-path retrieval.

This module provides various strategies for fusing scores from different retrieval methods.
"""

from typing import Dict, List

from ...models import QueryResult, StoreType


class ScoreFusionStrategy:
    """Base class for score fusion strategies."""

    def fuse_scores(
        self,
        results_by_id: Dict[str, List[QueryResult]],
        weights: Dict[StoreType, float]
    ) -> List[QueryResult]:
        """Fuse scores from different retrieval methods.

        Args:
            results_by_id: Dictionary mapping item IDs to lists of QueryResult objects
            weights: Dictionary mapping store types to weights

        Returns:
            List of merged QueryResult objects with fused scores
        """
        raise NotImplementedError


class SimpleWeightedSum(ScoreFusionStrategy):
    """Simple weighted sum fusion strategy.

    This is the original strategy used in MemFuse, which simply multiplies each score
    by its weight and adds them together.
    """

    def fuse_scores(
        self,
        results_by_id: Dict[str, List[QueryResult]],
        weights: Dict[StoreType, float]
    ) -> List[QueryResult]:
        """Fuse scores using simple weighted sum.

        Args:
            results_by_id: Dictionary mapping item IDs to lists of QueryResult objects
            weights: Dictionary mapping store types to weights

        Returns:
            List of merged QueryResult objects with fused scores
        """
        merged_results = []

        for item_id, result_list in results_by_id.items():
            # Use the first result as the base
            base_result = result_list[0]

            # Calculate weighted score
            weighted_score = 0.0
            store_types = set()

            for result in result_list:
                if result.store_type:
                    weight = weights.get(result.store_type, 0.0)
                    weighted_score += result.score * weight
                    store_types.add(result.store_type)

            # Create merged result
            merged_result = QueryResult(
                id=base_result.id,
                content=base_result.content,
                metadata=base_result.metadata,
                score=weighted_score,
                store_type=None  # No specific store type for merged results
            )

            # Add store types to metadata
            if "retrieval" not in merged_result.metadata:
                merged_result.metadata["retrieval"] = {}

            merged_result.metadata["retrieval"]["methods"] = [
                store_type.value for store_type in store_types
            ]

            # Add original scores to metadata for debugging
            merged_result.metadata["retrieval"]["original_scores"] = {
                result.store_type.value: result.score
                for result in result_list if result.store_type
            }

            merged_results.append(merged_result)

        # Sort by score and return
        merged_results.sort(key=lambda x: x.score, reverse=True)
        return merged_results


class NormalizedWeightedSum(ScoreFusionStrategy):
    """Normalized weighted sum fusion strategy.

    This strategy normalizes scores from each method before combining them.
    """

    def fuse_scores(
        self,
        results_by_id: Dict[str, List[QueryResult]],
        weights: Dict[StoreType, float]
    ) -> List[QueryResult]:
        """Fuse scores using normalized weighted sum.

        Args:
            results_by_id: Dictionary mapping item IDs to lists of QueryResult objects
            weights: Dictionary mapping store types to weights

        Returns:
            List of merged QueryResult objects with fused scores
        """
        # Group results by store type
        results_by_type: Dict[StoreType, List[QueryResult]] = {}

        for result_list in results_by_id.values():
            for result in result_list:
                if result.store_type:
                    if result.store_type not in results_by_type:
                        results_by_type[result.store_type] = []
                    results_by_type[result.store_type].append(result)

        # Normalize scores within each store type
        normalized_scores: Dict[str, Dict[StoreType, float]] = {}

        for store_type, results in results_by_type.items():
            # Find min and max scores
            scores = [result.score for result in results]
            min_score = min(scores) if scores else 0.0
            max_score = max(scores) if scores else 1.0

            # Avoid division by zero
            if max_score > min_score:
                # Apply min-max normalization
                for result in results:
                    if result.id not in normalized_scores:
                        normalized_scores[result.id] = {}

                    normalized_score = (result.score - min_score) / (max_score - min_score)
                    normalized_scores[result.id][store_type] = normalized_score
            else:
                # If all scores are the same, set them all to 1.0
                for result in results:
                    if result.id not in normalized_scores:
                        normalized_scores[result.id] = {}

                    normalized_scores[result.id][store_type] = 1.0

        # Merge results with normalized scores
        merged_results = []

        for item_id, result_list in results_by_id.items():
            # Use the first result as the base
            base_result = result_list[0]

            # Calculate weighted normalized score
            weighted_score = 0.0
            store_types = set()

            for result in result_list:
                if result.store_type:
                    # Get the weight for this store type
                    weight = weights.get(result.store_type, 0.0)

                    # Get the normalized score if available
                    norm_score = 0.0
                    if (result.id in normalized_scores
                            and result.store_type in normalized_scores[result.id]):
                        norm_score = normalized_scores[result.id][result.store_type]
                    else:
                        # If no normalized score (shouldn't happen), use original
                        norm_score = result.score

                    # Add to weighted score
                    weighted_score += norm_score * weight
                    store_types.add(result.store_type)

            # Create merged result
            merged_result = QueryResult(
                id=base_result.id,
                content=base_result.content,
                metadata=base_result.metadata,
                score=weighted_score,
                store_type=None  # No specific store type for merged results
            )

            # Add store types to metadata
            if "retrieval" not in merged_result.metadata:
                merged_result.metadata["retrieval"] = {}

            merged_result.metadata["retrieval"]["methods"] = [
                store_type.value for store_type in store_types
            ]

            # Add original scores to metadata for debugging
            merged_result.metadata["retrieval"]["original_scores"] = {
                result.store_type.value: result.score
                for result in result_list if result.store_type
            }

            # Add normalized scores to metadata for debugging
            norm_score_dict = {}
            for store_type in store_types:
                if item_id in normalized_scores and store_type in normalized_scores[item_id]:
                    norm_score_dict[store_type.value] = normalized_scores[item_id][store_type]

            merged_result.metadata["retrieval"]["normalized_scores"] = norm_score_dict

            merged_results.append(merged_result)

        # Sort by score and return
        merged_results.sort(key=lambda x: x.score, reverse=True)
        return merged_results


class ReciprocalRankFusion(ScoreFusionStrategy):
    """Reciprocal Rank Fusion (RRF) strategy.

    This strategy uses the reciprocal of the rank of each result in its respective
    retrieval method, with a constant k to prevent division by zero and to smooth
    the impact of high-ranking items.

    Reference:
    Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009).
    Reciprocal rank fusion outperforms condorcet and individual rank learning methods.
    In Proceedings of the 32nd international ACM SIGIR conference on Research and
    development in information retrieval (pp. 758-759).
    """

    def __init__(self, k: float = 1.0):
        """Initialize the RRF strategy.

        Args:
            k: Constant to prevent division by zero and smooth impact of high-ranking items
              Default is 1.0, which is more appropriate for small result sets
              Smaller values (e.g., 0.1-0.5) give more weight to top results
              Larger values (e.g., 2.0-60.0) give more weight to lower-ranked results
        """
        self.k = k

    def fuse_scores(
        self,
        results_by_id: Dict[str, List[QueryResult]],
        weights: Dict[StoreType, float]
    ) -> List[QueryResult]:
        """Fuse scores using Reciprocal Rank Fusion.

        Args:
            results_by_id: Dictionary mapping item IDs to lists of QueryResult objects
            weights: Dictionary mapping store types to weights

        Returns:
            List of merged QueryResult objects with fused scores
        """
        # Group results by store type
        results_by_type: Dict[StoreType, List[QueryResult]] = {}

        for result_list in results_by_id.values():
            for result in result_list:
                if result.store_type:
                    if result.store_type not in results_by_type:
                        results_by_type[result.store_type] = []
                    results_by_type[result.store_type].append(result)

        # Sort results by score within each store type
        for store_type, results in results_by_type.items():
            results.sort(key=lambda x: x.score, reverse=True)

        # Calculate RRF scores
        rrf_scores: Dict[str, Dict[StoreType, float]] = {}

        for store_type, results in results_by_type.items():
            for rank, result in enumerate(results, 1):  # 1-based ranking
                if result.id not in rrf_scores:
                    rrf_scores[result.id] = {}

                # RRF formula: 1 / (k + rank)
                rrf_score = 1.0 / (self.k + rank)
                rrf_scores[result.id][store_type] = rrf_score

        # Merge results with RRF scores
        merged_results = []

        for item_id, result_list in results_by_id.items():
            # Use the first result as the base
            base_result = result_list[0]

            # Calculate weighted RRF score
            weighted_score = 0.0
            store_types = set()

            for result in result_list:
                if result.store_type:
                    # Get the weight for this store type
                    weight = weights.get(result.store_type, 0.0)

                    # Get the RRF score if available
                    rrf_score = 0.0
                    if (result.id in rrf_scores
                            and result.store_type in rrf_scores[result.id]):
                        rrf_score = rrf_scores[result.id][result.store_type]
                    else:
                        # If no RRF score (shouldn't happen), use a default value
                        rrf_score = 1.0 / (self.k + 1)  # Default to rank 1

                    # Add to weighted score
                    weighted_score += rrf_score * weight
                    store_types.add(result.store_type)

            # Create merged result
            # Make a copy of metadata to avoid modifying the original
            metadata_copy = base_result.metadata.copy()

            merged_result = QueryResult(
                id=base_result.id,
                content=base_result.content,
                metadata=metadata_copy,
                score=weighted_score,
                store_type=None  # No specific store type for merged results
            )

            # Add store types to metadata
            if "retrieval" not in merged_result.metadata:
                merged_result.metadata["retrieval"] = {}

            # Set method to multi_path - this indicates server-side multi-path retrieval
            merged_result.metadata["retrieval"]["method"] = "multi_path"

            # Set the methods used for this result
            methods = [store_type.value for store_type in store_types]
            merged_result.metadata["retrieval"]["methods"] = methods

            # Collect original scores from result objects
            original_scores = {}
            for result in result_list:
                if result.store_type:
                    original_scores[result.store_type] = result.score

            # Add original scores to metadata for debugging
            merged_result.metadata["retrieval"]["original_scores"] = {
                store_type.value: original_scores.get(store_type, 0.0)
                for store_type in store_types
            }

            # Add RRF scores to metadata for debugging
            merged_result.metadata["retrieval"]["rrf_scores"] = {
                store_type.value: rrf_scores[item_id].get(store_type, 0.0)
                for store_type in store_types
            }

            # Add fusion strategy to metadata
            merged_result.metadata["retrieval"]["fusion_strategy"] = "rrf"

            # No additional processing needed

            # Add RRF scores to metadata for debugging
            rrf_score_dict = {}
            for store_type in store_types:
                if item_id in rrf_scores and store_type in rrf_scores[item_id]:
                    rrf_score_dict[store_type.value] = rrf_scores[item_id][store_type]

            merged_result.metadata["retrieval"]["rrf_scores"] = rrf_score_dict

            merged_results.append(merged_result)

        # Sort by score and return
        merged_results.sort(key=lambda x: x.score, reverse=True)
        return merged_results
