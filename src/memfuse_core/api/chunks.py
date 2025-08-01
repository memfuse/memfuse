"""Chunks API endpoints."""

import logging
from fastapi import APIRouter, Depends, Query
from typing import Optional, Dict, Any

from ..models import (
    ApiResponse,
    ErrorDetail,
)
from ..services.database_service import DatabaseService
from ..services.memory_service import MemoryService
from ..utils.auth import validate_api_key
from ..utils import (
    validate_user_exists,
    validate_session_exists,
    handle_api_errors,
)


router = APIRouter()

# Configure logging
logger = logging.getLogger(__name__)


def _apply_sorting_and_limit(chunks, sort_by: str, order: str, limit: int):
    """Apply sorting and limit to chunks list."""
    # Sort chunks
    if sort_by == "created_at":
        # Sort by created_at timestamp
        def sort_key(chunk):
            created_at = chunk.metadata.get("created_at")
            if created_at:
                try:
                    from datetime import datetime
                    return datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                except (ValueError, TypeError):
                    return datetime.min
            return datetime.min

        chunks.sort(key=sort_key, reverse=(order == "desc"))

    elif sort_by == "chunk_id":
        # Sort by chunk_id
        chunks.sort(key=lambda x: x.chunk_id, reverse=(order == "desc"))

    elif sort_by == "strategy":
        # Sort by strategy
        chunks.sort(key=lambda x: x.metadata.get("strategy", ""), reverse=(order == "desc"))

    # Apply limit
    return chunks[:limit]


@router.get("/sessions/{session_id}/chunks", response_model=ApiResponse)
@router.get("/sessions/{session_id}/chunks/", response_model=ApiResponse)
@handle_api_errors("get session chunks")
async def get_session_chunks(
    session_id: str,
    limit: Optional[str] = "20",  # Match Messages API default
    sort_by: str = "created_at",  # Default sort by creation time
    order: str = "desc",  # Default to newest first
    store_type: Optional[str] = Query(None, description="Store type: vector, keyword, graph, hybrid"),
    _: dict = Depends(validate_api_key),  # API key validation
) -> ApiResponse:
    """Get chunks for a session with sorting and filtering.

    Args:
        session_id: The ID of the session to get chunks from
        limit: Maximum number of chunks to return (default: 20, max: 100)
        sort_by: Field to sort chunks by (allowed values: created_at, chunk_id, strategy)
        order: Sort order (allowed values: asc, desc)
        store_type: Store type filter (vector, keyword, graph, hybrid)
    """
    try:
        # Validate query parameters
        # Validate and convert limit
        limit_value = 20  # Default value
        if limit is not None:
            try:
                limit_value = int(limit)
                if limit_value <= 0:
                    return ApiResponse.error(
                        message="Invalid limit parameter",
                        code=400,
                        errors=[ErrorDetail(
                            field="limit",
                            message="Limit must be greater than 0"
                        )],
                    )
                if limit_value > 100:
                    # Cap at 100 like Messages API
                    limit_value = 100
            except ValueError:
                return ApiResponse.error(
                    message="Invalid limit parameter",
                    code=400,
                    errors=[ErrorDetail(
                        field="limit",
                        message="Limit must be an integer"
                    )],
                )

        # Validate sort_by
        allowed_sort_fields = ["created_at", "chunk_id", "strategy"]
        if sort_by not in allowed_sort_fields:
            return ApiResponse.error(
                message="Invalid sort_by parameter",
                code=400,
                errors=[ErrorDetail(
                    field="sort_by",
                    message=f"sort_by must be one of: {', '.join(allowed_sort_fields)}"
                )],
            )

        # Validate order
        allowed_orders = ["asc", "desc"]
        if order not in allowed_orders:
            return ApiResponse.error(
                message="Invalid order parameter",
                code=400,
                errors=[ErrorDetail(
                    field="order",
                    message=f"order must be one of: {', '.join(allowed_orders)}"
                )],
            )

        # Validate store_type
        if store_type is not None:
            allowed_store_types = ["vector", "keyword", "graph", "hybrid"]
            if store_type not in allowed_store_types:
                return ApiResponse.error(
                    message="Invalid store_type parameter",
                    code=400,
                    errors=[ErrorDetail(
                        field="store_type",
                        message=f"store_type must be one of: {', '.join(allowed_store_types)}"
                    )],
                )

        # Validate session
        db = await DatabaseService.get_instance()
        is_valid, error_response, session = await validate_session_exists(db, session_id)
        if not is_valid:
            return error_response
        # Get user and agent info for MemoryService initialization
        user = await db.get_user(session["user_id"])
        agent = await db.get_agent(session["agent_id"])

        if not user or not agent:
            return ApiResponse.error(
                message="User or agent not found for session",
                code=404,
                errors=[ErrorDetail(field="session", message="Invalid session data")]
            )

        # Initialize memory service
        memory = MemoryService(
            user=user["name"],
            agent=agent["name"],
            session_id=session_id
        )
        await memory.initialize()

        # Get chunks from stores based on store_type
        all_chunks = []

        if store_type == "hybrid":
            # Hybrid: prioritize vector > keyword > graph
            stores_to_query = [
                ("vector", memory.vector_store),
                ("keyword", memory.keyword_store),
                ("graph", memory.graph_store)
            ]
        elif store_type is None:
            # All stores
            stores_to_query = [
                ("vector", memory.vector_store),
                ("keyword", memory.keyword_store),
                ("graph", memory.graph_store)
            ]
        else:
            # Specific store
            store_map = {
                "vector": memory.vector_store,
                "keyword": memory.keyword_store,
                "graph": memory.graph_store
            }
            stores_to_query = [(store_type, store_map.get(store_type))]

        for store_name, store in stores_to_query:
            if store and hasattr(store, 'get_chunks_by_session'):
                store_chunks = await store.get_chunks_by_session(session_id)
                for chunk in store_chunks:
                    chunk.metadata["store_type"] = store_name
                all_chunks.extend(store_chunks)

        # Apply sorting and limit
        sorted_chunks = _apply_sorting_and_limit(all_chunks, sort_by, order, limit_value)

        # Convert chunks to dict format
        chunks_data = []
        for chunk in sorted_chunks:
            chunks_data.append({
                "chunk_id": chunk.chunk_id,
                "content": chunk.content,
                "metadata": chunk.metadata
            })

        return ApiResponse.success(
            data={
                "chunks": chunks_data,
                "total_count": len(chunks_data),
                "session_id": session_id,
                "store_type": store_type or "all"
            },
            message=f"Retrieved {len(chunks_data)} chunks for session {session_id}",
        )

    except Exception as e:
        logger.error(f"Error getting chunks for session {session_id}: {e}")
        return ApiResponse.error(
            message="Failed to retrieve chunks",
            code=500,
            errors=[ErrorDetail(field="general", message=str(e))]
        )


@router.get("/rounds/{round_id}/chunks", response_model=ApiResponse)
@router.get("/rounds/{round_id}/chunks/", response_model=ApiResponse)
@handle_api_errors("get round chunks")
async def get_round_chunks(
    round_id: str,
    limit: Optional[str] = "20",  # Match Messages API default
    sort_by: str = "created_at",  # Default sort by creation time
    order: str = "desc",  # Default to newest first
    store_type: Optional[str] = Query(None, description="Store type: vector, keyword, graph, hybrid"),
    _: dict = Depends(validate_api_key),  # API key validation
) -> ApiResponse:
    """Get chunks for a round with sorting and filtering.

    Args:
        round_id: The ID of the round to get chunks from
        limit: Maximum number of chunks to return (default: 20, max: 100)
        sort_by: Field to sort chunks by (allowed values: created_at, chunk_id, strategy)
        order: Sort order (allowed values: asc, desc)
        store_type: Store type filter (vector, keyword, graph, hybrid)
    """
    try:
        # Validate query parameters (same as session chunks)
        limit_value = 20
        if limit is not None:
            try:
                limit_value = int(limit)
                if limit_value <= 0:
                    return ApiResponse.error(
                        message="Invalid limit parameter",
                        code=400,
                        errors=[ErrorDetail(field="limit", message="Limit must be greater than 0")],
                    )
                if limit_value > 100:
                    limit_value = 100
            except ValueError:
                return ApiResponse.error(
                    message="Invalid limit parameter",
                    code=400,
                    errors=[ErrorDetail(field="limit", message="Limit must be an integer")],
                )

        # Validate sort_by and order (same validation as session chunks)
        allowed_sort_fields = ["created_at", "chunk_id", "strategy"]
        if sort_by not in allowed_sort_fields:
            return ApiResponse.error(
                message="Invalid sort_by parameter",
                code=400,
                errors=[ErrorDetail(
                    field="sort_by",
                    message=f"sort_by must be one of: {', '.join(allowed_sort_fields)}"
                )],
            )

        allowed_orders = ["asc", "desc"]
        if order not in allowed_orders:
            return ApiResponse.error(
                message="Invalid order parameter",
                code=400,
                errors=[ErrorDetail(
                    field="order",
                    message=f"order must be one of: {', '.join(allowed_orders)}"
                )],
            )

        # Validate store_type
        if store_type is not None:
            allowed_store_types = ["vector", "keyword", "graph", "hybrid"]
            if store_type not in allowed_store_types:
                return ApiResponse.error(
                    message="Invalid store_type parameter",
                    code=400,
                    errors=[ErrorDetail(
                        field="store_type",
                        message=f"store_type must be one of: {', '.join(allowed_store_types)}"
                    )],
                )

        # Get round and session info
        db = await DatabaseService.get_instance()
        round_info = await db.get_round(round_id)
        if not round_info:
            return ApiResponse.error(
                message="Round not found",
                code=404,
                errors=[ErrorDetail(field="round_id", message="Round does not exist")]
            )

        session = await db.get_session(round_info["session_id"])
        if not session:
            return ApiResponse.error(
                message="Session not found for round",
                code=404,
                errors=[ErrorDetail(field="session", message="Session does not exist")]
            )

        # Get user and agent info
        user = await db.get_user(session["user_id"])
        agent = await db.get_agent(session["agent_id"])

        if not user or not agent:
            return ApiResponse.error(
                message="User or agent not found for session",
                code=404,
                errors=[ErrorDetail(field="session", message="Invalid session data")]
            )

        # Initialize memory service
        memory = MemoryService(
            user=user["name"],
            agent=agent["name"],
            session_id=session["id"]
        )
        await memory.initialize()

        # Get chunks using same logic as session chunks
        all_chunks = []
        if store_type == "hybrid":
            stores_to_query = [
                ("vector", memory.vector_store),
                ("keyword", memory.keyword_store),
                ("graph", memory.graph_store)
            ]
        elif store_type is None:
            stores_to_query = [
                ("vector", memory.vector_store),
                ("keyword", memory.keyword_store),
                ("graph", memory.graph_store)
            ]
        else:
            store_map = {
                "vector": memory.vector_store,
                "keyword": memory.keyword_store,
                "graph": memory.graph_store
            }
            stores_to_query = [(store_type, store_map.get(store_type))]

        for store_name, store in stores_to_query:
            if store and hasattr(store, 'get_chunks_by_round'):
                store_chunks = await store.get_chunks_by_round(round_id)
                for chunk in store_chunks:
                    chunk.metadata["store_type"] = store_name
                all_chunks.extend(store_chunks)

        # Apply sorting and limit
        sorted_chunks = _apply_sorting_and_limit(all_chunks, sort_by, order, limit_value)

        # Convert chunks to dict format
        chunks_data = []
        for chunk in sorted_chunks:
            chunks_data.append({
                "chunk_id": chunk.chunk_id,
                "content": chunk.content,
                "metadata": chunk.metadata
            })

        return ApiResponse.success(
            data={
                "chunks": chunks_data,
                "total_count": len(chunks_data),
                "round_id": round_id,
                "session_id": session["id"],
                "store_type": store_type or "all"
            },
            message=f"Retrieved {len(chunks_data)} chunks for round {round_id}",
        )

    except Exception as e:
        logger.error(f"Error getting chunks for round {round_id}: {e}")
        return ApiResponse.error(
            message="Failed to retrieve chunks",
            code=500,
            errors=[ErrorDetail(field="general", message=str(e))]
        )


@router.get("/chunks/stats", response_model=ApiResponse)
@router.get("/chunks/stats/", response_model=ApiResponse)
@handle_api_errors("get chunks stats")
async def get_chunks_stats(
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    session_id: Optional[str] = Query(None, description="Filter by session ID"),
    store_type: Optional[str] = Query(None, description="Store type: vector, keyword, graph"),
    _: dict = Depends(validate_api_key),  # API key validation
) -> ApiResponse:
    """Get statistics about chunks across all stores."""
    db = DatabaseService.get_instance()

    try:
        # Validate user if provided
        if user_id:
            is_valid, error_response, user = validate_user_exists(db, user_id)
            if not is_valid:
                return error_response
        else:
            # For stats, we need a default user to initialize MemoryService
            # This is a limitation of the current architecture
            return ApiResponse.error(
                message="user_id is required for chunks stats",
                code=400,
                errors=[ErrorDetail(field="user_id", message="user_id parameter is required")]
            )

        # Validate session if provided
        if session_id:
            is_valid, error_response, session = validate_session_exists(db, session_id)
            if not is_valid:
                return error_response
        
        # Get a default agent for MemoryService initialization
        agents = db.get_agents()
        if not agents:
            return ApiResponse.error(
                message="No agents found",
                code=500,
                errors=[ErrorDetail(field="general", message="No agents available")]
            )

        # Initialize memory service with first available agent
        memory = MemoryService(
            user=user["name"],
            agent=agents[0]["name"],
            session_id=session_id or "stats"
        )
        await memory.initialize()

        # Get stats from stores
        all_stats = {}
        
        if store_type is None or store_type == "vector":
            if memory.vector_store:
                vector_stats = await memory.vector_store.get_chunks_stats()
                all_stats["vector"] = vector_stats

        if store_type is None or store_type == "keyword":
            if memory.keyword_store:
                keyword_stats = await memory.keyword_store.get_chunks_stats()
                all_stats["keyword"] = keyword_stats

        if store_type is None or store_type == "graph":
            if memory.graph_store:
                graph_stats = await memory.graph_store.get_chunks_stats()
                all_stats["graph"] = graph_stats

        # Aggregate stats
        total_chunks = sum(stats.get("total_chunks", 0) for stats in all_stats.values())
        
        return ApiResponse.success(
            data={
                "total_chunks": total_chunks,
                "by_store": all_stats,
                "filters": {
                    "user_id": user_id,
                    "session_id": session_id,
                    "store_type": store_type or "all"
                }
            },
            message=f"Retrieved chunks statistics: {total_chunks} total chunks",
        )

    except Exception as e:
        logger.error(f"Error getting chunks stats: {e}")
        return ApiResponse.error(
            message="Failed to retrieve chunks statistics",
            code=500,
            errors=[ErrorDetail(field="general", message=str(e))]
        )
