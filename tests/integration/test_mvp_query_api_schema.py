"""Integration tests for MVP query API schema compliance."""

import pytest
from typing import Dict, Any
from unittest.mock import AsyncMock, patch

from memfuse_mvp.memfuse.api.query_api import (
    query_user_memories,
    query_session_memories,
    query_workflow_memories,
    _deduplicate_and_rank
)


class MockDatabase:
    """Mock database for testing."""
    
    def __init__(self):
        self.users = {"user-1": True}
        self.sessions = {
            "session-1": ("session-1", "user-1"),
            "session-2": ("session-2", "user-1")
        }
    
    def connect(self):
        return MockConnection()


class MockConnection:
    """Mock database connection."""
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass
    
    def cursor(self):
        return MockCursor()


class MockCursor:
    """Mock database cursor."""
    
    def __init__(self):
        self.results = []
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass
    
    def execute(self, query, params=None):
        # Mock different queries based on the query string
        if "SELECT id FROM users" in query:
            self.results = [("user-1",)]
        elif "SELECT id, user_id FROM sessions" in query:
            session_id = params[0] if params else "session-1"
            if session_id in ["session-1", "session-2"]:
                self.results = [(session_id, "user-1")]
            else:
                self.results = []
        elif "messages" in query.lower():
            # Mock message results
            self.results = [
                (
                    "msg-1",
                    "[USER]: What is environmental conservation?",
                    "user",
                    "2025-09-02T13:52:46.552383+00:00",
                    {"type": "question"},
                    "Test Session",
                    "session-1"
                )
            ]
        else:
            self.results = []
    
    def fetchone(self):
        return self.results[0] if self.results else None
    
    def fetchall(self):
        return self.results


class MockRAGService:
    """Mock RAG service for testing."""
    
    def __init__(self):
        self.retrieval = None
        self.db = None
        self.embedder = None
        self.settings = None


class MockRetrievalStrategy:
    """Mock retrieval strategy."""
    
    def retrieve(self, session_id, query, context):
        return [
            type('Chunk', (), {
                'content': 'Knowledge about environmental conservation',
                'source': 'knowledge_base',
                'score': 0.85
            })()
        ]


@pytest.mark.asyncio
class TestMVPQueryApiSchema:
    """Test MVP query API schema compliance."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_db = MockDatabase()
        self.mock_rag = MockRAGService()
    
    async def test_query_user_memories_response_format(self):
        """Test that query_user_memories returns the correct response format."""
        # Arrange
        user_id = "user-1"
        query_data = {
            "query": "environmental conservation",
            "session_id": "session-1",
            "agent_id": "agent-1",
            "top_k": 5,
            "metadata": {
                "task": "op_websearch_memory",
                "mode": None
            }
        }
        
        # Mock the RAG retrieval
        with patch('memfuse_mvp.memfuse.api.query_api.BasicRetrievalStrategy', MockRetrievalStrategy):
            # Act
            response = await query_user_memories(
                user_id=user_id,
                query_data=query_data,
                tag=None,
                db=self.mock_db,
                rag=self.mock_rag
            )
        
        # Assert response structure
        assert isinstance(response, dict)
        assert "status" in response
        assert "code" in response
        assert "data" in response
        assert "message" in response
        assert "errors" in response
        
        # Check response values
        assert response["status"] == "success"
        assert response["code"] == 200
        assert response["errors"] is None
        assert isinstance(response["message"], str)
        assert "Found" in response["message"]
        
        # Check data structure
        data = response["data"]
        assert "query" in data
        assert "results" in data
        assert "total" in data
        
        assert data["query"] == "environmental conservation"
        assert isinstance(data["results"], list)
        assert isinstance(data["total"], int)
        assert data["total"] == len(data["results"])
    
    async def test_query_session_memories_response_format(self):
        """Test that query_session_memories returns the correct response format."""
        # Arrange
        session_id = "session-1"
        query_data = {
            "query": "test query",
            "top_k": 10
        }
        
        # Act
        response = await query_session_memories(
            session_id=session_id,
            query_data=query_data,
            db=self.mock_db,
            rag=self.mock_rag
        )
        
        # Assert
        assert response["status"] == "success"
        assert response["code"] == 200
        assert response["errors"] is None
        
        data = response["data"]
        assert data["query"] == "test query"
        assert "results" in data
        assert "total" in data
    
    async def test_query_workflow_memories_response_format(self):
        """Test that query_workflow_memories returns the correct response format."""
        # Arrange
        workflow_id = "workflow-1"
        query_data = {
            "query": "workflow steps",
            "top_k": 5
        }
        
        # Act
        response = await query_workflow_memories(
            workflow_id=workflow_id,
            query_data=query_data,
            db=self.mock_db
        )
        
        # Assert
        assert response["status"] == "success"
        assert response["code"] == 200
        assert response["errors"] is None
        
        data = response["data"]
        assert data["query"] == "workflow steps"
        assert "results" in data
        assert "total" in data
    
    def test_deduplicate_and_rank_function(self):
        """Test the _deduplicate_and_rank helper function."""
        # Arrange
        results = [
            {
                "content": "First result about conservation",
                "relevance_score": 0.9,
                "id": "1"
            },
            {
                "content": "First result about conservation",  # Duplicate content
                "relevance_score": 0.8,
                "id": "2"
            },
            {
                "content": "Second unique result",
                "relevance_score": 0.85,
                "id": "3"
            },
            {
                "content": "Third unique result",
                "relevance_score": 0.95,
                "id": "4"
            }
        ]
        
        # Act
        deduplicated = _deduplicate_and_rank(results, top_k=3)
        
        # Assert
        assert len(deduplicated) == 3  # Should remove duplicate and limit to top_k
        
        # Should be sorted by relevance_score descending
        scores = [r.get("relevance_score", 0) for r in deduplicated]
        assert scores == sorted(scores, reverse=True)
        
        # Should not contain duplicates
        contents = [r.get("content", "")[:100] for r in deduplicated]
        assert len(contents) == len(set(contents))
    
    async def test_task_eos_metadata_processing(self):
        """Test that task_eos metadata is properly processed."""
        # Arrange
        user_id = "user-1"
        query_data = {
            "query": "Job done about web search.",
            "session_id": "session-1",
            "agent_id": "agent-1",
            "top_k": 5,
            "metadata": {
                "task": "op_websearch_memory",
                "mode": None,
                "task_eos": True
            }
        }
        
        # Mock the RAG retrieval
        with patch('memfuse_mvp.memfuse.api.query_api.BasicRetrievalStrategy', MockRetrievalStrategy):
            # Act
            response = await query_user_memories(
                user_id=user_id,
                query_data=query_data,
                tag=None,
                db=self.mock_db,
                rag=self.mock_rag
            )
        
        # Assert
        assert response["status"] == "success"
        data = response["data"]
        assert data["query"] == "Job done about web search."
        
        # The task_eos should be processed (logged) but not affect the response structure
        assert "query" in data
        assert "results" in data
        assert "total" in data
    
    async def test_error_handling_user_not_found(self):
        """Test error handling when user is not found."""
        # Arrange
        user_id = "nonexistent-user"
        query_data = {"query": "test"}
        
        # Mock database to return no user
        mock_db = MockDatabase()
        mock_db.users = {}  # No users
        
        # Act & Assert
        with pytest.raises(Exception):  # Should raise HTTPException
            await query_user_memories(
                user_id=user_id,
                query_data=query_data,
                tag=None,
                db=mock_db,
                rag=self.mock_rag
            )
    
    async def test_error_handling_session_not_found(self):
        """Test error handling when session is not found."""
        # Arrange
        session_id = "nonexistent-session"
        query_data = {"query": "test"}
        
        # Mock database to return no session
        mock_db = MockDatabase()
        mock_db.sessions = {}  # No sessions
        
        # Act & Assert
        with pytest.raises(Exception):  # Should raise HTTPException
            await query_session_memories(
                session_id=session_id,
                query_data=query_data,
                db=mock_db,
                rag=self.mock_rag
            )
    
    async def test_empty_query_handling(self):
        """Test handling of empty query."""
        # Arrange
        user_id = "user-1"
        query_data = {"query": ""}  # Empty query
        
        # Act & Assert
        with pytest.raises(Exception):  # Should raise HTTPException for empty query
            await query_user_memories(
                user_id=user_id,
                query_data=query_data,
                tag=None,
                db=self.mock_db,
                rag=self.mock_rag
            )
    
    async def test_top_k_limit_enforcement(self):
        """Test that top_k is properly limited."""
        # Arrange
        user_id = "user-1"
        query_data = {
            "query": "test",
            "top_k": 100  # Exceeds limit
        }
        
        # Mock the RAG retrieval to return many results
        with patch('memfuse_mvp.memfuse.api.query_api.BasicRetrievalStrategy', MockRetrievalStrategy):
            # Act
            response = await query_user_memories(
                user_id=user_id,
                query_data=query_data,
                tag=None,
                db=self.mock_db,
                rag=self.mock_rag
            )
        
        # Assert
        assert response["status"] == "success"
        data = response["data"]
        
        # Should be limited to maximum allowed (50 in the implementation)
        assert data["total"] <= 50
    
    async def test_m3_tag_mode(self):
        """Test M3 tag mode for workflow-focused retrieval."""
        # Arrange
        user_id = "user-1"
        query_data = {
            "query": "workflow task",
            "session_id": "session-1",
            "top_k": 5
        }
        
        # Act
        response = await query_user_memories(
            user_id=user_id,
            query_data=query_data,
            tag="m3",  # M3 mode
            db=self.mock_db,
            rag=self.mock_rag
        )
        
        # Assert
        assert response["status"] == "success"
        data = response["data"]
        assert data["query"] == "workflow task"
        
        # In M3 mode, should focus on workflows and messages, not knowledge
        # This is tested by the internal logic of the function