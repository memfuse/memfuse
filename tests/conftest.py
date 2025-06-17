"""Pytest configuration and shared fixtures for MemFuse tests."""

import asyncio
import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

# Add src to path for imports
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_messages():
    """Sample message data for testing."""
    return [
        {
            "role": "user",
            "content": "Hello, I'm interested in learning about space exploration.",
            "metadata": {"session_id": "test_session_1", "timestamp": "2024-01-01T10:00:00Z"}
        },
        {
            "role": "assistant", 
            "content": "Space exploration is a fascinating field! It involves the discovery and exploration of celestial structures in outer space.",
            "metadata": {"session_id": "test_session_1", "timestamp": "2024-01-01T10:00:30Z"}
        }
    ]


@pytest.fixture
def sample_message_batch():
    """Sample message batch data for testing."""
    return [
        [
            {"role": "user", "content": "What is Mars like?", "metadata": {"session_id": "session1"}},
            {"role": "assistant", "content": "Mars is the fourth planet from the Sun.", "metadata": {"session_id": "session1"}}
        ],
        [
            {"role": "user", "content": "Tell me about Jupiter.", "metadata": {"session_id": "session2"}},
            {"role": "assistant", "content": "Jupiter is the largest planet in our solar system.", "metadata": {"session_id": "session2"}}
        ]
    ]


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    return {
        "chunk": {
            "strategy": "message",
            "max_chunk_length": 1000,
            "overlap_length": 100
        },
        "vector_store": {
            "enabled": True,
            "dimension": 384
        },
        "keyword_store": {
            "enabled": True
        },
        "graph_store": {
            "enabled": True
        },
        "buffer": {
            "write_buffer_threshold": 5,
            "query_buffer_cache_size": 50
        }
    }


@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing."""
    store = AsyncMock()
    store.add_batch = AsyncMock(return_value={"status": "success", "count": 1})
    store.query = AsyncMock(return_value=[])
    return store


@pytest.fixture
def mock_keyword_store():
    """Mock keyword store for testing."""
    store = AsyncMock()
    store.add_batch = AsyncMock(return_value={"status": "success", "count": 1})
    store.query = AsyncMock(return_value=[])
    return store


@pytest.fixture
def mock_graph_store():
    """Mock graph store for testing."""
    store = AsyncMock()
    store.add_nodes = AsyncMock(return_value={"status": "success", "count": 1})
    store.query = AsyncMock(return_value=[])
    return store


@pytest.fixture
def mock_database():
    """Mock database for testing."""
    db = AsyncMock()
    db.execute = AsyncMock(return_value=None)
    db.fetch_all = AsyncMock(return_value=[])
    db.fetch_one = AsyncMock(return_value=None)
    return db


@pytest.fixture
async def mock_memory_service(mock_config, mock_vector_store, mock_keyword_store, mock_graph_store, mock_database):
    """Mock memory service for testing."""
    from memfuse_core.services.memory_service import MemoryService
    
    service = MemoryService(
        user="test_user",
        agent="test_agent", 
        session="test_session",
        config=mock_config
    )
    
    # Replace stores with mocks
    service.vector_store = mock_vector_store
    service.keyword_store = mock_keyword_store
    service.graph_store = mock_graph_store
    service.database = mock_database
    
    return service


@pytest.fixture
async def mock_buffer_service(mock_memory_service, mock_config):
    """Mock buffer service for testing."""
    from memfuse_core.services.buffer_service import BufferService
    
    service = BufferService(
        memory_service=mock_memory_service,
        config=mock_config
    )
    
    await service.initialize()
    return service


# Test markers
pytest_plugins = []

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "e2e: mark test as an end-to-end test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "chunking: mark test as chunking-related"
    )


# Test collection configuration
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on file paths."""
    for item in items:
        # Add markers based on file path
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
        
        # Add chunking marker for chunk-related tests
        if "chunk" in str(item.fspath) or "chunking" in item.name:
            item.add_marker(pytest.mark.chunking)
