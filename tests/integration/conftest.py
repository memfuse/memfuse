"""
Configuration and fixtures for integration tests.

This module provides shared fixtures and setup for integration testing,
including database management, service mocking, and test data loading.
"""

import pytest
import json
import subprocess
import sys
import os
import time
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock

# Add src to path so we can import the modules
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fastapi.testclient import TestClient
from memfuse_core.server import create_app


@pytest.fixture(scope="function", autouse=True)
def setup_integration_environment():
    """
    Set up integration test environment assuming database is already running.
    
    This fixture:
    1. Assumes PostgreSQL database is already started via memfuse_launcher.py
    2. Resets the database for clean state between tests
    3. Configures environment variables for PostgreSQL
    4. Validates database connectivity
    """
    print("\n🔄 Setting up integration test environment...")
    
    # Set environment variables for PostgreSQL
    os.environ["POSTGRES_HOST"] = "localhost"
    os.environ["POSTGRES_PORT"] = "5432"
    os.environ["POSTGRES_DB"] = "memfuse"
    os.environ["POSTGRES_USER"] = "postgres"
    os.environ["POSTGRES_PASSWORD"] = "postgres"
    
    # Check database connectivity first
    print("🔍 Checking database connectivity...")
    result = subprocess.run([
        sys.executable, "scripts/database_manager.py", "status"
    ], capture_output=True, text=True, cwd=PROJECT_ROOT)
    
    if result.returncode != 0:
        pytest.fail(f"Database is not available. Please start it first with: poetry run python scripts/memfuse_launcher.py --start-db --optimize-db\nError: {result.stderr}")
    
    # Reset database using database_manager.py for clean state
    print("🔄 Resetting database for clean state...")
    result = subprocess.run([
        sys.executable, "scripts/database_manager.py", "reset"
    ], capture_output=True, text=True, cwd=PROJECT_ROOT)
    
    if result.returncode != 0:
        pytest.fail(f"Failed to reset database: {result.stderr}")
    
    print("✅ Integration environment setup completed")
    yield
    
    # No cleanup - preserve database state for inspection
    print("📊 Database state preserved for inspection")


@pytest.fixture
def client():
    """Create test client for API testing with proper service initialization."""
    import asyncio
    # Import here to avoid circular imports
    from memfuse_core.services.service_initializer import ServiceInitializer
    from memfuse_core.utils.config import config_manager
    from omegaconf import DictConfig, OmegaConf
    
    # Create test configuration with PostgreSQL
    test_config = {
        "database": {
            "type": "postgres",
            "postgres": {
                "host": "localhost",
                "port": 5432,
                "database": "memfuse",
                "user": "postgres",
                "password": "postgres"
            }
        },
        "embedding": {
            "model": "all-MiniLM-L6-v2",
            "dimension": 384
        },
        "store": {
            "backend": "pgai"
        }
    }
    
    # Convert to DictConfig for ServiceInitializer
    cfg = OmegaConf.create(test_config)
    
    # Set configuration
    config_manager.set_config(test_config)
    
    # Initialize services with test configuration
    service_initializer = ServiceInitializer()
    
    # Run the async initialization in a synchronous context
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        success = loop.run_until_complete(service_initializer.initialize_all_services(cfg))
        if not success:
            raise RuntimeError("Failed to initialize services")
    finally:
        loop.close()
    
    # Create app with proper initialization
    app = create_app()
    
    return TestClient(app)


@pytest.fixture
def valid_api_key():
    """Return a valid API key for testing."""
    return "test-integration-api-key"


@pytest.fixture
def headers(valid_api_key):
    """Return headers with valid API key."""
    return {"X-API-Key": valid_api_key}


@pytest.fixture
def mock_embedding_service():
    """
    Mock embedding service for CRUD tests.
    
    This fixture provides a mock embedding service that returns
    consistent fake embeddings for testing purposes.
    """
    # Mock the embedding model creation and operations
    with patch('memfuse_core.rag.encode.MiniLM.MiniLMEncoder') as mock_encoder:
        # Mock embedding generation
        mock_encoder.return_value.encode.return_value = [0.1] * 384
        
        # Mock batch embedding generation
        def mock_batch_embeddings(texts):
            return [[0.1] * 384 for _ in texts]
        
        mock_encoder.return_value.encode_batch.side_effect = mock_batch_embeddings
        
        yield mock_encoder


@pytest.fixture
def real_embedding_service():
    """
    Use real embedding service for memory/retrieval tests.
    
    This fixture allows tests to use the actual embedding service
    for realistic similarity calculations.
    """
    # Don't mock anything - use real service
    yield


@pytest.fixture
def mock_llm_service():
    """
    Mock LLM service for tests that don't specifically test LLM functionality.
    """
    with patch('memfuse_core.llm.providers.openai.OpenAIProvider') as mock_provider:
        mock_provider.return_value.generate_response.return_value = "Mock LLM response"
        yield mock_provider


@pytest.fixture
def test_user_data():
    """Standard test user data."""
    import uuid
    unique_suffix = str(uuid.uuid4())[:8]
    return {
        "name": f"integration_test_user_{unique_suffix}",
        "description": "User for integration testing"
    }


@pytest.fixture
def test_agent_data():
    """Standard test agent data."""
    import uuid
    unique_suffix = str(uuid.uuid4())[:8]
    return {
        "name": f"integration_test_agent_{unique_suffix}", 
        "description": "Agent for integration testing"
    }


@pytest.fixture
def test_session_data():
    """Standard test session data generator."""
    def _generate_session_data(user_id: str, agent_id: str, name_suffix: str = ""):
        import uuid
        unique_suffix = str(uuid.uuid4())[:8]
        return {
            "user_id": user_id,
            "agent_id": agent_id,
            "name": f"integration_test_session_{unique_suffix}{name_suffix}"
        }
    return _generate_session_data


@pytest.fixture
def test_message_data():
    """Standard test message data."""
    return [
        {
            "role": "user",
            "content": "Hello, this is a test message for integration testing."
        },
        {
            "role": "assistant", 
            "content": "Hello! I'm responding to your test message."
        }
    ]


@pytest.fixture
def test_knowledge_data():
    """Standard test knowledge data."""
    return [
        "This is test knowledge for integration testing.",
        "Integration tests verify that components work together correctly.",
        "Database persistence is a key aspect of integration testing."
    ]


def load_fixture_data(filename: str) -> Dict[str, Any]:
    """Load test data from fixtures directory."""
    fixture_path = Path(__file__).parent / "fixtures" / filename
    
    if not fixture_path.exists():
        raise FileNotFoundError(f"Fixture file not found: {fixture_path}")
    
    with open(fixture_path, 'r') as f:
        return json.load(f)


@pytest.fixture
def msc_mc10_sample():
    """Load MSC-MC10 sample data for testing."""
    return load_fixture_data("msc_mc10_sample.json")


@pytest.fixture
def taylor_swift_test_data():
    """Load Taylor Swift test data for memory retrieval testing."""
    return load_fixture_data("taylor_swift_test.json")


@pytest.fixture
def database_connection():
    """
    Provide database connection for direct database validation.
    
    This fixture can be used to directly query the database
    to verify data persistence in integration tests.
    """
    # Import here to avoid circular imports
    from memfuse_core.database.postgres import PostgresDB
    
    # Create database connection with test configuration
    db = PostgresDB(
        host="localhost",
        port=5432,
        database="memfuse",
        user="postgres",
        password="postgres"
    )
    
    yield db
    
    # Close connection
    db.close()


class IntegrationTestHelper:
    """Helper class for common integration test operations."""
    
    @staticmethod
    def create_user_via_api(client: TestClient, headers: Dict[str, str], 
                           user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a user via API and return the response data."""
        response = client.post("/api/v1/users", json=user_data, headers=headers)
        assert response.status_code == 201
        return response.json()["data"]["user"]
    
    @staticmethod
    def create_agent_via_api(client: TestClient, headers: Dict[str, str], 
                            agent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create an agent via API and return the response data."""
        response = client.post("/api/v1/agents", json=agent_data, headers=headers)
        assert response.status_code == 201
        return response.json()["data"]["agent"]
    
    @staticmethod
    def create_session_via_api(client: TestClient, headers: Dict[str, str], 
                              session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a session via API and return the response data."""
        response = client.post("/api/v1/sessions", json=session_data, headers=headers)
        assert response.status_code == 201
        return response.json()["data"]["session"]
    
    @staticmethod
    def verify_database_record_exists(db, table: str, record_id: str) -> bool:
        """Verify that a record exists in the database."""
        cursor = db.conn.cursor()
        cursor.execute(f"SELECT id FROM {table} WHERE id = %s", (record_id,))
        result = cursor.fetchone()
        cursor.close()
        return result is not None
    
    @staticmethod
    def verify_database_record_count(db, table: str, expected_count: int) -> bool:
        """Verify the number of records in a table."""
        cursor = db.conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        result = cursor.fetchone()
        cursor.close()
        return result[0] == expected_count if result else False


@pytest.fixture
def integration_helper():
    """Provide integration test helper methods."""
    return IntegrationTestHelper()


# Test configuration
@pytest.fixture
def test_config():
    """Test-specific configuration."""
    return {
        "database": {
            "type": "postgres",
            "postgres": {
                "host": "localhost",
                "port": 5432,
                "database": "memfuse",
                "user": "postgres",
                "password": "postgres"
            }
        },
        "embedding": {
            "model": "all-MiniLM-L6-v2",
            "dimension": 384
        },
        "store": {
            "backend": "pgai"
        },
        "embedding_service": "mock_for_crud_real_for_memory",
        "llm_service": "mock_unless_specified",
        "vector_store": "real_always",
        "api_timeout": 30,
        "test_data_cleanup": False  # Don't cleanup for inspection
    } 