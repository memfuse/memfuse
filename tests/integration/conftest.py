"""
FIXED Configuration and fixtures for integration tests.

This module provides shared fixtures and setup for integration testing,
with PROPER database connection management and cleanup.
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
    Set up integration test environment with proper database cleanup.
    
    This fixture:
    1. Assumes PostgreSQL database is already started via memfuse_launcher.py
    2. Resets the database for clean state between tests
    3. Configures environment variables for PostgreSQL
    4. Validates database connectivity
    5. ENSURES proper cleanup of database connections
    """
    print("\n🔄 Setting up integration test environment...")

    # Set environment variables for PostgreSQL
    os.environ["POSTGRES_HOST"] = "localhost"
    os.environ["POSTGRES_PORT"] = "5432"
    os.environ["POSTGRES_DB"] = "memfuse"
    os.environ["POSTGRES_USER"] = "postgres"
    os.environ["POSTGRES_PASSWORD"] = "postgres"

    # CRITICAL: Disable pgai immediate triggers for testing to avoid connection pool issues
    os.environ["PGAI_IMMEDIATE_TRIGGER"] = "false"
    os.environ["PGAI_VECTORIZER_WORKER_ENABLED"] = "false"
    os.environ["MEMFUSE_TEST_MODE"] = "true"
    os.environ["DISABLE_PGAI_NOTIFICATIONS"] = "true"

    # Conservative connection pool settings for testing
    os.environ["POSTGRES_POOL_SIZE"] = "1"
    os.environ["POSTGRES_MAX_OVERFLOW"] = "2"
    os.environ["POSTGRES_POOL_TIMEOUT"] = "10.0"

    # Use default buffer configuration from config/buffer/default.yaml
    # This allows testing both buffer enabled and disabled scenarios
    
    # Check database connectivity first
    print("🔍 Checking database connectivity...")
    result = subprocess.run([
        sys.executable, "scripts/database_manager.py", "status"
    ], capture_output=True, text=True, cwd=PROJECT_ROOT)

    if result.returncode != 0:
        pytest.fail(f"Database is not available. Please start it first with: poetry run python scripts/memfuse_launcher.py --start-db --optimize-db\nError: {result.stderr}")

    # CRITICAL: Reset database content to avoid session name conflicts
    print("🗑️ Resetting database content...")
    reset_result = subprocess.run([
        sys.executable, "scripts/database_manager.py", "reset"
    ], capture_output=True, text=True, cwd=PROJECT_ROOT)

    if reset_result.returncode != 0:
        print(f"Warning: Database reset failed: {reset_result.stderr}")
    else:
        print("✅ Database content reset completed")

    # CRITICAL: Reset DatabaseService singleton before each test
    from memfuse_core.services.database_service import DatabaseService
    DatabaseService.reset_instance_sync()
    print("🧹 DatabaseService singleton reset")
    
    # CRITICAL: Close all connection pools to prevent connection leaks
    try:
        from memfuse_core.services.global_connection_manager import get_global_connection_manager
        connection_manager = get_global_connection_manager()
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(connection_manager.close_all_pools(force=True))
            print("🧹 Connection pools closed")
        finally:
            loop.close()
    except Exception as e:
        print(f"⚠️  Error closing connection pools: {e}")
    
    print("✅ Integration environment setup completed")
    
    yield
    
    # CRITICAL: Cleanup after each test
    print("🧹 Cleaning up database connections...")

    # Clean up any remaining transaction state first
    try:
        import psycopg
        conn = psycopg.connect(
            host="localhost",
            port=5432,
            dbname="memfuse",
            user="postgres",
            password="postgres"
        )
        try:
            conn.rollback()
        except:
            pass
        conn.close()
        print("🧹 Database transaction state cleaned")
    except:
        pass

    # Reset singleton to ensure connection is closed
    DatabaseService.reset_instance_sync()
    
    # CRITICAL: Close all connection pools to prevent connection leaks
    try:
        from memfuse_core.services.global_connection_manager import get_global_connection_manager
        connection_manager = get_global_connection_manager()
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(connection_manager.close_all_pools(force=True))
            print("🧹 Connection pools closed")
        finally:
            loop.close()
    except Exception as e:
        print(f"⚠️  Error closing connection pools: {e}")

    # Force garbage collection to ensure cleanup
    import gc
    gc.collect()

    print("✅ Database cleanup completed")


@pytest.fixture
def client():
    """Create configurable test client with PROPER cleanup."""
    import asyncio
    import requests
    
    # Check environment variable for client type
    client_type = os.environ.get("MEMFUSE_TEST_CLIENT_TYPE", "server")
    
    if client_type == "server":
        # Create real HTTP client for testing against running server
        print(f"🔗 Using real HTTP client against server at http://localhost:8000")
        
        class RealHTTPClient:
            def __init__(self, base_url="http://localhost:8000"):
                self.base_url = base_url
                self.session = requests.Session()
                
            def post(self, url, json=None, headers=None):
                """Make POST request to running server."""
                full_url = f"{self.base_url}{url}"
                return self.session.post(full_url, json=json, headers=headers)
                
            def get(self, url, headers=None):
                """Make GET request to running server."""
                full_url = f"{self.base_url}{url}"
                return self.session.get(full_url, headers=headers)
                
            def put(self, url, json=None, headers=None):
                """Make PUT request to running server."""
                full_url = f"{self.base_url}{url}"
                return self.session.put(full_url, json=json, headers=headers)
                
            def delete(self, url, headers=None):
                """Make DELETE request to running server."""
                full_url = f"{self.base_url}{url}"
                return self.session.delete(full_url, headers=headers)

            def request(self, method, url, content=None, headers=None):
                """Make generic request to running server."""
                full_url = f"{self.base_url}{url}"
                return self.session.request(method, full_url, data=content, headers=headers)

            def close(self):
                """Close the session."""
                self.session.close()
        
        client = RealHTTPClient()
        yield client
        client.close()
    
    else:
        # Create in-process TestClient with PROPER service management
        print(f"⚙️  Using in-process TestClient with proper cleanup")
        
        # Import here to avoid circular imports
        from memfuse_core.services.service_initializer import ServiceInitializer
        from memfuse_core.services.database_service import DatabaseService
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
                    "password": "postgres",
                    "pool_size": 10,  # Match new default
                    "max_overflow": 40,  # 50 - 10 = 40
                    "pool_timeout": 60.0,
                    "pool_recycle": 7200
                }
            },
            "embedding": {
                "model": "all-MiniLM-L6-v2",
                "dimension": 384
            },
            "store": {
                "backend": "pgai",
                "buffer_size": 10,
                "cache_size": 100
            },
            "embedding": {
                "model": "all-MiniLM-L6-v2",
                "dimension": 384
            },
            "data_dir": "/tmp/memfuse_test"
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
                
            # Create app with proper initialization
            app = create_app()
            client = TestClient(app)
            
            yield client
            
        finally:
            # CRITICAL: Proper cleanup
            print("🧹 Cleaning up services and database connections...")
            
            try:
                # Shutdown all services properly
                loop.run_until_complete(service_initializer.shutdown_all_services())
                print("✅ Services shutdown completed")
            except Exception as e:
                print(f"⚠️  Error during service shutdown: {e}")
            
            # Reset database singleton
            DatabaseService.reset_instance()
            print("✅ DatabaseService singleton reset")
            
            # Close event loop
            loop.close()
            
            # Force garbage collection
            import gc
            gc.collect()
            print("✅ TestClient cleanup completed")


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
    import time
    unique_suffix = f"{str(uuid.uuid4())}_{int(time.time() * 1000000) % 1000000}"
    return {
        "name": f"integration_test_user_{unique_suffix}",
        "description": "User for integration testing"
    }


@pytest.fixture
def test_agent_data():
    """Standard test agent data."""
    import uuid
    import time
    unique_suffix = f"{str(uuid.uuid4())}_{int(time.time() * 1000000) % 1000000}"
    return {
        "name": f"integration_test_agent_{unique_suffix}",
        "description": "Agent for integration testing"
    }


@pytest.fixture
def test_session_data():
    """Standard test session data generator."""
    def _generate_session_data(user_id: str, agent_id: str, name_suffix: str = ""):
        import uuid
        import time
        unique_suffix = f"{str(uuid.uuid4())}_{int(time.time() * 1000000) % 1000000}"
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

    IMPORTANT: This creates a separate synchronous connection for validation only.
    """
    import psycopg
    from psycopg.rows import dict_row

    # Create a direct synchronous connection for testing validation
    conn = psycopg.connect(
        host="localhost",
        port=5432,
        dbname="memfuse",
        user="postgres",
        password="postgres",
        row_factory=dict_row
    )

    yield conn

    # CRITICAL: Ensure clean connection close
    conn.close()
    print("✅ Database validation connection closed")


class IntegrationTestHelper:
    """Helper class for common integration test operations."""

    @staticmethod
    def validate_api_response(response, expected_status_code: int = 200) -> Dict[str, Any]:
        """Validate API response and return parsed JSON data with enhanced error handling."""
        # Check status code
        if response.status_code != expected_status_code:
            try:
                error_data = response.json()
                error_msg = f"Expected {expected_status_code}, got {response.status_code}. Error: {error_data.get('message', 'Unknown error')}"
            except Exception:
                error_msg = f"Expected {expected_status_code}, got {response.status_code}. Response: {response.text}"
            assert False, error_msg

        # Parse and validate JSON response
        try:
            response_data = response.json()
            if not response_data:
                assert False, f"Empty response data"

            # Validate response structure
            if "status" not in response_data:
                assert False, f"Missing 'status' field in response: {response_data}"

            if response_data["status"] != "success":
                assert False, f"API returned error status: {response_data.get('message', 'Unknown error')}"

            # For success responses, data field should exist (but can be None for some operations)
            if "data" not in response_data:
                assert False, f"Missing 'data' field in response: {response_data}"

            return response_data
        except Exception as e:
            assert False, f"Failed to parse response JSON: {e}. Response: {response.text}"

    @staticmethod
    def create_user_via_api(client, headers: Dict[str, str],
                           user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a user via API and return the response data."""
        response = client.post("/api/v1/users", json=user_data, headers=headers)
        response_data = IntegrationTestHelper.validate_api_response(response, 201)

        if not response_data.get("data") or not response_data["data"].get("user"):
            assert False, f"Invalid user creation response format: {response_data}"

        return response_data["data"]["user"]
    
    @staticmethod
    def create_agent_via_api(client, headers: Dict[str, str],
                            agent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create an agent via API and return the response data."""
        response = client.post("/api/v1/agents", json=agent_data, headers=headers)
        response_data = IntegrationTestHelper.validate_api_response(response, 201)

        if not response_data.get("data") or not response_data["data"].get("agent"):
            assert False, f"Invalid agent creation response format: {response_data}"

        return response_data["data"]["agent"]

    @staticmethod
    def create_session_via_api(client, headers: Dict[str, str],
                              session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a session via API and return the response data."""
        response = client.post("/api/v1/sessions", json=session_data, headers=headers)
        response_data = IntegrationTestHelper.validate_api_response(response, 201)

        if not response_data.get("data") or not response_data["data"].get("session"):
            assert False, f"Invalid session creation response format: {response_data}"

        return response_data["data"]["session"]
    
    @staticmethod
    def verify_database_record_exists(db, table: str, record_id: str) -> bool:
        """Verify that a record exists in the database."""
        cursor = db.execute(f"SELECT id FROM {table} WHERE id = %s", (record_id,))
        result = cursor.fetchone()
        cursor.close()
        return result is not None
    
    @staticmethod
    def verify_database_record_count(db, table: str, expected_count: int) -> bool:
        """Verify the number of records in a table."""
        cursor = db.execute(f"SELECT COUNT(*) as count FROM {table}")
        result = cursor.fetchone()
        cursor.close()
        return result["count"] == expected_count if result else False


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
            "backend": "pgai",
            "buffer_size": 10,
            "cache_size": 100
        },
        "data_dir": "/tmp/memfuse_test",
        "embedding_service": "mock_for_crud_real_for_memory",
        "llm_service": "mock_unless_specified",
        "vector_store": "real_always",
        "api_timeout": 30,
        "test_data_cleanup": False  # Don't cleanup for inspection
    } 