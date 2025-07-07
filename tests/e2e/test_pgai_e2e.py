"""End-to-end tests for PgaiStore with MemFuse system."""

import pytest
import asyncio
import os
import tempfile
from typing import List
from unittest.mock import patch, AsyncMock

from src.memfuse_core.store.pgai_store import PgaiStore
from src.memfuse_core.hierarchy.layers import M0EpisodicLayer
from src.memfuse_core.models.schema import MessageRecord
from src.memfuse_core.rag.chunk.base import ChunkData


class TestPgaiEndToEnd:
    """End-to-end tests for PgaiStore integration with MemFuse."""
    
    @pytest.fixture
    def e2e_config(self):
        """Configuration for end-to-end testing."""
        return {
            "database": {
                "postgres": {
                    "host": os.getenv("POSTGRES_HOST", "localhost"),
                    "port": int(os.getenv("POSTGRES_PORT", "5432")),
                    "database": os.getenv("POSTGRES_DB", "memfuse_e2e_test"),
                    "user": os.getenv("POSTGRES_USER", "postgres"),
                    "password": os.getenv("POSTGRES_PASSWORD", "password"),
                    "pool_size": 10
                },
                "pgai": {
                    "embedding_model": "text-embedding-3-small",
                    "embedding_dimensions": 1536,
                    "chunk_size": 1000,
                    "chunk_overlap": 200,
                    "vectorizer_worker_enabled": True,
                    "auto_embedding": True
                }
            },
            "store": {
                "backend": "pgai",
                "cache_size": 1000,
                "buffer_size": 100
            },
            "memory": {
                "m0": {
                    "enabled": True,
                    "chunk_strategy": "simple",
                    "max_chunks_per_message": 5
                }
            },
            "embedding": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "dimension": 384
            }
        }
    
    @pytest.fixture
    def conversation_data(self):
        """Sample conversation data for end-to-end testing."""
        return [
            MessageRecord(
                session_id="e2e-session-1",
                role="user",
                content="I'm working on a machine learning project and need help with data preprocessing. I have a dataset with missing values and categorical variables that need to be handled properly.",
                metadata={
                    "timestamp": "2024-01-01T10:00:00Z",
                    "user_id": "user-123",
                    "round_id": "round-1"
                }
            ),
            MessageRecord(
                session_id="e2e-session-1",
                role="assistant", 
                content="I'd be happy to help you with data preprocessing for your machine learning project. For handling missing values, you have several options: 1) Remove rows with missing values if the dataset is large enough, 2) Impute missing values using mean/median for numerical data or mode for categorical data, 3) Use advanced imputation techniques like KNN imputation. For categorical variables, you can use one-hot encoding, label encoding, or target encoding depending on your specific use case.",
                metadata={
                    "timestamp": "2024-01-01T10:01:00Z",
                    "user_id": "user-123",
                    "round_id": "round-1"
                }
            ),
            MessageRecord(
                session_id="e2e-session-1",
                role="user",
                content="That's very helpful! I think I'll go with mean imputation for numerical features and one-hot encoding for categorical features. Can you show me how to implement this in Python using pandas and scikit-learn?",
                metadata={
                    "timestamp": "2024-01-01T10:02:00Z",
                    "user_id": "user-123", 
                    "round_id": "round-2"
                }
            ),
            MessageRecord(
                session_id="e2e-session-1",
                role="assistant",
                content="Absolutely! Here's a complete example of how to implement data preprocessing with mean imputation and one-hot encoding:\n\n```python\nimport pandas as pd\nfrom sklearn.impute import SimpleImputer\nfrom sklearn.preprocessing import OneHotEncoder\nfrom sklearn.compose import ColumnTransformer\nfrom sklearn.pipeline import Pipeline\n\n# Load your data\ndf = pd.read_csv('your_dataset.csv')\n\n# Separate numerical and categorical columns\nnumerical_features = df.select_dtypes(include=['int64', 'float64']).columns\ncategorical_features = df.select_dtypes(include=['object']).columns\n\n# Create preprocessing pipelines\nnumerical_transformer = SimpleImputer(strategy='mean')\ncategorical_transformer = Pipeline(steps=[\n    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),\n    ('onehot', OneHotEncoder(handle_unknown='ignore'))\n])\n\n# Combine preprocessing steps\npreprocessor = ColumnTransformer(\n    transformers=[\n        ('num', numerical_transformer, numerical_features),\n        ('cat', categorical_transformer, categorical_features)\n    ])\n\n# Apply preprocessing\nX_processed = preprocessor.fit_transform(df)\n```\n\nThis approach creates a robust preprocessing pipeline that you can reuse for new data.",
                metadata={
                    "timestamp": "2024-01-01T10:03:00Z",
                    "user_id": "user-123",
                    "round_id": "round-2"
                }
            )
        ]
    
    @pytest.mark.asyncio
    async def test_full_conversation_flow(self, e2e_config, conversation_data):
        """Test complete conversation flow with PgaiStore."""
        with patch('src.memfuse_core.store.pgai_store.config_manager') as mock_config_manager:
            mock_config_manager.get_config.return_value = e2e_config
            
            # Mock pgai components for E2E testing
            with patch('src.memfuse_core.store.pgai_store.PGAI_AVAILABLE', True):
                with patch('src.memfuse_core.store.pgai_store.pgai') as mock_pgai:
                    with patch('src.memfuse_core.store.pgai_store.AsyncConnectionPool') as mock_pool:
                        # Setup store
                        store = PgaiStore(table_name="e2e_test_messages")
                        
                        # Mock database operations
                        mock_pool_instance = AsyncMock()
                        mock_cursor = AsyncMock()
                        mock_connection = AsyncMock()
                        mock_connection.cursor.return_value.__aenter__.return_value = mock_cursor
                        mock_pool_instance.connection.return_value.__aenter__.return_value = mock_connection
                        mock_pool.return_value = mock_pool_instance
                        
                        store.pool = mock_pool_instance
                        store.initialized = True
                        
                        # Process conversation through the system
                        all_chunks = []
                        
                        for message in conversation_data:
                            # Simulate chunking process
                            chunk = ChunkData(
                                content=message.content,
                                chunk_id=f"chunk-{len(all_chunks)+1}",
                                metadata={
                                    "session_id": message.session_id,
                                    "role": message.role,
                                    "user_id": message.metadata.get("user_id"),
                                    "round_id": message.metadata.get("round_id"),
                                    "timestamp": message.metadata.get("timestamp")
                                }
                            )
                            all_chunks.append(chunk)
                        
                        # Add all chunks to store
                        chunk_ids = await store.add(all_chunks)
                        
                        # Verify all chunks were added
                        assert len(chunk_ids) == 4
                        assert all(chunk_id.startswith("chunk-") for chunk_id in chunk_ids)
                        
                        # Verify database interactions
                        assert mock_cursor.execute.call_count == 4
                        mock_connection.commit.assert_called()
    
    @pytest.mark.asyncio
    async def test_session_based_retrieval(self, e2e_config, conversation_data):
        """Test session-based retrieval in end-to-end scenario."""
        with patch('src.memfuse_core.store.pgai_store.config_manager') as mock_config_manager:
            mock_config_manager.get_config.return_value = e2e_config
            
            with patch('src.memfuse_core.store.pgai_store.PGAI_AVAILABLE', True):
                with patch('src.memfuse_core.store.pgai_store.AsyncConnectionPool') as mock_pool:
                    store = PgaiStore(table_name="e2e_session_test")
                    
                    # Mock database setup
                    mock_pool_instance = AsyncMock()
                    mock_cursor = AsyncMock()
                    mock_connection = AsyncMock()
                    mock_connection.cursor.return_value.__aenter__.return_value = mock_cursor
                    mock_pool_instance.connection.return_value.__aenter__.return_value = mock_connection
                    mock_pool.return_value = mock_pool_instance
                    
                    store.pool = mock_pool_instance
                    store.initialized = True
                    
                    # Mock session query results
                    mock_cursor.__aiter__.return_value = [
                        ("chunk-1", conversation_data[0].content, 
                         f'{{"session_id": "e2e-session-1", "role": "user", "user_id": "user-123"}}', None, None),
                        ("chunk-2", conversation_data[1].content,
                         f'{{"session_id": "e2e-session-1", "role": "assistant", "user_id": "user-123"}}', None, None),
                        ("chunk-3", conversation_data[2].content,
                         f'{{"session_id": "e2e-session-1", "role": "user", "user_id": "user-123"}}', None, None),
                        ("chunk-4", conversation_data[3].content,
                         f'{{"session_id": "e2e-session-1", "role": "assistant", "user_id": "user-123"}}', None, None)
                    ]
                    
                    # Retrieve session data
                    session_chunks = await store.get_chunks_by_session("e2e-session-1")
                    
                    # Verify session retrieval
                    assert len(session_chunks) == 4
                    assert all(chunk.metadata.get("session_id") == "e2e-session-1" for chunk in session_chunks)
                    
                    # Verify conversation flow
                    roles = [chunk.metadata.get("role") for chunk in session_chunks]
                    assert roles == ["user", "assistant", "user", "assistant"]
    
    @pytest.mark.asyncio
    async def test_semantic_search_simulation(self, e2e_config, conversation_data):
        """Test semantic search capabilities in end-to-end scenario."""
        with patch('src.memfuse_core.store.pgai_store.config_manager') as mock_config_manager:
            mock_config_manager.get_config.return_value = e2e_config
            
            with patch('src.memfuse_core.store.pgai_store.PGAI_AVAILABLE', True):
                with patch('src.memfuse_core.store.pgai_store.AsyncConnectionPool') as mock_pool:
                    store = PgaiStore(table_name="e2e_search_test")
                    
                    # Mock database setup
                    mock_pool_instance = AsyncMock()
                    mock_cursor = AsyncMock()
                    mock_connection = AsyncMock()
                    mock_connection.cursor.return_value.__aenter__.return_value = mock_cursor
                    mock_pool_instance.connection.return_value.__aenter__.return_value = mock_connection
                    mock_pool.return_value = mock_pool_instance
                    
                    store.pool = mock_pool_instance
                    store.initialized = True
                    
                    # Mock search results for "data preprocessing" query
                    mock_cursor.__aiter__.return_value = [
                        ("chunk-1", conversation_data[0].content,  # User question about preprocessing
                         f'{{"session_id": "e2e-session-1", "role": "user"}}', None, None),
                        ("chunk-2", conversation_data[1].content,  # Assistant response about preprocessing
                         f'{{"session_id": "e2e-session-1", "role": "assistant"}}', None, None)
                    ]
                    
                    # Simulate semantic search
                    from src.memfuse_core.models import Query
                    query = Query(text="data preprocessing techniques")
                    
                    search_results = await store.query(query, top_k=5)
                    
                    # Verify search results
                    assert len(search_results) == 2
                    assert "preprocessing" in search_results[0].content.lower()
                    assert "preprocessing" in search_results[1].content.lower()
    
    @pytest.mark.asyncio
    async def test_multi_user_scenario(self, e2e_config):
        """Test multi-user scenario with PgaiStore."""
        with patch('src.memfuse_core.store.pgai_store.config_manager') as mock_config_manager:
            mock_config_manager.get_config.return_value = e2e_config
            
            with patch('src.memfuse_core.store.pgai_store.PGAI_AVAILABLE', True):
                with patch('src.memfuse_core.store.pgai_store.AsyncConnectionPool') as mock_pool:
                    store = PgaiStore(table_name="e2e_multiuser_test")
                    
                    # Mock database setup
                    mock_pool_instance = AsyncMock()
                    mock_cursor = AsyncMock()
                    mock_connection = AsyncMock()
                    mock_connection.cursor.return_value.__aenter__.return_value = mock_cursor
                    mock_pool_instance.connection.return_value.__aenter__.return_value = mock_connection
                    mock_pool.return_value = mock_pool_instance
                    
                    store.pool = mock_pool_instance
                    store.initialized = True
                    
                    # Create multi-user data
                    user1_chunks = [
                        ChunkData(content="User 1 message 1", chunk_id="u1-c1", 
                                metadata={"user_id": "user-1", "session_id": "session-1"}),
                        ChunkData(content="User 1 message 2", chunk_id="u1-c2",
                                metadata={"user_id": "user-1", "session_id": "session-1"})
                    ]
                    
                    user2_chunks = [
                        ChunkData(content="User 2 message 1", chunk_id="u2-c1",
                                metadata={"user_id": "user-2", "session_id": "session-2"}),
                        ChunkData(content="User 2 message 2", chunk_id="u2-c2", 
                                metadata={"user_id": "user-2", "session_id": "session-2"})
                    ]
                    
                    # Add chunks for both users
                    user1_ids = await store.add(user1_chunks)
                    user2_ids = await store.add(user2_chunks)
                    
                    # Verify chunks were added
                    assert len(user1_ids) == 2
                    assert len(user2_ids) == 2
                    
                    # Mock user-specific queries
                    mock_cursor.__aiter__.side_effect = [
                        # User 1 chunks
                        [("u1-c1", "User 1 message 1", '{"user_id": "user-1", "session_id": "session-1"}', None, None),
                         ("u1-c2", "User 1 message 2", '{"user_id": "user-1", "session_id": "session-1"}', None, None)],
                        # User 2 chunks  
                        [("u2-c1", "User 2 message 1", '{"user_id": "user-2", "session_id": "session-2"}', None, None),
                         ("u2-c2", "User 2 message 2", '{"user_id": "user-2", "session_id": "session-2"}', None, None)]
                    ]
                    
                    # Test user isolation
                    user1_data = await store.get_chunks_by_user("user-1")
                    user2_data = await store.get_chunks_by_user("user-2")
                    
                    # Verify user data isolation
                    assert len(user1_data) == 2
                    assert len(user2_data) == 2
                    assert all(chunk.metadata.get("user_id") == "user-1" for chunk in user1_data)
                    assert all(chunk.metadata.get("user_id") == "user-2" for chunk in user2_data)
    
    @pytest.mark.asyncio
    async def test_performance_simulation(self, e2e_config):
        """Test performance with larger dataset simulation."""
        with patch('src.memfuse_core.store.pgai_store.config_manager') as mock_config_manager:
            mock_config_manager.get_config.return_value = e2e_config
            
            with patch('src.memfuse_core.store.pgai_store.PGAI_AVAILABLE', True):
                with patch('src.memfuse_core.store.pgai_store.AsyncConnectionPool') as mock_pool:
                    store = PgaiStore(table_name="e2e_performance_test")
                    
                    # Mock database setup
                    mock_pool_instance = AsyncMock()
                    mock_cursor = AsyncMock()
                    mock_connection = AsyncMock()
                    mock_connection.cursor.return_value.__aenter__.return_value = mock_cursor
                    mock_pool_instance.connection.return_value.__aenter__.return_value = mock_connection
                    mock_pool.return_value = mock_pool_instance
                    
                    store.pool = mock_pool_instance
                    store.initialized = True
                    
                    # Create large batch of chunks
                    large_batch = []
                    for i in range(100):
                        chunk = ChunkData(
                            content=f"This is test message number {i} with some content for performance testing.",
                            chunk_id=f"perf-chunk-{i}",
                            metadata={
                                "user_id": f"user-{i % 10}",  # 10 different users
                                "session_id": f"session-{i % 20}",  # 20 different sessions
                                "batch_id": "performance-test"
                            }
                        )
                        large_batch.append(chunk)
                    
                    # Test batch insertion
                    import time
                    start_time = time.time()
                    result_ids = await store.add(large_batch)
                    end_time = time.time()
                    
                    # Verify batch processing
                    assert len(result_ids) == 100
                    processing_time = end_time - start_time
                    
                    # Log performance (in real test, this would be measured)
                    print(f"Processed 100 chunks in {processing_time:.2f} seconds")
                    
                    # Verify database calls were made
                    assert mock_cursor.execute.call_count == 100
                    mock_connection.commit.assert_called()
