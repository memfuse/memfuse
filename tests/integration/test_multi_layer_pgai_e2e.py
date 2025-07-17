"""
End-to-end integration tests for multi-layer PgAI embedding system.

Tests the complete flow from M0 data ingestion through M1 fact extraction
and automatic embedding generation for both layers.
"""

import pytest
import asyncio
import os
import tempfile
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List

from src.memfuse_core.store.pgai_store.multi_layer_store import (
    MultiLayerPgaiStore, LayerType
)
from src.memfuse_core.store.pgai_store.fact_extraction_processor import (
    FactExtractionProcessor
)
from src.memfuse_core.rag.chunk.base import ChunkData


@pytest.mark.integration
class TestMultiLayerPgaiE2E:
    """End-to-end integration tests for multi-layer PgAI system."""
    
    @pytest.fixture
    def integration_config(self) -> Dict[str, Any]:
        """Integration test configuration."""
        return {
            'database': {
                'host': os.getenv('TEST_POSTGRES_HOST', 'localhost'),
                'port': int(os.getenv('TEST_POSTGRES_PORT', '5432')),
                'database': os.getenv('TEST_POSTGRES_DB', 'test_memfuse'),
                'user': os.getenv('TEST_POSTGRES_USER', 'test_user'),
                'password': os.getenv('TEST_POSTGRES_PASSWORD', 'test_pass')
            },
            'memory_layers': {
                'm0': {
                    'enabled': True,
                    'table_name': 'test_m0_episodic',
                    'pgai': {
                        'auto_embedding': True,
                        'immediate_trigger': False,  # Use polling for tests
                        'embedding_model': 'all-MiniLM-L6-v2',
                        'embedding_dimensions': 384
                    },
                    'performance': {
                        'max_retries': 2,
                        'retry_interval': 1.0,
                        'worker_count': 1,
                        'queue_size': 10,
                        'batch_size': 5
                    }
                },
                'm1': {
                    'enabled': True,
                    'table_name': 'test_m1_semantic',
                    'pgai': {
                        'auto_embedding': True,
                        'immediate_trigger': False,  # Use polling for tests
                        'embedding_model': 'all-MiniLM-L6-v2',
                        'embedding_dimensions': 384
                    },
                    'fact_extraction': {
                        'enabled': True,
                        'llm_model': 'mock-model',
                        'temperature': 0.3,
                        'min_confidence_threshold': 0.6,  # Lower for testing
                        'batch_size': 3,
                        'context_window': 1
                    },
                    'performance': {
                        'max_retries': 2,
                        'retry_interval': 1.0,
                        'worker_count': 1,
                        'queue_size': 10,
                        'batch_size': 3
                    }
                }
            }
        }
    
    @pytest.fixture
    def conversation_chunks(self) -> List[ChunkData]:
        """Realistic conversation chunks for testing."""
        return [
            ChunkData(
                content="Hi! I'm Sarah and I work as a data scientist at TechCorp. I love working with machine learning models.",
                metadata={
                    'session_id': 'test_session_001',
                    'user_id': 'user_sarah',
                    'timestamp': '2024-01-15T10:00:00Z',
                    'message_type': 'introduction'
                }
            ),
            ChunkData(
                content="I really enjoy Python programming, especially pandas and scikit-learn. They're my favorite tools for data analysis.",
                metadata={
                    'session_id': 'test_session_001',
                    'user_id': 'user_sarah',
                    'timestamp': '2024-01-15T10:01:00Z',
                    'message_type': 'preference'
                }
            ),
            ChunkData(
                content="I've decided to learn deep learning this year. I plan to start with TensorFlow and then move to PyTorch.",
                metadata={
                    'session_id': 'test_session_001',
                    'user_id': 'user_sarah',
                    'timestamp': '2024-01-15T10:02:00Z',
                    'message_type': 'decision'
                }
            ),
            ChunkData(
                content="Tomorrow I have a meeting with my team to discuss our new recommendation system project.",
                metadata={
                    'session_id': 'test_session_001',
                    'user_id': 'user_sarah',
                    'timestamp': '2024-01-15T10:03:00Z',
                    'message_type': 'temporal'
                }
            ),
            ChunkData(
                content="I live in San Francisco and I graduated from Stanford University with a degree in Computer Science.",
                metadata={
                    'session_id': 'test_session_001',
                    'user_id': 'user_sarah',
                    'timestamp': '2024-01-15T10:04:00Z',
                    'message_type': 'personal'
                }
            )
        ]

    @pytest.fixture
    def mock_multi_layer_stores(self):
        """Shared fixture for multi-layer store mocking."""
        mock_m0_store = AsyncMock()
        mock_m1_store = AsyncMock()

        # Default successful responses
        mock_m0_store.initialize.return_value = True
        mock_m1_store.initialize.return_value = True
        mock_m0_store.add.return_value = ['m0_id_1', 'm0_id_2', 'm0_id_3']
        mock_m1_store.add.return_value = ['m1_id_1', 'm1_id_2']

        return mock_m0_store, mock_m1_store

    @pytest.mark.asyncio
    async def test_multi_layer_initialization(self, integration_config):
        """Test initialization of multi-layer system."""
        # Mock the database connections to avoid actual DB dependency
        with patch('src.memfuse_core.store.pgai_store.event_driven_store.EventDrivenPgaiStore') as mock_store_class:
            mock_store = AsyncMock()
            mock_store.initialize.return_value = True
            mock_store_class.return_value = mock_store
            
            store = MultiLayerPgaiStore(integration_config)
            
            # Test initialization
            success = await store.initialize()
            
            assert success is True
            assert store.initialized is True
            assert LayerType.M0 in store.enabled_layers
            assert LayerType.M1 in store.enabled_layers
            assert len(store.layer_stores) == 2
    
    @pytest.mark.asyncio
    async def test_m0_data_ingestion_mock(self, integration_config, conversation_chunks):
        """Test M0 data ingestion with mocked stores."""
        with patch('src.memfuse_core.store.pgai_store.event_driven_store.EventDrivenPgaiStore') as mock_store_class:
            # Setup mock stores
            mock_m0_store = AsyncMock()
            mock_m1_store = AsyncMock()
            
            mock_m0_store.initialize.return_value = True
            mock_m1_store.initialize.return_value = True
            
            mock_m0_store.add.return_value = ['m0_id_1', 'm0_id_2', 'm0_id_3']
            mock_m1_store.add.return_value = ['m1_id_1', 'm1_id_2']
            
            # Configure mock to return different instances
            mock_store_class.side_effect = [mock_m0_store, mock_m1_store]
            
            store = MultiLayerPgaiStore(integration_config)
            await store.initialize()
            
            # Test M0 data ingestion
            chunk_batch = conversation_chunks[:3]
            m0_ids = await store.write_to_layer(LayerType.M0, chunk_batch)
            
            assert len(m0_ids) == 3
            assert all(id.startswith('m0_id_') for id in m0_ids)
            
            # Verify M0 store was called
            mock_m0_store.add.assert_called_once()
            call_args = mock_m0_store.add.call_args[0][0]
            assert len(call_args) == 3
            assert all(isinstance(chunk, ChunkData) for chunk in call_args)
    
    @pytest.mark.asyncio
    async def test_parallel_layer_processing(self, integration_config, conversation_chunks):
        """Test parallel processing of M0 and M1 layers (new architecture)."""
        with patch('src.memfuse_core.store.pgai_store.event_driven_store.EventDrivenPgaiStore') as mock_store_class:
            # Setup mock stores
            mock_m0_store = AsyncMock()
            mock_m1_store = AsyncMock()

            mock_m0_store.initialize.return_value = True
            mock_m1_store.initialize.return_value = True

            mock_m0_store.add.return_value = ['m0_id_1', 'm0_id_2', 'm0_id_3']
            mock_m1_store.add.return_value = ['m1_id_1', 'm1_id_2']

            # Configure mock to return different instances
            mock_store_class.side_effect = [mock_m0_store, mock_m1_store]

            store = MultiLayerPgaiStore(integration_config)
            await store.initialize()

            # Test parallel processing (new method)
            results = await store.write_to_all_layers(conversation_chunks[:3])

            # Verify results structure
            assert 'm0' in results
            assert 'm1' in results
            assert isinstance(results['m0'], list)
            assert isinstance(results['m1'], list)

            # Verify both stores were called independently
            mock_m0_store.add.assert_called()
            # M1 should process original data, not M0 output

    @pytest.mark.asyncio
    async def test_fact_extraction_pipeline(self, integration_config, conversation_chunks):
        """Test fact extraction with flexible categorization."""
        processor = FactExtractionProcessor(integration_config['memory_layers']['m1']['fact_extraction'])

        # Test fact extraction from conversation chunks
        results = await processor.extract_facts_batch(conversation_chunks[:3])

        assert len(results) == 3

        successful_results = [r for r in results if r.success]
        assert len(successful_results) >= 2  # At least 2 should succeed

        # Verify facts were extracted with flexible categorization
        all_facts = []
        for result in successful_results:
            all_facts.extend(result.extracted_facts)

        # Verify fact structure (updated for flexible categorization)
        for fact in all_facts:
            assert 'id' in fact
            assert 'fact_content' in fact
            assert 'fact_type' in fact  # Should be flexible, not constrained
            assert 'confidence' in fact
            assert fact['confidence'] >= 0.6  # Above threshold
            # fact_category should be present for flexible categorization
            if 'fact_category' in fact:
                assert isinstance(fact['fact_category'], (dict, str))
            assert fact['needs_embedding'] is True
    
    @pytest.mark.asyncio
    async def test_dual_layer_write_with_fact_extraction(self, integration_config, conversation_chunks):
        """Test complete dual-layer write with fact extraction."""
        with patch('src.memfuse_core.store.pgai_store.event_driven_store.EventDrivenPgaiStore') as mock_store_class:
            # Setup mock stores
            mock_m0_store = AsyncMock()
            mock_m1_store = AsyncMock()
            
            mock_m0_store.initialize.return_value = True
            mock_m1_store.initialize.return_value = True
            
            mock_m0_store.add.return_value = ['m0_id_1']
            mock_m1_store.add.return_value = ['m1_id_1', 'm1_id_2']
            
            mock_store_class.side_effect = [mock_m0_store, mock_m1_store]
            
            store = MultiLayerPgaiStore(integration_config)
            await store.initialize()
            
            # Test dual-layer write with fact extraction
            test_chunk = conversation_chunks[1]  # Preference chunk
            results = await store.write_with_fact_extraction(test_chunk)
            
            assert 'm0' in results
            assert 'm1' in results
            assert len(results['m0']) == 1
            assert len(results['m1']) >= 1
            
            # Verify both stores were called
            mock_m0_store.add.assert_called_once()
            mock_m1_store.add.assert_called_once()
            
            # Verify M1 data contains extracted facts
            m1_call_args = mock_m1_store.add.call_args[0][0]
            assert len(m1_call_args) >= 1
            
            # Check that M1 chunks have fact metadata
            for chunk in m1_call_args:
                assert 'fact_type' in chunk.metadata
                assert 'confidence' in chunk.metadata
                assert 'source_chunk_id' in chunk.metadata
    
    @pytest.mark.asyncio
    async def test_query_all_layers(self, integration_config):
        """Test querying across all layers."""
        with patch('src.memfuse_core.store.pgai_store.event_driven_store.EventDrivenPgaiStore') as mock_store_class:
            # Setup mock stores with query results
            mock_m0_store = AsyncMock()
            mock_m1_store = AsyncMock()
            
            mock_m0_store.initialize.return_value = True
            mock_m1_store.initialize.return_value = True
            
            # Mock query results
            mock_m0_results = [
                ChunkData(content="M0 result about Python programming", metadata={'layer': 'm0'}),
                ChunkData(content="M0 result about data science", metadata={'layer': 'm0'})
            ]
            
            mock_m1_results = [
                ChunkData(content="User prefers Python for data analysis", metadata={'layer': 'm1', 'fact_type': 'preference'})
            ]
            
            mock_m0_store.query.return_value = mock_m0_results
            mock_m1_store.query.return_value = mock_m1_results
            
            mock_store_class.side_effect = [mock_m0_store, mock_m1_store]
            
            store = MultiLayerPgaiStore(integration_config)
            await store.initialize()
            
            # Test query all layers
            results = await store.query_all_layers("Python programming", top_k=5)
            
            assert 'm0' in results
            assert 'm1' in results
            assert len(results['m0']) == 2
            assert len(results['m1']) == 1
            
            # Verify query was called on both stores
            mock_m0_store.query.assert_called_once_with("Python programming", 5)
            mock_m1_store.query.assert_called_once_with("Python programming", 5)
            
            # Verify result content
            m0_contents = [chunk.content for chunk in results['m0']]
            assert any('Python programming' in content for content in m0_contents)
            
            m1_chunk = results['m1'][0]
            assert m1_chunk.metadata['fact_type'] == 'preference'
    
    @pytest.mark.asyncio
    async def test_layer_statistics_collection(self, integration_config):
        """Test comprehensive statistics collection across layers."""
        with patch('src.memfuse_core.store.pgai_store.event_driven_store.EventDrivenPgaiStore') as mock_store_class:
            # Setup mock stores with stats
            mock_m0_store = AsyncMock()
            mock_m1_store = AsyncMock()
            
            mock_m0_store.initialize.return_value = True
            mock_m1_store.initialize.return_value = True
            
            mock_m0_store.get_processing_stats.return_value = {
                'total_processed': 150,
                'success_rate': 0.95,
                'average_processing_time': 0.05,
                'embedding_stats': {
                    'total_embeddings': 150,
                    'pending_embeddings': 5
                }
            }
            
            mock_m1_store.get_processing_stats.return_value = {
                'total_processed': 75,
                'success_rate': 0.88,
                'average_processing_time': 0.12,
                'embedding_stats': {
                    'total_embeddings': 75,
                    'pending_embeddings': 2
                }
            }
            
            mock_store_class.side_effect = [mock_m0_store, mock_m1_store]
            
            store = MultiLayerPgaiStore(integration_config)
            await store.initialize()
            
            # Simulate some operations
            store.stats['total_operations'] = 200
            store.stats['fact_extractions'] = 75
            store.stats['layer_operations']['m0'] = 150
            store.stats['layer_operations']['m1'] = 75
            
            # Test comprehensive stats collection
            stats = await store.get_all_stats()
            
            assert 'overall' in stats
            assert 'enabled_layers' in stats
            assert 'layer_stats' in stats
            
            # Check overall stats
            assert stats['overall']['total_operations'] == 200
            assert stats['overall']['fact_extractions'] == 75
            assert stats['enabled_layers'] == ['m0', 'm1']
            
            # Check layer-specific stats
            m0_stats = stats['layer_stats']['m0']
            assert m0_stats['operations_count'] == 150
            assert m0_stats['store_stats']['total_processed'] == 150
            assert m0_stats['store_stats']['success_rate'] == 0.95
            
            m1_stats = stats['layer_stats']['m1']
            assert m1_stats['operations_count'] == 75
            assert m1_stats['store_stats']['total_processed'] == 75
            assert m1_stats['store_stats']['success_rate'] == 0.88
    
    @pytest.mark.asyncio
    async def test_error_handling_and_resilience(self, integration_config, conversation_chunks):
        """Test error handling and system resilience."""
        with patch('src.memfuse_core.store.pgai_store.event_driven_store.EventDrivenPgaiStore') as mock_store_class:
            # Setup mock stores where M1 fails but M0 succeeds
            mock_m0_store = AsyncMock()
            mock_m1_store = AsyncMock()
            
            mock_m0_store.initialize.return_value = True
            mock_m1_store.initialize.return_value = True
            
            mock_m0_store.add.return_value = ['m0_id_1']
            mock_m1_store.add.side_effect = Exception("M1 storage failed")
            
            mock_store_class.side_effect = [mock_m0_store, mock_m1_store]
            
            store = MultiLayerPgaiStore(integration_config)
            await store.initialize()
            
            # Test that M0 succeeds even when M1 fails
            test_chunk = conversation_chunks[0]
            
            # This should not raise an exception
            try:
                results = await store.write_with_fact_extraction(test_chunk)
                
                # M0 should succeed
                assert 'm0' in results
                assert len(results['m0']) == 1
                
                # M1 should fail but not crash the system
                assert 'm1' in results
                assert len(results['m1']) == 0  # No M1 results due to error
                
            except Exception as e:
                pytest.fail(f"System should handle M1 failure gracefully, but got: {e}")
            
            # Verify error was tracked
            assert store.stats['errors'] > 0
    
    @pytest.mark.asyncio
    async def test_configuration_driven_layer_management(self):
        """Test layer enable/disable through configuration."""
        # Test with only M0 enabled
        m0_only_config = {
            'memory_layers': {
                'm0': {
                    'enabled': True,
                    'table_name': 'test_m0_episodic'
                },
                'm1': {
                    'enabled': False,
                    'table_name': 'test_m1_semantic'
                }
            }
        }
        
        with patch('src.memfuse_core.store.pgai_store.event_driven_store.EventDrivenPgaiStore') as mock_store_class:
            mock_m0_store = AsyncMock()
            mock_m0_store.initialize.return_value = True
            mock_store_class.return_value = mock_m0_store
            
            store = MultiLayerPgaiStore(m0_only_config)
            
            # Only M0 should be enabled
            assert LayerType.M0 in store.enabled_layers
            assert LayerType.M1 not in store.enabled_layers
            
            await store.initialize()
            
            # Only M0 store should be created
            assert LayerType.M0 in store.layer_stores
            assert LayerType.M1 not in store.layer_stores
            assert len(store.layer_stores) == 1
    
    @pytest.mark.asyncio
    async def test_backward_compatibility(self, integration_config, conversation_chunks):
        """Test backward compatibility with single-layer usage."""
        with patch('src.memfuse_core.store.pgai_store.event_driven_store.EventDrivenPgaiStore') as mock_store_class:
            mock_m0_store = AsyncMock()
            mock_m1_store = AsyncMock()
            
            mock_m0_store.initialize.return_value = True
            mock_m1_store.initialize.return_value = True
            
            mock_m0_store.add.return_value = ['m0_id_1', 'm0_id_2']
            mock_m0_store.query.return_value = [
                ChunkData(content="Result 1"),
                ChunkData(content="Result 2")
            ]
            
            mock_store_class.side_effect = [mock_m0_store, mock_m1_store]
            
            store = MultiLayerPgaiStore(integration_config)
            await store.initialize()
            
            # Test backward compatibility methods
            chunks = conversation_chunks[:2]
            
            # Test add method (should use M0)
            ids = await store.add(chunks)
            assert len(ids) == 2
            mock_m0_store.add.assert_called_once()
            
            # Test query method (should use M0)
            results = await store.query("test query", top_k=5)
            assert len(results) == 2
            mock_m0_store.query.assert_called_once_with("test query", 5)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])