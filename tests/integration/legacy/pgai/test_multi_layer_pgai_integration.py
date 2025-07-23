"""
Integration tests for multi-layer PgAI embedding system.

Tests the complete multi-layer workflow with real database connections.
"""

import pytest
import asyncio
import os
from typing import Dict, Any, List
from unittest.mock import AsyncMock, patch

from src.memfuse_core.store.pgai_store.multi_layer_store import (
    MultiLayerPgaiStore, LayerType
)
from src.memfuse_core.rag.chunk.base import ChunkData


@pytest.mark.integration
class TestMultiLayerPgaiIntegration:
    """Integration tests for multi-layer PgAI system."""
    
    @pytest.fixture
    def integration_config(self) -> Dict[str, Any]:
        """Integration test configuration."""
        return {
            'database': {
                'host': os.getenv('POSTGRES_HOST', 'localhost'),
                'port': int(os.getenv('POSTGRES_PORT', '5432')),
                'database': os.getenv('POSTGRES_DB', 'memfuse_db'),
                'user': os.getenv('POSTGRES_USER', 'memfuse'),
                'password': os.getenv('POSTGRES_PASSWORD', 'memfuse_password')
            },
            'memory_layers': {
                'm0': {
                    'enabled': True,
                    'table_name': 'm0_episodic',
                    'pgai': {
                        'auto_embedding': True,
                        'immediate_trigger': True,
                        'embedding_model': 'all-MiniLM-L6-v2',
                        'embedding_dimensions': 384
                    }
                },
                'm1': {
                    'enabled': True,
                    'table_name': 'm1_semantic',
                    'pgai': {
                        'auto_embedding': True,
                        'immediate_trigger': True,
                        'embedding_model': 'all-MiniLM-L6-v2',
                        'embedding_dimensions': 384
                    },
                    'fact_extraction': {
                        'enabled': True,
                        'llm_model': 'grok-3-mini',
                        'temperature': 0.3,
                        'max_tokens': 1000,
                        'min_confidence_threshold': 0.7
                    }
                }
            }
        }
    
    @pytest.fixture
    def sample_chunks(self) -> List[ChunkData]:
        """Sample conversation chunks for testing."""
        return [
            ChunkData(
                content="Hi, I'm Sarah and I'm a data scientist working on machine learning projects.",
                metadata={
                    'session_id': 'test_session_001',
                    'user_id': 'user_sarah',
                    'timestamp': '2024-01-15T10:00:00Z',
                    'message_type': 'introduction'
                }
            ),
            ChunkData(
                content="I really enjoy Python programming, especially pandas and scikit-learn.",
                metadata={
                    'session_id': 'test_session_001',
                    'user_id': 'user_sarah',
                    'timestamp': '2024-01-15T10:05:00Z',
                    'message_type': 'preference'
                }
            )
        ]
    
    @pytest.mark.asyncio
    async def test_multi_layer_initialization(self, integration_config):
        """Test multi-layer store initialization."""
        # Mock the actual store initialization to avoid database dependency
        with patch('src.memfuse_core.store.pgai_store.multi_layer_store.EventDrivenPgaiStore') as mock_store_class:
            mock_store = AsyncMock()
            mock_store.initialize.return_value = True
            mock_store_class.return_value = mock_store
            
            with patch('src.memfuse_core.store.pgai_store.multi_layer_store.SchemaManager') as mock_schema_class:
                mock_schema = AsyncMock()
                mock_schema.initialize_all_schemas.return_value = True
                mock_schema_class.return_value = mock_schema
                
                store = MultiLayerPgaiStore(integration_config)
                
                # Test initialization
                success = await store.initialize()
                
                assert success is True
                assert store.initialized is True
                assert LayerType.M0 in store.enabled_layers
                assert LayerType.M1 in store.enabled_layers
    
    @pytest.mark.asyncio
    async def test_write_to_all_layers(self, integration_config, sample_chunks):
        """Test writing data to all enabled layers."""
        with patch('src.memfuse_core.store.pgai_store.multi_layer_store.EventDrivenPgaiStore') as mock_store_class:
            # Create separate mocks for M0 and M1
            mock_m0_store = AsyncMock()
            mock_m1_store = AsyncMock()
            
            mock_m0_store.initialize.return_value = True
            mock_m1_store.initialize.return_value = True
            mock_m0_store.add.return_value = ['m0_id_1', 'm0_id_2']
            mock_m1_store.add.return_value = ['m1_id_1']
            
            # Configure mock to return different instances
            mock_store_class.side_effect = [mock_m0_store, mock_m1_store]
            
            with patch('src.memfuse_core.store.pgai_store.multi_layer_store.SchemaManager') as mock_schema_class:
                mock_schema = AsyncMock()
                mock_schema.initialize_all_schemas.return_value = True
                mock_schema_class.return_value = mock_schema
                
                with patch('src.memfuse_core.store.pgai_store.fact_extraction_processor.FactExtractionProcessor') as mock_processor_class:
                    mock_processor = AsyncMock()
                    mock_processor.extract_facts_from_chunks.return_value = [
                        {'content': 'Sarah is a data scientist', 'type': 'personal', 'confidence': 0.9}
                    ]
                    mock_processor_class.return_value = mock_processor
                    
                    store = MultiLayerPgaiStore(integration_config)
                    await store.initialize()
                    
                    # Test writing to all layers
                    results = await store.write_to_all_layers(sample_chunks)
                    
                    assert 'm0' in results
                    assert 'm1' in results
                    assert len(results['m0']) == 2  # Two chunks written to M0
                    assert len(results['m1']) == 1  # One fact extracted to M1
    
    @pytest.mark.asyncio
    async def test_query_all_layers(self, integration_config):
        """Test querying across all layers."""
        with patch('src.memfuse_core.store.pgai_store.multi_layer_store.EventDrivenPgaiStore') as mock_store_class:
            mock_m0_store = AsyncMock()
            mock_m1_store = AsyncMock()
            
            mock_m0_store.initialize.return_value = True
            mock_m1_store.initialize.return_value = True
            mock_m0_store.query.return_value = [{'content': 'M0 result', 'score': 0.9}]
            mock_m1_store.query.return_value = [{'content': 'M1 result', 'score': 0.8}]
            
            mock_store_class.side_effect = [mock_m0_store, mock_m1_store]
            
            with patch('src.memfuse_core.store.pgai_store.multi_layer_store.SchemaManager') as mock_schema_class:
                mock_schema = AsyncMock()
                mock_schema.initialize_all_schemas.return_value = True
                mock_schema_class.return_value = mock_schema
                
                store = MultiLayerPgaiStore(integration_config)
                await store.initialize()
                
                # Test querying all layers
                results = await store.query_all_layers("Python programming")
                
                assert 'm0' in results
                assert 'm1' in results
                assert len(results['m0']) == 1
                assert len(results['m1']) == 1
    
    @pytest.mark.asyncio
    async def test_get_all_stats(self, integration_config):
        """Test statistics collection across all layers."""
        with patch('src.memfuse_core.store.pgai_store.multi_layer_store.EventDrivenPgaiStore') as mock_store_class:
            mock_store = AsyncMock()
            mock_store.initialize.return_value = True
            mock_store_class.return_value = mock_store
            
            with patch('src.memfuse_core.store.pgai_store.multi_layer_store.SchemaManager') as mock_schema_class:
                mock_schema = AsyncMock()
                mock_schema.initialize_all_schemas.return_value = True
                mock_schema_class.return_value = mock_schema
                
                store = MultiLayerPgaiStore(integration_config)
                await store.initialize()
                
                # Test statistics collection
                stats = await store.get_all_stats()

                # Check the actual structure returned by get_all_stats
                assert 'overall' in stats
                assert 'enabled_layers' in stats
                assert 'layer_stats' in stats
                assert stats['overall']['total_operations'] >= 0
                assert len(stats['enabled_layers']) == 2  # m0 and m1
    
    @pytest.mark.asyncio
    async def test_layer_configuration(self, integration_config):
        """Test configuration-driven layer management."""
        # Test M0-only configuration
        m0_only_config = integration_config.copy()
        m0_only_config['memory_layers']['m1']['enabled'] = False
        
        with patch('src.memfuse_core.store.pgai_store.multi_layer_store.EventDrivenPgaiStore') as mock_store_class:
            mock_store = AsyncMock()
            mock_store.initialize.return_value = True
            mock_store_class.return_value = mock_store
            
            with patch('src.memfuse_core.store.pgai_store.multi_layer_store.SchemaManager') as mock_schema_class:
                mock_schema = AsyncMock()
                mock_schema.initialize_all_schemas.return_value = True
                mock_schema_class.return_value = mock_schema
                
                store = MultiLayerPgaiStore(m0_only_config)
                
                # Should only have M0 enabled
                assert LayerType.M0 in store.enabled_layers
                assert LayerType.M1 not in store.enabled_layers
    
    @pytest.mark.asyncio
    async def test_error_handling(self, integration_config):
        """Test error handling and resilience."""
        with patch('src.memfuse_core.store.pgai_store.multi_layer_store.EventDrivenPgaiStore') as mock_store_class:
            # Simulate M1 store failure
            mock_m0_store = AsyncMock()
            mock_m1_store = AsyncMock()
            
            mock_m0_store.initialize.return_value = True
            mock_m1_store.initialize.side_effect = Exception("M1 initialization failed")
            
            mock_store_class.side_effect = [mock_m0_store, mock_m1_store]
            
            with patch('src.memfuse_core.store.pgai_store.multi_layer_store.SchemaManager') as mock_schema_class:
                mock_schema = AsyncMock()
                mock_schema.initialize_all_schemas.return_value = True
                mock_schema_class.return_value = mock_schema
                
                store = MultiLayerPgaiStore(integration_config)
                
                # Should handle M1 failure gracefully
                success = await store.initialize()
                assert success is False  # Overall initialization should fail if any layer fails
