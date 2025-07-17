"""
Unit tests for MultiLayerPgaiStore.

Tests the dual-layer PgAI store functionality including configuration,
layer management, and fact extraction integration.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List

from src.memfuse_core.store.pgai_store.multi_layer_store import (
    MultiLayerPgaiStore, LayerType
)
from src.memfuse_core.rag.chunk.base import ChunkData


class TestMultiLayerPgaiStore:
    """Test cases for MultiLayerPgaiStore."""
    
    @pytest.fixture
    def sample_config(self) -> Dict[str, Any]:
        """Sample configuration for testing."""
        return {
            'database': {
                'host': 'localhost',
                'port': 5432,
                'database': 'test_memfuse',
                'user': 'test_user',
                'password': 'test_pass'
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
                    },
                    'performance': {
                        'max_retries': 3,
                        'retry_interval': 5.0,
                        'worker_count': 2,
                        'queue_size': 100
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
                        'min_confidence_threshold': 0.7
                    },
                    'performance': {
                        'max_retries': 3,
                        'retry_interval': 5.0,
                        'worker_count': 2,
                        'queue_size': 100
                    }
                }
            }
        }
    
    @pytest.fixture
    def sample_chunks(self) -> List[ChunkData]:
        """Sample chunk data for testing."""
        return [
            ChunkData(
                content="I really like pizza and pasta. They are my favorite foods.",
                metadata={'session_id': 'test_session', 'user_id': 'test_user'}
            ),
            ChunkData(
                content="I decided to learn Python programming this year.",
                metadata={'session_id': 'test_session', 'user_id': 'test_user'}
            ),
            ChunkData(
                content="My name is John and I work as a software engineer.",
                metadata={'session_id': 'test_session', 'user_id': 'test_user'}
            )
        ]
    
    def test_initialization(self, sample_config):
        """Test MultiLayerPgaiStore initialization."""
        store = MultiLayerPgaiStore(sample_config)

        # ConfigManager applies defaults, so we check key components instead of exact equality
        assert 'memory_layers' in store.config
        assert 'm0' in store.config['memory_layers']
        assert 'm1' in store.config['memory_layers']
        assert store.config['memory_layers']['m0']['enabled'] == True
        assert store.config['memory_layers']['m1']['enabled'] == True

        assert not store.initialized
        assert LayerType.M0 in store.enabled_layers
        assert LayerType.M1 in store.enabled_layers
        assert len(store.layer_stores) == 0
    
    def test_get_enabled_layers(self, sample_config):
        """Test enabled layers detection from configuration."""
        store = MultiLayerPgaiStore(sample_config)
        enabled = store._get_enabled_layers()
        
        assert LayerType.M0 in enabled
        assert LayerType.M1 in enabled
        assert len(enabled) == 2
    
    def test_get_enabled_layers_m0_only(self):
        """Test with only M0 layer enabled."""
        config = {
            'memory_layers': {
                'm0': {'enabled': True, 'table_name': 'm0_episodic'},
                'm1': {'enabled': False, 'table_name': 'm1_semantic'}
            }
        }
        
        store = MultiLayerPgaiStore(config)
        enabled = store._get_enabled_layers()
        
        assert LayerType.M0 in enabled
        assert LayerType.M1 not in enabled
        assert len(enabled) == 1
    
    def test_create_store_config(self, sample_config):
        """Test store configuration creation for layers."""
        store = MultiLayerPgaiStore(sample_config)
        
        # Test M0 configuration
        m0_config = store._create_store_config(LayerType.M0, sample_config['memory_layers']['m0'])
        
        assert 'database' in m0_config
        assert 'pgai' in m0_config
        assert m0_config['pgai']['auto_embedding'] is True
        assert m0_config['pgai']['embedding_model'] == 'all-MiniLM-L6-v2'
        assert m0_config['pgai']['max_retries'] == 3
        
        # Test M1 configuration
        m1_config = store._create_store_config(LayerType.M1, sample_config['memory_layers']['m1'])
        
        assert 'database' in m1_config
        assert 'pgai' in m1_config
        assert m1_config['pgai']['auto_embedding'] is True
        assert m1_config['pgai']['worker_count'] == 2
    
    @pytest.mark.skip(reason="Mock fact extraction methods removed - functionality moved to FactExtractionProcessor")
    async def test_mock_fact_extraction(self, sample_chunks):
        """Test mock fact extraction functionality."""
        config = {
            'memory_layers': {
                'm0': {'enabled': True, 'table_name': 'm0_episodic'},
                'm1': {'enabled': True, 'table_name': 'm1_semantic'}
            }
        }
        
        store = MultiLayerPgaiStore(config)
        
        # Test preference extraction
        preference_chunk = sample_chunks[0]  # "I really like pizza..."
        facts = await store._mock_fact_extraction(preference_chunk.content, {})
        
        assert len(facts) > 0
        preference_facts = [f for f in facts if f['type'] == 'preference']
        assert len(preference_facts) > 0
        assert 'pizza' in preference_facts[0]['content'].lower()
        
        # Test decision extraction
        decision_chunk = sample_chunks[1]  # "I decided to learn..."
        facts = await store._mock_fact_extraction(decision_chunk.content, {})
        
        decision_facts = [f for f in facts if f['type'] == 'decision']
        assert len(decision_facts) > 0
        assert 'python' in decision_facts[0]['content'].lower()
        
        # Test personal information extraction
        personal_chunk = sample_chunks[2]  # "My name is John..."
        facts = await store._mock_fact_extraction(personal_chunk.content, {})
        
        personal_facts = [f for f in facts if f['type'] == 'personal']
        assert len(personal_facts) > 0
        assert 'john' in personal_facts[0]['content'].lower()
    
    @pytest.mark.skip(reason="Method _extract_facts_from_data no longer exists")
    async def test_extract_facts_from_data(self, sample_chunks):
        """Test fact extraction from chunk data."""
        config = {
            'memory_layers': {
                'm0': {'enabled': True, 'table_name': 'm0_episodic'},
                'm1': {'enabled': True, 'table_name': 'm1_semantic'}
            }
        }
        
        store = MultiLayerPgaiStore(config)
        
        # Test single chunk extraction
        single_chunk = sample_chunks[0]
        fact_chunks = await store._extract_facts_from_data(single_chunk, {'session_id': 'test'})
        
        assert isinstance(fact_chunks, list)
        assert len(fact_chunks) > 0
        
        for fact_chunk in fact_chunks:
            assert isinstance(fact_chunk, ChunkData)
            assert 'fact_type' in fact_chunk.metadata
            assert 'confidence' in fact_chunk.metadata
            assert 'source_chunk_id' in fact_chunk.metadata
            assert fact_chunk.metadata['source_chunk_id'] == single_chunk.chunk_id
        
        # Test multiple chunks extraction
        fact_chunks = await store._extract_facts_from_data(sample_chunks, {'session_id': 'test'})
        
        assert len(fact_chunks) >= len(sample_chunks)  # Should extract at least one fact per chunk
        
        # Verify fact types are distributed
        fact_types = [chunk.metadata['fact_type'] for chunk in fact_chunks]
        assert 'preference' in fact_types
        assert 'decision' in fact_types
        assert 'personal' in fact_types
    
    def test_stats_initialization(self, sample_config):
        """Test statistics initialization."""
        store = MultiLayerPgaiStore(sample_config)
        
        assert store.stats['total_operations'] == 0
        assert store.stats['fact_extractions'] == 0
        assert store.stats['errors'] == 0
        assert 'm0' in store.stats['layer_operations']
        assert 'm1' in store.stats['layer_operations']
        assert store.stats['layer_operations']['m0'] == 0
        assert store.stats['layer_operations']['m1'] == 0
    
    @pytest.mark.skip(reason="Method write_with_fact_extraction no longer exists - use write_to_all_layers")
    async def test_write_with_fact_extraction_mock(self, sample_chunks):
        """Test write with fact extraction using mocked stores."""
        config = {
            'memory_layers': {
                'm0': {'enabled': True, 'table_name': 'm0_episodic'},
                'm1': {'enabled': True, 'table_name': 'm1_semantic'}
            }
        }
        
        store = MultiLayerPgaiStore(config)
        
        # Mock the layer stores
        mock_m0_store = AsyncMock()
        mock_m1_store = AsyncMock()
        
        mock_m0_store.add.return_value = ['m0_id_1', 'm0_id_2']
        mock_m1_store.add.return_value = ['m1_id_1', 'm1_id_2']
        
        store.layer_stores = {
            LayerType.M0: mock_m0_store,
            LayerType.M1: mock_m1_store
        }
        store.enabled_layers = [LayerType.M0, LayerType.M1]
        
        # Test write with fact extraction
        results = await store.write_with_fact_extraction(sample_chunks[0])
        
        assert 'm0' in results
        assert 'm1' in results
        assert len(results['m0']) > 0
        assert len(results['m1']) > 0
        
        # Verify M0 store was called
        mock_m0_store.add.assert_called_once()
        
        # Verify M1 store was called
        mock_m1_store.add.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_query_all_layers_mock(self):
        """Test querying all layers with mocked stores."""
        config = {
            'memory_layers': {
                'm0': {'enabled': True, 'table_name': 'm0_episodic'},
                'm1': {'enabled': True, 'table_name': 'm1_semantic'}
            }
        }
        
        store = MultiLayerPgaiStore(config)
        
        # Mock the layer stores
        mock_m0_store = AsyncMock()
        mock_m1_store = AsyncMock()
        
        mock_m0_results = [ChunkData(content="M0 result 1"), ChunkData(content="M0 result 2")]
        mock_m1_results = [ChunkData(content="M1 result 1")]
        
        mock_m0_store.query.return_value = mock_m0_results
        mock_m1_store.query.return_value = mock_m1_results
        
        store.layer_stores = {
            LayerType.M0: mock_m0_store,
            LayerType.M1: mock_m1_store
        }
        store.enabled_layers = [LayerType.M0, LayerType.M1]
        
        # Test query all layers
        results = await store.query_all_layers("test query", top_k=5)
        
        assert 'm0' in results
        assert 'm1' in results
        assert len(results['m0']) == 2
        assert len(results['m1']) == 1
        
        # Verify both stores were queried
        mock_m0_store.query.assert_called_once_with("test query", 5)
        mock_m1_store.query.assert_called_once_with("test query", 5)
    
    @pytest.mark.asyncio
    async def test_get_all_stats_mock(self):
        """Test comprehensive statistics collection."""
        config = {
            'memory_layers': {
                'm0': {'enabled': True, 'table_name': 'm0_episodic'},
                'm1': {'enabled': True, 'table_name': 'm1_semantic'}
            }
        }
        
        store = MultiLayerPgaiStore(config)
        
        # Mock the layer stores
        mock_m0_store = AsyncMock()
        mock_m1_store = AsyncMock()
        
        mock_m0_store.get_processing_stats.return_value = {
            'total_processed': 100,
            'success_rate': 0.95
        }
        mock_m1_store.get_processing_stats.return_value = {
            'total_processed': 50,
            'success_rate': 0.90
        }
        
        store.layer_stores = {
            LayerType.M0: mock_m0_store,
            LayerType.M1: mock_m1_store
        }
        store.enabled_layers = [LayerType.M0, LayerType.M1]
        
        # Update some stats
        store.stats['total_operations'] = 150
        store.stats['fact_extractions'] = 25
        
        # Test get all stats
        stats = await store.get_all_stats()
        
        assert 'overall' in stats
        assert 'enabled_layers' in stats
        assert 'layer_stats' in stats
        
        assert stats['overall']['total_operations'] == 150
        assert stats['overall']['fact_extractions'] == 25
        assert stats['enabled_layers'] == ['m0', 'm1']
        
        assert 'm0' in stats['layer_stats']
        assert 'm1' in stats['layer_stats']
        
        # Verify store stats were collected
        mock_m0_store.get_processing_stats.assert_called_once()
        mock_m1_store.get_processing_stats.assert_called_once()
    
    def test_backward_compatibility_methods(self, sample_config):
        """Test backward compatibility methods."""
        store = MultiLayerPgaiStore(sample_config)
        
        # Ensure backward compatibility methods exist
        assert hasattr(store, 'add')
        assert hasattr(store, 'query')
        assert callable(store.add)
        assert callable(store.query)
    
    @pytest.mark.asyncio
    async def test_cleanup(self):
        """Test cleanup functionality."""
        config = {
            'memory_layers': {
                'm0': {'enabled': True, 'table_name': 'm0_episodic'}
            }
        }
        
        store = MultiLayerPgaiStore(config)
        
        # Mock a layer store
        mock_store = AsyncMock()
        store.layer_stores = {LayerType.M0: mock_store}
        store.initialized = True
        
        # Test cleanup
        await store.cleanup()
        
        assert not store.initialized
        assert len(store.layer_stores) == 0
        mock_store.cleanup.assert_called_once()


if __name__ == '__main__':
    pytest.main([__file__])