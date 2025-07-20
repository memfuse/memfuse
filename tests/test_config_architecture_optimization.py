"""Test script for configuration architecture optimization.

This script tests the optimized configuration architecture with autonomous
component configuration management through the BufferConfigManager.
"""

import asyncio
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from memfuse_core.buffer.config_factory import ComponentConfigFactory, BufferConfigManager
from memfuse_core.services.buffer_service import BufferService


class MockMemoryService:
    """Mock memory service for testing."""
    
    def __init__(self, user_id: str = "test_user"):
        self._user_id = user_id
    
    async def query(self, query: str, **kwargs):
        """Mock query method."""
        return {
            "status": "success",
            "data": {"results": [], "total": 0}
        }
    
    async def add_batch(self, rounds):
        """Mock add_batch method."""
        return {"status": "success", "message": "Added rounds"}


async def test_component_config_factory():
    """Test ComponentConfigFactory functionality."""
    print("\n=== Test: ComponentConfigFactory ===")
    
    # Test default configuration creation
    round_config = ComponentConfigFactory.create_component_config('round_buffer')
    print(f"Default round_buffer config: {round_config}")
    
    # Verify default values
    assert round_config['max_tokens'] == 800
    assert round_config['max_size'] == 5
    assert round_config['token_model'] == 'gpt-4o-mini'
    
    # Test user override
    user_override = {'max_tokens': 1000, 'custom_setting': 'test'}
    custom_config = ComponentConfigFactory.create_component_config(
        'round_buffer', 
        user_override
    )
    print(f"Custom round_buffer config: {custom_config}")
    
    # Verify override worked
    assert custom_config['max_tokens'] == 1000
    assert custom_config['custom_setting'] == 'test'
    assert custom_config['max_size'] == 5  # Default preserved
    
    print("✅ ComponentConfigFactory test passed")


async def test_global_context_application():
    """Test global configuration context application."""
    print("\n=== Test: Global Context Application ===")
    
    # Create global config with model settings
    global_config = {
        'model': {
            'default_model': 'gpt-4',
            'embedding_model': 'custom-embedding-model'
        },
        'performance': {
            'max_workers': 5,
            'flush_interval': 30.0
        }
    }
    
    # Test round_buffer with global context
    round_config = ComponentConfigFactory.create_component_config(
        'round_buffer',
        None,
        global_config
    )
    print(f"Round buffer with global context: {round_config}")
    
    # Should use global model setting
    assert round_config['token_model'] == 'gpt-4'
    
    # Test hybrid_buffer with global context
    hybrid_config = ComponentConfigFactory.create_component_config(
        'hybrid_buffer',
        None,
        global_config
    )
    print(f"Hybrid buffer with global context: {hybrid_config}")
    
    # Should use global embedding model
    assert hybrid_config['embedding_model'] == 'custom-embedding-model'
    
    # Test flush_manager with global context
    flush_config = ComponentConfigFactory.create_component_config(
        'flush_manager',
        None,
        global_config
    )
    print(f"Flush manager with global context: {flush_config}")
    
    # Should use global performance settings
    assert flush_config['max_workers'] == 5
    assert flush_config['flush_interval'] == 30.0
    
    print("✅ Global context application test passed")


async def test_buffer_config_manager():
    """Test BufferConfigManager functionality."""
    print("\n=== Test: BufferConfigManager ===")
    
    # Create global config
    global_config = {
        'buffer': {
            'round_buffer': {'max_tokens': 1200},
            'hybrid_buffer': {'max_size': 8},
            'query': {'max_size': 20},
            'performance': {'max_flush_workers': 4}
        },
        'retrieval': {'use_rerank': False}
    }
    
    # Create config manager
    config_manager = BufferConfigManager(global_config)
    
    # Test complete BufferService configuration
    buffer_service_config = config_manager.get_buffer_service_config()
    print(f"Complete BufferService config: {buffer_service_config}")
    
    # Verify structure
    assert 'write_buffer' in buffer_service_config
    assert 'query_buffer' in buffer_service_config
    assert 'speculative_buffer' in buffer_service_config
    assert 'retrieval' in buffer_service_config
    
    # Verify WriteBuffer sub-components
    write_config = buffer_service_config['write_buffer']
    assert 'round_buffer' in write_config
    assert 'hybrid_buffer' in write_config
    assert 'flush_manager' in write_config
    
    # Verify user overrides were applied
    assert write_config['round_buffer']['max_tokens'] == 1200
    assert write_config['hybrid_buffer']['max_size'] == 8
    assert write_config['flush_manager']['max_workers'] == 4
    
    # Verify query buffer config
    query_config = buffer_service_config['query_buffer']
    assert query_config['max_size'] == 20
    
    # Verify retrieval config
    retrieval_config = buffer_service_config['retrieval']
    assert retrieval_config['use_rerank'] == False
    
    print("✅ BufferConfigManager test passed")


async def test_configuration_validation():
    """Test configuration validation."""
    print("\n=== Test: Configuration Validation ===")
    
    # Test valid configuration
    valid_config = {
        'buffer': {
            'round_buffer': {'max_tokens': 800, 'max_size': 5},
            'hybrid_buffer': {'max_size': 5, 'chunk_strategy': 'message'},
            'query': {'max_size': 15, 'cache_size': 100}
        }
    }
    
    config_manager = BufferConfigManager(valid_config)
    assert config_manager.validate_configuration() == True
    
    # Test invalid configuration (should handle gracefully)
    invalid_config = {
        'buffer': {
            'round_buffer': {'max_tokens': -1, 'max_size': 0},  # Invalid values
            'hybrid_buffer': {'max_size': -5}
        }
    }
    
    try:
        invalid_config_manager = BufferConfigManager(invalid_config)
        # Should handle validation errors gracefully
        result = invalid_config_manager.validate_configuration()
        print(f"Invalid config validation result: {result}")
    except Exception as e:
        print(f"Validation error handled: {e}")
    
    print("✅ Configuration validation test passed")


async def test_buffer_service_with_optimized_config():
    """Test BufferService with optimized configuration architecture."""
    print("\n=== Test: BufferService with Optimized Config ===")
    
    # Create mock memory service
    mock_memory_service = MockMemoryService("test_user_config")
    
    # Create custom configuration
    custom_config = {
        'buffer': {
            'round_buffer': {
                'max_tokens': 1000,
                'max_size': 8,
                'token_model': 'gpt-4'
            },
            'hybrid_buffer': {
                'max_size': 10,
                'chunk_strategy': 'semantic',
                'embedding_model': 'custom-embedding'
            },
            'query': {
                'max_size': 25,
                'cache_size': 200,
                'default_sort_by': 'timestamp'
            },
            'performance': {
                'max_flush_workers': 5,
                'flush_interval': 45.0
            }
        },
        'retrieval': {
            'use_rerank': True
        }
    }
    
    # Create BufferService with optimized configuration
    buffer_service = BufferService(
        memory_service=mock_memory_service,
        user="test_user_config",
        config=custom_config
    )
    
    # Verify configuration was applied correctly
    assert buffer_service.use_rerank == True
    
    # Check WriteBuffer configuration through stats
    write_stats = buffer_service.write_buffer.get_stats()
    print(f"WriteBuffer stats: {write_stats}")

    # Access nested stats structure
    if 'write_buffer' in write_stats:
        nested_stats = write_stats['write_buffer']
        round_stats = nested_stats.get('round_buffer', {})
        hybrid_stats = nested_stats.get('hybrid_buffer', {})
    else:
        round_stats = write_stats.get('round_buffer', {})
        hybrid_stats = write_stats.get('hybrid_buffer', {})

    # Verify round buffer configuration
    assert round_stats.get('max_tokens') == 1000
    assert round_stats.get('max_size') == 8
    assert round_stats.get('token_model') == 'gpt-4'

    # Verify hybrid buffer configuration
    assert hybrid_stats.get('max_size') == 10
    assert hybrid_stats.get('chunk_strategy') == 'semantic'
    assert hybrid_stats.get('embedding_model') == 'custom-embedding'
    
    # Check QueryBuffer configuration
    query_stats = buffer_service.query_buffer.get_stats()
    print(f"QueryBuffer stats: {query_stats}")
    
    assert query_stats['max_size'] == 25
    assert query_stats['cache_size'] == 200
    assert query_stats['default_sort_by'] == 'timestamp'
    
    print("✅ BufferService with optimized config test passed")


async def test_component_autonomy():
    """Test component configuration autonomy."""
    print("\n=== Test: Component Configuration Autonomy ===")
    
    # Test that components can be configured independently
    config_manager = BufferConfigManager()
    
    # Get individual component configurations
    round_config = config_manager.get_component_config('round_buffer')
    hybrid_config = config_manager.get_component_config('hybrid_buffer')
    query_config = config_manager.get_component_config('query_buffer')
    
    print(f"Independent round_buffer config: {round_config}")
    print(f"Independent hybrid_buffer config: {hybrid_config}")
    print(f"Independent query_buffer config: {query_config}")
    
    # Verify each component has its own complete configuration
    assert 'max_tokens' in round_config
    assert 'max_size' in hybrid_config
    assert 'cache_size' in query_config
    
    # Test component-specific overrides
    custom_round_config = config_manager.get_component_config(
        'round_buffer',
        {'max_tokens': 2000, 'custom_param': 'test'}
    )
    
    assert custom_round_config['max_tokens'] == 2000
    assert custom_round_config['custom_param'] == 'test'
    assert 'max_size' in custom_round_config  # Default preserved
    
    print("✅ Component configuration autonomy test passed")


async def run_all_tests():
    """Run all configuration architecture optimization tests."""
    print("Starting Configuration Architecture Optimization Tests...")
    
    try:
        await test_component_config_factory()
        await test_global_context_application()
        await test_buffer_config_manager()
        await test_configuration_validation()
        await test_buffer_service_with_optimized_config()
        await test_component_autonomy()
        
        print("\n🎉 All configuration architecture optimization tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    # Run tests
    success = asyncio.run(run_all_tests())
    
    if success:
        print("\n✅ Configuration architecture optimization implementation is working correctly!")
        exit(0)
    else:
        print("\n❌ Configuration architecture optimization tests failed!")
        exit(1)
