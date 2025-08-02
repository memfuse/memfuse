#!/usr/bin/env python3
"""Simple performance test for GlobalModelManager."""

import asyncio
import sys
import os
import time

# Add src to path
sys.path.insert(0, 'src')

from memfuse_core.services.global_model_manager import GlobalModelManager, ModelType


async def test_model_performance():
    """Test GlobalModelManager performance."""
    # Sample config
    config = {
        'embedding': {
            'enabled': True,
            'model': 'all-MiniLM-L6-v2',
            'dimension': 384
        },
        'retrieval': {
            'use_rerank': True,
            'rerank': {
                'model': 'cross-encoder/ms-marco-MiniLM-L-6-v2'
            }
        },
        'memory': {
            'layers': {
                'm1': {'enabled': False},  # Disable to avoid LLM loading
                'm2': {'enabled': False}
            }
        }
    }
    
    print("Testing GlobalModelManager performance...")
    
    # Reset singleton for testing
    GlobalModelManager._instance = None
    GlobalModelManager._initialized = False
    
    # Test model initialization
    start = time.time()
    model_manager = GlobalModelManager()
    await model_manager.initialize_models(config)
    init_time = time.time() - start
    
    print(f'Model initialization time: {init_time:.4f}s')
    
    # Test model access performance
    start = time.time()
    for _ in range(1000):
        embedding_model = model_manager.get_embedding_model()
        reranking_model = model_manager.get_reranking_model()
    access_time = time.time() - start
    
    print(f'Model access time (2000 ops): {access_time:.4f}s')
    
    # Get performance statistics
    stats = model_manager.get_performance_stats()
    print(f'Total models loaded: {stats["total_models"]}')
    print(f'Total load time: {stats["total_load_time_seconds"]:.4f}s')
    print(f'Total model usage: {stats["total_model_usage"]}')
    
    # Test model health checking
    start = time.time()
    for _ in range(100):
        await model_manager.check_model_health("embedding")
        await model_manager.check_model_health("reranking")
    health_check_time = time.time() - start
    
    print(f'Health check time (200 ops): {health_check_time:.4f}s')
    
    # List all models
    models = model_manager.list_models()
    print(f'Loaded models: {list(models.keys())}')
    
    for key, model_info in models.items():
        print(f'  {key}: {model_info.model_name} ({model_info.model_type.value}) - {model_info.load_time:.4f}s load time')
    
    print("✓ GlobalModelManager performance test completed!")


if __name__ == "__main__":
    asyncio.run(test_model_performance())