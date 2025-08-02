import asyncio
import sys
import os
import time
sys.path.insert(0, 'src')

from memfuse_core.services.global_model_manager import GlobalModelManager

async def test():
    config = {
        'embedding': {'enabled': True, 'model': 'all-MiniLM-L6-v2'},
        'retrieval': {'use_rerank': True},
        'memory': {'layers': {'m1': {'enabled': False}, 'm2': {'enabled': False}}}
    }
    
    GlobalModelManager._instance = None
    GlobalModelManager._initialized = False
    
    start = time.time()
    manager = GlobalModelManager()
    await manager.initialize_models(config)
    init_time = time.time() - start
    
    start = time.time()
    for _ in range(100):
        manager.get_embedding_model()
        manager.get_reranking_model()
    access_time = time.time() - start
    
    stats = manager.get_performance_stats()
    print(f'Init time: {init_time:.4f}s')
    print(f'Access time (200 ops): {access_time:.4f}s')
    print(f'Models loaded: {stats["total_models"]}')
    print(f'Total usage: {stats["total_model_usage"]}')

asyncio.run(test())