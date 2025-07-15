#!/usr/bin/env python3
"""Test script to verify contextual chunking configuration."""

import asyncio
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from memfuse_core.utils.config import config_manager
from memfuse_core.buffer.hybrid_buffer import HybridBuffer


async def test_contextual_chunking_config():
    """Test contextual chunking configuration."""
    print("🔍 Testing Contextual Chunking Configuration")
    print("=" * 50)
    
    # Test 1: Load configuration
    print("1. Loading configuration...")
    try:
        config = config_manager.get_config()
        buffer_config = config.get("buffer", {})
        chunking_config = buffer_config.get("chunking", {})
        
        print(f"   ✅ Buffer config loaded: {bool(buffer_config)}")
        print(f"   ✅ Chunking config loaded: {bool(chunking_config)}")
        
        # Print chunking strategies available
        if chunking_config:
            print(f"   📋 Available strategies: {list(chunking_config.keys())}")
            for strategy, config_data in chunking_config.items():
                print(f"      - {strategy}: {config_data}")
        
    except Exception as e:
        print(f"   ❌ Configuration loading failed: {e}")
        return False
    
    # Test 2: Test HybridBuffer with different strategies
    print("\n2. Testing HybridBuffer strategy loading...")
    
    strategies_to_test = ["message", "contextual"]
    
    for strategy in strategies_to_test:
        print(f"\n   Testing strategy: {strategy}")
        try:
            buffer = HybridBuffer(chunk_strategy=strategy)
            await buffer._load_chunk_strategy()
            
            strategy_class = buffer.chunk_strategy.__class__.__name__
            print(f"   ✅ {strategy} -> {strategy_class}")
            
            # Check if it's the advanced strategy for contextual options
            if strategy == "contextual":
                if strategy_class == "ContextualChunkStrategy":
                    print(f"   ✅ Correctly using advanced contextual strategy")
                    
                    # Check configuration
                    if hasattr(buffer.chunk_strategy, 'enable_contextual'):
                        contextual_enabled = buffer.chunk_strategy.enable_contextual
                        print(f"   📋 Contextual enhancement: {contextual_enabled}")
                    
                    if hasattr(buffer.chunk_strategy, 'context_window_size'):
                        window_size = buffer.chunk_strategy.context_window_size
                        print(f"   📋 Context window size: {window_size}")
                        
                else:
                    print(f"   ⚠️  Expected ContextualChunkStrategy, got {strategy_class}")
            
        except Exception as e:
            print(f"   ❌ Strategy {strategy} failed: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Contextual chunking configuration test completed!")
    return True


if __name__ == "__main__":
    asyncio.run(test_contextual_chunking_config())
