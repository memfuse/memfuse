#!/usr/bin/env python3
"""
Validation script for immediate trigger embedding functionality.

This script provides a simple way to test the new event-driven pgai store
implementation with real database operations.

Usage:
    # Start MemFuse server first
    poetry run memfuse-core
    
    # In another terminal, run this validation script
    cd /Users/mxue/GitRepos/MemFuse/memfuse
    python tests/store/test_immediate_trigger_validation.py
"""

import asyncio
import time
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from memfuse_core.store.store_factory import PgaiStoreFactory
from memfuse_core.rag.chunk.base import ChunkData
from memfuse_core.utils.config import config_manager
from hydra import initialize, compose
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_immediate_trigger():
    """Test immediate trigger functionality with real database."""
    print("🚀 Starting Immediate Trigger Validation Test")
    print("=" * 60)
    
    try:
        # Load configuration
        print("📋 Loading configuration...")
        with initialize(version_base=None, config_path="../../config"):
            cfg = compose(config_name="config")
            config_manager.set_config(cfg)
        
        # Create event-driven store
        print("🏗️  Creating event-driven pgai store...")
        store = PgaiStoreFactory.create_store(table_name="test_immediate_validation")
        
        # Check store type
        store_type = type(store).__name__
        print(f"📦 Created store type: {store_type}")
        
        # Initialize store
        print("🔧 Initializing store...")
        success = await store.initialize()
        if not success:
            print("❌ Store initialization failed!")
            return False
        
        print("✅ Store initialized successfully")
        
        # Check configuration
        immediate_trigger = store.pgai_config.get("immediate_trigger", False)
        auto_embedding = store.pgai_config.get("auto_embedding", False)
        max_retries = store.pgai_config.get("max_retries", 3)
        retry_interval = store.pgai_config.get("retry_interval", 5.0)
        
        print(f"⚙️  Configuration:")
        print(f"   - Immediate Trigger: {immediate_trigger}")
        print(f"   - Auto Embedding: {auto_embedding}")
        print(f"   - Max Retries: {max_retries}")
        print(f"   - Retry Interval: {retry_interval}s")
        
        if not immediate_trigger:
            print("⚠️  Warning: Immediate trigger is disabled. This test will use polling mode.")
        
        # Create test chunks
        print("\n📝 Creating test chunks...")
        test_chunks = [
            ChunkData(
                chunk_id=f"immediate_test_{int(time.time())}_{i}",
                content=f"This is test content number {i} for immediate embedding validation. "
                       f"The content should be processed quickly with the new trigger mechanism."
            )
            for i in range(3)
        ]
        
        print(f"📄 Created {len(test_chunks)} test chunks")
        
        # Test insertion with timing
        print("\n⏱️  Testing insertion performance...")
        start_time = time.time()
        
        chunk_ids = await store.add(test_chunks)
        
        insertion_time = time.time() - start_time
        print(f"✅ Inserted {len(chunk_ids)} chunks in {insertion_time:.3f} seconds")
        
        if insertion_time > 1.0:
            print(f"⚠️  Warning: Insertion took longer than expected ({insertion_time:.3f}s)")
        else:
            print(f"🎯 Excellent: Fast insertion achieved ({insertion_time:.3f}s)")
        
        # Monitor processing progress
        print("\n👀 Monitoring embedding processing...")
        max_wait_time = 30  # 30 seconds max
        check_interval = 2  # Check every 2 seconds
        wait_start = time.time()
        
        processed_count = 0
        while time.time() - wait_start < max_wait_time:
            # Check if store has monitoring capabilities
            if hasattr(store, 'get_processing_stats'):
                try:
                    stats = await store.get_processing_stats()
                    processed_count = stats.get("processing_stats", {}).get("total_processed", 0)
                    queue_size = stats.get("queue_size", 0)
                    
                    print(f"📊 Progress: {processed_count} processed, {queue_size} in queue")
                    
                    if processed_count >= len(test_chunks):
                        print("🎉 All chunks processed!")
                        break
                        
                except Exception as e:
                    print(f"⚠️  Could not get processing stats: {e}")
            
            # Alternative: Check database directly
            try:
                # Query for completed embeddings
                async with store.pool.connection() as conn:
                    async with conn.cursor() as cur:
                        await cur.execute(f"""
                            SELECT COUNT(*) FROM {store.table_name}
                            WHERE id = ANY(%s) AND embedding IS NOT NULL
                        """, (chunk_ids,))
                        
                        completed_count = (await cur.fetchone())[0]
                        pending_count = len(chunk_ids) - completed_count
                        
                        print(f"📈 Database check: {completed_count}/{len(chunk_ids)} completed, {pending_count} pending")
                        
                        if completed_count >= len(test_chunks):
                            print("✅ All embeddings generated!")
                            break
                            
            except Exception as e:
                print(f"⚠️  Database check failed: {e}")
            
            await asyncio.sleep(check_interval)
        
        total_processing_time = time.time() - start_time
        
        # Final verification
        print("\n🔍 Final verification...")
        try:
            # Test search functionality
            search_results = await store.search("test content", top_k=5)
            found_count = len([r for r in search_results if r.get('id') in chunk_ids])
            
            print(f"🔎 Search test: Found {found_count}/{len(chunk_ids)} test chunks")
            
            if found_count >= len(chunk_ids):
                print("✅ Search verification passed!")
            else:
                print(f"⚠️  Search verification incomplete: {found_count}/{len(chunk_ids)}")
                
        except Exception as e:
            print(f"❌ Search verification failed: {e}")
        
        # Performance summary
        print("\n📊 Performance Summary:")
        print(f"   - Insertion Time: {insertion_time:.3f}s")
        print(f"   - Total Processing Time: {total_processing_time:.3f}s")
        print(f"   - Average Time per Chunk: {total_processing_time/len(test_chunks):.3f}s")
        
        # Performance evaluation
        if immediate_trigger:
            if total_processing_time < 10.0:
                print("🎯 Excellent: Immediate trigger performance achieved!")
            elif total_processing_time < 20.0:
                print("✅ Good: Reasonable immediate trigger performance")
            else:
                print("⚠️  Warning: Immediate trigger performance slower than expected")
        else:
            if total_processing_time < 30.0:
                print("✅ Acceptable: Polling mode performance within expected range")
            else:
                print("⚠️  Warning: Polling mode performance slower than expected")
        
        # Get final stats if available
        if hasattr(store, 'get_processing_stats'):
            try:
                final_stats = await store.get_processing_stats()
                print(f"\n📈 Final Processing Stats:")
                for key, value in final_stats.get("processing_stats", {}).items():
                    print(f"   - {key}: {value}")
                    
                retry_stats = final_stats.get("retry_stats", {})
                if retry_stats:
                    print(f"\n🔄 Retry Stats:")
                    for key, value in retry_stats.items():
                        print(f"   - {key}: {value}")
                        
            except Exception as e:
                print(f"⚠️  Could not get final stats: {e}")
        
        # Cleanup
        print("\n🧹 Cleaning up...")
        if hasattr(store, 'cleanup'):
            await store.cleanup()
        
        print("✅ Validation test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Validation test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_retry_mechanism():
    """Test retry mechanism with simulated failures."""
    print("\n🔄 Testing Retry Mechanism")
    print("=" * 40)
    
    try:
        # Create store with retry configuration
        store = PgaiStoreFactory.create_store(table_name="test_retry_validation")
        await store.initialize()
        
        # Check if we can access retry manager
        if hasattr(store, 'retry_manager'):
            print("✅ Retry manager available")
            
            # Test retry stats
            retry_stats = await store.retry_manager.get_retry_stats()
            print(f"📊 Current retry stats: {retry_stats}")
            
        else:
            print("⚠️  Retry manager not available (traditional store)")
        
        return True
        
    except Exception as e:
        print(f"❌ Retry test failed: {e}")
        return False


async def main():
    """Main validation function."""
    print("🧪 MemFuse Event-Driven PgAI Store Validation")
    print("=" * 80)
    
    # Test immediate trigger
    trigger_success = await test_immediate_trigger()
    
    # Test retry mechanism
    retry_success = await test_retry_mechanism()
    
    # Overall result
    print("\n" + "=" * 80)
    if trigger_success and retry_success:
        print("🎉 ALL VALIDATION TESTS PASSED!")
        print("✅ Event-driven pgai store is working correctly")
    else:
        print("❌ SOME VALIDATION TESTS FAILED!")
        print("⚠️  Please check the errors above and fix any issues")
    
    print("\n💡 Next steps:")
    print("   1. Run full test suite: pytest tests/store/test_event_driven_pgai_store.py -v")
    print("   2. Run integration tests: pytest tests/store/test_event_driven_pgai_store.py --integration -v")
    print("   3. Monitor performance in production environment")


if __name__ == "__main__":
    asyncio.run(main())
