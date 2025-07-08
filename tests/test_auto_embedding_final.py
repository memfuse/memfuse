#!/usr/bin/env python3
"""
Auto-Embedding Final Test

Simplified test to validate auto-embedding functionality.
Can be run directly without pytest dependencies.
"""

import asyncio
import sys
import os
import time
import subprocess

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))


def run_sql(sql: str) -> str:
    """Execute SQL command and return result"""
    try:
        cmd = [
            'docker', 'exec', '-i', 'memfuse-pgai-postgres-1',
            'psql', '-U', 'postgres', '-d', 'memfuse', '-t', '-c', sql
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception as e:
        print(f"SQL Error: {e}")
        return ""


def get_record_state(record_id: str) -> dict:
    """Get detailed state of a specific record"""
    sql = f"""
    SELECT 
        CASE WHEN embedding IS NOT NULL THEN 'YES' ELSE 'NO' END as has_embedding,
        needs_embedding,
        CASE WHEN embedding IS NOT NULL THEN array_length(embedding::real[], 1) ELSE 0 END as dimension
    FROM m0_messages 
    WHERE id = '{record_id}';
    """
    
    result = run_sql(sql)
    if result:
        parts = [part.strip() for part in result.split('|')]
        if len(parts) >= 3:
            return {
                'has_embedding': parts[0] == 'YES',
                'needs_embedding': parts[1] == 't',
                'dimension': int(parts[2]) if parts[2].isdigit() else 0
            }
    return {}


async def test_manual_mode():
    """Test manual embedding mode"""
    print("🔬 Testing Manual Mode")
    print("-" * 40)

    try:
        # Load configuration
        from hydra import initialize, compose

        with initialize(version_base=None, config_path="../config"):
            cfg = compose(config_name="config")
            cfg.database.pgai.auto_embedding = False

            from memfuse_core.utils.config import config_manager
            config_manager.set_config(cfg)

        from memfuse_core.store.pgai_store import PgaiStore
        from memfuse_core.rag.chunk.base import ChunkData

        store = PgaiStore()
        await store.initialize()

        # Verify manual mode
        auto_embedding = store.pgai_config.get("auto_embedding", False)
        print(f"Auto-embedding disabled: {not auto_embedding}")

        # Warm up the model first (to get realistic performance)
        print("Warming up model...")
        warmup_chunk = ChunkData(
            chunk_id=f"warmup-{int(time.time())}",
            content="Warmup content to load model",
            metadata={"test": "warmup"}
        )
        await store.add([warmup_chunk])
        print("Model warmed up")

        # Test actual insertion (warm performance)
        test_id = f"manual-final-{int(time.time())}"
        test_chunk = ChunkData(
            chunk_id=test_id,
            content="Manual mode final test content",
            metadata={"test": "manual_final"}
        )

        start_time = time.time()
        chunk_ids = await store.add([test_chunk])
        insert_time = time.time() - start_time

        print(f"Warm insertion time: {insert_time:.3f}s")

        # Check result
        state = get_record_state(test_id)
        print(f"Has embedding: {state.get('has_embedding', False)}")
        print(f"Needs embedding: {state.get('needs_embedding', False)}")
        print(f"Dimension: {state.get('dimension', 0)}")

        success = (
            len(chunk_ids) == 1 and
            state.get('has_embedding', False) and
            not state.get('needs_embedding', False) and
            state.get('dimension', 0) == 384
        )

        print(f"✅ Manual mode: {'PASS' if success else 'FAIL'}")
        return success, insert_time

    except Exception as e:
        print(f"❌ Manual mode failed: {e}")
        return False, 0


async def test_auto_mode():
    """Test auto embedding mode"""
    print("\n🔬 Testing Auto Mode")
    print("-" * 40)
    
    try:
        # Load configuration
        from hydra import initialize, compose
        
        with initialize(version_base=None, config_path="../config"):
            cfg = compose(config_name="config")
            cfg.database.pgai.auto_embedding = True
            
            from memfuse_core.utils.config import config_manager
            config_manager.set_config(cfg)
        
        from memfuse_core.store.pgai_store import PgaiStore
        from memfuse_core.rag.chunk.base import ChunkData
        
        store = PgaiStore()
        await store.initialize()
        
        # Verify auto mode
        auto_embedding = store.pgai_config.get("auto_embedding", False)
        print(f"Auto-embedding enabled: {auto_embedding}")
        
        # Test insertion
        test_id = f"auto-final-{int(time.time())}"
        test_chunk = ChunkData(
            chunk_id=test_id,
            content="Auto mode final test content",
            metadata={"test": "auto_final"}
        )
        
        start_time = time.time()
        chunk_ids = await store.add([test_chunk])
        insert_time = time.time() - start_time
        
        print(f"Insertion time: {insert_time:.3f}s")
        
        # Check immediate state
        immediate_state = get_record_state(test_id)
        print(f"Immediate - Has embedding: {immediate_state.get('has_embedding', False)}")
        print(f"Immediate - Needs embedding: {immediate_state.get('needs_embedding', False)}")
        
        # Wait for background processing
        print("Waiting for background processing...")
        max_wait = 30
        background_time = 0
        
        for i in range(max_wait):
            await asyncio.sleep(1)
            current_state = get_record_state(test_id)
            
            if current_state.get('has_embedding', False) and not current_state.get('needs_embedding', False):
                background_time = i + 1
                print(f"Background processing completed in {background_time}s")
                break
        else:
            print(f"⚠️ Background processing did not complete within {max_wait}s")
        
        # Check final state
        final_state = get_record_state(test_id)
        print(f"Final - Has embedding: {final_state.get('has_embedding', False)}")
        print(f"Final - Needs embedding: {final_state.get('needs_embedding', False)}")
        print(f"Final - Dimension: {final_state.get('dimension', 0)}")
        
        success = (
            len(chunk_ids) == 1 and
            insert_time < 0.1 and  # Fast insertion
            not immediate_state.get('has_embedding', False) and  # No immediate embedding
            immediate_state.get('needs_embedding', False) and  # Needs embedding initially
            final_state.get('has_embedding', False) and  # Eventually has embedding
            not final_state.get('needs_embedding', False) and  # No longer needs embedding
            final_state.get('dimension', 0) == 384  # Correct dimension
        )
        
        print(f"✅ Auto mode: {'PASS' if success else 'FAIL'}")
        return success, insert_time, background_time
        
    except Exception as e:
        print(f"❌ Auto mode failed: {e}")
        return False, 0, 0


async def main():
    """Main test function"""
    print("🧪 Auto-Embedding Final Validation Test")
    print("=" * 50)
    
    # Test manual mode
    manual_success, manual_time = await test_manual_mode()
    
    # Test auto mode
    auto_success, auto_time, bg_time = await test_auto_mode()
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 30)
    print(f"Manual mode (warm): {'✅ PASS' if manual_success else '❌ FAIL'} ({manual_time:.3f}s)")
    print(f"Auto mode: {'✅ PASS' if auto_success else '❌ FAIL'} ({auto_time:.3f}s insertion, {bg_time}s background)")

    if manual_success and auto_success:
        print("\n🎉 All tests passed! Auto-embedding functionality is working correctly.")

        # Performance comparison (using warm performance for fair comparison)
        if manual_time > 0 and auto_time > 0:
            speedup = manual_time / auto_time if auto_time > 0 else float('inf')
            print(f"\n📈 Performance Analysis (Warm Performance):")
            print(f"  Manual mode (warm): {manual_time:.3f}s per insertion")
            print(f"  Auto mode insertion: {auto_time:.3f}s (non-blocking)")
            print(f"  Auto mode background: {bg_time}s (asynchronous)")
            print(f"  Insertion speedup: {speedup:.1f}x (auto vs manual warm)")
            print(f"\n💡 Key Benefits:")
            print(f"  - Auto mode: Non-blocking insertion, good for high throughput")
            print(f"  - Manual mode: Immediate embedding availability, good for real-time")

        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
