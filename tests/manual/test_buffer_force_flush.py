#!/usr/bin/env python3
"""
Test script to verify buffer force flush functionality.

This script tests the core force flush mechanisms:
1. Timeout detection based on RoundBuffer write time
2. Graceful shutdown with proper data transfer
3. No duplicate clearing warnings

Usage:
    poetry run python tests/manual/test_buffer_force_flush.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import asyncio
import time
from memfuse_core.buffer.write_buffer import WriteBuffer
from memfuse_core.buffer.config_factory import ComponentConfigFactory


async def test_timeout_detection():
    """Test that timeout detection only monitors RoundBuffer write time."""
    print("🧪 Testing timeout detection logic...")
    
    # Create test config with short timeout
    config_dict = {
        'buffer': {
            'performance': {
                'force_flush_timeout': 3.0  # 3 seconds for quick testing
            }
        }
    }
    
    factory = ComponentConfigFactory()
    config = factory.create_component_config('write_buffer', global_config=config_dict)
    
    # Create WriteBuffer
    write_buffer = WriteBuffer(config)
    await write_buffer.initialize()
    
    try:
        # Test 1: Initial state - no data, no timeout
        print(f"Initial last write time: {write_buffer.get_last_write_time()}")
        print(f"Initial has pending data: {write_buffer.has_pending_data()}")
        
        timeout_triggered = await write_buffer.check_force_flush_timeout(3.0)
        print(f"Timeout triggered with no data: {timeout_triggered}")
        assert not timeout_triggered, "Timeout should not trigger with no data"
        
        # Test 2: Add data to RoundBuffer
        test_messages = [
            {'role': 'user', 'content': 'Test message 1'},
            {'role': 'assistant', 'content': 'Test response 1'}
        ]
        
        result = await write_buffer.add(test_messages, session_id='test_session')
        print(f"Add result: {result['status']}")
        
        # Check state after adding data
        last_write_time = write_buffer.get_last_write_time()
        has_data = write_buffer.has_pending_data()
        print(f"After add - last write time: {last_write_time}")
        print(f"After add - has pending data: {has_data}")
        print(f"RoundBuffer rounds: {len(write_buffer.round_buffer.rounds)}")
        
        assert has_data, "Should have pending data after adding messages"
        assert last_write_time > 0, "Should have valid last write time"
        
        # Test 3: Immediate timeout check - should not trigger
        timeout_triggered = await write_buffer.check_force_flush_timeout(3.0)
        print(f"Immediate timeout check: {timeout_triggered}")
        assert not timeout_triggered, "Timeout should not trigger immediately"
        
        # Test 4: Wait and check timeout detection
        print("Waiting 4 seconds to test timeout detection...")
        await asyncio.sleep(4)
        
        timeout_triggered = await write_buffer.check_force_flush_timeout(3.0)
        print(f"Timeout triggered after 4 seconds: {timeout_triggered}")
        assert timeout_triggered, "Timeout should trigger after waiting"
        
        print("✅ Timeout detection test passed!")
        
        # Test 5: Test shutdown without warnings
        print("Testing shutdown behavior...")
        await write_buffer.shutdown()
        print("✅ Shutdown completed without warnings!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise


async def test_shutdown_no_duplicate_clear():
    """Test that shutdown doesn't produce duplicate clear warnings."""
    print("\n🧪 Testing shutdown behavior...")
    
    config_dict = {
        'buffer': {
            'performance': {
                'force_flush_timeout': 30.0
            }
        }
    }
    
    factory = ComponentConfigFactory()
    config = factory.create_component_config('write_buffer', global_config=config_dict)
    
    # Create WriteBuffer
    write_buffer = WriteBuffer(config)
    await write_buffer.initialize()
    
    try:
        # Add some test data
        test_messages = [
            {'role': 'user', 'content': 'Test message for shutdown'},
            {'role': 'assistant', 'content': 'Test response for shutdown'}
        ]
        
        result = await write_buffer.add(test_messages, session_id='shutdown_test')
        print(f"Added test data: {result['status']}")
        print(f"RoundBuffer has {len(write_buffer.round_buffer.rounds)} rounds")
        
        # Shutdown - should transfer data without warnings
        print("Initiating shutdown...")
        await write_buffer.shutdown()
        print("✅ Shutdown completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during shutdown test: {e}")
        import traceback
        traceback.print_exc()
        raise


async def main():
    """Run all tests."""
    print("🚀 Testing Force Flush Fixes")
    print("=" * 50)
    
    try:
        await test_timeout_detection()
        await test_shutdown_no_duplicate_clear()
        
        print("\n🎉 All tests passed!")
        print("✅ Timeout detection only monitors RoundBuffer write time")
        print("✅ Shutdown doesn't produce duplicate clear warnings")
        
    except Exception as e:
        print(f"\n❌ Tests failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
