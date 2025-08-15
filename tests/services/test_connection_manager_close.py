#!/usr/bin/env python3
"""
Test GlobalConnectionManager close functionality.

This test specifically verifies the fix for the "Event loop is closed" error
when shutting down connection pools.
"""

import asyncio
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from memfuse_core.services.global_connection_manager import get_global_connection_manager


async def test_connection_manager_close_fix():
    """Test that the GlobalConnectionManager close fix works correctly."""
    print("🧪 Testing GlobalConnectionManager close fix...")
    
    try:
        # Initialize connection manager
        manager = get_global_connection_manager()
        db_url = "postgresql://postgres:postgres@localhost:5432/memfuse"
        
        print("1. Creating connection pool...")
        pool = await manager.get_connection_pool(db_url)
        print("   ✅ Connection pool created")
        
        print("2. Testing normal close...")
        await manager.close_all_pools(force=True)
        print("   ✅ Normal close successful - no 'Event loop is closed' error!")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_multiple_pools_close():
    """Test closing multiple connection pools."""
    print("\n🧪 Testing multiple connection pools close...")
    
    try:
        manager = get_global_connection_manager()
        
        # Create multiple pools
        db_urls = [
            "postgresql://postgres:postgres@localhost:5432/memfuse",
            "postgresql://postgres:postgres@localhost:5432/memfuse?application_name=test1",
            "postgresql://postgres:postgres@localhost:5432/memfuse?application_name=test2"
        ]
        
        print("1. Creating multiple connection pools...")
        pools = []
        for i, db_url in enumerate(db_urls):
            pool = await manager.get_connection_pool(db_url)
            pools.append(pool)
            print(f"   ✅ Pool {i+1} created")
        
        print("2. Testing close all pools...")
        await manager.close_all_pools(force=True)
        print("   ✅ All pools closed successfully!")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test function."""
    print("🚀 GlobalConnectionManager Close Fix Tests")
    print("=" * 50)
    
    success_count = 0
    total_tests = 2
    
    # Test 1: Basic close fix
    if await test_connection_manager_close_fix():
        success_count += 1
    
    # Test 2: Multiple pools close
    if await test_multiple_pools_close():
        success_count += 1
    
    # Summary
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("🎉 All connection manager close tests passed!")
        return 0
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 Test interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
