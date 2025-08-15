#!/usr/bin/env python3
"""
Test service shutdown functionality.

This test verifies that all MemFuse services can be shut down gracefully
without event loop errors or connection pool issues.
"""

import asyncio
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from memfuse_core.services.global_connection_manager import get_global_connection_manager
from memfuse_core.services.sync_connection_pool import sync_connection_pool


async def test_global_connection_manager_shutdown():
    """Test GlobalConnectionManager shutdown without event loop errors."""
    print("🧪 Testing GlobalConnectionManager shutdown...")
    
    try:
        # Initialize connection manager
        manager = get_global_connection_manager()
        db_url = "postgresql://postgres:postgres@localhost:5432/memfuse"
        
        print("1. Creating connection pool...")
        pool = await manager.get_connection_pool(db_url)
        print("   ✅ Connection pool created")
        
        print("2. Testing graceful shutdown...")
        await manager.close_all_pools(force=True)
        print("   ✅ Shutdown successful - no 'Event loop is closed' error!")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_sync_connection_pool_shutdown():
    """Test SyncConnectionPool shutdown."""
    print("\n🧪 Testing SyncConnectionPool shutdown...")
    
    try:
        # Initialize sync connection pool
        db_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'memfuse',
            'user': 'postgres',
            'password': 'postgres'
        }
        
        print("1. Initializing sync connection pool...")
        sync_connection_pool.initialize(db_config)
        print("   ✅ Sync connection pool initialized")
        
        print("2. Testing graceful shutdown...")
        sync_connection_pool.close()
        print("   ✅ Sync connection pool shutdown successful!")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test function."""
    print("🚀 MemFuse Service Shutdown Tests")
    print("=" * 50)
    
    success_count = 0
    total_tests = 2
    
    # Test 1: GlobalConnectionManager shutdown
    if await test_global_connection_manager_shutdown():
        success_count += 1
    
    # Test 2: SyncConnectionPool shutdown
    if await test_sync_connection_pool_shutdown():
        success_count += 1
    
    # Summary
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("🎉 All service shutdown tests passed!")
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
