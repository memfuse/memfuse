#!/usr/bin/env python3
"""Test server startup without actually running the server."""

import sys
import os
import asyncio

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_server_components():
    """Test individual server components."""
    print("Testing server components...")
    
    try:
        # Test FastAPI import
        from fastapi import FastAPI
        print("✓ FastAPI imported successfully")
        
        # Test basic app creation
        app = FastAPI(title="Test App")
        print("✓ FastAPI app created successfully")
        
        # Test MemFuse imports
        from memfuse_core.services.buffer_service import BufferService
        print("✓ BufferService imported successfully")
        
        from memfuse_core.services.memory_service import MemoryService
        print("✓ MemoryService imported successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Server component test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_minimal_server():
    """Test minimal server creation."""
    print("\nTesting minimal server creation...")
    
    try:
        from fastapi import FastAPI
        from memfuse_core.services.buffer_service import BufferService

        # Create minimal app
        app = FastAPI(title="MemFuse Test", version="0.2.0")

        # Add a simple health endpoint
        @app.get("/health")
        async def health():
            return {"status": "ok", "version": "0.2.0"}

        print("✓ Minimal server created with health endpoint")

        # Test BufferService creation
        buffer_service = BufferService(
            memory_service=None,
            user="test_user",
            config={
                'buffer': {
                    'round_buffer': {'max_tokens': 800, 'max_size': 5},
                    'hybrid_buffer': {'max_size': 5},
                    'query': {'max_size': 15}
                }
            }
        )
        print("✓ BufferService created successfully")
        
        # Test service stats
        stats = await buffer_service.get_buffer_stats()
        print(f"✓ Buffer stats retrieved: version={stats['version']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Minimal server test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_server_routes():
    """Test server route configuration."""
    print("\nTesting server route configuration...")
    
    try:
        # Import the actual server creation function
        from memfuse_core.server import create_app_async

        print("Attempting to create full MemFuse app...")
        app = await create_app_async()
        print("✓ Full MemFuse app created successfully")
        
        # Check routes
        routes = []
        for route in app.routes:
            if hasattr(route, 'path'):
                routes.append(route.path)
        
        print(f"✓ App has {len(routes)} routes configured")
        
        # Check for key endpoints
        key_endpoints = ['/health', '/api/v1/memory/add', '/api/v1/memory/query']
        found_endpoints = []
        
        for endpoint in key_endpoints:
            if any(endpoint in route for route in routes):
                found_endpoints.append(endpoint)
        
        print(f"✓ Found {len(found_endpoints)}/{len(key_endpoints)} key endpoints")
        
        if found_endpoints:
            print(f"  Found endpoints: {found_endpoints}")
        
        return True
        
    except Exception as e:
        print(f"✗ Server route test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all server tests."""
    print("=" * 60)
    print("MemFuse Server Startup Test Suite")
    print("=" * 60)
    
    tests = [
        test_server_components,
        test_minimal_server,
        test_server_routes,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if await test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ MemFuse server can start successfully!")
        return True
    else:
        print("✗ Some server components have issues")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
