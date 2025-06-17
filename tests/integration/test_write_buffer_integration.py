"""Integration test for WriteBuffer implementation."""

import asyncio
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

async def test_write_buffer_basic_functionality():
    """Test basic WriteBuffer functionality without external dependencies."""
    
    print("🧪 Testing WriteBuffer Integration...")
    
    try:
        # Test 1: Import WriteBuffer
        print("1. Testing WriteBuffer import...")
        from memfuse_core.buffer.write_buffer import WriteBuffer
        print("   ✅ WriteBuffer imported successfully")
        
        # Test 2: Create WriteBuffer instance
        print("2. Testing WriteBuffer initialization...")
        config = {
            'round_buffer': {
                'max_tokens': 100,
                'max_size': 2,
                'token_model': 'gpt-4o-mini'
            },
            'hybrid_buffer': {
                'max_size': 2,
                'chunk_strategy': 'message',
                'embedding_model': 'all-MiniLM-L6-v2'
            }
        }
        
        write_buffer = WriteBuffer(config=config)
        print("   ✅ WriteBuffer initialized successfully")
        
        # Test 3: Check component access
        print("3. Testing component access...")
        round_buffer = write_buffer.get_round_buffer()
        hybrid_buffer = write_buffer.get_hybrid_buffer()
        
        assert round_buffer is not None, "RoundBuffer should not be None"
        assert hybrid_buffer is not None, "HybridBuffer should not be None"
        print("   ✅ Component access working correctly")
        
        # Test 4: Check initial state
        print("4. Testing initial state...")
        assert write_buffer.total_writes == 0, "Initial writes should be 0"
        assert write_buffer.total_transfers == 0, "Initial transfers should be 0"
        assert write_buffer.is_empty() == True, "Buffer should be empty initially"
        print("   ✅ Initial state correct")
        
        # Test 5: Test statistics
        print("5. Testing statistics...")
        stats = write_buffer.get_stats()
        assert 'write_buffer' in stats, "Stats should contain write_buffer"
        assert stats['write_buffer']['total_writes'] == 0, "Stats should show 0 writes"
        print("   ✅ Statistics working correctly")
        
        print("\n🎉 All WriteBuffer integration tests passed!")
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_buffer_service_integration():
    """Test BufferService integration with WriteBuffer."""
    
    print("\n🧪 Testing BufferService Integration...")
    
    try:
        # Test 1: Import BufferService
        print("1. Testing BufferService import...")
        from memfuse_core.services.buffer_service import BufferService
        print("   ✅ BufferService imported successfully")
        
        # Test 2: Check if BufferService uses WriteBuffer
        print("2. Testing BufferService WriteBuffer integration...")
        
        # Create a mock memory service
        class MockMemoryService:
            def __init__(self):
                self._user_id = "test_user"
        
        mock_memory_service = MockMemoryService()
        config = {
            'buffer': {
                'round_buffer': {
                    'max_tokens': 100,
                    'max_size': 2,
                    'token_model': 'gpt-4o-mini'
                },
                'hybrid_buffer': {
                    'max_size': 2,
                    'chunk_strategy': 'message',
                    'embedding_model': 'all-MiniLM-L6-v2'
                },
                'query': {
                    'max_size': 15,
                    'cache_size': 100,
                    'default_sort_by': 'score',
                    'default_order': 'desc'
                }
            },
            'retrieval': {
                'use_rerank': True
            }
        }
        
        buffer_service = BufferService(
            memory_service=mock_memory_service,
            user="test_user",
            config=config
        )
        print("   ✅ BufferService initialized with WriteBuffer")
        
        # Test 3: Check WriteBuffer integration
        print("3. Testing WriteBuffer access in BufferService...")
        assert hasattr(buffer_service, 'write_buffer'), "BufferService should have write_buffer"
        assert buffer_service.write_buffer is not None, "write_buffer should not be None"
        print("   ✅ WriteBuffer properly integrated in BufferService")
        
        # Test 4: Check component access through WriteBuffer
        print("4. Testing component access through WriteBuffer...")
        round_buffer = buffer_service.write_buffer.get_round_buffer()
        hybrid_buffer = buffer_service.write_buffer.get_hybrid_buffer()
        
        assert round_buffer is not None, "RoundBuffer should be accessible"
        assert hybrid_buffer is not None, "HybridBuffer should be accessible"
        print("   ✅ Components accessible through WriteBuffer")
        
        print("\n🎉 All BufferService integration tests passed!")
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all integration tests."""
    print("🚀 Starting WriteBuffer Integration Tests\n")
    
    test1_result = await test_write_buffer_basic_functionality()
    test2_result = await test_buffer_service_integration()
    
    if test1_result and test2_result:
        print("\n✅ All integration tests passed successfully!")
        print("🎯 WriteBuffer implementation is ready for production use.")
        return 0
    else:
        print("\n❌ Some integration tests failed.")
        print("🔧 Please review the errors above and fix the issues.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
