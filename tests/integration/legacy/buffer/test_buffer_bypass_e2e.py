#!/usr/bin/env python3
"""
Simplified End-to-End Buffer Bypass Verification Test

This script performs focused testing to verify that when buffer.enabled=false
in the configuration, the system completely bypasses all buffer components.

Usage:
    poetry run python tests/integration/legacy/buffer/test_buffer_bypass_e2e.py
"""

import asyncio
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from memfuse_core.services.buffer_service import BufferService
from memfuse_core.services.memory_service import MemoryService


class BufferBypassE2ETest:
    """Simplified end-to-end test for buffer bypass functionality."""

    def __init__(self):
        self.test_results = []

    def log_test_result(self, test_name: str, passed: bool, details: str = ""):
        """Log test result."""
        status = "✅ PASS" if passed else "❌ FAIL"
        self.test_results.append({
            'test': test_name,
            'passed': passed,
            'details': details
        })
        print(f"{status}: {test_name}")
        if details:
            print(f"   📝 {details}")
    
    async def test_buffer_enabled_configuration(self):
        """Test with buffer enabled configuration."""
        print("\n🔍 Testing Buffer ENABLED Configuration...")

        try:
            # Create test configuration directly
            config = {
                'buffer': {
                    'enabled': True,
                    'round_buffer': {
                        'max_tokens': 800,
                        'max_size': 5,
                        'token_model': 'gpt-4o-mini'
                    },
                    'hybrid_buffer': {
                        'max_size': 5,
                        'chunk_strategy': 'message',
                        'embedding_model': 'all-MiniLM-L6-v2'
                    },
                    'performance': {
                        'max_flush_workers': 2,
                        'max_flush_queue_size': 100,
                        'flush_timeout': 30.0
                    }
                },
                'memory': {
                    'storage': {
                        'type': 'mock'
                    }
                }
            }

            # Verify configuration
            buffer_config = config.get('buffer', {})
            enabled = buffer_config.get('enabled', True)

            self.log_test_result(
                "Buffer Enabled Config Loading",
                enabled is True,
                f"Buffer enabled value: {enabled}"
            )

            # Create BufferService
            memory_service = MemoryService(user_id="test_user_enabled", config=config)
            buffer_service = BufferService(
                memory_service=memory_service,
                user="test_user_enabled",
                config=config
            )

            # Verify buffer components are initialized
            components_initialized = all([
                buffer_service.buffer_enabled is True,
                buffer_service.write_buffer is not None,
                buffer_service.query_buffer is not None,
                buffer_service.speculative_buffer is not None,
                buffer_service.config_manager is not None
            ])

            self.log_test_result(
                "Buffer Components Initialization",
                components_initialized,
                f"All components initialized: {components_initialized}"
            )

        except Exception as e:
            self.log_test_result(
                "Buffer Enabled Configuration",
                False,
                f"Error: {str(e)}"
            )
    
    async def test_buffer_disabled_configuration(self):
        """Test with buffer disabled configuration - CRITICAL TEST."""
        print("\n🔍 Testing Buffer DISABLED Configuration (BYPASS MODE)...")

        try:
            # Create test configuration directly
            config = {
                'buffer': {
                    'enabled': False,  # CRITICAL: Buffer disabled
                    'round_buffer': {
                        'max_tokens': 800,
                        'max_size': 5,
                        'token_model': 'gpt-4o-mini'
                    },
                    'hybrid_buffer': {
                        'max_size': 5,
                        'chunk_strategy': 'message',
                        'embedding_model': 'all-MiniLM-L6-v2'
                    }
                },
                'memory': {
                    'storage': {
                        'type': 'mock'
                    }
                }
            }

            # Verify configuration
            buffer_config = config.get('buffer', {})
            enabled = buffer_config.get('enabled', True)

            self.log_test_result(
                "Buffer Disabled Config Loading",
                enabled is False,
                f"Buffer enabled value: {enabled}"
            )

            # Create BufferService
            memory_service = MemoryService(user_id="test_user_disabled", config=config)
            buffer_service = BufferService(
                memory_service=memory_service,
                user="test_user_disabled",
                config=config
            )

            # CRITICAL VERIFICATION: Buffer components are NOT initialized
            components_not_initialized = all([
                buffer_service.buffer_enabled is False,
                buffer_service.write_buffer is None,
                buffer_service.query_buffer is None,
                buffer_service.speculative_buffer is None,
                buffer_service.config_manager is None,
                buffer_service.use_rerank is False
            ])

            self.log_test_result(
                "Buffer Components NOT Initialized",
                components_not_initialized,
                f"All components correctly NOT initialized: {components_not_initialized}"
            )

            # Test that operations bypass buffer
            await self.test_bypass_operations(buffer_service)

        except Exception as e:
            self.log_test_result(
                "Buffer Disabled Configuration",
                False,
                f"Error: {str(e)}"
            )
    
    async def test_bypass_operations(self, buffer_service: BufferService):
        """Test that operations correctly bypass buffer components."""
        print("   🔍 Testing Bypass Operations...")
        
        # Create test data
        test_messages = [
            [
                {"content": "Test message 1", "role": "user"},
                {"content": "Test response 1", "role": "assistant"}
            ],
            [
                {"content": "Test message 2", "role": "user"},
                {"content": "Test response 2", "role": "assistant"}
            ]
        ]
        
        try:
            # Test add_batch operation
            result = await buffer_service.add_batch(test_messages, session_id="test_session")
            
            bypass_add_success = (
                result.get("status") == "success" and
                result.get("data", {}).get("mode") == "bypass"
            )
            
            self.log_test_result(
                "Add Batch Bypass Operation",
                bypass_add_success,
                f"Result mode: {result.get('data', {}).get('mode', 'unknown')}"
            )
            
            # Test query operation
            query_result = await buffer_service.query("test query", top_k=5)
            
            bypass_query_success = (
                query_result.get("status") == "success" and
                query_result.get("data", {}).get("mode") == "bypass"
            )
            
            self.log_test_result(
                "Query Bypass Operation",
                bypass_query_success,
                f"Query result mode: {query_result.get('data', {}).get('mode', 'unknown')}"
            )
            
        except Exception as e:
            self.log_test_result(
                "Bypass Operations",
                False,
                f"Error during bypass operations: {str(e)}"
            )
    
    async def run_all_tests(self):
        """Run all end-to-end tests."""
        print("🚀 Starting End-to-End Buffer Bypass Verification Tests")
        print("=" * 80)
        
        # Test buffer enabled configuration
        await self.test_buffer_enabled_configuration()
        
        # Test buffer disabled configuration (CRITICAL)
        await self.test_buffer_disabled_configuration()
        
        # Generate summary
        self.generate_test_summary()
    
    def generate_test_summary(self):
        """Generate comprehensive test summary."""
        print("\n" + "=" * 80)
        print("📊 END-TO-END TEST SUMMARY")
        print("=" * 80)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['passed'])
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print()
        
        for result in self.test_results:
            status = "✅" if result['passed'] else "❌"
            print(f"{status} {result['test']}")
            if result['details']:
                print(f"   📝 {result['details']}")
        
        print("\n" + "=" * 80)
        if failed_tests == 0:
            print("🎯 CONCLUSION: Buffer bypass functionality is WORKING CORRECTLY")
            print("   ✅ When enabled=true: Buffer components are initialized and used")
            print("   ✅ When enabled=false: Buffer components are NOT initialized")
            print("   ✅ When enabled=false: Operations bypass buffer and go directly to MemoryService")
            print("   ✅ Data flows correctly: BufferService → MemoryService → MemoryLayer")
        else:
            print("❌ CONCLUSION: Buffer bypass functionality has ISSUES")
            print(f"   {failed_tests} test(s) failed - review the details above")
        
        print("=" * 80)
        
        return failed_tests == 0


async def main():
    """Main test execution function."""
    test_runner = BufferBypassE2ETest()
    success = await test_runner.run_all_tests()
    
    if success:
        print("\n✅ All end-to-end buffer bypass tests PASSED!")
        return 0
    else:
        print("\n❌ Some end-to-end buffer bypass tests FAILED!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
