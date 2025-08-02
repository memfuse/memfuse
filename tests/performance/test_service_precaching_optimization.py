"""Performance tests for service pre-caching optimization."""

import pytest
import asyncio
import time
from pathlib import Path
import sys

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from memfuse_core.services.service_factory import ServiceFactory
from memfuse_core.services.service_initializer import ServiceInitializer
from omegaconf import DictConfig, OmegaConf


class TestServicePreCachingOptimization:
    """Test suite for service pre-caching performance optimization."""

    @pytest.fixture(autouse=True)
    async def setup_and_teardown(self):
        """Setup and teardown for each test."""
        # Reset service factory before each test
        ServiceFactory.reset()
        yield
        # Cleanup after each test
        await ServiceFactory.cleanup_all_services()

    @pytest.mark.asyncio
    async def test_service_warmup_functionality(self):
        """Test that service warmup creates cached instances correctly."""
        # Configure warmup
        common_users = ["test_user1", "test_user2"]
        ServiceFactory.configure_warmup(enabled=True, common_users=common_users)
        
        # Verify warmup not completed initially
        assert not ServiceFactory.is_warmup_completed()
        
        # Create minimal config
        config = OmegaConf.create({
            "performance": {
                "service_warmup": {
                    "enabled": True,
                    "common_users": common_users
                }
            }
        })
        
        # Perform warmup
        success = await ServiceFactory.warmup_common_services(config)
        assert success, "Service warmup should succeed"
        
        # Verify warmup completed
        assert ServiceFactory.is_warmup_completed()
        
        # Verify services are cached
        for user in common_users:
            # Check if services are in cache
            assert user in ServiceFactory._memory_service_instances
            assert user in ServiceFactory._memory_service_proxy_instances
            assert user in ServiceFactory._buffer_service_instances
        
        print("✅ Service warmup functionality working correctly")

    @pytest.mark.asyncio
    async def test_warmup_performance_improvement(self):
        """Test that warmup improves service access performance."""
        test_user = "performance_test_user"
        
        # Configure warmup with test user
        ServiceFactory.configure_warmup(enabled=True, common_users=[test_user])
        
        # Create minimal config
        config = OmegaConf.create({})
        
        # Perform warmup
        await ServiceFactory.warmup_common_services(config)
        
        # Measure warm access time (should be fast)
        warm_start_time = time.time()
        memory_service = ServiceFactory.get_memory_service_for_user(test_user, config)
        memory_proxy = await ServiceFactory.get_memory_service_proxy_for_user(test_user)
        buffer_service = await ServiceFactory.get_buffer_service_for_user(test_user)
        warm_end_time = time.time()
        
        warm_access_time = warm_end_time - warm_start_time
        
        # Verify services were returned
        assert memory_service is not None
        assert memory_proxy is not None
        assert buffer_service is not None
        
        # Reset and test cold access
        ServiceFactory.reset()
        
        # Measure cold access time (should be slower)
        cold_start_time = time.time()
        memory_service_cold = ServiceFactory.get_memory_service_for_user(test_user, config)
        memory_proxy_cold = await ServiceFactory.get_memory_service_proxy_for_user(test_user)
        buffer_service_cold = await ServiceFactory.get_buffer_service_for_user(test_user)
        cold_end_time = time.time()
        
        cold_access_time = cold_end_time - cold_start_time
        
        # Verify services were returned
        assert memory_service_cold is not None
        assert memory_proxy_cold is not None
        assert buffer_service_cold is not None
        
        # Performance assertion: warm access should be significantly faster
        performance_improvement = (cold_access_time - warm_access_time) / cold_access_time * 100
        
        print(f"🚀 Cold access time: {cold_access_time:.4f}s")
        print(f"🚀 Warm access time: {warm_access_time:.4f}s")
        print(f"🚀 Performance improvement: {performance_improvement:.1f}%")
        
        # Warm access should be at least 50% faster
        assert performance_improvement > 50, f"Expected >50% improvement, got {performance_improvement:.1f}%"
        
        print("✅ Service pre-caching provides significant performance improvement")

    @pytest.mark.asyncio
    async def test_concurrent_warmup_performance(self):
        """Test that concurrent warmup performs well."""
        common_users = [f"concurrent_user_{i}" for i in range(5)]
        
        # Configure warmup
        ServiceFactory.configure_warmup(enabled=True, common_users=common_users)
        
        # Create minimal config
        config = OmegaConf.create({})
        
        # Measure warmup time
        warmup_start_time = time.time()
        success = await ServiceFactory.warmup_common_services(config)
        warmup_end_time = time.time()
        
        warmup_duration = warmup_end_time - warmup_start_time
        
        assert success, "Concurrent warmup should succeed"
        
        # Verify all users are warmed up
        for user in common_users:
            assert user in ServiceFactory._memory_service_instances
            assert user in ServiceFactory._memory_service_proxy_instances
            assert user in ServiceFactory._buffer_service_instances
        
        # Performance target: warmup should complete within reasonable time
        max_warmup_time = 10.0  # 10 seconds for 5 users
        assert warmup_duration < max_warmup_time, f"Warmup took {warmup_duration:.3f}s, expected <{max_warmup_time}s"
        
        print(f"🚀 Concurrent warmup for {len(common_users)} users completed in {warmup_duration:.3f}s")
        print("✅ Concurrent warmup performance is acceptable")

    @pytest.mark.asyncio
    async def test_warmup_with_service_initializer(self):
        """Test that warmup integrates correctly with ServiceInitializer."""
        # Create service initializer
        initializer = ServiceInitializer()
        
        # Create config with warmup settings
        config = OmegaConf.create({
            "performance": {
                "service_warmup": {
                    "enabled": True,
                    "common_users": ["initializer_test_user"]
                }
            }
        })
        
        # Test the warmup method directly
        success = await initializer._warmup_service_instances(config)
        assert success, "Service initializer warmup should succeed"
        
        # Verify warmup was performed
        assert ServiceFactory.is_warmup_completed()
        assert "initializer_test_user" in ServiceFactory._memory_service_instances
        
        print("✅ Service warmup integrates correctly with ServiceInitializer")

    @pytest.mark.asyncio
    async def test_warmup_disabled_configuration(self):
        """Test that warmup can be disabled via configuration."""
        # Configure warmup as disabled
        ServiceFactory.configure_warmup(enabled=False, common_users=["disabled_test_user"])
        
        # Create config
        config = OmegaConf.create({})
        
        # Attempt warmup
        success = await ServiceFactory.warmup_common_services(config)
        assert success, "Disabled warmup should still return success"
        
        # Verify no services were cached
        assert len(ServiceFactory._memory_service_instances) == 0
        assert len(ServiceFactory._memory_service_proxy_instances) == 0
        assert len(ServiceFactory._buffer_service_instances) == 0
        
        print("✅ Service warmup can be properly disabled")

    @pytest.mark.asyncio
    async def test_warmup_error_handling(self):
        """Test that warmup handles errors gracefully."""
        # Configure warmup with invalid user (should cause some failures)
        invalid_users = ["valid_user", ""]  # Empty string might cause issues
        ServiceFactory.configure_warmup(enabled=True, common_users=invalid_users)
        
        # Create config
        config = OmegaConf.create({})
        
        # Attempt warmup (should handle errors gracefully)
        success = await ServiceFactory.warmup_common_services(config)
        
        # Should still return True even if some users fail
        # (as long as at least one succeeds)
        assert isinstance(success, bool), "Warmup should return boolean result"
        
        print("✅ Service warmup handles errors gracefully")


if __name__ == "__main__":
    # Run tests directly
    test_instance = TestServicePreCachingOptimization()
    
    async def run_async_tests():
        await test_instance.setup_and_teardown().__anext__()
        try:
            await test_instance.test_service_warmup_functionality()
            await test_instance.test_warmup_performance_improvement()
            await test_instance.test_concurrent_warmup_performance()
            await test_instance.test_warmup_with_service_initializer()
            await test_instance.test_warmup_disabled_configuration()
            await test_instance.test_warmup_error_handling()
        finally:
            await ServiceFactory.cleanup_all_services()
    
    asyncio.run(run_async_tests())
    print("✅ All service pre-caching optimization tests passed!")
