#!/usr/bin/env python3
"""
Configuration validation test for immediate trigger functionality.

This test validates the configuration logic without requiring actual pgai dependencies.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

def test_store_type_selection():
    """Test store type selection logic."""
    print("🧪 Testing Store Type Selection Logic")
    print("=" * 50)
    
    try:
        from memfuse_core.store.store_factory import PgaiStoreFactory
        
        # Test 1: Event-driven configuration
        config_event_driven = {
            "pgai": {
                "auto_embedding": True,
                "immediate_trigger": True
            }
        }
        
        store_type = PgaiStoreFactory.get_store_type(config_event_driven)
        assert store_type == "event_driven", f"Expected 'event_driven', got '{store_type}'"
        print("✅ Event-driven configuration detected correctly")
        
        # Test 2: Traditional configuration
        config_traditional = {
            "pgai": {
                "auto_embedding": True,
                "immediate_trigger": False
            }
        }
        
        store_type = PgaiStoreFactory.get_store_type(config_traditional)
        assert store_type == "traditional", f"Expected 'traditional', got '{store_type}'"
        print("✅ Traditional configuration detected correctly")
        
        # Test 3: Auto-embedding disabled
        config_disabled = {
            "pgai": {
                "auto_embedding": False,
                "immediate_trigger": True
            }
        }
        
        store_type = PgaiStoreFactory.get_store_type(config_disabled)
        assert store_type == "traditional", f"Expected 'traditional', got '{store_type}'"
        print("✅ Disabled auto-embedding handled correctly")
        
        # Test 4: Empty configuration
        config_empty = {}
        
        store_type = PgaiStoreFactory.get_store_type(config_empty)
        assert store_type == "traditional", f"Expected 'traditional', got '{store_type}'"
        print("✅ Empty configuration handled correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Store type selection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_configuration_validation():
    """Test configuration validation and normalization."""
    print("\n🧪 Testing Configuration Validation")
    print("=" * 50)
    
    try:
        from memfuse_core.store.store_factory import PgaiStoreFactory
        
        # Test invalid configuration
        invalid_config = {
            "pgai": {
                "worker_count": "invalid",  # Should be int
                "queue_size": -1,          # Should be positive
                "max_retries": "abc",      # Should be int
                "retry_interval": -5.0     # Should be non-negative
            }
        }
        
        validated_config = PgaiStoreFactory.validate_configuration(invalid_config)
        pgai_config = validated_config["pgai"]
        
        # Check corrections
        assert isinstance(pgai_config["worker_count"], int), "worker_count should be corrected to int"
        assert pgai_config["worker_count"] > 0, "worker_count should be positive"
        
        assert isinstance(pgai_config["queue_size"], int), "queue_size should be corrected to int"
        assert pgai_config["queue_size"] > 0, "queue_size should be positive"
        
        assert isinstance(pgai_config["max_retries"], int), "max_retries should be corrected to int"
        assert pgai_config["max_retries"] >= 0, "max_retries should be non-negative"
        
        assert isinstance(pgai_config["retry_interval"], (int, float)), "retry_interval should be numeric"
        assert pgai_config["retry_interval"] >= 0, "retry_interval should be non-negative"
        
        print("✅ Invalid configuration corrected successfully")
        print(f"   - worker_count: {pgai_config['worker_count']}")
        print(f"   - queue_size: {pgai_config['queue_size']}")
        print(f"   - max_retries: {pgai_config['max_retries']}")
        print(f"   - retry_interval: {pgai_config['retry_interval']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backward_compatibility():
    """Test backward compatibility migration."""
    print("\n🧪 Testing Backward Compatibility")
    print("=" * 50)
    
    try:
        from memfuse_core.store.store_factory import BackwardCompatibilityManager
        
        # Test legacy configuration migration
        legacy_config = {
            "pgai": {
                "retry_delay": 10.0,                    # Legacy key
                "vectorizer_worker_enabled": True       # Legacy key
            }
        }
        
        migrated_config = BackwardCompatibilityManager.migrate_legacy_config(legacy_config)
        pgai_config = migrated_config["pgai"]
        
        # Check migrations
        assert "retry_interval" in pgai_config, "retry_delay should be migrated to retry_interval"
        assert pgai_config["retry_interval"] == 10.0, "retry_interval value should be preserved"
        
        assert "auto_embedding" in pgai_config, "vectorizer_worker_enabled should be migrated to auto_embedding"
        assert pgai_config["auto_embedding"] is True, "auto_embedding value should be preserved"
        
        print("✅ Legacy configuration migrated successfully")
        print(f"   - retry_delay -> retry_interval: {pgai_config['retry_interval']}")
        print(f"   - vectorizer_worker_enabled -> auto_embedding: {pgai_config['auto_embedding']}")
        
        # Test compatibility check
        test_config = {
            "pgai": {
                "immediate_trigger": True,
                "auto_embedding": False  # Conflicting setting
            }
        }
        
        report = BackwardCompatibilityManager.check_compatibility(test_config)
        assert not report["compatible"], "Should detect incompatible configuration"
        assert len(report["warnings"]) > 0, "Should have warnings for incompatible settings"
        
        print("✅ Compatibility checking works correctly")
        print(f"   - Detected {len(report['warnings'])} warnings")
        
        return True
        
    except Exception as e:
        print(f"❌ Backward compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_monitoring_components():
    """Test monitoring components."""
    print("\n🧪 Testing Monitoring Components")
    print("=" * 50)
    
    try:
        from memfuse_core.store.monitoring import EmbeddingMonitor, HealthChecker
        
        # Test EmbeddingMonitor
        monitor = EmbeddingMonitor("test_store")
        
        # Test basic functionality
        monitor.start_processing("test_record_1", "worker-1", 0)
        assert "test_record_1" in monitor.active_processing, "Should track active processing"
        
        monitor.complete_processing("test_record_1", True, None)
        assert "test_record_1" not in monitor.active_processing, "Should remove from active processing"
        assert monitor.metrics["success_count"] == 1, "Should increment success count"
        
        # Test failure case
        monitor.start_processing("test_record_2", "worker-1", 1)
        monitor.complete_processing("test_record_2", False, "Test error")
        assert monitor.metrics["failure_count"] == 1, "Should increment failure count"
        assert "Test error" in monitor.error_patterns, "Should track error patterns"
        
        print("✅ EmbeddingMonitor basic functionality works")

        return True

    except Exception as e:
        print(f"❌ Monitoring components test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all configuration validation tests."""
    print("🚀 MemFuse Immediate Trigger Configuration Validation")
    print("=" * 80)
    
    tests = [
        ("Store Type Selection", test_store_type_selection),
        ("Configuration Validation", test_configuration_validation),
        ("Backward Compatibility", test_backward_compatibility),
        ("Monitoring Components", test_monitoring_components),
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 Test Results Summary:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL CONFIGURATION TESTS PASSED!")
        print("✅ Immediate trigger configuration system is working correctly")
    else:
        print("⚠️  SOME CONFIGURATION TESTS FAILED!")
        print("❌ Please review the failed tests above")
    
    print("\n💡 Next Steps:")
    print("   1. Start MemFuse server: poetry run memfuse-core")
    print("   2. Test with actual database: python tests/store/test_immediate_trigger_validation.py")
    print("   3. Run full test suite: pytest tests/store/ -v")


if __name__ == "__main__":
    main()
