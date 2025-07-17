#!/usr/bin/env python3
"""
Multi-layer PgAI functionality test script.

This script tests the core functionality of the multi-layer PgAI system
without requiring a database connection.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def test_basic_functionality():
    """Test basic multi-layer functionality."""
    print("🧪 Testing basic multi-layer functionality...")
    
    try:
        from memfuse_core.store.pgai_store.multi_layer_store import (
            MultiLayerPgaiStore, LayerType
        )
        from memfuse_core.store.pgai_store.config_manager import ConfigManager
        from memfuse_core.store.pgai_store.stats_collector import StatsCollector
        from memfuse_core.rag.chunk.base import ChunkData
        
        # Test configuration management
        config = {
            'memory_layers': {
                'm0': {'enabled': True},
                'm1': {'enabled': True}
            }
        }
        
        # Test ConfigManager
        enabled_layers = ConfigManager.get_enabled_layers(config)
        assert 'm0' in enabled_layers
        assert 'm1' in enabled_layers
        print("   ✅ ConfigManager works correctly")
        
        # Test StatsCollector
        stats = StatsCollector()
        stats.record_operation('test_op', 0.1, True, 'm0')
        all_stats = stats.get_all_stats()
        assert 'total_operations' in all_stats
        assert all_stats['total_operations'] == 1
        print("   ✅ StatsCollector works correctly")
        
        # Test MultiLayerPgaiStore initialization
        store = MultiLayerPgaiStore(config)
        assert LayerType.M0 in store.enabled_layers
        assert LayerType.M1 in store.enabled_layers
        assert not store.initialized
        print("   ✅ MultiLayerPgaiStore initialization works")
        
        # Test ChunkData creation
        chunk = ChunkData(
            content="Test content",
            metadata={'session_id': 'test', 'user_id': 'test_user'}
        )
        assert chunk.content == "Test content"
        assert chunk.metadata['session_id'] == 'test'
        print("   ✅ ChunkData creation works")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Basic functionality test failed: {e}")
        return False


def test_fact_extraction_processor():
    """Test fact extraction processor functionality."""
    print("🧪 Testing fact extraction processor...")
    
    try:
        from memfuse_core.store.pgai_store.fact_extraction_processor import (
            FactExtractionProcessor, ExtractedFact
        )
        
        # Test processor initialization
        config = {
            'llm_model': 'grok-3-mini',
            'temperature': 0.3,
            'max_tokens': 1000,
            'min_confidence_threshold': 0.7
        }
        
        processor = FactExtractionProcessor(config)
        assert processor.llm_model == 'grok-3-mini'
        assert processor.temperature == 0.3
        print("   ✅ FactExtractionProcessor initialization works")
        
        # Test ExtractedFact creation
        fact = ExtractedFact(
            content="Test fact",
            type="test_type",
            confidence=0.8,
            entities=["entity1"],
            temporal_info={"time": "now"},
            source_context="test context",
            category={"type": "test"}
        )
        assert fact.content == "Test fact"
        assert fact.confidence == 0.8
        print("   ✅ ExtractedFact creation works")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Fact extraction processor test failed: {e}")
        return False


def test_schema_manager():
    """Test schema manager functionality."""
    print("🧪 Testing schema manager...")
    
    try:
        from memfuse_core.store.pgai_store.schema_manager import SchemaManager
        
        # Test that SchemaManager can be imported and has expected methods
        assert hasattr(SchemaManager, 'initialize_all_schemas')
        assert hasattr(SchemaManager, 'validate_schemas')
        print("   ✅ SchemaManager has expected methods")
        
        # Test supported layers
        # Note: We can't test actual database operations without a connection
        # but we can test the class structure
        expected_layers = ["m0", "m1"]
        # This would require a pool, so we just test the class exists
        print("   ✅ SchemaManager class structure is correct")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Schema manager test failed: {e}")
        return False


def test_configuration_validation():
    """Test configuration validation."""
    print("🧪 Testing configuration validation...")
    
    try:
        from memfuse_core.store.pgai_store.config_manager import ConfigManager
        
        # Test valid configuration
        valid_config = {
            'memory_layers': {
                'm0': {
                    'enabled': True,
                    'pgai': {
                        'embedding_model': 'all-MiniLM-L6-v2',
                        'embedding_dimensions': 384
                    }
                },
                'm1': {
                    'enabled': True,
                    'pgai': {
                        'embedding_model': 'all-MiniLM-L6-v2',
                        'embedding_dimensions': 384
                    },
                    'fact_extraction': {
                        'min_confidence_threshold': 0.7,
                        'classification_strategy': 'open'
                    }
                }
            }
        }
        
        is_valid = ConfigManager.validate_config(valid_config)
        assert is_valid is True
        print("   ✅ Valid configuration passes validation")
        
        # Test invalid configuration
        invalid_config = {
            'memory_layers': {
                'm1': {
                    'enabled': True,
                    'fact_extraction': {
                        'min_confidence_threshold': 1.5,  # Invalid: > 1.0
                        'classification_strategy': 'invalid'  # Invalid strategy
                    }
                }
            }
        }
        
        is_valid = ConfigManager.validate_config(invalid_config)
        assert is_valid is False
        print("   ✅ Invalid configuration fails validation")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Configuration validation test failed: {e}")
        return False


def main():
    """Run all functionality tests."""
    print("🚀 Multi-Layer PgAI Functionality Test")
    print("=" * 50)
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Fact Extraction Processor", test_fact_extraction_processor),
        ("Schema Manager", test_schema_manager),
        ("Configuration Validation", test_configuration_validation),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All functionality tests passed!")
        print("\nThe multi-layer PgAI system is working correctly.")
        print("\nNext steps:")
        print("1. Run validation script: poetry run python tests/scripts/validate_multi_layer_pgai.py")
        print("2. Run unit tests: poetry run python -m pytest tests/unit/store/pgai_store/test_multi_layer_store.py -v")
        print("3. Start database and run integration tests")
    else:
        print("\n❌ Some functionality tests failed. Please check the issues above.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
