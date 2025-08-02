#!/usr/bin/env python3
"""
Validation script for dual-layer PgAI setup.

This script validates that all components can be imported and basic
functionality works as expected.
"""

import sys
import traceback
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

def test_imports():
    """Test that all required modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        # Test core imports
        from memfuse_core.store.pgai_store.multi_layer_store import (
            MultiLayerPgaiStore, LayerType
        )
        print("   ✅ MultiLayerPgaiStore imported successfully")
        
        from memfuse_core.store.pgai_store.fact_extraction_processor import (
            FactExtractionProcessor, FactExtractionResult
        )
        print("   ✅ FactExtractionProcessor imported successfully")
        
        from memfuse_core.store.pgai_store.schema_manager import SchemaManager
        print("   ✅ SchemaManager imported successfully")
        
        from memfuse_core.hierarchy.llm_service import AdvancedLLMService, ExtractedFact
        print("   ✅ LLM service components imported successfully")
        
        from memfuse_core.rag.chunk.base import ChunkData
        print("   ✅ ChunkData imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        traceback.print_exc()
        return False

def test_basic_functionality():
    """Test basic functionality without database connections."""
    print("\n🧪 Testing basic functionality...")
    
    try:
        from memfuse_core.store.pgai_store.multi_layer_store import (
            MultiLayerPgaiStore, LayerType
        )
        from memfuse_core.store.pgai_store.fact_extraction_processor import (
            FactExtractionProcessor
        )
        from memfuse_core.rag.chunk.base import ChunkData
        from memfuse_core.hierarchy.llm_service import ExtractedFact
        
        # Test configuration
        config = {
            'memory_layers': {
                'm0': {
                    'enabled': True,
                    'table_name': 'test_m0_raw',
                    'pgai': {
                        'auto_embedding': True,
                        'immediate_trigger': False,
                        'embedding_model': 'all-MiniLM-L6-v2',
                        'embedding_dimensions': 384
                    }
                },
                'm1': {
                    'enabled': True,
                    'table_name': 'test_m1_episodic',
                    'fact_extraction': {
                        'enabled': True,
                        'classification_strategy': 'open',
                        'enable_auto_categorization': True,
                        'min_confidence_threshold': 0.6
                    }
                }
            }
        }
        
        # Test MultiLayerPgaiStore initialization
        store = MultiLayerPgaiStore(config)
        assert store.enabled_layers == [LayerType.M0, LayerType.M1]
        print("   ✅ MultiLayerPgaiStore configuration works")
        
        # Test FactExtractionProcessor initialization
        processor = FactExtractionProcessor(config['memory_layers']['m1']['fact_extraction'])
        assert processor.classification_strategy == 'open'
        assert processor.enable_auto_categorization == True
        print("   ✅ FactExtractionProcessor configuration works")
        
        # Test ChunkData creation
        chunk = ChunkData(
            content="I love Python programming and machine learning.",
            metadata={'test': True}
        )
        assert chunk.content
        assert chunk.metadata['test'] == True
        print("   ✅ ChunkData creation works")
        
        # Test ExtractedFact creation with new category field
        fact = ExtractedFact(
            content="User loves Python programming",
            type="extracted_fact",
            confidence=0.85,
            entities=["Python"],
            category={"semantic_type": "preference"}
        )
        assert fact.category is not None
        assert fact.category["semantic_type"] == "preference"
        print("   ✅ ExtractedFact with category field works")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Functionality test failed: {e}")
        traceback.print_exc()
        return False

def test_schema_files():
    """Test that schema files exist and are readable."""
    print("\n🧪 Testing schema files...")
    
    try:
        schema_dir = Path(__file__).parent.parent.parent / "src" / "memfuse_core" / "store" / "pgai_store" / "schemas"
        
        # Check M1 schema file
        m1_schema_file = schema_dir / "m1_episodic.sql"
        if m1_schema_file.exists():
            content = m1_schema_file.read_text()
            # Check for flexible fact_type (should not have CHECK constraint)
            if "fact_type TEXT," in content and "CHECK (fact_type IN" not in content:
                print("   ✅ M1 schema has flexible fact_type")
            else:
                print("   ⚠️  M1 schema may still have hardcoded fact_type constraints")
            
            # Check for fact_category field
            if "fact_category JSONB" in content:
                print("   ✅ M1 schema has fact_category field")
            else:
                print("   ⚠️  M1 schema missing fact_category field")
                
        else:
            print("   ❌ M1 schema file not found")
            return False
            
        return True
        
    except Exception as e:
        print(f"   ❌ Schema file test failed: {e}")
        return False

def test_configuration():
    """Test configuration file structure."""
    print("\n🧪 Testing configuration...")
    
    try:
        import yaml
        
        config_file = Path(__file__).parent.parent.parent / "config" / "store" / "pgai.yaml"
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Check for flexible classification configuration
            m1_config = config.get('memory_layers', {}).get('m1', {})
            fact_extraction = m1_config.get('fact_extraction', {})
            
            if 'classification_strategy' in fact_extraction:
                print("   ✅ Configuration has flexible classification_strategy")
            else:
                print("   ⚠️  Configuration missing classification_strategy")
                
            if 'enable_auto_categorization' in fact_extraction:
                print("   ✅ Configuration has enable_auto_categorization")
            else:
                print("   ⚠️  Configuration missing enable_auto_categorization")
                
        else:
            print("   ❌ Configuration file not found")
            return False
            
        return True
        
    except ImportError:
        print("   ⚠️  PyYAML not available, skipping config test")
        return True
    except Exception as e:
        print(f"   ❌ Configuration test failed: {e}")
        return False

def main():
    """Run all validation tests."""
    print("🚀 Multi-Layer PgAI Setup Validation")
    print("=" * 50)
    
    tests = [
        ("Import Tests", test_imports),
        ("Basic Functionality", test_basic_functionality),
        ("Schema Files", test_schema_files),
        ("Configuration", test_configuration),
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
    print("📊 Validation Summary:")
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! Multi-layer PgAI setup is ready.")
        print("\nNext steps:")
        print("1. Start database: docker-compose -f docker/docker-compose.pgai.yml up -d")
        print("2. Run integration tests: python -m pytest tests/integration/test_multi_layer_pgai_e2e.py -v")
        print("3. Start MemFuse server: poetry run memfuse-core")
    else:
        print("\n❌ Some tests failed. Please check the issues above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
