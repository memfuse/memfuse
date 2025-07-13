#!/usr/bin/env python3
"""
Validate reorganized pgai_store directory structure and functionality.

This test validates:
1. File structure is correct
2. Import paths work properly
3. Backward compatibility
4. Basic functionality is available
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

def test_file_structure():
    """Test if file structure is correct."""
    print("🧪 Testing File Structure")
    print("=" * 50)
    
    base_path = Path(__file__).parent.parent.parent / "src" / "memfuse_core" / "store" / "pgai_store"
    
    expected_files = [
        "__init__.py",
        "pgai_store.py", 
        "simplified_event_driven_store.py",
        "event_driven_pgai_store.py",
        "store_factory.py",
        "immediate_trigger_components.py",
        "monitoring.py",
        "simple_error_handling.py",
        "pgai_vector_wrapper.py"
    ]
    
    missing_files = []
    for file_name in expected_files:
        file_path = base_path / file_name
        if not file_path.exists():
            missing_files.append(file_name)
        else:
            # Check if file is not empty
            if file_path.stat().st_size == 0:
                missing_files.append(f"{file_name} (empty)")
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All expected files present and non-empty")
        return True


def test_imports():
    """Test if imports work properly."""
    print("\n🧪 Testing Imports")
    print("=" * 50)
    
    try:
        # Test direct imports from pgai_store package
        from memfuse_core.store.pgai_store.pgai_store import PgaiStore
        from memfuse_core.store.pgai_store.simplified_event_driven_store import SimplifiedEventDrivenPgaiStore
        from memfuse_core.store.pgai_store.store_factory import PgaiStoreFactory
        from memfuse_core.store.pgai_store.immediate_trigger_components import TriggerManager
        from memfuse_core.store.pgai_store.monitoring import EmbeddingMonitor
        
        print("✅ Direct imports successful")
        
        # Test package-level imports
        from memfuse_core.store.pgai_store import (
            PgaiStore as PackagePgaiStore,
            SimplifiedEventDrivenPgaiStore as PackageSimplified,
            PgaiStoreFactory as PackageFactory
        )
        
        print("✅ Package-level imports successful")
        
        # Test backward compatibility
        from memfuse_core.store.pgai_store import EventDrivenPgaiStore
        
        if EventDrivenPgaiStore is SimplifiedEventDrivenPgaiStore:
            print("✅ Backward compatibility alias working")
        else:
            print("❌ Backward compatibility alias not working")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_class_instantiation():
    """Test if class instantiation works properly."""
    print("\n🧪 Testing Class Instantiation")
    print("=" * 50)
    
    try:
        from memfuse_core.store.pgai_store import (
            PgaiStore, SimplifiedEventDrivenPgaiStore, 
            PgaiStoreFactory, TriggerManager, EmbeddingMonitor
        )
        
        # Test basic instantiation (without actual initialization)
        config = {
            "pgai": {
                "enabled": True,
                "auto_embedding": True,
                "immediate_trigger": True
            }
        }
        
        # Test PgaiStore instantiation
        store = PgaiStore(config, "test_table")
        print(f"✅ PgaiStore instantiated: {type(store).__name__}")
        
        # Test SimplifiedEventDrivenPgaiStore instantiation
        event_store = SimplifiedEventDrivenPgaiStore(config, "test_table")
        print(f"✅ SimplifiedEventDrivenPgaiStore instantiated: {type(event_store).__name__}")
        
        # Test factory
        factory_store = PgaiStoreFactory.create_store(config, "test_table")
        print(f"✅ Factory created store: {type(factory_store).__name__}")
        
        # Test components (without pool)
        monitor = EmbeddingMonitor("test_monitor")
        print(f"✅ EmbeddingMonitor instantiated: {type(monitor).__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ Instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_table_name_update():
    """Test if table name update is correct."""
    print("\n🧪 Testing Table Name Update (m0_messages -> m0_episodic)")
    print("=" * 50)
    
    try:
        from memfuse_core.store.pgai_store import PgaiStore, SimplifiedEventDrivenPgaiStore
        
        # Test default table name
        store1 = PgaiStore()
        if store1.table_name == "m0_episodic":
            print("✅ PgaiStore default table name updated to m0_episodic")
        else:
            print(f"❌ PgaiStore table name is {store1.table_name}, expected m0_episodic")
            return False
        
        store2 = SimplifiedEventDrivenPgaiStore()
        if store2.table_name == "m0_episodic":
            print("✅ SimplifiedEventDrivenPgaiStore default table name updated to m0_episodic")
        else:
            print(f"❌ SimplifiedEventDrivenPgaiStore table name is {store2.table_name}, expected m0_episodic")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Table name test failed: {e}")
        return False


def test_configuration_files():
    """Test if configuration files are updated."""
    print("\n🧪 Testing Configuration Files")
    print("=" * 50)
    
    try:
        config_files = [
            Path(__file__).parent.parent.parent / "config" / "memory" / "default.yaml",
            Path(__file__).parent.parent.parent / "config" / "store" / "pgai.yaml"
        ]
        
        for config_file in config_files:
            if config_file.exists():
                content = config_file.read_text()
                if "m0_episodic" in content:
                    print(f"✅ {config_file.name} updated with m0_episodic")
                else:
                    print(f"⚠️  {config_file.name} may not be updated")
            else:
                print(f"⚠️  {config_file.name} not found")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def main():
    """Run all validation tests."""
    print("🚀 MemFuse PgAI Store Reorganization Validation")
    print("=" * 60)
    
    tests = [
        test_file_structure,
        test_imports,
        test_class_instantiation,
        test_table_name_update,
        test_configuration_files
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{i+1}. {test.__name__}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Reorganization successful!")
        return True
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
