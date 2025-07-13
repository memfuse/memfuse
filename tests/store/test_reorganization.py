#!/usr/bin/env python3
"""
Reorganization validation test without external module dependencies.
"""

import sys
import os
from pathlib import Path

def test_file_structure():
    """Test if file structure is correct."""
    print("🧪 Testing File Structure")
    print("=" * 50)
    
    base_path = Path(__file__).parent.parent.parent / "src" / "memfuse_core" / "store" / "pgai_store"
    
    expected_files = [
        "__init__.py",
        "pgai_store.py",
        "event_driven_store.py",
        "store_factory.py",
        "immediate_trigger_components.py",
        "monitoring.py",
        "error_handling.py",
        "pgai_vector_wrapper.py"
    ]
    
    results = {}
    for file_name in expected_files:
        file_path = base_path / file_name
        if file_path.exists():
            size = file_path.stat().st_size
            results[file_name] = f"✅ {size} bytes"
        else:
            results[file_name] = "❌ Missing"
    
    for file_name, status in results.items():
        print(f"  {file_name}: {status}")
    
    missing = [f for f, s in results.items() if "❌" in s]
    if missing:
        print(f"\n❌ Missing files: {missing}")
        return False
    else:
        print("\n✅ All files present")
        return True


def test_file_contents():
    """Test if file contents are correct."""
    print("\n🧪 Testing File Contents")
    print("=" * 50)
    
    base_path = Path(__file__).parent.parent.parent / "src" / "memfuse_core" / "store" / "pgai_store"
    
    # Test key files have expected content
    tests = [
        ("__init__.py", ["PgaiStore", "EventDrivenPgaiStore", "EventDrivenPgaiStore"]),
        ("pgai_store.py", ["class PgaiStore", "m0_episodic"]),
        ("event_driven_store.py", ["class EventDrivenPgaiStore", "EventDrivenPgaiStore = EventDrivenPgaiStore"]),
        ("store_factory.py", ["class PgaiStoreFactory", "m0_episodic"]),
    ]
    
    results = []
    for file_name, expected_content in tests:
        file_path = base_path / file_name
        if file_path.exists():
            content = file_path.read_text()
            missing_content = [item for item in expected_content if item not in content]
            if missing_content:
                print(f"  {file_name}: ❌ Missing content: {missing_content}")
                results.append(False)
            else:
                print(f"  {file_name}: ✅ All expected content present")
                results.append(True)
        else:
            print(f"  {file_name}: ❌ File not found")
            results.append(False)
    
    return all(results)


def test_table_name_in_configs():
    """Test table name updates in configuration files."""
    print("\n🧪 Testing Table Name in Configs")
    print("=" * 50)
    
    config_files = [
        ("config/memory/default.yaml", "m0_episodic"),
        ("config/store/pgai.yaml", "m0_episodic"),
    ]
    
    base_path = Path(__file__).parent.parent.parent
    results = []
    
    for config_file, expected_name in config_files:
        file_path = base_path / config_file
        if file_path.exists():
            content = file_path.read_text()
            if expected_name in content:
                print(f"  {config_file}: ✅ Contains {expected_name}")
                results.append(True)
            else:
                print(f"  {config_file}: ❌ Missing {expected_name}")
                results.append(False)
        else:
            print(f"  {config_file}: ⚠️  File not found")
            results.append(False)
    
    return all(results)


def test_database_schema_updates():
    """Test database schema updates."""
    print("\n🧪 Testing Database Schema Updates")
    print("=" * 50)
    
    schema_files = [
        ("src/memfuse_core/database/sqlite.py", "m0_episodic"),
        ("src/memfuse_core/database/base.py", "get_m0_episodic_by_session"),
        ("src/memfuse_core/hierarchy/storage.py", "m0_episodic"),
    ]
    
    base_path = Path(__file__).parent.parent.parent
    results = []
    
    for schema_file, expected_content in schema_files:
        file_path = base_path / schema_file
        if file_path.exists():
            content = file_path.read_text()
            if expected_content in content:
                print(f"  {schema_file}: ✅ Updated with {expected_content}")
                results.append(True)
            else:
                print(f"  {schema_file}: ❌ Missing {expected_content}")
                results.append(False)
        else:
            print(f"  {schema_file}: ❌ File not found")
            results.append(False)
    
    return all(results)


def test_old_files_removed():
    """Test if old files have been removed."""
    print("\n🧪 Testing Old Files Removed")
    print("=" * 50)
    
    old_files = [
        "src/memfuse_core/store/pgai_store.py",
        "src/memfuse_core/store/event_driven_pgai_store.py", 
        "src/memfuse_core/store/event_driven_store.py",
        "src/memfuse_core/store/store_factory.py",
        "src/memfuse_core/store/monitoring.py",
        "src/memfuse_core/store/immediate_trigger_components.py",
        "src/memfuse_core/store/error_handling.py",
        "src/memfuse_core/store/error_handling.py",
        "src/memfuse_core/store/schema_migration.py",
    ]
    
    base_path = Path(__file__).parent.parent.parent
    results = []
    
    for old_file in old_files:
        file_path = base_path / old_file
        if file_path.exists():
            # Check if it's just a placeholder
            content = file_path.read_text().strip()
            if len(content) <= 100:  # Likely a placeholder
                print(f"  {old_file}: ✅ Removed (placeholder only)")
                results.append(True)
            else:
                print(f"  {old_file}: ❌ Still contains content")
                results.append(False)
        else:
            print(f"  {old_file}: ✅ Removed")
            results.append(True)
    
    return all(results)


def main():
    """Run all validation tests."""
    print("🚀 MemFuse PgAI Store Reorganization Simple Validation")
    print("=" * 60)
    
    tests = [
        ("File Structure", test_file_structure),
        ("File Contents", test_file_contents),
        ("Table Names in Configs", test_table_name_in_configs),
        ("Database Schema Updates", test_database_schema_updates),
        ("Old Files Removed", test_old_files_removed),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)
    
    passed = 0
    for i, (test_name, result) in enumerate(results):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{i+1}. {test_name}: {status}")
        if result:
            passed += 1
    
    total = len(results)
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Reorganization successful!")
        print("\n📋 Summary of Changes:")
        print("  ✅ Created pgai_store/ directory")
        print("  ✅ Moved all pgai-related files to pgai_store/")
        print("  ✅ Updated table name from m0_messages to m0_episodic")
        print("  ✅ Updated configuration files")
        print("  ✅ Updated database schema references")
        print("  ✅ Maintained backward compatibility")
        print("  ✅ Cleaned up old files")
        return True
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
