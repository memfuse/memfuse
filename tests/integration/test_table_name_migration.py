"""Test for table name migration from m0_messages to m0_raw.

This test validates that the table name changes have been properly applied
across all relevant files and configurations.
"""

import asyncio
from pathlib import Path
from loguru import logger


def test_config_files_updated():
    """Test that configuration files use the new table name."""
    logger.info("🧪 Testing configuration files for table name updates")
    
    base_path = Path(__file__).parent.parent.parent
    config_files = [
        "config/memory/default.yaml",
        "config/store/pgai.yaml"
    ]
    
    for config_file in config_files:
        file_path = base_path / config_file
        if file_path.exists():
            content = file_path.read_text()
            assert "m0_raw" in content, f"m0_raw not found in {config_file}"
            assert "m0_messages" not in content, f"Old m0_messages still found in {config_file}"
            logger.info(f"✅ {config_file}: Updated to use m0_raw")
        else:
            logger.warning(f"⚠️ {config_file}: File not found")


def test_database_code_updated():
    """Test that database-related code uses the new table name."""
    logger.info("🧪 Testing database code for table name updates")
    
    base_path = Path(__file__).parent.parent.parent / "src" / "memfuse_core"
    
    # Check database/base.py
    db_base_file = base_path / "database" / "base.py"
    if db_base_file.exists():
        content = db_base_file.read_text()
        assert "m0_raw" in content, "m0_raw not found in database/base.py"
        logger.info("✅ database/base.py: Updated to use m0_raw")
    
    # Check store schema manager
    schema_manager_file = base_path / "store" / "pgai_store" / "schema_manager.py"
    if schema_manager_file.exists():
        content = schema_manager_file.read_text()
        assert "m0_raw" in content, "m0_raw not found in schema_manager.py"
        logger.info("✅ schema_manager.py: Updated to use m0_raw")


def test_docker_scripts_updated():
    """Test that Docker initialization scripts use the new table name."""
    logger.info("🧪 Testing Docker scripts for table name updates")
    
    base_path = Path(__file__).parent.parent.parent
    docker_files = [
        "docker/pgai/init-scripts/01-init-extensions.sh",
        "docker/pgvectorscale/init-scripts/00-init-memfuse-pgai.sql"
    ]
    
    for docker_file in docker_files:
        file_path = base_path / docker_file
        if file_path.exists():
            content = file_path.read_text()
            assert "m0_raw" in content, f"m0_raw not found in {docker_file}"
            logger.info(f"✅ {docker_file}: Updated to use m0_raw")
        else:
            logger.warning(f"⚠️ {docker_file}: File not found")


def test_m1_schema_exists():
    """Test that M1 episodic schema file exists and is properly defined."""
    logger.info("🧪 Testing M1 episodic schema file")
    
    base_path = Path(__file__).parent.parent.parent / "src" / "memfuse_core" / "store" / "pgai_store" / "schemas"
    m1_schema_file = base_path / "m1_episodic.sql"
    
    assert m1_schema_file.exists(), "m1_episodic.sql schema file not found"
    
    content = m1_schema_file.read_text()
    assert "CREATE TABLE IF NOT EXISTS m1_episodic" in content, "M1 table creation not found"
    assert "embedding VECTOR(384)" in content, "Vector embedding column not found"
    assert "needs_embedding BOOLEAN" in content, "Embedding flag not found"
    
    logger.info("✅ m1_episodic.sql: Schema file exists and is properly defined")


def test_documentation_updated():
    """Test that documentation reflects the new table names."""
    logger.info("🧪 Testing documentation for table name updates")
    
    base_path = Path(__file__).parent.parent.parent
    doc_file = base_path / "docs" / "architecture" / "pgai" / "overview.md"
    
    if doc_file.exists():
        content = doc_file.read_text()
        assert "m0_raw" in content, "m0_raw not found in documentation"
        assert "m1_episodic" in content, "m1_episodic not found in documentation"
        logger.info("✅ Documentation: Updated to reflect new table names")
    else:
        logger.warning("⚠️ Documentation file not found")


def run_all_tests():
    """Run all table name migration tests."""
    logger.info("🚀 Starting Table Name Migration Tests")
    logger.info("=" * 60)
    
    tests = [
        ("Configuration Files", test_config_files_updated),
        ("Database Code", test_database_code_updated),
        ("Docker Scripts", test_docker_scripts_updated),
        ("M1 Schema", test_m1_schema_exists),
        ("Documentation", test_documentation_updated),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running {test_name} Test...")
        try:
            test_func()
            logger.info(f"✅ {test_name} Test: PASSED")
            passed += 1
        except Exception as e:
            logger.error(f"❌ {test_name} Test: FAILED - {e}")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 Table Name Migration Test Results:")
    logger.info(f"🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All table name migration tests passed!")
        return True
    else:
        logger.error("💥 Some tests failed. Please check the logs above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
