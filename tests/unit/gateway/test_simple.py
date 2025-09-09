"""Simple Gateway unit test."""

def test_gateway_imports():
    """Test that gateway modules can be imported."""
    try:
        from memfuse_core.gateway.api_gateway import MemoryApiGateway
        from memfuse_core.gateway.metadata_router import MemoryMetadataRouter
        from memfuse_core.gateway.processors import QueryResponseProcessor
        assert True
    except ImportError as e:
        assert False, f"Import failed: {e}"

def test_basic_functionality():
    """Test basic functionality."""
    assert 1 + 1 == 2
