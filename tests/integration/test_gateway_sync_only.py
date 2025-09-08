"""Gateway integration tests - synchronous only to avoid event loop issues."""

from unittest.mock import AsyncMock

from memfuse_core.gateway.api_gateway import MemoryApiGateway, create_memory_gateway


def test_gateway_creation_sync():
    """Test gateway creation with proper dependencies."""
    # Mock dependencies
    buffer_service = AsyncMock()
    db_service = AsyncMock()
    
    # Create gateway
    gateway = create_memory_gateway(
        buffer_service=buffer_service,
        db_service=db_service
    )
    
    assert isinstance(gateway, MemoryApiGateway)
    assert gateway.buffer_service == buffer_service
    assert gateway.db_service == db_service


def test_gateway_transformers_sync():
    """Test Gateway transformation components."""
    from memfuse_core.gateway.processors import (
        QueryResponseProcessor,
        MetadataEnricher,
        ScopeCalculator,
        FieldRemover
    )

    # Create processor instances
    response_processor = QueryResponseProcessor()
    metadata_enricher = MetadataEnricher()
    scope_calculator = ScopeCalculator()
    field_remover = FieldRemover(fields_to_remove=["source", "similarity_score"])

    # Verify they exist and have expected methods
    assert hasattr(response_processor, 'transform')
    assert hasattr(metadata_enricher, 'transform')
    assert hasattr(scope_calculator, 'transform')
    assert hasattr(field_remover, 'transform')
