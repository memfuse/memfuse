"""Gateway layer for MemFuse API.

This module provides a gateway layer that sits between the API endpoints and the
underlying services (Buffer, Memory, etc.). The gateway handles:

1. Request parsing and validation
2. Metadata enrichment and transformation
3. Response formatting and standardization
4. Scope calculation and field mapping

The gateway architecture provides:
- Separation of concerns between API and business logic
- Extensible metadata handling
- Consistent response formats
- Easy maintenance and testing
"""

from ..interfaces.gateway_interface import GatewayInterface, RequestContext
from .api_gateway import MemoryApiGateway, create_memory_gateway
from .processors import (
    QueryRequestProcessor,
    QueryResponseProcessor,
    MetadataEnricher,
    ScopeCalculator,
    FieldRemover
)

__all__ = [
    "GatewayInterface",
    "RequestContext",
    "QueryResponseProcessor",
    "MemoryApiGateway",
    "create_memory_gateway",
    "QueryRequestProcessor",
    "MetadataEnricher",
    "ScopeCalculator",
    "FieldRemover"
]
