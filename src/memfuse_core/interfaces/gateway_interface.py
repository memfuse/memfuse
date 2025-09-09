"""Gateway and Router interfaces for MemFuse API layer.

This module defines the core interfaces for the API gateway architecture,
including request routing, response transformation, and guardrails.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol, Union
from dataclasses import dataclass
from enum import Enum


class ServiceType(str, Enum):
    """Types of services that can be routed to."""
    BUFFER_SERVICE = "buffer_service"
    MEMORY_SERVICE = "memory_service"
    HYBRID_SERVICE = "hybrid_service"
    SEMANTIC_SERVICE = "semantic_service"


class OperationType(str, Enum):
    """Types of operations supported."""
    QUERY = "query"
    ADD = "add"
    UPDATE = "update"
    DELETE = "delete"


@dataclass
class RequestContext:
    """Context information extracted from a request."""
    user_id: str
    user_name: Optional[str] = None
    agent_id: Optional[str] = None
    agent_name: Optional[str] = None
    session_id: Optional[str] = None
    session_name: Optional[str] = None
    operation_type: Optional[OperationType] = None
    request_metadata: Optional[Dict[str, Any]] = None
    query: Optional[str] = None  # Add query field for composite filters


class QueryResponseProcessor(Protocol):
    """Protocol for response processors."""

    def transform(self, data: Any, context: RequestContext) -> Any:
        """Transform response data."""
        ...


class SchemaValidator(ABC):
    """Base class for schema validators."""

    @abstractmethod
    def validate(self, data: Any, context: RequestContext) -> bool:
        """Validate data against schema."""
        pass


class ResponseGuardrail(ABC):
    """Base class for response guardrails."""

    @abstractmethod
    def check(self, data: Any, context: RequestContext) -> bool:
        """Check if response passes guardrail."""
        pass


class ResponseAuditor(ABC):
    """Base class for response auditors."""

    @abstractmethod
    def audit(self, data: Any, context: RequestContext) -> None:
        """Audit response data."""
        pass


@dataclass
class RoutingDecision:
    """Decision made by the router about which service to use."""
    service_type: ServiceType
    service_params: Dict[str, Any]
    transformation_hints: Dict[str, Any]


class RequestParser(Protocol):
    """Protocol for parsing requests and extracting context."""
    
    def parse_request(self, request_data: Dict[str, Any]) -> RequestContext:
        """Parse request data and extract context."""
        ...


class ServiceRouter(Protocol):
    """Protocol for routing requests to appropriate services."""
    
    def route_request(self, context: RequestContext) -> RoutingDecision:
        """Determine which service should handle the request."""
        ...





class ResponseGuardrail(Protocol):
    """Protocol for validating and auditing responses."""
    
    def validate_response(self, response: Dict[str, Any], context: RequestContext) -> bool:
        """Validate response format and content."""
        ...
    
    def audit_response(self, response: Dict[str, Any], context: RequestContext) -> None:
        """Audit response for compliance and logging."""
        ...


class GatewayInterface(ABC):
    """Main gateway interface that orchestrates the entire request/response flow."""
    
    @abstractmethod
    async def process_request(
        self,
        request_data: Dict[str, Any],
        operation_type: OperationType
    ) -> Dict[str, Any]:
        """Process a complete request through the gateway pipeline.
        
        Pipeline:
        1. Parse request -> RequestContext
        2. Route request -> RoutingDecision  
        3. Call appropriate service
        4. Transform response
        5. Validate response (guardrails)
        6. Return final response
        """
        pass


class MetadataBasedRouter(ABC):
    """Abstract router that makes routing decisions based on metadata."""
    
    @abstractmethod
    def extract_routing_metadata(self, context: RequestContext) -> Dict[str, Any]:
        """Extract metadata relevant for routing decisions."""
        pass
    
    @abstractmethod
    def determine_service_type(self, routing_metadata: Dict[str, Any]) -> ServiceType:
        """Determine service type based on routing metadata."""
        pass
    
    @abstractmethod
    def build_service_params(
        self, 
        context: RequestContext, 
        service_type: ServiceType
    ) -> Dict[str, Any]:
        """Build parameters for the selected service."""
        pass


class SchemaValidator(ABC):
    """Abstract validator for response schema compliance."""
    
    @abstractmethod
    def get_expected_schema(self, context: RequestContext) -> Dict[str, Any]:
        """Get expected response schema based on context."""
        pass
    
    @abstractmethod
    def validate_schema(self, response: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
        """Validate response against schema, return list of errors."""
        pass


class ResponseAuditor(ABC):
    """Abstract auditor for response compliance and logging."""
    
    @abstractmethod
    def audit_metadata_completeness(self, response: Dict[str, Any]) -> List[str]:
        """Audit metadata completeness, return list of issues."""
        pass
    
    @abstractmethod
    def audit_field_compliance(self, response: Dict[str, Any]) -> List[str]:
        """Audit field naming and structure compliance."""
        pass
    
    @abstractmethod
    def log_response_metrics(self, response: Dict[str, Any], context: RequestContext) -> None:
        """Log response metrics for monitoring."""
        pass
