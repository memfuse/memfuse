"""Metadata-based request router for MemFuse Gateway."""

from typing import Any, Dict, Optional
from loguru import logger

from ..interfaces.gateway_interface import (
    MetadataBasedRouter,
    RequestContext,
    RoutingDecision,
    ServiceType
)


class MemoryMetadataRouter(MetadataBasedRouter):
    """Router that determines service based on request metadata."""
    
    def __init__(self):
        """Initialize the metadata router."""
        self.routing_rules = self._initialize_routing_rules()
    
    def _initialize_routing_rules(self) -> Dict[str, Any]:
        """Initialize routing rules based on metadata patterns."""
        return {
            # Task-based routing
            "task_routing": {
                "search": ServiceType.HYBRID_SERVICE,
                "retrieval": ServiceType.MEMORY_SERVICE,
                "recent": ServiceType.BUFFER_SERVICE,
                "semantic": ServiceType.SEMANTIC_SERVICE,
                None: ServiceType.HYBRID_SERVICE  # Default
            },
            
            # Mode-based routing
            "mode_routing": {
                "episodic": ServiceType.BUFFER_SERVICE,
                "semantic": ServiceType.SEMANTIC_SERVICE,
                "hybrid": ServiceType.HYBRID_SERVICE,
                None: ServiceType.HYBRID_SERVICE  # Default
            },
            
            # Session-based routing preferences
            "session_routing": {
                "with_session": {
                    "prefer_buffer": True,
                    "include_cross_session": True
                },
                "without_session": {
                    "prefer_semantic": True,
                    "include_cross_session": False
                }
            }
        }
    
    def extract_routing_metadata(self, context: RequestContext) -> Dict[str, Any]:
        """Extract metadata relevant for routing decisions."""
        request_metadata = context.request_metadata or {}
        
        routing_metadata = {
            "task": request_metadata.get("task"),
            "mode": request_metadata.get("mode"),
            "has_session": context.session_id is not None,
            "has_agent": context.agent_id is not None,
            "operation_type": context.operation_type,
            "custom_fields": {}
        }
        
        # Extract any custom routing fields
        for key, value in request_metadata.items():
            if key.startswith("routing_"):
                routing_metadata["custom_fields"][key] = value
        
        logger.info(f"Extracted routing metadata: {routing_metadata}")
        return routing_metadata
    
    def determine_service_type(self, routing_metadata: Dict[str, Any]) -> ServiceType:
        """Determine service type based on routing metadata."""
        task = routing_metadata.get("task")
        mode = routing_metadata.get("mode")
        has_session = routing_metadata.get("has_session", False)
        
        # Priority 1: Explicit mode specification
        if mode:
            service_type = self.routing_rules["mode_routing"].get(mode, ServiceType.HYBRID_SERVICE)
            logger.info(f"Routing by mode '{mode}' -> {service_type}")
            return service_type
        
        # Priority 2: Task-based routing
        if task:
            service_type = self.routing_rules["task_routing"].get(task, ServiceType.HYBRID_SERVICE)
            logger.info(f"Routing by task '{task}' -> {service_type}")
            return service_type
        
        # Priority 3: Session-based default routing
        if has_session:
            # With session, prefer buffer service for recent context
            logger.info("Routing with session -> BUFFER_SERVICE")
            return ServiceType.BUFFER_SERVICE
        else:
            # Without session, prefer semantic service for general knowledge
            logger.info("Routing without session -> SEMANTIC_SERVICE")
            return ServiceType.SEMANTIC_SERVICE
    
    def build_service_params(
        self,
        context: RequestContext,
        service_type: ServiceType
    ) -> Dict[str, Any]:
        """Build parameters for the selected service."""
        # Base parameters
        params = {
            "user_id": context.user_id,
            "agent_id": context.agent_id,
            "session_id": context.session_id,
        }

        # Add service-specific parameters based on service type
        if service_type == ServiceType.SEMANTIC_SERVICE:
            # For semantic queries, focus on knowledge
            params.update({
                "include_messages": False,
                "include_knowledge": True,
                "store_type": "semantic"
            })
        elif service_type == ServiceType.BUFFER_SERVICE:
            # For episodic/buffer queries, focus on messages
            params.update({
                "include_messages": True,
                "include_knowledge": False,
                "store_type": "buffer"
            })
        else:
            # Default: include both
            params.update({
                "include_messages": True,
                "include_knowledge": True
            })

        return params
    
    def route_request(self, context: RequestContext) -> RoutingDecision:
        """Determine which service should handle the request."""
        # Extract routing metadata
        routing_metadata = self.extract_routing_metadata(context)
        
        # Determine service type
        service_type = self.determine_service_type(routing_metadata)
        
        # Build service parameters
        service_params = self.build_service_params(context, service_type)
        
        # Create transformation hints
        transformation_hints = {
            "service_type": service_type,
            "has_session": routing_metadata.get("has_session", False),
            "requested_mode": routing_metadata.get("mode"),
            "requested_task": routing_metadata.get("task")
        }
        
        decision = RoutingDecision(
            service_type=service_type,
            service_params=service_params,
            transformation_hints=transformation_hints
        )
        
        logger.info(f"Routing decision: {decision}")
        return decision


class CustomMetadataRouter(MemoryMetadataRouter):
    """Extended router that supports custom metadata field routing."""
    
    def __init__(self, custom_rules: Optional[Dict[str, Any]] = None):
        """Initialize with optional custom routing rules."""
        super().__init__()
        if custom_rules:
            self.routing_rules.update(custom_rules)
    
    def add_custom_routing_rule(self, field_name: str, field_value: Any, service_type: ServiceType):
        """Add a custom routing rule for a specific metadata field."""
        if "custom_routing" not in self.routing_rules:
            self.routing_rules["custom_routing"] = {}
        
        if field_name not in self.routing_rules["custom_routing"]:
            self.routing_rules["custom_routing"][field_name] = {}
        
        self.routing_rules["custom_routing"][field_name][field_value] = service_type
        logger.info(f"Added custom routing rule: {field_name}={field_value} -> {service_type}")
    
    def determine_service_type(self, routing_metadata: Dict[str, Any]) -> ServiceType:
        """Enhanced service type determination with custom rules."""
        # Check custom routing rules first
        if "custom_routing" in self.routing_rules:
            request_metadata = routing_metadata.get("custom_fields", {})
            
            for field_name, field_rules in self.routing_rules["custom_routing"].items():
                if field_name in request_metadata:
                    field_value = request_metadata[field_name]
                    if field_value in field_rules:
                        service_type = field_rules[field_value]
                        logger.info(f"Custom routing: {field_name}={field_value} -> {service_type}")
                        return service_type
        
        # Fall back to parent logic
        return super().determine_service_type(routing_metadata)
