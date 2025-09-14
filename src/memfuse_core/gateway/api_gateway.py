"""Main API Gateway implementation with routing and guardrails."""

from typing import Any, Dict, Optional
from loguru import logger

from ..interfaces.gateway_interface import (
    GatewayInterface,
    RequestContext,
    OperationType,
    ServiceType
)
from ..services.buffer_service import BufferService
from ..services.database_service import DatabaseService
from .metadata_router import MemoryMetadataRouter
from ..validators.guardrails import MemoryGuardrail
from .processors import (
    QueryResponseProcessor,
    MetadataEnricher,
    ScopeCalculator,
    FieldRemover
)
from .m3_processor import M3Processor, M3ResponseEnricher, M3MetadataExtractor


class MemoryRequestParser:
    """Parser for memory service requests."""
    
    def parse_request(self, request_data: Dict[str, Any]) -> RequestContext:
        """Parse request data and extract context."""
        return RequestContext(
            user_id=request_data.get("user_id", ""),
            user_name=request_data.get("user_name"),
            agent_id=request_data.get("agent_id"),
            agent_name=request_data.get("agent_name"),
            session_id=request_data.get("session_id"),
            session_name=request_data.get("session_name"),
            operation_type=OperationType.QUERY,  # Default for now
            request_metadata=request_data.get("metadata", {})
        )


class MemoryApiGateway(GatewayInterface):
    """Main API Gateway for memory operations with routing and guardrails."""
    
    def __init__(
        self,
        buffer_service: Optional[BufferService] = None,
        db_service: Optional[DatabaseService] = None
    ):
        """Initialize the API Gateway."""
        self.buffer_service = buffer_service
        self.db_service = db_service
        
        # Initialize components
        self.request_parser = MemoryRequestParser()
        self.router = MemoryMetadataRouter()
        self.guardrail = MemoryGuardrail()

        # Initialize processors
        self.response_processor = QueryResponseProcessor()
        self.metadata_enricher = MetadataEnricher()
        self.scope_calculator = ScopeCalculator()

        # Remove unused fields (QueryResponseProcessor handles most top-level fields)
        unused_fields = [
            'metadata.level',
            'metadata.retrieval',
            'metadata.source'
        ]
        self.field_remover = FieldRemover(fields_to_remove=unused_fields)
        
        # Initialize M3 processors
        self.m3_processor = M3Processor()
        self.m3_response_enricher = M3ResponseEnricher()
        self.m3_metadata_extractor = M3MetadataExtractor()
    
    async def process_request(
        self,
        request_data: Dict[str, Any],
        operation_type: OperationType = OperationType.QUERY
    ) -> Dict[str, Any]:
        """Process a complete request through the gateway pipeline."""
        try:
            # Step 1: Parse request and create context
            context = self.request_parser.parse_request(request_data)
            context.operation_type = operation_type

            # Add gateway marker to track processing
            request_data['_gateway_entry'] = True
            
            # Enrich context with database information
            context = await self._enrich_context(context)
            
            logger.info(f"Processing request for user {context.user_id}, operation: {operation_type}")
            
            # Check if this should trigger M3 workflow
            if self.m3_processor.should_trigger_m3(request_data, context):
                logger.info("Request triggers M3 workflow processing")
                m3_response = await self.m3_processor.process_m3_request(request_data, context)
                
                # Apply minimal guardrails to M3 response
                if not self.guardrail.validate_response(m3_response, context):
                    logger.error("M3 response failed validation")
                    return self._create_error_response("M3 response validation failed")
                
                self.guardrail.audit_response(m3_response, context)
                return m3_response
            
            # Step 2: Route request to appropriate service
            routing_decision = self.router.route_request(context)
            logger.info(f"Routing to {routing_decision.service_type}")
            
            # Step 3: Call appropriate service
            service_response = await self._call_service(
                routing_decision.service_type,
                request_data,
                routing_decision.service_params
            )
            
            # Step 4: Transform response
            transformed_response = await self._transform_response(
                service_response,
                context,
                routing_decision,
                request_data
            )
            
            # Step 5: Apply guardrails
            if not self.guardrail.validate_response(transformed_response, context):
                logger.error("Response failed validation")
                return self._create_error_response("Response validation failed")
            
            # Step 6: Audit response
            self.guardrail.audit_response(transformed_response, context)
            
            return transformed_response
            
        except Exception as e:
            logger.error(f"Gateway processing error: {e}")
            import traceback
            logger.error(f"Gateway traceback: {traceback.format_exc()}")
            return self._create_error_response(f"Gateway error: {str(e)}")
    
    async def _enrich_context(self, context: RequestContext) -> RequestContext:
        """Enrich context with database information."""
        if not self.db_service:
            return context
        
        try:
            # Get user name if not provided
            if not context.user_name and context.user_id:
                user = await self.db_service.get_user(context.user_id)
                if user:
                    context.user_name = user.get("name")
            
            # Get agent name if not provided
            if not context.agent_name and context.agent_id:
                agent = await self.db_service.get_agent(context.agent_id)
                if agent:
                    context.agent_name = agent.get("name")
            
            # Get session name if not provided
            if not context.session_name and context.session_id:
                session = await self.db_service.get_session(context.session_id)
                if session:
                    context.session_name = session.get("name")
            
        except Exception as e:
            logger.warning(f"Failed to enrich context: {e}")
        
        return context
    
    async def _call_service(
        self,
        service_type: ServiceType,
        request_data: Dict[str, Any],
        service_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Call the appropriate service based on routing decision."""
        query = request_data.get("query", "")
        top_k = request_data.get("top_k", 5)

        # All service types now just use BufferService with basic parameters
        if not self.buffer_service:
            raise ValueError("Buffer service not available")

        # Only pass session_id if available; also pass task from request metadata
        buffer_params = {}
        if service_params.get("session_id"):
            buffer_params["session_id"] = service_params["session_id"]
        try:
            req_task = (request_data.get("metadata") or {}).get("task")
            if req_task is not None:
                buffer_params["task"] = req_task
        except Exception:
            pass

        return await self.buffer_service.query(
            query=query,
            top_k=top_k,
            **buffer_params
        )
    
    async def _transform_response(
        self,
        service_response: Dict[str, Any],
        context: RequestContext,
        routing_decision,
        request_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Transform service response through the transformation pipeline."""
        if service_response.get("status") != "success":
            return service_response
        
        # Apply transformation pipeline
        data = service_response.get("data", {})

        logger.info(f"Gateway: Starting transformation pipeline with {len(data.get('results', []))} results")

        # 1. Transform response format (field renaming, memory type handling)
        data = self.response_processor.transform(data, context)
        # 2. Enrich metadata (merge request_data metadata into context for this transform)
        try:
            req_meta = (request_data or {}).get("metadata") or {}
            # Create a shallow copy context to avoid mutating original
            ctx_for_meta = RequestContext(
                user_id=context.user_id,
                user_name=context.user_name,
                agent_id=context.agent_id,
                agent_name=context.agent_name,
                session_id=context.session_id,
                session_name=context.session_name,
                operation_type=context.operation_type,
                request_metadata={**(context.request_metadata or {}), **req_meta},
            )
        except Exception:
            ctx_for_meta = context
        data = self.metadata_enricher.transform(data, ctx_for_meta)

        # 3. Calculate scope
        data = self.scope_calculator.transform(data, context)

        # 4. Remove unwanted fields
        data = self.field_remover.transform(data, context)
        
        # 5. Apply M3 enrichment if relevant
        data = self.m3_response_enricher.transform(data, context)

        # 6. Echo query back in response data for clarity
        try:
            if isinstance(data, dict) and request_data and request_data.get("query"):
                data.setdefault("query", request_data.get("query"))
        except Exception:
            pass

        # Return transformed response with defaults to satisfy API contract
        status = service_response.get("status", "success")
        code = service_response.get("code") if service_response.get("code") is not None else 200
        message = service_response.get("message") if service_response.get("message") is not None else "Success"
        # For successful responses, errors should be None per contract
        errors = None if status == "success" else service_response.get("errors")

        response = {
            "status": status,
            "code": code,
            "data": data,
            "message": message,
            "errors": errors
        }
        return response
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create a standardized error response."""
        return {
            "status": "error",
            "code": 500,
            "data": {"results": [], "total": 0},
            "message": error_message,
            "errors": [error_message]
        }


# Convenience function for creating gateway instances
def create_memory_gateway(
    buffer_service: Optional[BufferService] = None,
    db_service: Optional[DatabaseService] = None
) -> MemoryApiGateway:
    """Create a configured memory gateway instance."""
    return MemoryApiGateway(
        buffer_service=buffer_service,
        db_service=db_service
    )
