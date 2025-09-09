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
from .filters import (
    InboundFilter,
    OutboundFilter,
    build_filters_from_config,
)
from ..utils.global_config_manager import get_global_config_manager
from ..observability.tracing import get_tracer


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
        # Filters registration points (inbound/outbound)
        # By default empty; may be populated from config if available
        self.inbound_filters: list[InboundFilter] = []
        self.outbound_filters: list[OutboundFilter] = []

        # Try to load filters from configuration (if global config is initialized)
        try:
            gcm = get_global_config_manager()
            if gcm.is_initialized():
                gateway_cfg = gcm.get_section("gateway")
                inbound, outbound = build_filters_from_config(gateway_cfg)
                if inbound:
                    self.inbound_filters = inbound
                if outbound:
                    self.outbound_filters = outbound
        except Exception:
            # Best-effort; keep empty if config not available
            pass

        self.field_remover = FieldRemover(fields_to_remove=unused_fields)

    async def process_request(
        self,
        request_data: Dict[str, Any],
        operation_type: OperationType = OperationType.QUERY
    ) -> Dict[str, Any]:
        """Process a complete request through the gateway pipeline."""
        import time
        overall_start = time.perf_counter()

        # Initialize distributed tracing
        tracer = get_tracer()

        async with tracer.trace_async_operation(
            "gateway.process_request",
            attributes={
                "operation_type": str(operation_type),
                "has_query": "query" in request_data,
                "top_k": request_data.get("top_k", 5)
            }
        ) as span:
            try:
                # Step 1: Parse request and create context
                context = self.request_parser.parse_request(request_data)
                context.operation_type = operation_type
                context.query = request_data.get("query", "")  # Add query to context for tracing

                # Add request attributes to span
                tracer.add_request_attributes(span, context, request_data)

                # Inbound filters (pre-routing)
                for flt in self.inbound_filters:
                    try:
                        request_data = flt.apply(request_data, context)
                    except Exception as e:
                        logger.warning(f"Inbound filter {type(flt).__name__} failed: {e}")

                # Add gateway marker to track processing
                request_data['_gateway_entry'] = True

                # Minimal request validation via Guardrail (after inbound normalization)
                if hasattr(self.guardrail, "validate_request"):
                    if not self.guardrail.validate_request(request_data, context):
                        return self._create_error_response("Request validation failed")

                # Enrich context with database information
                context = await self._enrich_context(context)

                logger.info(f"Processing request for user {context.user_id}, operation: {operation_type}")

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

                # Outbound filters (post-transformation, pre-guardrail)
                for flt in self.outbound_filters:
                    try:
                        transformed_response = flt.apply(transformed_response, context)
                    except Exception as e:
                        logger.warning(f"Outbound filter {type(flt).__name__} failed: {e}")

                # Step 5: Apply guardrails
                if not self.guardrail.validate_response(transformed_response, context):
                    logger.error("Response failed validation")
                    return self._create_error_response("Response validation failed")

                # Step 6: Audit response
                self.guardrail.audit_response(transformed_response, context)

                # Optional: Add overall timing to debug metadata
                overall_duration = time.perf_counter() - overall_start
                try:
                    gcm = get_global_config_manager()
                    if gcm.is_initialized():
                        gw_cfg = gcm.get_section("gateway") or {}
                        dbg = gw_cfg.get("debug") or {}
                        if dbg.get("enabled") and dbg.get("include_durations"):
                            data = transformed_response.get("data", {})
                            md_top = data.setdefault("metadata", {}) if isinstance(data, dict) else {}
                            obs_top = md_top.setdefault("observability", {}) if isinstance(md_top, dict) else {}
                            if isinstance(obs_top, dict) and "durations" in obs_top:
                                obs_top["durations"]["overall"] = round(overall_duration * 1000, 3)
                except Exception:
                    pass

                # Add response attributes to tracing span
                tracer.add_response_attributes(span, transformed_response)

                # Add trace ID to response metadata for correlation
                trace_id = tracer.get_current_trace_id()
                if trace_id:
                    data = transformed_response.get("data", {})
                    if isinstance(data, dict):
                        md_top = data.setdefault("metadata", {})
                        if isinstance(md_top, dict):
                            obs_top = md_top.setdefault("observability", {})
                            if isinstance(obs_top, dict):
                                obs_top["trace_id"] = trace_id

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
        tracer = get_tracer()

        async with tracer.trace_async_operation(
            "gateway.call_service",
            attributes={
                "service_type": str(service_type),
                "has_session_id": bool(service_params.get("session_id"))
            }
        ):
            query = request_data.get("query", "")
            top_k = request_data.get("top_k", 5)

            # All service types now just use BufferService with basic parameters
            if not self.buffer_service:
                raise ValueError("Buffer service not available")

            # Only pass session_id if available, let BufferService use its defaults for everything else
            buffer_params = {}
            if service_params.get("session_id"):
                buffer_params["session_id"] = service_params["session_id"]

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

        # Optional timing collection for debug
        durations = {}
        import time

        # 1. Transform response format (field renaming, memory type handling)
        start_time = time.perf_counter()
        data = self.response_processor.transform(data, context)
        durations["response_processor"] = time.perf_counter() - start_time

        # 2. Enrich metadata
        start_time = time.perf_counter()
        data = self.metadata_enricher.transform(data, context)
        durations["metadata_enricher"] = time.perf_counter() - start_time

        # 3. Calculate scope
        start_time = time.perf_counter()
        data = self.scope_calculator.transform(data, context)
        durations["scope_calculator"] = time.perf_counter() - start_time

        # 4. Remove unwanted fields
        start_time = time.perf_counter()
        data = self.field_remover.transform(data, context)
        durations["field_remover"] = time.perf_counter() - start_time

        # Optional debug metadata aggregation (controlled by gateway.debug)
        try:
            gcm = get_global_config_manager()
            if gcm.is_initialized():
                gw_cfg = gcm.get_section("gateway") or {}
                dbg = gw_cfg.get("debug") or {}
                results = data.get("results", []) or []
                # Aggregate rerank cache hit
                if dbg.get("enabled") and dbg.get("include_rerank_cache_hit"):
                    def _has_cache_hit(item: Dict[str, Any]) -> bool:
                        if not isinstance(item, dict):
                            return False
                        md = item.get("metadata")
                        if not isinstance(md, dict):
                            return False
                        obs = md.get("observability")
                        return isinstance(obs, dict) and obs.get("rerank_cache_hit") is True
                    agg_hit = any(_has_cache_hit(it) for it in results)
                    md_top = data.setdefault("metadata", {}) if isinstance(data, dict) else {}
                    obs_top = md_top.setdefault("observability", {}) if isinstance(md_top, dict) else {}
                    if isinstance(obs_top, dict):
                        obs_top["rerank_cache_hit"] = agg_hit
                # Aggregate plugin order (first non-empty list)
                if dbg.get("enabled") and dbg.get("include_plugin_order"):
                    def _get_order(item: Dict[str, Any]):
                        if not isinstance(item, dict):
                            return None
                        md = item.get("metadata")
                        if not isinstance(md, dict):
                            return None
                        obs = md.get("observability")
                        if not isinstance(obs, dict):
                            return None
                        order = obs.get("plugin_order")
                        if isinstance(order, list) and len(order) > 0:
                            return order
                        return None
                    first_order = None
                    for it in results:
                        first_order = _get_order(it)
                        if first_order:
                            break
                    if first_order:
                        md_top = data.setdefault("metadata", {}) if isinstance(data, dict) else {}
                        obs_top = md_top.setdefault("observability", {}) if isinstance(md_top, dict) else {}
                        if isinstance(obs_top, dict):
                            obs_top["plugin_order"] = first_order
                # Aggregate score range (min/max of scores present)
                if dbg.get("enabled") and dbg.get("include_score_range"):
                    scores = []
                    for it in results:
                        if isinstance(it, dict):
                            s = it.get("relevance_score")
                            if not isinstance(s, (int, float)):
                                s = it.get("score")
                            if isinstance(s, (int, float)):
                                scores.append(float(s))
                    if scores:
                        md_top = data.setdefault("metadata", {}) if isinstance(data, dict) else {}
                        obs_top = md_top.setdefault("observability", {}) if isinstance(md_top, dict) else {}
                        if isinstance(obs_top, dict):
                            obs_top["score_range"] = {"min": min(scores), "max": max(scores)}
                # Aggregate dedup removed count (first available)
                if dbg.get("enabled") and dbg.get("include_dedup_removed_count"):
                    dedup_val = None
                    for it in results:
                        if not isinstance(it, dict):
                            continue
                        md = it.get("metadata")
                        obs = md.get("observability") if isinstance(md, dict) else None
                        val = obs.get("dedup_removed_count") if isinstance(obs, dict) else None
                        if isinstance(val, int):
                            dedup_val = val
                            break
                    if dedup_val is not None:
                        md_top = data.setdefault("metadata", {}) if isinstance(data, dict) else {}
                        obs_top = md_top.setdefault("observability", {}) if isinstance(md_top, dict) else {}
                        if isinstance(obs_top, dict):
                            obs_top["dedup_removed_count"] = dedup_val
                # Aggregate dedup unique count (first available)
                if dbg.get("enabled") and dbg.get("include_dedup_unique_count"):
                    uniq_val = None
                    for it in results:
                        if not isinstance(it, dict):
                            continue
                        md = it.get("metadata")
                        obs = md.get("observability") if isinstance(md, dict) else None
                        val = obs.get("dedup_unique_count") if isinstance(obs, dict) else None
                        if isinstance(val, int):
                            uniq_val = val
                            break
                    if uniq_val is not None:
                        md_top = data.setdefault("metadata", {}) if isinstance(data, dict) else {}
                        obs_top = md_top.setdefault("observability", {}) if isinstance(md_top, dict) else {}
                        if isinstance(obs_top, dict):
                            obs_top["dedup_unique_count"] = uniq_val
                # Aggregate dedup key source (first available)
                if dbg.get("enabled") and dbg.get("include_dedup_key_source"):
                    key_src = None
                    for it in results:
                        if not isinstance(it, dict):
                            continue
                        md = it.get("metadata")
                        obs = md.get("observability") if isinstance(md, dict) else None
                        val = obs.get("dedup_key_source") if isinstance(obs, dict) else None
                        if isinstance(val, str) and val:
                            key_src = val
                            break
                    if key_src is not None:
                        md_top = data.setdefault("metadata", {}) if isinstance(data, dict) else {}
                        obs_top = md_top.setdefault("observability", {}) if isinstance(md_top, dict) else {}
                        if isinstance(obs_top, dict):
                            obs_top["dedup_key_source"] = key_src
                # Aggregate score clip stats (first available)
                if dbg.get("enabled") and dbg.get("include_score_clip_stats"):
                    clip_stats = None
                    for it in results:
                        if not isinstance(it, dict):
                            continue
                        md = it.get("metadata")
                        obs = md.get("observability") if isinstance(md, dict) else None
                        st = obs.get("score_clip_stats") if isinstance(obs, dict) else None
                        if isinstance(st, dict):
                            clip_stats = st
                            break
                    if clip_stats is not None:
                        md_top = data.setdefault("metadata", {}) if isinstance(data, dict) else {}
                        obs_top = md_top.setdefault("observability", {}) if isinstance(md_top, dict) else {}
                        if isinstance(obs_top, dict):
                            obs_top["score_clip_stats"] = clip_stats
                # Aggregate durations (if enabled)
                if dbg.get("enabled") and dbg.get("include_durations"):
                    md_top = data.setdefault("metadata", {}) if isinstance(data, dict) else {}
                    obs_top = md_top.setdefault("observability", {}) if isinstance(md_top, dict) else {}
                    if isinstance(obs_top, dict):
                        # Convert to milliseconds and round to 3 decimal places
                        obs_top["durations"] = {
                            k: round(v * 1000, 3) for k, v in durations.items()
                        }
        except Exception:
            # Best-effort: do not break pipeline on debug enrich failures
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
