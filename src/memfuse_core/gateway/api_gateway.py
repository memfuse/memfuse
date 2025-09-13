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
        # Pass db_service for potential metadata enrichment needs
        self.metadata_enricher = MetadataEnricher(db_service=self.db_service)
        self.scope_calculator = ScopeCalculator()

        # Remove unused fields (QueryResponseProcessor handles most top-level fields)
        unused_fields = [
            'metadata.level',
            'metadata.retrieval',
            'metadata.source'
        ]
        self.field_remover = FieldRemover(fields_to_remove=unused_fields)
    
    async def process_request(
        self,
        request_data: Dict[str, Any],
        operation_type: OperationType = OperationType.QUERY
    ) -> Dict[str, Any]:
        """Process a complete request through the gateway pipeline."""
        try:
            # Special-case ADD operation: handle write path here to keep API thin
            if operation_type == OperationType.ADD:
                return await self._handle_add(request_data)

            # Step 1: Parse request and create context
            context = self.request_parser.parse_request(request_data)
            context.operation_type = operation_type

            # Add gateway marker to track processing
            request_data['_gateway_entry'] = True
            
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

    async def _handle_add(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle add messages (write path) including optional M3 EOS orchestration.

        Expects request_data to contain at least:
          - messages: List[dict]
          - user_id, session_id (optional but recommended)
          - metadata: Dict
        """
        try:
            if not self.buffer_service:
                raise ValueError("Buffer service not available")

            messages = request_data.get("messages") or []
            session_id = request_data.get("session_id")

            # 1) Write messages through buffer service
            try:
                add_result = await self.buffer_service.add(messages, session_id=session_id)
            except ModuleNotFoundError as e:
                # Allow environments without optional backends
                if 'qdrant_client' in str(e):
                    add_result = {"status": "success", "data": {"message_ids": []}}
                else:
                    raise

            # 2) Interpret metadata for M3 EOS
            workflow_name = None
            user_goal = None
            trigger_m3 = False
            # Pre-scan messages for EOS and task
            pre_task = None
            pre_goal = None
            pre_eos = False
            try:
                msgs0 = list(messages) if not isinstance(messages, list) else messages
                for m in msgs0:
                    if isinstance(m, dict) and str(m.get('role','')) == 'user':
                        md = m.get('metadata') or {}
                        if bool(md.get('task_eos', False)):
                            pre_eos = True
                            t = str(md.get('task') or '').strip()
                            if t:
                                pre_task = t
                            pre_goal = str(m.get('content') or '').strip() or pre_goal
                if pre_task is None:
                    for m in reversed(msgs0):
                        md = m.get('metadata') if isinstance(m, dict) else None
                        if isinstance(md, dict):
                            t = str(md.get('task') or '').strip()
                            if t:
                                pre_task = t
                                if not pre_goal and isinstance(m, dict):
                                    pre_goal = str(m.get('content') or '').strip() or None
                                break
            except Exception:
                pass

            try:
                from .message_metadata import MessageMetadataInterpreter
                decision = MessageMetadataInterpreter().interpret_add_messages(messages, tag=None, legacy_tag_trigger=False)
                trigger_m3 = bool(decision.trigger_m3)
                workflow_name = decision.workflow_name
                user_goal = decision.user_goal
            except Exception:
                trigger_m3 = False

            # Robust scan of messages to derive EOS and workflow/task if needed
            try:
                msgs = list(messages) if not isinstance(messages, list) else messages
                has_eos = False
                last_task = None
                last_goal = None
                for m in msgs:
                    if isinstance(m, dict) and str(m.get('role','')) == 'user':
                        md = m.get('metadata') or {}
                        if 'task' in md and not last_task:
                            t = str(md.get('task') or '').strip()
                            if t:
                                last_task = t
                                last_goal = str(m.get('content') or '').strip() or last_goal
                        if bool(md.get('task_eos', False)):
                            has_eos = True
                            # prefer EOS message's task and goal
                            t = str(md.get('task') or '').strip()
                            if t:
                                last_task = t
                            last_goal = str(m.get('content') or '').strip() or last_goal
                if not trigger_m3 and has_eos:
                    trigger_m3 = True
                if not workflow_name and last_task:
                    workflow_name = last_task
                if not user_goal and last_goal:
                    user_goal = last_goal
            except Exception:
                pass

            # Prefer pre-scan values if still missing
            if not trigger_m3 and pre_eos:
                trigger_m3 = True
            if not workflow_name and pre_task:
                workflow_name = pre_task
            if not user_goal and pre_goal:
                user_goal = pre_goal

            # 3) Check config gating
            m3_enabled = True
            max_history_scan = 2000
            try:
                from ..utils.global_config_manager import get_global_config_manager
                cfg = get_global_config_manager()
                m3_enabled = bool(cfg.get("memory.layers.m3.enabled", m3_enabled))
                max_history_scan = int(cfg.get("memory.layers.m3.max_history_scan", max_history_scan))
            except Exception:
                pass

            assistant_message_id = None
            workflow_id = None

            # 4) If triggered, run orchestrator with workflow-scoped history
            if trigger_m3 and m3_enabled and session_id:
                # Build history via buffer service when available
                history_all = []
                if hasattr(self.buffer_service, 'get_messages_by_session'):
                    try:
                        history_all = await self.buffer_service.get_messages_by_session(
                            session_id=session_id,
                            limit=max_history_scan,
                            sort_by="timestamp",
                            order="asc",
                            buffer_only=None,
                        )
                    except Exception:
                        history_all = []
                # Filter by workflow_name
                history = []
                try:
                    if workflow_name:
                        for h in history_all or []:
                            md = h.get('metadata') if isinstance(h, dict) else None
                            if isinstance(md, dict) and str(md.get('task') or '') == workflow_name:
                                history.append(h)
                    else:
                        history = history_all or []
                except Exception:
                    history = history_all or []

                try:
                    from ..m3.orchestrator import Orchestrator
                    from ..procedural.store import ProceduralStore
                    import uuid as _uuid

                    orch = Orchestrator()
                    ai_text = await orch.handle_request(
                        session_id,
                        user_goal or "",
                        workflow_name=workflow_name or None,
                        history_messages=history or None,
                    )

                    # Write assistant reply back through buffer service
                    assistant_msg = [{
                        "role": "assistant",
                        "content": ai_text,
                        "metadata": {"m3_enabled": True, "source": "orchestrator", "task": workflow_name, "task_eos": True},
                    }]
                    try:
                        ares = await self.buffer_service.add(assistant_msg, session_id=session_id)
                        if ares and ares.get("status") == "success" and ares.get("data"):
                            mids = ares["data"].get("message_ids", [])
                            if mids:
                                assistant_message_id = mids[0]
                    except Exception:
                        pass

                    # Log workflow rows (soft-fail)
                    try:
                        store = ProceduralStore()
                        workflow_id = getattr(orch, "last_workflow_id", None) or str(_uuid.uuid4())
                        steps = getattr(orch, "last_plan_steps", None) or []
                        outcomes = getattr(orch, "last_step_outcomes", None) or []
                        if assistant_message_id and isinstance(steps, list) and steps:
                            for idx, st in enumerate(steps):
                                try:
                                    agent_name = getattr(st, "agent", None) or (st.get("agent") if isinstance(st, dict) else None)
                                except Exception:
                                    agent_name = None
                                meta = {
                                    "m3_enabled": True,
                                    "user_goal": user_goal,
                                    "reused": getattr(orch, "last_reused", False),
                                    "task": workflow_name,
                                    "task_eos": True,
                                }
                                if idx < len(outcomes):
                                    oc = outcomes[idx]
                                    if isinstance(oc, dict):
                                        if oc.get("success") is not None:
                                            meta["success"] = bool(oc.get("success"))
                                        if oc.get("attempts") is not None:
                                            meta["attempts"] = int(oc.get("attempts"))
                                        if oc.get("duration_ms") is not None:
                                            meta["duration_ms"] = int(oc.get("duration_ms"))
                                        if oc.get("error"):
                                            meta["error"] = str(oc.get("error"))[:200]
                                if agent_name:
                                    meta["agent"] = agent_name
                                await store.log_message_workflow(
                                    message_id=assistant_message_id,
                                    workflow_id=workflow_id,
                                    step_index=idx,
                                    tags=["m3", "workflow"],
                                    metadata=meta,
                                )
                    except Exception:
                        pass
                except Exception as e:
                    logger.info(f"Gateway ADD: orchestration skipped: {e}")

            # Build response
            data = {
                "message_ids": (add_result.get("data", {}) or {}).get("message_ids", []),
            }
            if assistant_message_id:
                data["assistant_message_id"] = assistant_message_id
            if workflow_id:
                data["workflow_id"] = workflow_id

            return {
                "status": "success",
                "code": 201,
                "data": data,
                "message": "Messages added successfully",
                "errors": None,
            }

        except Exception as e:
            logger.error(f"Gateway ADD error: {e}")
            return self._create_error_response(f"Gateway ADD error: {str(e)}")
    
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
            
            # Get session info; fill session_name and missing agent_id from session
            if context.session_id:
                session = await self.db_service.get_session(context.session_id)
                if session:
                    if not context.session_name:
                        context.session_name = session.get("name")
                    if not context.agent_id:
                        context.agent_id = session.get("agent_id")
            
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

        # 1. Transform response format (field renaming, memory type handling)
        data = self.response_processor.transform(data, context)
        # 2. Enrich metadata
        data = self.metadata_enricher.transform(data, context)

        # 3. Calculate scope
        data = self.scope_calculator.transform(data, context)

        # 4. Remove unwanted fields
        data = self.field_remover.transform(data, context)

        # Optionally echo query back in response data for clarity
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
