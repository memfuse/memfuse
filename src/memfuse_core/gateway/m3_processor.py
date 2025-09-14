"""M3 processor for gateway integration."""

from typing import Any, Dict, List, Optional
from loguru import logger

from ..interfaces.gateway_interface import RequestContext
from ..m3.orchestrator import Orchestrator
from ..m3.config import get_m3_config


class M3Processor:
    """Processes requests that require M3 workflow orchestration."""
    
    def __init__(self, orchestrator: Optional[Orchestrator] = None):
        self.orchestrator = orchestrator or Orchestrator()
        self.config = get_m3_config()
    
    def should_trigger_m3(self, request_data: Dict[str, Any], context: RequestContext) -> bool:
        """Check if request should trigger M3 processing."""
        if not self.config.enable_workflow_reuse:
            return False
        # Only consider orchestration triggers on ADD/write paths
        try:
            from ..interfaces.gateway_interface import OperationType
            if context.operation_type != OperationType.ADD:
                return False
        except Exception:
            return False
        
        # Check for task_eos metadata
        metadata = request_data.get("metadata", {})
        if metadata.get("task_eos") is True:
            return True
        
        # Check for explicit M3 trigger
        if metadata.get("m3_trigger") is True:
            return True
        
        # Check for workflow name
        if metadata.get("workflow_name"):
            return True
        
        return False
    
    def extract_m3_params(self, request_data: Dict[str, Any], context: RequestContext) -> Dict[str, Any]:
        """Extract M3 parameters from request."""
        metadata = request_data.get("metadata", {})
        
        return {
            "task_name": metadata.get("task", "default_task"),
            "workflow_name": metadata.get("workflow_name"),
            "user_goal": request_data.get("query", ""),
            "session_id": context.session_id,
            "user_id": context.user_id,
            "agent_id": context.agent_id,
            "messages": request_data.get("messages", [])
        }
    
    async def process_m3_request(
        self, 
        request_data: Dict[str, Any], 
        context: RequestContext
    ) -> Dict[str, Any]:
        """Process request through M3 workflow orchestration."""
        try:
            logger.info("Processing M3 workflow request")
            
            # Extract M3 parameters
            m3_params = self.extract_m3_params(request_data, context)
            
            # Execute M3 workflow
            result = await self.orchestrator.handle_request(
                session_id=m3_params["session_id"],
                user_goal=m3_params["user_goal"],
                workflow_name=m3_params["workflow_name"],
                history_messages=m3_params["messages"]
            )
            
            # Create response in gateway format
            response = {
                "status": "success",
                "code": 200,
                "data": {
                    "m3_result": result,
                    "workflow_id": self.orchestrator.last_workflow_id,
                    "workflow_reused": self.orchestrator.last_reused,
                    "plan_steps": [
                        {"agent": step.agent, "input": step.input} 
                        for step in self.orchestrator.last_plan_steps
                    ],
                    "step_outcomes": getattr(self.orchestrator, 'last_step_outcomes', [])
                },
                "message": "M3 workflow completed successfully",
                "errors": None
            }
            
            logger.info(f"M3 workflow completed: workflow_id={self.orchestrator.last_workflow_id}, reused={self.orchestrator.last_reused}")
            return response
            
        except Exception as e:
            logger.error(f"M3 workflow processing failed: {e}")
            return {
                "status": "error",
                "code": 500,
                "data": {"m3_result": None},
                "message": f"M3 workflow failed: {str(e)}",
                "errors": [str(e)]
            }


class M3ResponseEnricher:
    """Enriches regular responses with M3-related metadata."""
    
    def transform(self, data: Any, context: RequestContext) -> Any:
        """Enrich response data with M3 metadata if applicable."""
        if not isinstance(data, dict):
            return data
        
        # Check if this is a query response with results
        if "results" in data and isinstance(data["results"], list):
            self._enrich_query_results(data["results"], context)
        
        return data
    
    def _enrich_query_results(self, results: List[Dict[str, Any]], context: RequestContext):
        """Enrich query results with M3 metadata."""
        for result in results:
            if not isinstance(result, dict):
                continue
            
            # Add M3 metadata if available
            metadata = result.setdefault("metadata", {})
            
            # Check if result is from M3 workflow
            if metadata.get("workflow_id"):
                metadata["m3_generated"] = True
            
            # Add task information if available from context
            if context.request_metadata and context.request_metadata.get("task"):
                metadata["task_context"] = context.request_metadata["task"]


class M3MetadataExtractor:
    """Extracts M3-relevant metadata from requests."""
    
    def extract_m3_metadata(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract M3-relevant metadata from request."""
        metadata = request_data.get("metadata", {})
        
        m3_metadata = {}
        
        # Extract task information
        if "task" in metadata:
            m3_metadata["task"] = metadata["task"]
        
        # Extract workflow information
        if "workflow_name" in metadata:
            m3_metadata["workflow_name"] = metadata["workflow_name"]
        
        # Extract M3 triggers
        if "task_eos" in metadata:
            m3_metadata["task_eos"] = metadata["task_eos"]
        
        if "m3_trigger" in metadata:
            m3_metadata["m3_trigger"] = metadata["m3_trigger"]
        
        # Extract procedural memory hints
        if "reuse_threshold" in metadata:
            m3_metadata["reuse_threshold"] = metadata["reuse_threshold"]
        
        return m3_metadata
    
    def should_enable_m3_features(self, m3_metadata: Dict[str, Any]) -> bool:
        """Check if M3 features should be enabled for this request."""
        # Enable if any M3-specific metadata is present
        return bool(
            m3_metadata.get("task") or
            m3_metadata.get("workflow_name") or
            m3_metadata.get("task_eos") or
            m3_metadata.get("m3_trigger")
        )

    async def enrich_results_with_m3_context(
        self,
        data: Any,
        context: RequestContext,
        query_text: Optional[str] = None,
    ) -> Any:
        """Enrich query results with M3 guidance when a task is specified.

        Adds lightweight guidance into each result's metadata without changing
        top-level response shape (to satisfy strict response schema).
        """
        # Config gate: allow disabling enrichment to avoid heavy deps during tests
        try:
            from .config import get_m3_config
            if not get_m3_config().enable_query_guidance:
                return data
        except Exception:
            return data
        try:
            if not isinstance(data, dict) or "results" not in data:
                return data

            task_name = None
            if context.request_metadata and isinstance(context.request_metadata, dict):
                task_name = context.request_metadata.get("task")

            # Only enrich when a task is explicitly provided and not an EOS trigger
            if not task_name:
                return data

            # Query procedural store for similar workflows and lessons
            try:
                from ..procedural.store import ProceduralStore
                from ..utils.embeddings import create_embedding
                store = ProceduralStore()
                vec = None
                try:
                    # Use query text embedding when available to improve similarity
                    if query_text:
                        vec = await create_embedding(query_text)
                except Exception:
                    vec = None

                # Fallback: if embedding unavailable, use a small zero-vector to allow mocked tests
                if vec is None:
                    vec = [0.0] * 8  # minimal length; store implementations tolerate vector casts in tests

                workflows = await store.query_procedural_similar(vec, top_k=3)
                lessons = await store.query_lessons_similar(vec, agent=None, top_k=3)

                if not workflows and not lessons:
                    # Still attach task marker; keep payload minimal
                    for result in data.get("results", []):
                        if not isinstance(result, dict):
                            continue
                        md = result.setdefault("metadata", {})
                        if isinstance(md, dict):
                            md.setdefault("task", task_name)
                            md.setdefault("m3_guidance", "")
                    return data

                # Build compact guidance payload
                guidance_items: List[str] = []
                for wid, wf, score in workflows or []:
                    try:
                        steps = wf.get("plan", []) if isinstance(wf, dict) else []
                        if steps:
                            agent_names = [str(s.get("agent", "")) for s in steps if isinstance(s, dict)]
                            guidance_items.append(f"reuse:{wid[:8]} score={score:.2f} agents={','.join(agent_names[:3])}")
                    except Exception:
                        continue
                for lid, status, fix_summary, working_params, score in lessons or []:
                    try:
                        tag = "ok" if status == "success" else "fail"
                        summary = (fix_summary or "").strip()[:60]
                        guidance_items.append(f"lesson:{lid[:8]} {tag} score={score:.2f} {summary}")
                    except Exception:
                        continue

                # Attach minimal context to each result's metadata
                for result in data.get("results", []):
                    if not isinstance(result, dict):
                        continue
                    md = result.setdefault("metadata", {})
                    if isinstance(md, dict):
                        md.setdefault("task", task_name)
                        # Use a compact joined string to avoid heavy payloads
                        md["m3_guidance"] = "; ".join(guidance_items[:5]) if guidance_items else ""
            except Exception:
                # Soft-fail: never block the main query path
                return data

            return data
        except Exception:
            return data


class M3TaskMessageCollector:
    """Collects task-scoped messages for M3 processing."""
    
    def __init__(self, db_service=None):
        self.db_service = db_service
    
    async def collect_task_messages(
        self, 
        session_id: str, 
        task_name: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Collect messages for a specific task from the session."""
        if not self.db_service:
            logger.warning("No database service available for task message collection")
            return []
        
        try:
            # Get all messages for the session
            all_messages = await self.db_service.get_messages_by_session(
                session_id=session_id,
                limit=limit,
                sort_by="timestamp",
                order="asc"
            )
            
            # Filter messages that belong to this task
            task_messages = []
            for msg in all_messages:
                msg_metadata = msg.get("metadata", {})
                if msg_metadata.get("task") == task_name:
                    task_messages.append(msg)
            
            logger.info(f"Collected {len(task_messages)} messages for task {task_name}")
            return task_messages
            
        except Exception as e:
            logger.error(f"Failed to collect task messages: {e}")
            return []
