"""Response validation and guardrails for MemFuse Gateway."""

from typing import Any, Dict, List, Optional, Set
from datetime import datetime
from loguru import logger

from ..interfaces.gateway_interface import (
    RequestContext,
    SchemaValidator,
    ResponseGuardrail,
    ResponseAuditor
)
from ..utils.global_config_manager import get_global_config_manager


class MemoryValidator(SchemaValidator):
    """Validator for memory service response schemas."""
    
    def __init__(self):
        """Initialize the response validator."""
        self.required_fields = {
            "root": {"status", "code", "data", "message"},
            "data": {"results", "total"},
            "result_episodic": {"id", "content", "relevance_score", "memory_type", "created_at", "metadata"},
            "result_semantic": {"id", "fact", "relevance_score", "memory_type", "created_at", "metadata"},
            "metadata": {"user_id", "scope"},
            "fact": {"text", "triples"}
        }
        
        self.forbidden_fields = {
            "result": {"score", "type", "role"},  # Old field names
            "metadata": {"level", "retrieval", "source"}  # Internal fields
        }
        
        self.valid_memory_types = {"episodic", "semantic", "message", "knowledge", "chunk"}
        self.valid_scopes = {"in_session", "cross_session", None}
    
    def get_expected_schema(self, context: RequestContext) -> Dict[str, Any]:
        """Get expected response schema based on context."""
        return {
            "type": "object",
            "required": list(self.required_fields["root"]),
            "properties": {
                "status": {"type": "string", "enum": ["success", "error"]},
                "code": {"type": "integer"},
                "data": {
                    "type": "object",
                    "required": list(self.required_fields["data"]),
                    "properties": {
                        "results": {
                            "type": "array",
                            "items": {"$ref": "#/definitions/result"}
                        },
                        "total": {"type": "integer", "minimum": 0}
                    }
                },
                "message": {"type": "string"}
            },
            "definitions": {
                "result": {
                    "type": "object",
                    "required": ["id", "relevance_score", "memory_type", "metadata"],
                    "properties": {
                        "id": {"type": "string"},
                        "relevance_score": {"type": "number", "minimum": 0, "maximum": 1},
                        "memory_type": {"type": "string", "enum": list(self.valid_memory_types)},
                        "created_at": {"type": ["string", "null"]},
                        "updated_at": {"type": ["string", "null"]},
                        "metadata": {"$ref": "#/definitions/metadata"}
                    }
                },
                "metadata": {
                    "type": "object",
                    "required": ["user_id", "scope"],
                    "properties": {
                        "user_id": {"type": "string"},
                        "agent_id": {"type": ["string", "null"]},
                        "session_id": {"type": ["string", "null"]},
                        "session_name": {"type": ["string", "null"]},
                        "scope": {"type": ["string", "null"], "enum": ["in_session", "cross_session", None]}
                    }
                }
            }
        }
    
    def validate_schema(self, response: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
        """Validate response against schema, return list of errors."""
        errors = []
        
        # Validate root structure
        errors.extend(self._validate_required_fields(response, self.required_fields["root"], "root"))
        
        if "data" in response:
            data = response["data"]
            errors.extend(self._validate_required_fields(data, self.required_fields["data"], "data"))
            
            if "results" in data and isinstance(data["results"], list):
                for i, result in enumerate(data["results"]):
                    errors.extend(self._validate_result(result, i))
        
        return errors
    
    def _validate_required_fields(self, obj: Dict[str, Any], required: Set[str], context: str) -> List[str]:
        """Validate that required fields are present."""
        errors = []
        for field in required:
            if field not in obj:
                errors.append(f"Missing required field '{field}' in {context}")
        return errors
    
    def _validate_result(self, result: Dict[str, Any], index: int) -> List[str]:
        """Validate a single result object."""
        errors = []
        context = f"result[{index}]"
        
        # Check memory type specific requirements
        memory_type = result.get("memory_type")
        if memory_type == "episodic" or memory_type == "message":
            errors.extend(self._validate_required_fields(
                result, self.required_fields["result_episodic"], context
            ))
            if "content" not in result:
                errors.append(f"Missing 'content' field for episodic memory in {context}")
        
        elif memory_type == "semantic":
            errors.extend(self._validate_required_fields(
                result, self.required_fields["result_semantic"], context
            ))
            if "fact" in result:
                errors.extend(self._validate_required_fields(
                    result["fact"], self.required_fields["fact"], f"{context}.fact"
                ))
        
        # Validate forbidden fields
        for field in self.forbidden_fields["result"]:
            if field in result:
                errors.append(f"Forbidden field '{field}' found in {context}")
        
        # Validate metadata
        if "metadata" in result:
            metadata = result["metadata"]
            errors.extend(self._validate_required_fields(
                metadata, self.required_fields["metadata"], f"{context}.metadata"
            ))
            
            # Validate scope value
            scope = metadata.get("scope")
            if scope not in self.valid_scopes:
                errors.append(f"Invalid scope value '{scope}' in {context}.metadata")
            
            # Check forbidden metadata fields
            for field in self.forbidden_fields["metadata"]:
                if field in metadata:
                    errors.append(f"Forbidden metadata field '{field}' found in {context}.metadata")
        
        return errors


class MemoryResponseAuditor(ResponseAuditor):
    """Auditor for memory service responses."""
    
    def __init__(self):
        """Initialize the response auditor."""
        self.audit_log = []
    
    def audit_metadata_completeness(self, response: Dict[str, Any]) -> List[str]:
        """Audit metadata completeness, return list of issues."""
        issues = []
        
        if "data" not in response or "results" not in response["data"]:
            return ["Response missing data.results structure"]
        
        results = response["data"]["results"]
        for i, result in enumerate(results):
            if "metadata" not in result:
                issues.append(f"Result[{i}] missing metadata")
                continue
            
            metadata = result["metadata"]
            
            # Check for recommended fields
            recommended_fields = ["agent_id", "session_id", "session_name"]
            for field in recommended_fields:
                if field not in metadata:
                    issues.append(f"Result[{i}].metadata missing recommended field '{field}'")
            
            # Check scope consistency
            scope = metadata.get("scope")
            session_id = metadata.get("session_id")
            
            if scope == "in_session" and not session_id:
                issues.append(f"Result[{i}] has scope 'in_session' but no session_id")
            elif scope == "cross_session" and not session_id:
                issues.append(f"Result[{i}] has scope 'cross_session' but no session_id")
        
        return issues
    
    def audit_field_compliance(self, response: Dict[str, Any]) -> List[str]:
        """Audit field naming and structure compliance."""
        issues = []
        
        if "data" not in response or "results" not in response["data"]:
            return ["Response missing data.results structure"]
        
        results = response["data"]["results"]
        for i, result in enumerate(results):
            # Check for old field names that should have been renamed
            old_fields = {"score": "relevance_score", "type": "memory_type"}
            for old_field, new_field in old_fields.items():
                if old_field in result:
                    issues.append(f"Result[{i}] contains old field name '{old_field}', should be '{new_field}'")
            
            # Check memory type specific structure
            memory_type = result.get("memory_type")
            if memory_type == "semantic":
                if "content" in result:
                    issues.append(f"Result[{i}] semantic memory should not have 'content' field")
                if "fact" not in result:
                    issues.append(f"Result[{i}] semantic memory missing 'fact' structure")
            elif memory_type in ["episodic", "message"]:
                if "fact" in result:
                    issues.append(f"Result[{i}] episodic memory should not have 'fact' field")
                if "content" not in result:
                    issues.append(f"Result[{i}] episodic memory missing 'content' field")
        
        return issues
    
    def log_response_metrics(self, response: Dict[str, Any], context: RequestContext) -> None:
        """Log response metrics for monitoring."""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "user_id": context.user_id,
            "agent_id": context.agent_id,
            "session_id": context.session_id,
            "status": response.get("status"),
            "code": response.get("code"),
            "result_count": 0,
            "memory_types": {},
            "scopes": {}
        }
        
        if "data" in response and "results" in response["data"]:
            results = response["data"]["results"]
            metrics["result_count"] = len(results)
            
            for result in results:
                # Count memory types
                memory_type = result.get("memory_type", "unknown")
                metrics["memory_types"][memory_type] = metrics["memory_types"].get(memory_type, 0) + 1
                
                # Count scopes
                scope = result.get("metadata", {}).get("scope", "unknown")
                metrics["scopes"][scope] = metrics["scopes"].get(scope, 0) + 1
        
        self.audit_log.append(metrics)
        logger.info(f"Response metrics: {metrics}")


class MemoryGuardrail(ResponseGuardrail):
    """Main guardrail implementation for memory responses."""

    def __init__(self):
        """Initialize the guardrail and load config-driven settings."""
        self.validator = MemoryValidator()
        self.auditor = MemoryResponseAuditor()
        # Config-driven toggles
        self._output_cfg: Dict[str, Any] = {}
        self._pii_cfg: Dict[str, Any] = {}
        self._toxicity_cfg: Dict[str, Any] = {}
        self._quota_cfg: Dict[str, Any] = {}
        try:
            gcm = get_global_config_manager()
            if gcm.is_initialized():
                guardrail_cfg = gcm.get_section("guardrail") or {}
                # Normalize shapes from either guardrail/default.yaml or split files
                self._output_cfg = guardrail_cfg.get("output", guardrail_cfg.get("output_filter", {})) or {}
                self._pii_cfg = guardrail_cfg.get("pii", {}) or {}
                self._toxicity_cfg = guardrail_cfg.get("toxicity", guardrail_cfg.get("thresholds", {})) or {}
                self._quota_cfg = guardrail_cfg.get("quota", {}) or {}
        except Exception:
            # Best-effort: keep defaults if config not available
            pass

    def validate_response(self, response: Dict[str, Any], context: RequestContext) -> bool:
        """Validate response format and content."""
        schema = self.validator.get_expected_schema(context)
        errors = self.validator.validate_schema(response, schema)

        if errors:
            logger.error(f"Response validation failed: {errors}")
            return False

        logger.info("Response validation passed")
        return True

    def validate_request(self, request: Dict[str, Any], context: RequestContext) -> bool:
        """Minimal request validation to catch obvious issues early.

        Checks:
        - query present and is string
        - top_k is coercible to int
        """
        try:
            q = request.get("query", "")
            if not isinstance(q, str):
                return False
            tk = request.get("top_k", 5)
            _ = int(tk)
            return True
        except Exception:
            return False

    def _remove_path(self, obj: Dict[str, Any], dotted: str) -> None:
        parts = dotted.split('.') if dotted else []
        if not parts:
            return

        def rec(cur: Any, idx: int) -> None:
            if idx >= len(parts) or cur is None:
                return
            key = parts[idx]
            is_last = (idx == len(parts) - 1)

            if isinstance(cur, dict):
                if key not in cur:
                    return
                if is_last:
                    try:
                        del cur[key]
                    except Exception:
                        pass
                else:
                    rec(cur.get(key), idx + 1)
            elif isinstance(cur, list):
                if key == "*":
                    for item in cur:
                        rec(item, idx + 1)
                else:
                    try:
                        i = int(key)
                    except Exception:
                        return
                    if 0 <= i < len(cur):
                        rec(cur[i], idx + 1)
            else:
                return

        rec(obj, 0)

    def _apply_output_filters(self, response: Dict[str, Any]) -> None:
        enabled = bool(self._output_cfg.get("enabled", False))
        fields = list(self._output_cfg.get("remove_fields", []) or [])
        if not enabled or not fields:
            return
        try:
            results = response.get("data", {}).get("results", [])
            for r in results:
                for f in fields:
                    self._remove_path(r, f)
        except Exception as e:
            logger.warning(f"Output filter application failed: {e}")

    def audit_response(self, response: Dict[str, Any], context: RequestContext) -> None:
        """Audit response for compliance and logging, then apply config-driven output filters."""
        # Audit metadata completeness
        metadata_issues = self.auditor.audit_metadata_completeness(response)
        if metadata_issues:
            logger.warning(f"Metadata completeness issues: {metadata_issues}")

        # Audit field compliance
        field_issues = self.auditor.audit_field_compliance(response)
        if field_issues:
            logger.warning(f"Field compliance issues: {field_issues}")

        # Apply output field removal as a last step prior to logging metrics
        self._apply_output_filters(response)

        # Log metrics
        self.auditor.log_response_metrics(response, context)

        logger.info("Response audit completed")


class SecurityGuardrail(ResponseGuardrail):
    """Security guardrail for response validation."""

    def __init__(self):
        """Initialize security guardrail."""
        self.blocked_fields = {"password", "token", "secret", "key"}

    def check(self, data: Any, context: RequestContext) -> bool:
        """Check for security violations."""
        try:
            if isinstance(data, dict):
                return self._check_dict_security(data)
            elif isinstance(data, list):
                return all(self._check_dict_security(item) for item in data if isinstance(item, dict))
            return True
        except Exception as e:
            logger.error(f"Security check failed: {e}")
            return False

    def _check_dict_security(self, data: Dict[str, Any]) -> bool:
        """Check dictionary for security violations."""
        for key in data.keys():
            if key.lower() in self.blocked_fields:
                logger.warning(f"Blocked security-sensitive field: {key}")
                return False
        return True


class AuditLogger(ResponseAuditor):
    """Simple audit logger for responses."""

    def __init__(self):
        """Initialize audit logger."""
        self.audit_log = []

    def audit(self, data: Any, context: RequestContext) -> None:
        """Log response for audit purposes."""
        try:
            audit_entry = {
                "timestamp": datetime.now().isoformat(),
                "user_id": context.user_id,
                "session_id": context.session_id,
                "response_size": len(str(data)) if data else 0,
                "has_results": bool(data and isinstance(data, dict) and data.get("data", {}).get("results"))
            }
            self.audit_log.append(audit_entry)
            logger.debug(f"Audit entry: {audit_entry}")
        except Exception as e:
            logger.error(f"Audit logging failed: {e}")

    def audit_metadata_completeness(self, response: Dict[str, Any]) -> List[str]:
        """Audit metadata completeness - simplified implementation."""
        return []  # No issues for simple logger

    def audit_field_compliance(self, response: Dict[str, Any]) -> List[str]:
        """Audit field compliance - simplified implementation."""
        return []  # No issues for simple logger

    def log_response_metrics(self, response: Dict[str, Any], context: RequestContext) -> None:
        """Log response metrics - simplified implementation."""
        self.audit(response, context)
