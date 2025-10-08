"""Response processors for the gateway layer."""

from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
from loguru import logger

from ..interfaces.gateway_interface import RequestContext


class QueryRequestProcessor:
    """Transforms query requests before sending to services."""
    
    def transform(self, request_data: Dict[str, Any], context: RequestContext) -> Dict[str, Any]:
        """Transform request data.
        
        Args:
            request_data: Original request data
            context: Request context
            
        Returns:
            Transformed request data
        """
        # Add context information to request
        transformed = request_data.copy()
        
        # Ensure metadata exists
        if 'metadata' not in transformed:
            transformed['metadata'] = {}
        
        # Add context information
        if context.user_id:
            transformed['metadata']['user_id'] = context.user_id
        if context.agent_id:
            transformed['metadata']['agent_id'] = context.agent_id
        if context.session_id:
            transformed['metadata']['session_id'] = context.session_id
        
        return transformed


class QueryResponseProcessor:
    """Transforms query responses to match the new schema."""

    def transform(self, data: Any, context: RequestContext) -> Any:
        """Transform response data to new schema format.

        Args:
            data: Response data from services
            context: Request context

        Returns:
            Transformed response data
        """
        if isinstance(data, dict) and 'results' in data:
            # Transform each result
            transformed_results = []
            for result in data['results']:
                transformed_result = self._transform_result(result, context)
                if transformed_result:  # Only add valid results
                    transformed_results.append(transformed_result)

            # Update the response
            data['results'] = transformed_results

        return data

    def _transform_result(self, result: Dict[str, Any], context: RequestContext) -> Dict[str, Any]:
        """Transform a single result."""
        if not isinstance(result, dict):
            logger.warning(f"QueryResponseProcessor: Skipping non-dict result: {type(result)}")
            return None
        transformed = result.copy()

        # Rename fields according to new schema
        if 'score' in transformed:
            transformed['relevance_score'] = transformed.pop('score')

        if 'type' in transformed:
            transformed['memory_type'] = transformed.pop('type')

        # Handle different memory types
        memory_type = transformed.get('memory_type', 'episodic')  # Default to episodic
        # Normalize memory_type: map granular to canonical
        if memory_type in ['chunk', 'message']:
            memory_type = 'episodic'
            transformed['memory_type'] = 'episodic'

        # For M1 (episodic) memories - keep content field
        if memory_type in ['episodic', 'message', 'chunk']:
            # Ensure content field exists
            if 'content' not in transformed:
                logger.warning(f"QueryResponseProcessor: Missing content field in result {transformed.get('id')}")
                return None

        # For M2 (semantic) memories - use fact structure
        elif memory_type in ['semantic', 'M2 Semantic', 'knowledge']:
            # Transform to semantic format with fact structure
            content = transformed.get('content', '')
            transformed['fact'] = {
                'text': content,
                'triples': transformed.get('triples')  # Use existing triples if available
            }
            # Remove content field for semantic memories
            transformed.pop('content', None)
            # Normalize memory_type to 'semantic'
            transformed['memory_type'] = 'semantic'
            
            # Handle derived_from metadata for M2 memories
            if 'metadata' in transformed and isinstance(transformed['metadata'], dict):
                metadata = transformed['metadata']
                # Move derived_from to correct location if it exists
                if 'derived_from' not in metadata and 'derived_from' in transformed:
                    metadata['derived_from'] = transformed.pop('derived_from')

        # Ensure updated_at field exists; allow null if unknown
        if 'updated_at' not in transformed:
            transformed['updated_at'] = transformed.get('created_at') or None

        # Normalize timestamp fields to ISO 8601 strings
        for ts_field in ('created_at', 'updated_at'):
            if ts_field in transformed:
                transformed[ts_field] = self._normalize_timestamp(transformed.get(ts_field))

        # Remove unused fields at top level (done early to ensure clean data)
        unused_top_level_fields = ['role', 'source', 'similarity_score', 'scope', 'distance']
        for field in unused_top_level_fields:
            transformed.pop(field, None)

        # Handle metadata cleanup
        if 'metadata' in transformed and isinstance(transformed['metadata'], dict):
            metadata = transformed['metadata']

            # Remove unused metadata fields
            unused_metadata_fields = ['level', 'retrieval', 'source']
            for field in unused_metadata_fields:
                metadata.pop(field, None)

        # Ensure metadata timestamps (if any) are normalized as well
        if 'metadata' in transformed and isinstance(transformed['metadata'], dict):
            for ts_field in ('created_at', 'updated_at'):
                if ts_field in transformed['metadata']:
                    transformed['metadata'][ts_field] = self._normalize_timestamp(
                        transformed['metadata'].get(ts_field)
                    )

        return transformed

    @staticmethod
    def _normalize_timestamp(value: Any) -> Optional[str]:
        """Convert various timestamp representations to ISO 8601 strings."""
        if value is None:
            return None

        if isinstance(value, datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            return value.isoformat()

        if isinstance(value, (int, float)):
            try:
                return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()
            except (OverflowError, OSError, ValueError):
                logger.warning(f"QueryResponseProcessor: Invalid timestamp value {value}")
                return None

        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            try:
                numeric = float(stripped)
                return datetime.fromtimestamp(numeric, tz=timezone.utc).isoformat()
            except (ValueError, OverflowError, OSError):
                # Assume already ISO formatted or acceptable string
                return stripped

        try:
            return str(value)
        except Exception:
            return None


class MetadataEnricher:
    """Enriches results with additional metadata."""
    
    def __init__(self, db_service=None):
        self.db_service = db_service
        self._session_cache: Dict[str, Dict[str, Optional[str]]] = {}
    
    async def transform(self, data: Any, context: RequestContext) -> Any:
        """Enrich results with metadata.
        
        Args:
            data: Response data
            context: Request context
            
        Returns:
            Data with enriched metadata
        """
        if isinstance(data, dict) and 'results' in data:
            for result in data['results']:
                await self._enrich_result_metadata(result, context)
        
        return data
    
    async def _enrich_result_metadata(self, result: Dict[str, Any], context: RequestContext):
        """Enrich a single result with metadata."""
        if 'metadata' not in result:
            result['metadata'] = {}

        metadata = result['metadata']

        # Ensure required fields are present
        if 'user_id' not in metadata or not metadata.get('user_id'):
            if context.user_id:
                metadata['user_id'] = str(context.user_id)

        # Resolve session_id from multiple sources
        result_session_id = metadata.get('session_id') or result.get('session_id') or context.session_id
        if result_session_id:
            metadata['session_id'] = str(result_session_id)

        # Fetch session info when possible
        session_info = await self._resolve_session_info(metadata.get('session_id'), context)

        # Ensure user_id aligns with session user if available
        if (not metadata.get('user_id')) and session_info and session_info.get('user_id'):
            metadata['user_id'] = str(session_info.get('user_id'))

        # Ensure agent_id is populated from context or session info
        if not metadata.get('agent_id'):
            if context.agent_id:
                metadata['agent_id'] = str(context.agent_id)
            elif session_info and session_info.get('agent_id'):
                metadata['agent_id'] = str(session_info.get('agent_id'))

        # Ensure session_name reflects the originating session
        session_name = metadata.get('session_name')
        if not session_name:
            if session_info and session_info.get('name'):
                metadata['session_name'] = session_info.get('name')
            elif metadata.get('session_id') == context.session_id and context.session_name:
                metadata['session_name'] = context.session_name
            else:
                metadata['session_name'] = None
        else:
            metadata['session_name'] = str(session_name)

        # Guarantee presence of key fields even when null
        metadata.setdefault('session_id', None)
        metadata.setdefault('agent_id', None)
        metadata.setdefault('session_name', None)
        if 'user_id' not in metadata or metadata['user_id'] is None:
            metadata['user_id'] = str(context.user_id) if context.user_id else None

        # Add task and mode from request metadata if available
        if context.request_metadata:
            if 'task' in context.request_metadata:
                metadata['task'] = context.request_metadata['task']
            if 'mode' in context.request_metadata:
                metadata['mode'] = context.request_metadata['mode']

    async def _resolve_session_info(
        self,
        session_id: Optional[str],
        context: RequestContext
    ) -> Optional[Dict[str, Optional[str]]]:
        """Resolve session metadata (name, agent, user) via cache or database."""
        if not session_id:
            return None

        session_id_str = str(session_id)

        if session_id_str in self._session_cache:
            return self._session_cache[session_id_str]

        # Prefer context information when session matches request context
        if session_id_str == (context.session_id or ""):
            info = {
                "name": context.session_name,
                "agent_id": context.agent_id,
                "user_id": context.user_id
            }
            self._session_cache[session_id_str] = info
            return info

        if not self.db_service:
            return None

        try:
            session = await self.db_service.get_session(session_id_str)
        except Exception as exc:
            logger.debug(f"MetadataEnricher: Failed to load session {session_id_str}: {exc}")
            return None

        if not session:
            return None

        info = {
            "name": session.get("name"),
            "agent_id": session.get("agent_id"),
            "user_id": session.get("user_id")
        }
        self._session_cache[session_id_str] = info
        return info


class ScopeCalculator:
    """Calculates scope field based on session context."""
    
    def transform(self, data: Any, context: RequestContext) -> Any:
        """Calculate and set scope for each result.
        
        Args:
            data: Response data
            context: Request context
            
        Returns:
            Data with scope fields set
        """
        if isinstance(data, dict) and 'results' in data:
            for result in data['results']:
                self._calculate_scope(result, context)
        
        return data
    
    def _calculate_scope(self, result: Dict[str, Any], context: RequestContext):
        """Calculate scope for a single result."""
        if 'metadata' not in result:
            result['metadata'] = {}

        metadata = result['metadata']

        # Get result's session_id from multiple possible locations
        # Priority: metadata.session_id > top-level session_id
        result_session_id = (
            metadata.get('session_id') or
            result.get('session_id')
        )

        if context.session_id:
            # Session ID was provided in request
            if result_session_id == context.session_id:
                metadata['scope'] = 'in_session'
            elif result_session_id and result_session_id != context.session_id:
                metadata['scope'] = 'cross_session'
            else:
                metadata['scope'] = None
        else:
            # No session ID in request - scope should be null
            metadata['scope'] = None


class FieldRemover:
    """Removes unused fields from responses."""
    
    def __init__(self, fields_to_remove: List[str]):
        self.fields_to_remove = fields_to_remove
    
    def transform(self, data: Any, context: RequestContext) -> Any:
        """Remove specified fields from response.
        
        Args:
            data: Response data
            context: Request context
            
        Returns:
            Data with fields removed
        """
        if isinstance(data, dict) and 'results' in data:
            for result in data['results']:
                self._remove_fields(result)
        
        return data
    
    def _remove_fields(self, result: Dict[str, Any]):
        """Remove fields from a single result."""
        for field in self.fields_to_remove:
            if '.' in field:
                # Handle nested fields like 'metadata.level'
                parts = field.split('.')
                current = result
                for part in parts[:-1]:
                    if part in current and isinstance(current[part], dict):
                        current = current[part]
                    else:
                        break
                else:
                    current.pop(parts[-1], None)
            else:
                result.pop(field, None)
