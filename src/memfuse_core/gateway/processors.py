"""Response processors for the gateway layer."""

from typing import Any, Dict, List
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
                'triples': None  # Could be populated later if available
            }
            # Remove content field for semantic memories
            transformed.pop('content', None)
            # Normalize memory_type to 'semantic'
            transformed['memory_type'] = 'semantic'

        # Ensure updated_at field exists; allow null if unknown
        if 'updated_at' not in transformed:
            transformed['updated_at'] = transformed.get('created_at') or None

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

        return transformed


class MetadataEnricher:
    """Enriches results with additional metadata."""
    
    def __init__(self, db_service=None):
        self.db_service = db_service
    
    def transform(self, data: Any, context: RequestContext) -> Any:
        """Enrich results with metadata.
        
        Args:
            data: Response data
            context: Request context
            
        Returns:
            Data with enriched metadata
        """
        if isinstance(data, dict) and 'results' in data:
            for result in data['results']:
                self._enrich_result_metadata(result, context)
        
        return data
    
    def _enrich_result_metadata(self, result: Dict[str, Any], context: RequestContext):
        """Enrich a single result with metadata."""
        if 'metadata' not in result:
            result['metadata'] = {}

        metadata = result['metadata']

        # Ensure required fields are present
        if 'user_id' not in metadata and context.user_id:
            metadata['user_id'] = context.user_id

        if 'agent_id' not in metadata and context.agent_id:
            metadata['agent_id'] = context.agent_id

        if 'session_id' not in metadata and context.session_id:
            metadata['session_id'] = context.session_id

        if 'session_name' not in metadata and context.session_name:
            metadata['session_name'] = context.session_name

        # Add task and mode from request metadata if available
        if context.request_metadata:
            if 'task' in context.request_metadata:
                metadata['task'] = context.request_metadata['task']
            if 'mode' in context.request_metadata:
                metadata['mode'] = context.request_metadata['mode']


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
