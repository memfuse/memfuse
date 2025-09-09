# Gateway Architecture Implementation Summary

## Overview

This document summarizes the implementation of the new Gateway/Processor architecture for MemFuse API schema updates. The implementation addresses the requirements to update interface schemas and response formats while providing a scalable, extensible architecture.

## Recent Updates (2025-09-09)

### Advanced Features Completed

1. **Bidirectional Filter Pipeline**: Complete inbound and outbound filter system with configuration-driven loading
2. **Semantic Validation**: Advanced embedding-based content validation with similarity analysis and conflict detection
3. **Performance Caching**: Comprehensive caching system for filter operations, regex patterns, and content validation
4. **Prometheus Metrics**: Full observability with detailed metrics for monitoring filter performance and system health
5. **Enhanced Configuration**: Complete configuration system with environment-specific overrides and guardrail integration

## Requirements Addressed

### Request Schema Updates
- Added `agent_id` and `session_id` fields to request models
- Added `metadata` field for extensible request parameters
- Updated both `MemoryQuery` (core) and `QueryRequest` (client) models

### Response Schema Updates
- **Field Renaming**: `score` → `relevance_score`, `type` → `memory_type`
- **Field Removal**: Removed unused fields (`role`, `metadata.level`, `metadata.retrieval`, `metadata.source`)
- **Metadata Enhancement**: Added missing fields (`metadata.agent_id`, `metadata.session_id`, `metadata.session_name`)
- **Scope Calculation**: Proper `scope` field logic based on session context
- **Memory Type Support**: Different formats for M1 (episodic) vs M2 (semantic) memories

## Architecture Design

### Gateway Layer
The new architecture introduces a Gateway layer between API endpoints and services:

```
API Endpoint → Gateway → Service (Buffer/Memory) → Database
     ↑           ↑
   Request    Response
 Validation  Transformation
```

### Key Components

#### 1. Gateway Interface (`src/memfuse_core/gateway/interfaces.py`)
- `GatewayInterface`: Abstract base for gateway implementations
- `RequestContext`: Encapsulates request context (user, agent, session info)
- `QueryResponseProcessor`: Protocol for response processing components

#### 2. Memory Gateway (`src/memfuse_core/gateway/memory_gateway.py`)
- Main gateway implementation for memory operations
- Orchestrates request/response transformations
- Handles service delegation and error management

#### 3. Processors (`src/memfuse_core/gateway/processors.py`)
- `QueryRequestProcessor`: Enriches requests with context
- `QueryResponseProcessor`: Handles field renaming and memory type formatting
- `MetadataEnricher`: Adds missing metadata fields
- `ScopeCalculator`: Calculates scope based on session context
- `FieldRemover`: Removes unused fields

## Implementation Details

### Request Processing Flow
1. API endpoint receives request
2. Gateway creates `RequestContext` from request parameters
3. Request processors enrich the request data
4. Gateway delegates to appropriate service (Buffer/Memory)
5. Response processors process the service response
6. Standardized response returned to client

### Response Transformation Pipeline
1. **Field Renaming**: `score` → `relevance_score`, `type` → `memory_type`
2. **Memory Type Handling**:
   - **M1 (Episodic)**: Keep `content` field, set `memory_type` to "episodic"
   - **M2 (Semantic)**: Transform to `fact` structure, set `memory_type` to "semantic"
3. **Metadata Enrichment**: Add user, agent, session information
4. **Scope Calculation**: Set scope based on session context
5. **Field Cleanup**: Remove unused fields

### Scope Logic
- **`in_session`**: Result's session_id matches request's session_id
- **`cross_session`**: Result has different session_id than request
- **`null`**: No session_id in request OR no session_id in result

## Files Modified/Created

### New Files
- `src/memfuse_core/gateway/__init__.py`
- `src/memfuse_core/gateway/interfaces.py`
- `src/memfuse_core/gateway/memory_gateway.py`
- `src/memfuse_core/gateway/processors.py`
- `tests/unit/test_gateway.py`
- `tests/integration/test_new_schema.py`

### Modified Files
- `src/memfuse_core/models/api.py`: Added metadata field to MemoryQuery
- `src/memfuse/models/requests.py`: Added agent_id, session_id, metadata to QueryRequest
- `src/memfuse_core/api/users.py`: Simplified to use Gateway instead of complex transformation logic

## Example Response Formats

### M1 Episodic Memory (with session_id in request)
```json
{
  "status": "success",
  "code": 200,
  "data": {
    "query": "What did you know about environmental conservation?",
    "results": [
      {
        "id": "00004017-625d-4adc-97f7-1b4f7cb7895f",
        "content": "[ASSISTANT]: Yes, Baha'i communities have been involved...",
        "relevance_score": 0.7356168329715745,
        "memory_type": "episodic",
        "created_at": "2025-09-02T13:52:46.552383+00:00",
        "updated_at": null,
        "metadata": {
          "user_id": "8a0d893f-21bd-450c-b337-735645324a6a",
          "agent_id": "5a3a351c-18a1-9996-af50-149089234ca",
          "session_id": "1bcd351c-88b8-4f56-af50-14908955edcb",
          "session_name": "some-session-name",
          "scope": "in_session",
          "task": null,
          "mode": null
        }
      }
    ],
    "total": 1
  },
  "message": "Found 1 results"
}
```

### M2 Semantic Memory (no session_id in request)
```json
{
  "status": "success",
  "code": 200,
  "data": {
    "query": "What's the status of the 'auth-refactor' ticket?",
    "results": [
      {
        "id": "m2_jira_ticket_753",
        "memory_type": "semantic",
        "relevance_score": 0.98,
        "created_at": "2025-09-02T13:52:46.552383+00:00",
        "updated_at": null,
        "fact": {
          "text": "The goal is to update the legacy authentication service...",
          "triples": null
        },
        "metadata": {
          "user_id": "8a0d893f-21bd-450c-b337-735645324a6a",
          "agent_id": "5a3a351c-18a1-9996-af50-149089234ca",
          "session_id": "2209aabc-c2a6-4b76-9530-f4ef7f57a521",
          "session_name": "some-session-name",
          "scope": null,
          "derived_from": ["m1_conversation_log_102", "some_other_source_memory_id"],
          "task": null,
          "mode": null
        }
      }
    ],
    "total": 1
  },
  "message": "Found 1 results"
}
```

## Benefits

### 1. Separation of Concerns
- API layer focuses on request validation and routing
- Gateway handles transformation logic
- Services focus on business logic

### 2. Extensibility
- Easy to add new processors for additional metadata fields
- Pluggable processor architecture
- Support for different memory types (M1, M2, future M3)

### 3. Maintainability
- Centralized transformation logic
- Clear interfaces and abstractions
- Comprehensive test coverage

### 4. Backward Compatibility
- Existing services continue to work unchanged
- Gradual migration path
- Fallback mechanisms for error handling

## Testing

### Unit Tests (`tests/unit/test_gateway.py`)
- Gateway functionality
- Individual transformer behavior
- Context creation and enrichment

### Integration Tests (`tests/integration/test_new_schema.py`)
- End-to-end schema validation
- Response format verification
- Scope calculation scenarios

## Future Extensibility

## Bidirectional Filter Pipeline

Gateway supports a comprehensive, config-driven bidirectional filtering system that processes both incoming requests (inbound filters) and outgoing responses (outbound filters). This provides flexible content processing without changing core service logic.

### Filter Execution Flow

```text
Client Request → Inbound Filters → Service Processing → Outbound Filters → Guardrail Validation → Client Response
```

### Inbound Filters (Request Processing)

Applied to incoming requests before service processing:

- **Request Validator**: Validates request structure and required fields
- **Rate Limiter**: Enforces request rate limits per user/session
- **Input Sanitizer**: Cleans and normalizes input content
- **Content Filter**: Detects and handles sensitive content in requests

### Outbound Filters (Response Processing)

Applied to responses after service processing but before guardrail validation:

- **Field Remover**: Removes internal/debug fields from responses
- **Max Length**: Truncates content beyond configured limits
- **Sensitive Word**: Masks or removes sensitive content
- **Composite Content**: Advanced multi-dimensional content validation
- **Metadata Enricher**: Adds processing metadata to responses

### Advanced Features

#### Semantic Validation
- **Similarity Detection**: Identifies duplicate or highly similar content
- **Relevance Scoring**: Validates contextual relevance of responses
- **Coherence Analysis**: Checks internal consistency of content
- **Conflict Detection**: Identifies contradictory statements

#### Performance Caching
- **Regex Pattern Cache**: Caches compiled regular expressions
- **Content Hash Cache**: Caches validation results for identical content
- **Quality Score Cache**: Caches expensive quality assessments

#### Observability
- **Prometheus Metrics**: Comprehensive metrics for filter performance
- **Distributed Tracing**: OpenTelemetry integration for request tracking
- **Debug Information**: Optional detailed timing and execution stats

### Configuration

Filters are configured via `config/gateway/pipeline.yaml`:

```yaml
gateway:
  inbound_filters:
    - name: "request_validator"
      enabled: true
      params:
        max_query_length: 10000
        required_fields: ["query"]

    - name: "rate_limiter"
      enabled: true
      params:
        requests_per_minute: 100
        per_user: true

  outbound_filters:
    - name: "sensitive_word"
      enabled: true
      params:
        words: ["password", "secret", "token"]
        action: "mask"
        mask_token: "[REDACTED]"

    - name: "max_length"
      enabled: true
      params:
        max_content_length: 5000
        action: "truncate"
```

### Environment-Specific Configuration

Different environments can have different filter configurations:

```yaml
# Development - more permissive
development:
  gateway:
    inbound_filters:
      - name: "rate_limiter"
        enabled: false  # No rate limiting in dev

# Production - more restrictive
production:
  gateway:
    outbound_filters:
      - name: "sensitive_word"
        params:
          action: "drop"  # More aggressive in production
```

See also: `docs/architecture/outbound_filters.md` for detailed examples and best practices.

### Configuration example

```yaml
gateway:
  pipeline:
    inbound: []
    outbound:
      - name: max_length
        enabled: true
      - name: sensitive_word
        enabled: true

guardrail:
  length:
    enabled: true
    max_content_length: 120
    suffix: "..."
  sensitive:
    enabled: true
    words: ["forbidden", "secret"]
    mask_token: "[SENSITIVE]"
    case_insensitive: true
```

Notes:
- Filters are ordered; max_length runs before sensitive_word in this example.
- Both filters are no-ops unless corresponding `guardrail.*.enabled` is true.
- Additional filter behaviors may be added incrementally (e.g., metadata recursion, different actions).

The Gateway architecture is designed to easily accommodate future requirements:

1. **New Metadata Fields**: Add new processors to the pipeline
2. **Additional Memory Types**: Extend QueryResponseProcessor
3. **Custom Business Logic**: Implement new transformer classes
4. **Service Integration**: Add new service types to Gateway

## Migration Notes

- The old complex transformation logic in `users.py` has been replaced with clean Gateway calls
- Services continue to return their existing formats
- All transformation happens in the Gateway layer
- No breaking changes to existing service interfaces
