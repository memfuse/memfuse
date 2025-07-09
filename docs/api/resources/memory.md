# Memory Query API

This document describes the Memory Query API endpoint for MemFuse, which allows querying memory across all sessions for a user.

## Overview

The Memory Query API provides powerful search capabilities across a user's entire memory space, including messages from all sessions and knowledge items. It supports advanced features like scope tagging, multi-store querying, and intelligent result deduplication.

## Key Features

- **Cross-session querying**: Search across all sessions for a user
- **Scope tagging**: Results are tagged as `in_session` or `cross_session` when a session context is provided
- **Multi-store support**: Query vector, keyword, and graph stores
- **Intelligent deduplication**: Handles results from both buffer (in-memory) and persistent storage
- **Flexible filtering**: Include/exclude messages and knowledge items
- **Agent filtering**: Optionally filter by specific agent

## Endpoint

### Query User Memory

```
POST /api/v1/users/{user_id}/query
```

Queries memory across all sessions for a user. This endpoint searches through messages and knowledge items, returning ranked results based on relevance.

**Parameters:**

- `user_id` (string, required): The user's unique ID

**Request Body:**

```json
{
  "query": "search terms",
  "session_id": "session-123",
  "agent_id": "agent-456",
  "top_k": 10,
  "store_type": "vector",
  "include_messages": true,
  "include_knowledge": true
}
```

**Request Parameters:**

- `query` (string, required): The search query text
- `session_id` (string, optional): Session ID for scope tagging. When provided, results are tagged as `in_session` or `cross_session`
- `agent_id` (string, optional): Filter results to sessions with this agent
- `top_k` (integer, optional): Maximum number of results to return. Default: 5
- `store_type` (string, optional): Type of store to query. Options: `vector`, `keyword`, `graph`, `hybrid`. Default: searches all stores
- `include_messages` (boolean, optional): Whether to include message results. Default: true
- `include_knowledge` (boolean, optional): Whether to include knowledge results. Default: true

## Response Format

### Success Response

```json
{
  "status": "success",
  "code": 200,
  "data": {
    "results": [
      {
        "id": "msg-123",
        "content": "This is a message content",
        "score": 0.95,
        "type": "message",
        "role": "user",
        "created_at": "2023-01-01T12:00:00Z",
        "updated_at": "2023-01-01T12:00:00Z",
        "metadata": {
          "user_id": "user-123",
          "agent_id": "agent-456",
          "session_id": "session-789",
          "session_name": "My Session",
          "scope": "in_session",
          "level": 0,
          "retrieval": {
            "source": "vector_store",
            "similarity": 0.95
          }
        }
      },
      {
        "id": "knowledge-456",
        "content": "This is a knowledge item",
        "score": 0.88,
        "type": "knowledge",
        "role": null,
        "created_at": "2023-01-01T10:00:00Z",
        "updated_at": "2023-01-01T10:00:00Z",
        "metadata": {
          "user_id": "user-123",
          "agent_id": null,
          "session_id": null,
          "session_name": null,
          "scope": null,
          "level": 0,
          "retrieval": {
            "source": "keyword_store",
            "similarity": 0.88
          }
        }
      }
    ],
    "total": 2
  },
  "message": "Found 2 results",
  "errors": null
}
```

### Result Object Schema

Each result in the `results` array contains:

- `id` (string): Unique identifier for the result
- `content` (string): The actual content/text
- `score` (number): Relevance score (0.0 to 1.0)
- `type` (string): Result type - `message` or `knowledge`
- `role` (string|null): For messages: `user`, `assistant`, etc. For knowledge: `null`
- `created_at` (string|null): ISO timestamp when created
- `updated_at` (string|null): ISO timestamp when last updated
- `metadata` (object): Additional metadata about the result

### Metadata Schema

The metadata object contains:

- `user_id` (string): ID of the user who owns this result
- `agent_id` (string|null): ID of the agent (for messages) or `null` (for knowledge)
- `session_id` (string|null): ID of the session (for messages) or `null` (for knowledge)
- `session_name` (string|null): Name of the session (for messages) or `null` (for knowledge)
- `scope` (string|null): Scope tag when `session_id` is provided in request:
  - `"in_session"`: Result belongs to the specified session
  - `"cross_session"`: Result belongs to a different session
  - `null`: No session context provided in request
- `level` (integer): Hierarchical level (currently always 0)
- `retrieval` (object): Information about how this result was retrieved:
  - `source` (string): Source store (`vector_store`, `keyword_store`, `graph_store`, `write_buffer`, `speculative_buffer`, `query_buffer`, `hybrid_buffer`)
  - `similarity` (number): Similarity score from the retrieval algorithm

## Store Types

The `store_type` parameter allows targeting specific storage backends:

- `vector`: Search using vector/semantic similarity
- `keyword`: Search using keyword/lexical matching
- `graph`: Search using graph-based relationships
- `hybrid`: Search across multiple store types (default behavior)

## Scope Behavior

When `session_id` is provided in the request:

- Results from the specified session are tagged with `scope: "in_session"`
- Results from other sessions are tagged with `scope: "cross_session"`
- Knowledge items always have `scope: null` (not session-specific)

When `session_id` is not provided:

- All results have `scope: null`
- Search covers all sessions equally

## Error Responses

### User Not Found

```json
{
  "status": "error",
  "code": 404,
  "data": null,
  "message": "User not found",
  "errors": [
    {
      "field": "user_id",
      "message": "User with ID 'user-123' not found"
    }
  ]
}
```

### Session Not Found

```json
{
  "status": "error",
  "code": 404,
  "data": null,
  "message": "Session 'session-123' not found for user 'user-456'",
  "errors": [
    {
      "field": "session_id",
      "message": "Session 'session-123' not found for user 'user-456'"
    }
  ]
}
```

### Agent Not Found

```json
{
  "status": "error",
  "code": 404,
  "data": null,
  "message": "Agent 'agent-123' not found",
  "errors": [
    {
      "field": "agent_id",
      "message": "Agent 'agent-123' not found"
    }
  ]
}
```

### Invalid Request

```json
{
  "status": "error",
  "code": 400,
  "data": null,
  "message": "Invalid request parameters",
  "errors": [
    {
      "field": "query",
      "message": "Query parameter is required"
    }
  ]
}
```

## Examples

### Basic Query

Query all memory for a user:

```bash
curl -X POST "https://api.memfuse.com/api/v1/users/user-123/query" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "query": "machine learning concepts",
    "top_k": 5
  }'
```

### Query with Session Context

Query with session context for scope tagging:

```bash
curl -X POST "https://api.memfuse.com/api/v1/users/user-123/query" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "query": "database optimization",
    "session_id": "session-456",
    "top_k": 10
  }'
```

### Query Specific Store Type

Query only vector store:

```bash
curl -X POST "https://api.memfuse.com/api/v1/users/user-123/query" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "query": "neural networks",
    "store_type": "vector",
    "top_k": 15
  }'
```

### Query with Agent Filter

Query sessions from a specific agent:

```bash
curl -X POST "https://api.memfuse.com/api/v1/users/user-123/query" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "query": "code review feedback",
    "agent_id": "agent-789",
    "top_k": 8
  }'
```

### Query Only Knowledge Items

Query only knowledge items (exclude messages):

```bash
curl -X POST "https://api.memfuse.com/api/v1/users/user-123/query" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "query": "API documentation",
    "include_messages": false,
    "include_knowledge": true,
    "top_k": 20
  }'
```

## Performance Considerations

- **Result Deduplication**: The system automatically deduplicates results from buffer (in-memory) and persistent storage
- **Cross-session Search**: Queries all sessions for comprehensive results but may be slower for users with many sessions
- **Store Type Selection**: Specifying a `store_type` can improve performance by avoiding unnecessary searches
- **Top-K Limiting**: Use appropriate `top_k` values to balance result quality with response time

## Integration with Other APIs

The Memory Query API complements other MemFuse APIs:

- **Sessions API**: Use session IDs from the Sessions API for scope tagging
- **Agents API**: Use agent IDs from the Agents API for filtering
- **Messages API**: Results may include messages that can be updated via the Messages API
- **Knowledge API**: Results may include knowledge items that can be managed via the Knowledge API

## Error Handling

The Memory Query API uses standard HTTP status codes:

- `200 OK`: Successful query with results
- `400 Bad Request`: Invalid request parameters
- `401 Unauthorized`: Missing or invalid API key
- `404 Not Found`: User, session, or agent not found
- `500 Internal Server Error`: Server error during query processing

All error responses follow the standard MemFuse API error format with detailed error information in the `errors` field.
