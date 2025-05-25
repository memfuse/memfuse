# MemFuse RESTful API Design

This document outlines the comprehensive RESTful API design for MemFuse, following industry best practices and implementing proper authentication and security validation mechanisms.

## Table of Contents

1. [Introduction](#introduction)
2. [Design Principles](#design-principles)
   - [Resource Identification Pattern](#resource-identification-pattern)
3. [Authentication](#authentication)
4. [API Structure](#api-structure)
5. [Resource Endpoints](#resource-endpoints)
6. [Request/Response Formats](#requestresponse-formats)
7. [Error Handling](#error-handling)
8. [Query Operations](#query-operations)
9. [Implementation Considerations](#implementation-considerations)
10. [Client SDK Updates](#client-sdk-updates)
    - [Client API Implementation](#client-api-implementation)
11. [Migration Strategy](#migration-strategy)
12. [Client Usage Pattern](#client-usage-pattern)
13. [Conclusion](#conclusion)

## Introduction

MemFuse currently uses a session-centric API design where all operations require a session ID. The client SDK requires an API key, but the server doesn't validate it. This document proposes a more robust API design that follows industry best practices, implements proper authentication, and shifts from a session-centric to a user-centric approach.

## Design Principles

The new API design follows these principles:

1. **RESTful Resource Modeling**: Treat entities (users, agents, sessions) as resources with standard CRUD operations
2. **User-Centric Design**: Shift from session-centric to user-centric design
3. **Proper Authentication**: Implement API key validation and security best practices
4. **Consistent Response Format**: Standardize error and success responses
5. **Clear Resource Relationships**: Respect the many-to-many relationships between users and agents
6. **Flexible Query Operations**: Support querying across sessions or within specific sessions
7. **Resource Identification Pattern**: Use IDs for all operations except GET queries, which can use names

### Resource Identification Pattern

MemFuse API follows a specific pattern for resource identification:

1. **Resource Identification**:

   - Each resource (users, agents, sessions) has a unique ID and a human-readable name
   - IDs are system-generated unique identifiers that never change
   - Names are user-friendly identifiers that can be changed

2. **Query Operations**:

   - Resources can be queried by ID: `GET /api/v1/users/{user_id}`
   - Resources can be queried by name: `GET /api/v1/users?name={name}`
   - The purpose of queries is to retrieve complete resource information, including IDs

3. **Create Operations**:

   - When creating resources, names must be provided: `POST /api/v1/users` with `{"name": "example-user"}`
   - The system generates an ID and returns it in the response

4. **Update and Delete Operations**:
   - Update and delete operations **must** use IDs, not names
   - Update: `PUT /api/v1/users/{user_id}` with `{"name": "new-name"}`
   - Delete: `DELETE /api/v1/users/{user_id}`

This pattern ensures security, reliability, and consistency across all API operations.

## Authentication

All API requests will require authentication using API keys.

### API Key Authentication

```
Authorization: Bearer {api_key}
```

API keys will be associated with users and have specific permissions. The server will validate API keys for every request.

### API Key Generation

API keys can be generated through the API. Each API key is associated with a user and can have specific permissions.

```python
# Generate an API key for a user
api_key_id = db.create_api_key(
    user_id=user_id,
    name="API Key Name",
    permissions="read,write",  # Optional
    expires_at="2025-12-31T23:59:59"  # Optional
)

# The API key is stored in the database and can be retrieved
api_key_data = db.get_api_key(api_key_id)
```

### API Key Validation

The server validates API keys for every request using the following logic:

1. Extract the API key from the Authorization header
2. Check if the API key exists in the database
3. Check if the API key has expired
4. Return the API key data if valid, including the associated user ID

```python
def validate_api_key(key: str) -> Optional[Dict[str, Any]]:
    """Validate an API key.

    Args:
        key: API key

    Returns:
        API key data if valid, None otherwise
    """
    # Get the API key from the database
    api_key = get_api_key_by_key(key)

    if api_key is None:
        return None

    # Check if the API key has expired
    if api_key["expires_at"] is not None:
        expires_at = datetime.fromisoformat(api_key["expires_at"])
        if expires_at < datetime.now():
            return None

    return api_key
```

### Authentication Middleware

The authentication middleware extracts the API key from the Authorization header and validates it:

```python
# Authentication Middleware
from fastapi import Request, HTTPException, Depends
from fastapi.security import APIKeyHeader
from typing import Dict, Any, Optional

# API key header
API_KEY_HEADER = APIKeyHeader(name="Authorization", auto_error=False)

async def validate_api_key(api_key_header: str = Depends(API_KEY_HEADER)) -> Dict[str, Any]:
    """Validate the API key from the Authorization header.

    Args:
        api_key_header: The Authorization header value

    Returns:
        The API key data including user_id

    Raises:
        HTTPException: If the API key is invalid
    """
    if not api_key_header:
        raise HTTPException(
            status_code=401,
            detail="Missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Extract the API key from the header
    # Format: "Bearer {api_key}"
    parts = api_key_header.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=401,
            detail="Invalid API key format",
            headers={"WWW-Authenticate": "Bearer"},
        )

    api_key = parts[1]

    # Validate the API key
    db = Database()
    api_key_data = db.validate_api_key(api_key)

    if api_key_data is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return api_key_data
```

## API Structure

The API will be structured around resources with standard HTTP methods:

- `GET`: Retrieve resources
- `POST`: Create resources
- `PUT`: Update resources
- `DELETE`: Delete resources

### Base URL

```
/api/v1
```

## Resource Endpoints

### Users

```
GET    /users                  - List users
GET    /users?name={name}      - Get user by name
POST   /users                  - Create a new user
GET    /users/{user_id}        - Get user details by ID
PUT    /users/{user_id}        - Update user
DELETE /users/{user_id}        - Delete user
```

### API Keys

```
POST   /users/{user_id}/api-keys           - Create a new API key for a user
GET    /users/{user_id}/api-keys           - List all API keys for a user
DELETE /users/{user_id}/api-keys/{key_id}  - Delete an API key
```

Note: API Keys endpoints are implemented in a separate module (`api_keys.py`) but maintain the same URL structure for consistency.

### Agents

```
GET    /agents                 - List all agents
GET    /agents?name={name}     - Get agent by name
POST   /agents                 - Create a new agent
GET    /agents/{agent_id}      - Get agent details by ID
PUT    /agents/{agent_id}      - Update agent
DELETE /agents/{agent_id}      - Delete agent
```

### Sessions

```
GET    /sessions                                        - List all sessions
GET    /sessions?name={name}                            - Get session by name
GET    /sessions?user_id={user_id}&agent_id={agent_id}  - List sessions (filter by user and/or agent)
POST   /sessions                                        - Create a new session
GET    /sessions/{session_id}                           - Get session details by ID
PUT    /sessions/{session_id}                           - Update session
DELETE /sessions/{session_id}                           - Delete session
```

### Messages

```
GET    /sessions/{session_id}/messages                - List messages in a session
POST   /sessions/{session_id}/messages                - Add messages to a session
GET    /sessions/{session_id}/messages/{message_id}   - Get message details
PUT    /sessions/{session_id}/messages/{message_id}   - Update message
DELETE /sessions/{session_id}/messages/{message_id}   - Delete message
```

### Knowledge

```
GET    /users/{user_id}/knowledge                   - List knowledge items for a user
POST   /users/{user_id}/knowledge                   - Add knowledge items for a user
GET    /users/{user_id}/knowledge/{knowledge_id}    - Get knowledge item details
PUT    /users/{user_id}/knowledge/{knowledge_id}    - Update knowledge item
DELETE /users/{user_id}/knowledge/{knowledge_id}    - Delete knowledge item
```

### Memory Operations

```
POST   /users/{user_id}/query                       - Query memory across all sessions or within specific sessions for a user
```

## Request/Response Formats

### Standard Response Format

All API responses will follow this format:

```json
{
  "status": "success" | "error",
  "code": 200 | 400 | 401 | 403 | 404 | 500,
  "data": { ... } | null,
  "message": "Human-readable message",
  "errors": [
    {
      "field": "field_name",
      "message": "Error message"
    }
  ] | null
}
```

### Success Response Example

```json
{
  "status": "success",
  "code": 200,
  "data": {
    "user": {
      "id": "user-123",
      "name": "example-user",
      "description": "Example user description"
    }
  },
  "message": "User retrieved successfully",
  "errors": null
}
```

### Error Response Example

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

## API Examples

### Creating a User

**Request:**

```http
POST /api/v1/users HTTP/1.1
Host: localhost:8000
Authorization: Bearer your-api-key
Content-Type: application/json

{
  "name": "example-user",
  "description": "Example user description"
}
```

**Response:**

```json
{
  "status": "success",
  "code": 200,
  "data": {
    "user": {
      "id": "user-123",
      "name": "example-user",
      "description": "Example user description",
      "created_at": "2023-01-01T12:00:00Z",
      "updated_at": "2023-01-01T12:00:00Z"
    }
  },
  "message": "User created successfully",
  "errors": null
}
```

### Creating an API Key

**Request:**

```http
POST /api/v1/users/user-123/api-keys HTTP/1.1
Host: localhost:8000
Authorization: Bearer your-api-key
Content-Type: application/json

{
  "name": "Example API Key",
  "permissions": {
    "read": true,
    "write": true
  },
  "expires_at": "2025-12-31T23:59:59Z"
}
```

**Response:**

```json
{
  "status": "success",
  "code": 200,
  "data": {
    "api_key": {
      "id": "key-123",
      "user_id": "user-123",
      "key": "generated-api-key",
      "name": "Example API Key",
      "permissions": {
        "read": true,
        "write": true
      },
      "created_at": "2023-01-01T12:00:00Z",
      "expires_at": "2025-12-31T23:59:59Z"
    }
  },
  "message": "API key created successfully",
  "errors": null
}
```

### Creating a Session

**Request:**

```http
POST /api/v1/sessions HTTP/1.1
Host: localhost:8000
Authorization: Bearer your-api-key
Content-Type: application/json

{
  "user_id": "user-123",
  "agent_id": "agent-123",
  "name": "example-session"
}
```

**Response:**

```json
{
  "status": "success",
  "code": 200,
  "data": {
    "session": {
      "id": "session-123",
      "user_id": "user-123",
      "agent_id": "agent-123",
      "name": "example-session",
      "created_at": "2023-01-01T12:00:00Z",
      "updated_at": "2023-01-01T12:00:00Z"
    }
  },
  "message": "Session created successfully",
  "errors": null
}
```

### Adding Messages

**Request:**

```http
POST /api/v1/sessions/session-123/messages HTTP/1.1
Host: localhost:8000
Authorization: Bearer your-api-key
Content-Type: application/json

{
  "messages": [
    {
      "role": "user",
      "content": "Hello, how are you?"
    },
    {
      "role": "assistant",
      "content": "I'm doing well, thank you for asking!"
    }
  ]
}
```

**Response:**

```json
{
  "status": "success",
  "code": 200,
  "data": {
    "message_ids": ["message-123", "message-124"]
  },
  "message": "Messages added successfully",
  "errors": null
}
```

### Querying Memory

**Request:**

```http
POST /api/v1/users/user-123/query HTTP/1.1
Host: localhost:8000
Authorization: Bearer your-api-key
Content-Type: application/json

{
  "query": "How are you?",
  "top_k": 5,
  "store_type": "vector",
  "include_messages": true,
  "include_knowledge": true,
  "session_id": "session-123"
}
```

**Response:**

```json
{
  "status": "success",
  "code": 200,
  "data": {
    "results": [
      {
        "id": "message-123",
        "content": "Hello, how are you?",
        "score": 0.95,
        "type": "message",
        "role": "user",
        "created_at": "2023-01-01T12:00:00Z",
        "updated_at": "2023-01-01T12:00:00Z",
        "metadata": {
          "session_id": "session-123",
          "agent_id": "agent-123",
          "session_name": "example-session"
        }
      }
    ],
    "total": 1
  },
  "message": "Query successful",
  "errors": null
}
```

### Adding Knowledge

**Request:**

```http
POST /api/v1/users/user-123/knowledge HTTP/1.1
Host: localhost:8000
Authorization: Bearer your-api-key
Content-Type: application/json

{
  "knowledge": [
    "Python is a programming language.",
    "Python was created by Guido van Rossum."
  ]
}
```

**Response:**

```json
{
  "status": "success",
  "code": 200,
  "data": {
    "knowledge_ids": ["knowledge-123", "knowledge-124"]
  },
  "message": "Knowledge items added successfully",
  "errors": null
}
```

## Error Handling

The API will use standard HTTP status codes:

- `200 OK`: Successful request
- `201 Created`: Resource created successfully
- `400 Bad Request`: Invalid request parameters
- `401 Unauthorized`: Missing or invalid API key
- `403 Forbidden`: API key doesn't have permission
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server error

All error responses will include detailed error messages in the `errors` field.

## Query Operations

The query operation is a special case that needs to support both session-specific queries and queries across all sessions. This is one of the most important features of MemFuse, allowing users to retrieve relevant information from their memory.

### Query Endpoint

```
POST /api/v1/users/{user_id}/query
```

### Query Parameters

The query operation supports several parameters to customize the search:
| Parameter | Type | Required | Default | Description |
|----------------------|----------|----------|---------|--------------------------------------------------------------------------------------------------------------------------------------|
| **query** | string | Yes | – | The search query text. |
| **session_id** | string | No | null | If provided with a specific session ID, only queries that specific session. If null or omitted, queries across all sessions. |
| **agent_id** | string | No | null | If provided, only memories created by this agent are returned. |
| **top_k** | integer | No | 5 | Number of results to return. |
| **store_type** | string | No | null | Which store(s) to query: `"vector"`, `"graph"`, or `null` for both. |
| **include_messages** | boolean | No | true | Whether to include message‑type memories in the results. |
| **include_knowledge**| boolean | No | true | Whether to include knowledge‑type memories in the results. |

> **Note**: The parameter order in the API and client SDK is standardized as follows: `query`, `session_id`, `agent_id`, `top_k`, `store_type`, `include_messages`, `include_knowledge`. This order is maintained consistently across all interfaces to reduce confusion.

### Query Request Examples

**Example 1: Query across all sessions for a user**

No session_id → global user‑scoped query, metadata.scope: null

```http
POST /api/v1/users/user-123/query HTTP/1.1
Host: localhost:8000
Authorization: Bearer your-api-key
Content-Type: application/json

{
  "query": "What is the capital of France?",
  "top_k": 3,
  "include_messages": true,
  "include_knowledge": true
}
```

**Example 2: Query a specific session**

Returns both in‑session and cross‑session hits, each tagged in metadata.scope

```http
POST /api/v1/users/user-123/query HTTP/1.1
Host: localhost:8000
Authorization: Bearer your-api-key
Content-Type: application/json

{
  "query": "What is the capital of France?",
  "session_id": "session-456",
  "top_k": 5,
  "store_type": "vector"
}
```

**Example 3: Query sessions with a specific agent**

Filters by agent across all sessions (no session_id), metadata.scope: null

```http
POST /api/v1/users/user-123/query HTTP/1.1
Host: localhost:8000
Authorization: Bearer your-api-key
Content-Type: application/json

{
  "query": "What is the capital of France?",
  "agent_id": "agent-789",
  "include_knowledge": false
}
```

### Query Response Format

The query response includes a list of results, each with metadata about the source:

#### Response Fields

| Field          | Type                      | Description                                                        |
| -------------- | ------------------------- | ------------------------------------------------------------------ |
| **status**     | `string`                  | `"success"` or `"error"`.                                          |
| **code**       | `integer`                 | HTTP‑style status code: `200`, `400`, `500`.                       |
| **data**       | `object`                  | Payload container.                                                 |
| └─ **results** | `array`                   | List of matching memory items (`results[]` fields below).          |
| └─ **total**   | `integer`                 | Total number of hits (before paging/truncation).                   |
| **message**    | `string`                  | Human‑readable summary, e.g. `"Found 3 results"` or error detail.  |
| **errors**     | `null` or `array[string]` | If `status === "error"`, list of error messages; otherwise `null`. |

#### `results[]` fields

Each result in the response includes the following fields:

| Field      | Type   | Description                                    |
| ---------- | ------ | ---------------------------------------------- |
| id         | string | Unique identifier for the result               |
| content    | string | The content of the message or knowledge item   |
| score      | float  | Relevance score (0-1)                          |
| type       | string | Type of result: "message" or "knowledge"       |
| role       | string | For messages: "user", "assistant", or "system" |
| created_at | string | ISO timestamp of creation                      |
| updated_at | string | ISO timestamp of last update                   |
| metadata   | object | Additional metadata about the result           |

#### `metadata` sub‑fields

| Field            | Type                                          | Description                                                                                                                 |
| ---------------- | --------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| **user_id**      | `string`                                      | Echo of the `user` who owns this memory.                                                                                    |
| **agent_id**     | `string` or `null`                            | Creator agent’s ID, if any.                                                                                                 |
| **session_id**   | `string` or `null`                            | Session ID where this memory was recorded, if any.                                                                          |
| **session_name** | `string` or `null`                            | Human name of the session, if known.                                                                                        |
| **scope**        | `"in_session"` \| `"cross_session"` \| `null` | If request included `session_id`: `"in_session"` when equal, `"cross_session"` otherwise. If no session in request: `null`. |
| **level**        | `0` \| `1` \| `2`                             | Memory hierarchy level: 0=L0 (message), 1=L1 (semantic/episodic), 2=L2 (reflective).                                        |
| **retrieval**    | `object`                                      | Retrieval metadata.                                                                                                         |
| └─ **method**    | `"embedding"` \| `"keyword"` \| `"graph"`     | Which retrieval method matched this item.                                                                                   |

### Response Schema

```
{
  "status": "success" | "error",
  "code": 200 | 400 | 500,
  "data": {
    "results": [
      {
        "id": "string",
        "content": "string",
        "score": number,
        "type": "message" | "knowledge",
        "role": "user" | "assistant" | null,
        "created_at": "ISO8601 timestamp",
        "updated_at": "ISO8601 timestamp",
        "metadata": {
          "user_id": "string",
          "agent_id": "string|null",
          "session_id": "string|null",
          "session_name": "string|null",
          "scope": "in_session" | "cross_session" | null,
          "level": 0 | 1 | 2,
          "retrieval": {
            "method": "embedding" | "keyword" | "graph"
          }
          // future fields can live here without breaking clients
        }
      }
      // …more results…
    ],
    "total": number             // total hits before paging/truncation
  },
  "message": "string",
  "errors": null | [ "string", … ]
}
```

### Example Response

````json
{
  "status": "success",
  "code": 200,
  "data": {
    "results": [
      {
        "id": "msg-001",
        "content": "The capital of France is Paris.",
        "score": 0.95,
        "type": "message",
        "role": "assistant",
        "created_at": "2025-04-30T10:00:00Z",
        "updated_at": "2025-04-30T10:00:00Z",
        "metadata": {
          "user_id": "user-123",
          "agent_id": "agent-789",
          "session_id": "session-456",
          "session_name": "geography-session",
          "scope": "in_session",
          "level": 0,
          "retrieval": {
            "method": "embedding"
          }
        }
      },
      {
        "id": "kn-002",
        "content": "Paris is the capital and most populous city of France.",
        "score": 0.87,
        "type": "knowledge",
        "role": null,
        "created_at": "2025-04-25T09:30:00Z",
        "updated_at": "2025-04-25T09:30:00Z",
        "metadata": {
          "user_id": "user-123",
          "agent_id": "agent-789",
          "session_id": "session-999",
          "session_name": "history-session",
          "scope": "cross_session",
          "level": 1,
          "retrieval": {
            "method": "keyword"
          }
        }
      }
    ],
    "total": 2
  },
  "message": "Found 2 results",
  "errors": null
}

### Query Implementation

The query implementation follows these steps:

1. Accept a user ID as a path parameter
2. Accept query parameters in the request body
3. Validate that the user exists
4. Apply filters based on session_id and agent_id if provided
5. Execute the query against the appropriate stores (vector, graph, or both)
6. Return results with metadata about which session they came from
7. Sort results by relevance score

The query operation is optimized for performance, with a target response time of under 500ms for most queries.

```python
@router.post("/users/{user_id}/query", response_model=ApiResponse)
async def query_memory(
    user_id: str,
    request: QueryRequest,
    api_key: str = Depends(validate_api_key),
) -> ApiResponse:
    """Query memory across sessions for a user."""
    # Validate user exists
    user = await UserService.get_user(user_id)
    if not user:
        return ApiResponse.error(
            message=f"User {user_id} not found",
            code=404,
        )

    # Build query filters
    filters = {"user_id": user_id}

    if request.session_id:
        # Validate session exists and belongs to user
        session = await SessionService.get_session(request.session_id)
        if not session or session["user_id"] != user_id:
            return ApiResponse.error(
                message=f"Session {request.session_id} not found for user {user_id}",
                code=404,
            )
        filters["session_id"] = request.session_id

    if request.agent_id:
        # Validate agent exists
        agent = await AgentService.get_agent(request.agent_id)
        if not agent:
            return ApiResponse.error(
                message=f"Agent {request.agent_id} not found",
                code=404,
            )
        filters["agent_id"] = request.agent_id

    # Execute query with filters
    results = await MemoryService.query(
        query=request.query,
        filters=filters,
        top_k=request.top_k,
        store_type=request.store_type,
        include_messages=request.include_messages,
        include_knowledge=request.include_knowledge,
    )

    return ApiResponse.success(
        data={"results": results},
        message="Query successful",
    )
````

## Implementation Considerations

### Database Schema

The current database schema already supports the many-to-many relationships between users and agents:

- Users and agents are independent entities
- Sessions link users and agents (a session belongs to one user and one agent)
- Knowledge is associated with users only

This schema works well with the proposed API design.

### API Key Storage

API keys will be stored in a new table in the database:

```sql
CREATE TABLE IF NOT EXISTS api_keys (
    id TEXT PRIMARY KEY,
    user_id TEXT,
    key TEXT UNIQUE,
    name TEXT,
    permissions TEXT,  -- Stored as JSON string representing permissions object
    created_at TIMESTAMP,
    expires_at TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users (id)
)
```

### Rate Limiting

To protect the API from abuse, rate limiting will be implemented:

```python
from fastapi import Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
import time
from typing import Dict, Tuple

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, rate_limit_per_minute: int = 60):
        super().__init__(app)
        self.rate_limit = rate_limit_per_minute
        self.requests: Dict[str, Tuple[int, float]] = {}  # {ip: (count, reset_time)}

    async def dispatch(self, request: Request, call_next):
        ip = request.client.host
        now = time.time()

        # Check if IP exists and if the reset time has passed
        if ip in self.requests:
            count, reset_time = self.requests[ip]
            if now > reset_time:
                # Reset counter if the minute has passed
                self.requests[ip] = (1, now + 60)
            else:
                # Increment counter
                count += 1
                if count > self.rate_limit:
                    # Rate limit exceeded
                    return Response(
                        content='{"status":"error","code":429,"message":"Rate limit exceeded","errors":[{"field":"general","message":"Too many requests, please try again later"}]}',
                        status_code=429,
                        media_type="application/json"
                    )
                self.requests[ip] = (count, reset_time)
        else:
            # First request from this IP
            self.requests[ip] = (1, now + 60)

        return await call_next(request)
```

## Client SDK Updates

The client SDK has been updated to match the new API design, with a focus on the resource identification pattern:

```python
class MemFuseClient:
    """MemFuse client for interacting with the MemFuse server API."""

    def __init__(self, api_key: Optional[str] = None,
                 base_url: str = "http://localhost:8000"):
        """Initialize the MemFuse client.

        Args:
            api_key: API key for authentication (required)
            base_url: URL of the MemFuse server API
        """
        self.api_key = api_key or os.environ.get("MEMFUSE_API_KEY")
        if not self.api_key:
            raise ValueError("API key is required")
        self.base_url = base_url

        # Initialize API clients
        self.health = HealthApi(self)
        self.users = UsersApi(self)
        self.agents = AgentsApi(self)
        self.sessions = SessionsApi(self)
        self.knowledge = KnowledgeApi(self)
        self.messages = MessagesApi(self)
        self.api_keys = ApiKeysApi(self)

    async def init(
        self,
        user: str,
        agent: Optional[str] = None,
        session: Optional[str] = None,
    ) -> "ClientMemory":
        """Initialize a memory instance.

        Args:
            user: User name (required)
            agent: Agent name (optional)
            session: Session name (optional, will be auto-generated if not provided)

        Returns:
            ClientMemory: A client memory instance for the specified user, agent, and session
        """
        # Get or create user
        user_name = user
        user_response = await self.users.get_by_name(user_name)

        if user_response["status"] == "error" or not user_response.get("data", {}).get("users"):
            # User doesn't exist, create it
            user_response = await self.users.create(
                name=user_name,
                description="User created by MemFuse client"
            )
            user_id = user_response["data"]["user"]["id"]
        else:
            # User exists, get the first one
            user_id = user_response["data"]["users"][0]["id"]

        # Get or create agent
        agent_name = agent or "agent_default"
        agent_response = await self.agents.get_by_name(agent_name)

        if agent_response["status"] == "error" or not agent_response.get("data", {}).get("agents"):
            # Agent doesn't exist, create it
            agent_response = await self.agents.create(
                name=agent_name,
                description="Agent created by MemFuse client"
            )
            agent_id = agent_response["data"]["agent"]["id"]
        else:
            # Agent exists, get the first one
            agent_id = agent_response["data"]["agents"][0]["id"]

        # Check if session with the given name already exists
        session_name = session
        if session_name:
            session_response = await self.sessions.get_by_name(session_name)
            if session_response["status"] == "success" and session_response.get("data", {}).get("sessions"):
                # Session exists, get the first one
                session_data = session_response["data"]["sessions"][0]
                session_id = session_data["id"]
            else:
                # Session doesn't exist, create it
                session_response = await self.sessions.create(
                    user_id=user_id,
                    agent_id=agent_id,
                    name=session_name
                )
                session_data = session_response["data"]["session"]
                session_id = session_data["id"]
        else:
            # No session name provided, create a new session
            session_response = await self.sessions.create(
                user_id=user_id,
                agent_id=agent_id
            )
            session_data = session_response["data"]["session"]
            session_id = session_data["id"]
            session_name = session_data["name"]

        # Create ClientMemory with all necessary parameters
        memory = ClientMemory(
            client=self,
            session_id=session_id,
            user_id=user_id,
            agent_id=agent_id,
            user_name=user_name,
            agent_name=agent_name,
            session_name=session_name
        )

        return memory
```

### Client API Implementation

The client API implementation follows the resource identification pattern with specialized API classes for each resource type:

```python
class UsersApi:
    """Users API client for MemFuse."""

    async def get_by_name(self, name: str) -> Dict[str, Any]:
        """Get a user by name.

        Args:
            name: User name

        Returns:
            Response data
        """
        return await self.client._request("GET", f"/api/v1/users?name={name}")

    async def update(
        self, user_id: str, name: Optional[str] = None, description: Optional[str] = None
    ) -> Dict[str, Any]:
        """Update a user.

        Args:
            user_id: User ID (required)
            name: New user name (optional)
            description: New user description (optional)

        Returns:
            Response data
        """
        return await self.client._request(
            "PUT",
            f"/api/v1/users/{user_id}",
            {
                "name": name,
                "description": description,
            },
        )
```

This implementation ensures that:

1. GET operations can use names to retrieve IDs
2. All other operations (POST, PUT, DELETE) use IDs for resource identification
3. The client handles the translation between names and IDs transparently

## Migration Strategy

To migrate from the current API to the new API, we'll follow these steps:

1. **Implement New API Endpoints**: Add the new endpoints alongside the existing ones
2. **Update Client SDK**: Update the client SDK to use the new endpoints but maintain backward compatibility
3. **Deprecate Old Endpoints**: Mark the old endpoints as deprecated
4. **Remove Old Endpoints**: After a transition period, remove the old endpoints

During the transition period, both APIs will be available, allowing users to migrate at their own pace.

### Backward Compatibility

To maintain backward compatibility, the client SDK will:

1. Detect which API version is being used
2. Use the appropriate endpoints
3. Convert between the old and new parameter names

```python
async def init(
    self,
    user: str,
    agent: Optional[str] = None,
    session: Optional[str] = None,
    # Backward compatibility parameters
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> "ClientMemory":
    """Initialize a memory instance."""
    # Use new parameters if provided, fall back to old ones
    user_name = user or user_id
    agent_name = agent or agent_id
    session_name = session or session_id

    if not user_name:
        raise ValueError("User name is required")

    # Rest of the implementation...
```

## Client Usage Pattern

The client code should follow this pattern when using the API:

```python
# First, get the user ID by name (or create if it doesn't exist)
user_response = await client.users.get_by_name("example-user")
if user_response["status"] == "success" and user_response.get("data", {}).get("users"):
    user_id = user_response["data"]["users"][0]["id"]
else:
    # User doesn't exist, create it
    user_response = await client.users.create(name="example-user")
    user_id = user_response["data"]["user"]["id"]

# Then, use the ID for all subsequent operations
await client.users.update(user_id, name="new-name")
await client.users.delete(user_id)
```

This pattern ensures that:

1. Names are only used for initial lookup to get IDs
2. All subsequent operations use IDs for reliability and security
3. The client code is consistent with the API design principles

## Conclusion

This RESTful API design provides a robust, user-centric approach that follows industry best practices. It respects the many-to-many relationships between users and agents, implements proper authentication, and provides flexible query operations. The resource identification pattern ensures security and reliability by using IDs for all operations except GET queries, which can use names.

By implementing this design, MemFuse has a more intuitive, secure, and maintainable API that better serves its users' needs. The client SDK provides a clean interface to the API, handling the translation between names and IDs transparently.
