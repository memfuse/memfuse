# Messages API

This document describes the Messages API endpoints for MemFuse, following the RESTful design principles and resource identification patterns.

## Overview

Messages represent individual communications within sessions between users and agents. Each message has a role (user, assistant, or system) and content. Messages are the primary way conversation data is stored and retrieved in MemFuse.

## Resource Identification Pattern

The Messages API follows MemFuse's standard resource identification pattern:

- **Query Operations**: Messages are accessed within the context of a session
  - List messages: `GET /api/v1/sessions/{session_id}/messages`
  - Read messages by IDs: `POST /api/v1/sessions/{session_id}/messages/read`
- **Create Operations**: Messages are added to sessions with role and content
- **Update/Delete Operations**: Must use message IDs within session context

## Endpoints

### List Messages in Session

```
GET /api/v1/sessions/{session_id}/messages
```

Lists all messages in a specific session with optional pagination, sorting, and filtering.

**Parameters:**

- `session_id` (string, required): The session's unique ID

**Query Parameters:**

- `limit` (integer, optional): Maximum number of messages to return (default: 20, max: 100)
- `sort_by` (string, optional): Field to sort messages by (default: "timestamp", allowed values: "timestamp", "id")
- `order` (string, optional): Sort order (default: "desc", allowed values: "asc", "desc")
- `buffer_only` (boolean, optional): Buffer filtering - if "true", only return RoundBuffer data; if "false" or omitted, return data from all sources (RoundBuffer + HybridBuffer + Database)

**Example URLs:**

- `GET /api/v1/sessions/session-456/messages` - Get latest 20 messages
- `GET /api/v1/sessions/session-456/messages?limit=10&order=asc` - Get oldest 10 messages
- `GET /api/v1/sessions/session-456/messages?sort_by=id&order=desc&limit=50` - Get latest 50 messages sorted by ID
- `GET /api/v1/sessions/session-456/messages?buffer_only=true` - Get messages from round buffer only

**Response:**

The API **always** returns a consistent response format regardless of the underlying service implementation (MemoryService, BufferService, or database fallback).

```json
{
  "status": "success",
  "code": 200,
  "data": {
    "messages": [
      {
        "id": "message-123",
        "session_id": "session-456",
        "role": "user",
        "content": "Hello, how are you?",
        "created_at": "2023-01-01T12:00:00Z",
        "updated_at": "2023-01-01T12:00:00Z",
        "metadata": {
          "user_id": "user-789",
          "agent_id": "agent-101"
        }
      },
      {
        "id": "message-124",
        "session_id": "session-456",
        "role": "assistant",
        "content": "I'm doing well, thank you for asking!",
        "created_at": "2023-01-01T12:01:00Z",
        "updated_at": "2023-01-01T12:01:00Z",
        "metadata": {
          "user_id": "user-789",
          "agent_id": "agent-101"
        }
      }
    ]
  },
  "message": "Messages retrieved successfully",
  "errors": null
}
```

**Response Format Guarantees:**

- `data.messages` is **always** an array (`List[Dict[str, Any]]`)
- Each message object **always** contains the required fields: `id`, `role`, `content`, `created_at`, `updated_at`
- The `metadata` field may contain additional context information depending on the source
- Empty results return `data.messages: []` (empty array), never `null` or other types
- Response format is consistent across all service implementations and buffer configurations

**Buffer Behavior:**

The `buffer_only` parameter controls data source selection and behaves differently based on buffer configuration:

#### When Buffer is Enabled (`buffer.enabled=true`)

- **`buffer_only=true`**: Only returns messages from RoundBuffer (latest, in-memory data)
  - Fastest response time
  - Most recent messages only
  - May miss older messages that have been flushed to HybridBuffer or Database

- **`buffer_only=false` or omitted**: Returns messages from all sources with intelligent merging:
  1. **RoundBuffer** (highest priority) - Latest, in-memory data
  2. **HybridBuffer** (medium priority) - Intermediate cached data
  3. **Database** (lowest priority) - Persisted data
  4. Messages are merged and deduplicated by ID, with RoundBuffer taking precedence

#### When Buffer is Disabled (`buffer.enabled=false`)

- **`buffer_only` parameter is ignored** - All requests query the database directly through MemoryService
- Provides consistent behavior regardless of buffer_only value
- Ensures backward compatibility when switching between buffer modes

## API Contract Consistency

The list messages endpoint maintains strict API contract consistency to ensure reliable client integration:

### Response Format Guarantees

1. **Consistent Data Structure**: The response always follows the same JSON schema regardless of:
   - Underlying service implementation (MemoryService vs BufferService)
   - Buffer configuration (enabled/disabled)
   - Data source (RoundBuffer, HybridBuffer, or Database)
   - Error conditions or empty results

2. **Type Safety**:
   - `data.messages` is always an array, never a dictionary or other type
   - Each message is always a dictionary with consistent field types
   - Empty results return `[]`, never `null` or undefined

3. **Field Consistency**:
   - Required fields (`id`, `role`, `content`, `created_at`, `updated_at`) are always present
   - Field types are consistent across all messages
   - Optional fields like `metadata` may vary but maintain consistent structure when present

4. **Error Handling**:
   - Invalid service responses are normalized to empty arrays rather than propagating errors
   - Malformed data is filtered out to maintain response integrity
   - Client code can rely on consistent structure even in edge cases

### Breaking Change Prevention

This API endpoint is designed to prevent breaking changes that could affect client applications:

- **No Format Variations**: Clients never need to handle different response formats
- **Backward Compatibility**: New fields may be added but existing structure remains stable
- **Graceful Degradation**: Service failures result in empty arrays, not broken responses

**Error Response (Session Not Found):**

```json
{
  "status": "error",
  "code": 404,
  "data": null,
  "message": "Session not found",
  "errors": [
    {
      "field": "session_id",
      "message": "Session with ID 'session-456' not found"
    }
  ]
}
```

**Error Response (Invalid Parameters):**

```json
{
  "status": "error",
  "code": 400,
  "data": null,
  "message": "Invalid limit parameter",
  "errors": [
    {
      "field": "limit",
      "message": "Limit must be an integer"
    }
  ]
}
```

**Error Response (Invalid Sort Field):**

```json
{
  "status": "error",
  "code": 400,
  "data": null,
  "message": "Invalid sort_by parameter",
  "errors": [
    {
      "field": "sort_by",
      "message": "sort_by must be one of: timestamp, id"
    }
  ]
}
```

**Error Response (Invalid Order):**

```json
{
  "status": "error",
  "code": 400,
  "data": null,
  "message": "Invalid order parameter",
  "errors": [
    {
      "field": "order",
      "message": "order must be one of: asc, desc"
    }
  ]
}
```

**Error Response (Invalid Buffer Parameter):**

```json
{
  "status": "error",
  "code": 400,
  "data": null,
  "message": "Invalid buffer_only parameter",
  "errors": [
    {
      "field": "buffer_only",
      "message": "buffer_only must be 'true' or 'false'"
    }
  ]
}
```

### Read Messages by IDs

```
POST /api/v1/sessions/{session_id}/messages/read
```

Retrieves multiple specific messages by their IDs within a session context. This endpoint is useful for bulk retrieval of messages when you know their IDs.

**Parameters:**

- `session_id` (string, required): The session's unique ID

**Request Body:**

```json
{
  "message_ids": ["message-123", "message-124", "message-125"]
}
```

**Request Parameters:**

- `message_ids` (array, required): Array of message IDs to retrieve

**Response:**

```json
{
  "status": "success",
  "code": 200,
  "data": {
    "messages": [
      {
        "id": "message-123",
        "session_id": "session-456",
        "role": "user",
        "content": "Hello, how are you?",
        "created_at": "2023-01-01T12:00:00Z",
        "updated_at": "2023-01-01T12:00:00Z"
      },
      {
        "id": "message-124",
        "session_id": "session-456",
        "role": "assistant",
        "content": "I'm doing well, thank you for asking!",
        "created_at": "2023-01-01T12:01:00Z",
        "updated_at": "2023-01-01T12:01:00Z"
      },
      {
        "id": "message-125",
        "session_id": "session-456",
        "role": "user",
        "content": "What's the weather like today?",
        "created_at": "2023-01-01T12:02:00Z",
        "updated_at": "2023-01-01T12:02:00Z"
      }
    ]
  },
  "message": "Messages read successfully",
  "errors": null
}
```

**Error Response (Session Not Found):**

```json
{
  "status": "error",
  "code": 404,
  "data": null,
  "message": "Session not found",
  "errors": [
    {
      "field": "session_id",
      "message": "Session with ID 'session-456' not found"
    }
  ]
}
```

**Error Response (Some Messages Not Found):**

```json
{
  "status": "error",
  "code": 404,
  "data": null,
  "message": "Some message IDs were not found",
  "errors": [
    {
      "field": "message_ids",
      "message": "Messages with IDs ['message-999'] not found in session 'session-456'"
    }
  ]
}
```

**Error Response (Service Unavailable):**

```json
{
  "status": "error",
  "code": 500,
  "data": null,
  "message": "Failed to get service",
  "errors": [
    {
      "field": "general",
      "message": "Memory or buffer service unavailable"
    }
  ]
}
```

### Add Messages to Session

```
POST /api/v1/sessions/{session_id}/messages
```

Adds one or more messages to a session. This endpoint accepts an array of messages to support adding conversation exchanges in a single request.

**Parameters:**

- `session_id` (string, required): The session's unique ID

**Request Body:**

```json
{
  "messages": [
    {
      "role": "user",
      "content": "What's the weather like today?"
    },
    {
      "role": "assistant",
      "content": "I don't have access to current weather data, but I'd be happy to help you find weather information!"
    }
  ]
}
```

**Request Parameters:**

- `messages` (array, required): Array of message objects to add
  - `role` (string, required): Message role - "user", "assistant", or "system"
  - `content` (string, required): Message content text

**Response:**

```json
{
  "status": "success",
  "code": 201,
  "data": {
    "message_ids": ["message-125", "message-126"]
  },
  "message": "Messages added successfully",
  "errors": null
}
```

**Error Response (Invalid Session):**

```json
{
  "status": "error",
  "code": 404,
  "data": null,
  "message": "Session not found",
  "errors": [
    {
      "field": "session_id",
      "message": "Session with ID 'session-456' not found"
    }
  ]
}
```

**Error Response (Invalid Message Data):**

```json
{
  "status": "error",
  "code": 400,
  "data": null,
  "message": "Invalid message data",
  "errors": [
    {
      "field": "role",
      "message": "Role must be 'user', 'assistant', or 'system'"
    },
    {
      "field": "content",
      "message": "Content cannot be empty"
    }
  ]
}
```

### Update Message

```
PUT /api/v1/sessions/{session_id}/messages/{message_id}
```

Updates an existing message. Only the message content can be updated, not the role.

**Parameters:**

- `session_id` (string, required): The session's unique ID
- `message_id` (string, required): The message's unique ID

**Request Body:**

```json
{
  "content": "Updated message content"
}
```

**Request Parameters:**

- `content` (string, required): New content for the message

**Response (Success):**

```json
{
  "status": "success",
  "code": 200,
  "data": {
    "message": {
      "id": "message-123",
      "session_id": "session-456",
      "role": "user",
      "content": "Updated message content",
      "created_at": "2023-01-01T12:00:00Z",
      "updated_at": "2023-01-01T12:30:00Z"
    }
  },
  "message": "Message updated successfully",
  "errors": null
}
```

**Error Response (Message Not Found):**

```json
{
  "status": "error",
  "code": 404,
  "data": null,
  "message": "Message not found",
  "errors": [
    {
      "field": "message_id",
      "message": "Message with ID 'message-123' not found in session 'session-456'"
    }
  ]
}
```

**Error Response (Update Failed):**

```json
{
  "status": "error",
  "code": 500,
  "data": null,
  "message": "Failed to update message",
  "errors": [
    {
      "field": "general",
      "message": "Database update failed"
    }
  ]
}
```

### Delete Message

```
DELETE /api/v1/sessions/{session_id}/messages/{message_id}
```

Deletes a specific message from a session.

**Parameters:**

- `session_id` (string, required): The session's unique ID
- `message_id` (string, required): The message's unique ID

**Response (Success):**

```json
{
  "status": "success",
  "code": 200,
  "data": {
    "message_id": "message-123"
  },
  "message": "Message deleted successfully",
  "errors": null
}
```

**Error Response (Message Not Found):**

```json
{
  "status": "error",
  "code": 404,
  "data": null,
  "message": "Message not found",
  "errors": [
    {
      "field": "message_id",
      "message": "Message with ID 'message-123' not found in session 'session-456'"
    }
  ]
}
```

**Error Response (Delete Failed):**

```json
{
  "status": "error",
  "code": 500,
  "data": null,
  "message": "Failed to delete message",
  "errors": [
    {
      "field": "general",
      "message": "Database delete failed"
    }
  ]
}
```

## Error Handling

The Messages API uses standard HTTP status codes:

- `200 OK`: Successful operation
- `201 Created`: Messages added successfully
- `400 Bad Request`: Invalid request parameters (invalid role, empty content)
- `401 Unauthorized`: Missing or invalid API key
- `404 Not Found`: Session or message not found
- `500 Internal Server Error`: Server error

All error responses include detailed error messages in the `errors` field following the standard MemFuse API response format.

## Message Roles

Messages support three roles:

- **user**: Messages from the human user
- **assistant**: Messages from the AI agent/assistant
- **system**: System messages for context or instructions

## Message Content

Message content characteristics:

- Content is stored as plain text
- Content cannot be empty or null
- Content is indexed for memory search and retrieval
- Content can be updated after creation (role cannot be changed)
