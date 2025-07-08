# Sessions API

This document describes the Sessions API endpoints for MemFuse, following the RESTful design principles and resource identification patterns.

## Overview

Sessions represent conversations between users and agents in MemFuse. Each session contains a series of messages and serves as a context boundary for conversations. Sessions link users and agents together, enabling organized conversation management and memory retrieval.

## Resource Identification Pattern

The Sessions API follows MemFuse's standard resource identification pattern:

* **Query Operations**: Resources can be queried by ID or name
  * By ID: `GET /api/v1/sessions/{session_id}`
  * By name: `GET /api/v1/sessions?name={name}`
  * With filters: `GET /api/v1/sessions?user_id={user_id}&agent_id={agent_id}`
* **Create Operations**: Names can be provided when creating sessions (auto-generated if not provided)
* **Update/Delete Operations**: Must use IDs, not names

## Endpoints

### List Sessions

```
GET /api/v1/sessions
```

Lists all sessions accessible to the authenticated API key.

**Response:**

```json
{
  "status": "success",
  "code": 200,
  "data": {
    "sessions": [
      {
        "id": "session-123",
        "user_id": "user-456",
        "agent_id": "agent-789",
        "name": "conversation-about-weather",
        "created_at": "2023-01-01T12:00:00Z",
        "updated_at": "2023-01-01T12:00:00Z"
      }
    ]
  },
  "message": "Sessions retrieved successfully",
  "errors": null
}
```

### Get Session by Name

```
GET /api/v1/sessions?name={name}
```

Retrieves a session by its name. This is primarily used for getting the session ID from a known name.

**Parameters:**

* `name` (string, required): The session's name

**Response:**

```json
{
  "status": "success",
  "code": 200,
  "data": {
    "sessions": [
      {
        "id": "session-123",
        "user_id": "user-456",
        "agent_id": "agent-789",
        "name": "conversation-about-weather",
        "created_at": "2023-01-01T12:00:00Z",
        "updated_at": "2023-01-01T12:00:00Z"
      }
    ]
  },
  "message": "Session retrieved successfully",
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
      "field": "name",
      "message": "Session with name 'conversation-about-weather' not found"
    }
  ]
}
```

### List Sessions with Filters

```
GET /api/v1/sessions?user_id={user_id}&agent_id={agent_id}
```

Lists sessions filtered by user and/or agent. Both parameters are optional and can be used independently.

**Parameters:**

* `user_id` (string, optional): Filter sessions by user ID
* `agent_id` (string, optional): Filter sessions by agent ID

**Response:**

```json
{
  "status": "success",
  "code": 200,
  "data": {
    "sessions": [
      {
        "id": "session-123",
        "user_id": "user-456",
        "agent_id": "agent-789",
        "name": "conversation-about-weather",
        "created_at": "2023-01-01T12:00:00Z",
        "updated_at": "2023-01-01T12:00:00Z"
      }
    ]
  },
  "message": "Sessions retrieved successfully",
  "errors": null
}
```

### Create Session

```
POST /api/v1/sessions
```

Creates a new session between a user and an agent. The session name is optional and will be auto-generated if not provided.

**Request Body:**

```json
{
  "user_id": "user-456",
  "agent_id": "agent-789",
  "name": "conversation-about-weather"
}
```

**Parameters:**

* `user_id` (string, required): ID of the user participating in the session
* `agent_id` (string, required): ID of the agent participating in the session
* `name` (string, optional): Name for the session (auto-generated if not provided)

**Response:**

```json
{
  "status": "success",
  "code": 201,
  "data": {
    "session": {
      "id": "session-123",
      "user_id": "user-456",
      "agent_id": "agent-789",
      "name": "conversation-about-weather",
      "created_at": "2023-01-01T12:00:00Z",
      "updated_at": "2023-01-01T12:00:00Z"
    }
  },
  "message": "Session created successfully",
  "errors": null
}
```

**Error Response (Invalid User/Agent):**

```json
{
  "status": "error",
  "code": 400,
  "data": null,
  "message": "Invalid user or agent",
  "errors": [
    {
      "field": "user_id",
      "message": "User with ID 'user-456' not found"
    }
  ]
}
```

### Get Session by ID

```
GET /api/v1/sessions/{session_id}
```

Retrieves a specific session by its ID.

**Parameters:**

* `session_id` (string, required): The session's unique ID

**Response:**

```json
{
  "status": "success",
  "code": 200,
  "data": {
    "session": {
      "id": "session-123",
      "user_id": "user-456",
      "agent_id": "agent-789",
      "name": "conversation-about-weather",
      "created_at": "2023-01-01T12:00:00Z",
      "updated_at": "2023-01-01T12:00:00Z"
    }
  },
  "message": "Session retrieved successfully",
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
      "message": "Session with ID 'session-123' not found"
    }
  ]
}
```

### Update Session

```
PUT /api/v1/sessions/{session_id}
```

Updates an existing session. Only the session ID can be used for updates, not the name.

**Parameters:**

* `session_id` (string, required): The session's unique ID

**Request Body:**

```json
{
  "name": "updated-conversation-name"
}
```

**Request Parameters:**

* `name` (string, optional): New name for the session

**Response (Success):**

```json
{
  "status": "success",
  "code": 200,
  "data": {
    "session": {
      "id": "session-123",
      "user_id": "user-456",
      "agent_id": "agent-789",
      "name": "updated-conversation-name",
      "created_at": "2023-01-01T12:00:00Z",
      "updated_at": "2023-01-01T12:30:00Z"
    }
  },
  "message": "Session updated successfully",
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
      "message": "Session with ID 'session-123' not found"
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
  "message": "Failed to update session",
  "errors": [
    {
      "field": "general",
      "message": "Database update failed"
    }
  ]
}
```

### Delete Session

```
DELETE /api/v1/sessions/{session_id}
```

Deletes a session and all associated messages.

**Parameters:**

* `session_id` (string, required): The session's unique ID

**Response (Success):**

```json
{
  "status": "success",
  "code": 200,
  "data": {
    "session_id": "session-123"
  },
  "message": "Session deleted successfully",
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
      "message": "Session with ID 'session-123' not found"
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
  "message": "Failed to delete session",
  "errors": [
    {
      "field": "general",
      "message": "Database delete failed"
    }
  ]
}
```

## Error Handling

The Sessions API uses standard HTTP status codes:

* `200 OK`: Successful operation
* `201 Created`: Session created successfully
* `400 Bad Request`: Invalid request parameters (invalid user/agent IDs)
* `401 Unauthorized`: Missing or invalid API key
* `404 Not Found`: Session not found
* `500 Internal Server Error`: Server error

All error responses include detailed error messages in the `errors` field following the standard MemFuse API response format.

## Session Names

Session names have the following characteristics:

* Names are optional when creating sessions (auto-generated if not provided)
* Auto-generated names follow a pattern like "session-{timestamp}" or similar
* Names should be descriptive of the conversation topic
* Names can be updated after session creation