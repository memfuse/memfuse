# Agents API

This document describes the Agents API endpoints for MemFuse, following the RESTful design principles and resource identification patterns.

## Overview

Agents are entities in MemFuse that participate in conversations with users. They represent AI assistants, chatbots, or other conversational entities that can interact with users through sessions. The Agents API provides endpoints for managing agent definitions and their associated resources.

## Resource Identification Pattern

The Agents API follows MemFuse's standard resource identification pattern:

* **Query Operations**: Resources can be queried by ID or name
  * By ID: `GET /api/v1/agents/{agent_id}`
  * By name: `GET /api/v1/agents?name={name}`
* **Create Operations**: Names must be provided when creating agents
* **Update/Delete Operations**: Must use IDs, not names

## Endpoints

### List Agents

```
GET /api/v1/agents
```

Lists all agents accessible to the authenticated API key.

**Response:**

```json
{
  "status": "success",
  "code": 200,
  "data": {
    "agents": [
      {
        "id": "agent-123",
        "name": "helpful-assistant",
        "description": "A helpful AI assistant",
        "created_at": "2023-01-01T12:00:00Z",
        "updated_at": "2023-01-01T12:00:00Z"
      }
    ]
  },
  "message": "Agents retrieved successfully",
  "errors": null
}
```

### Get Agent by Name

```
GET /api/v1/agents?name={name}
```

Retrieves an agent by their name. This is primarily used for getting the agent ID from a known name.

**Parameters:**

* `name` (string, required): The agent's name

**Response:**

```json
{
  "status": "success",
  "code": 200,
  "data": {
    "agents": [
      {
        "id": "agent-123",
        "name": "helpful-assistant",
        "description": "A helpful AI assistant",
        "created_at": "2023-01-01T12:00:00Z",
        "updated_at": "2023-01-01T12:00:00Z"
      }
    ]
  },
  "message": "Agent retrieved successfully",
  "errors": null
}
```

**Error Response (Agent Not Found):**

```json
{
  "status": "error",
  "code": 404,
  "data": null,
  "message": "Agent not found",
  "errors": [
    {
      "field": "name",
      "message": "Agent with name 'helpful-assistant' not found"
    }
  ]
}
```

### Create Agent

```
POST /api/v1/agents
```

Creates a new agent. The agent name must be unique.

**Request Body:**

```json
{
  "name": "helpful-assistant",
  "description": "A helpful AI assistant"
}
```

**Parameters:**

* `name` (string, required): Unique name for the agent
* `description` (string, optional): Agent description

**Response:**

```json
{
  "status": "success",
  "code": 201,
  "data": {
    "agent": {
      "id": "agent-123",
      "name": "helpful-assistant",
      "description": "A helpful AI assistant",
      "created_at": "2023-01-01T12:00:00Z",
      "updated_at": "2023-01-01T12:00:00Z"
    }
  },
  "message": "Agent created successfully",
  "errors": null
}
```

**Error Response (Duplicate Name):**

```json
{
  "status": "error",
  "code": 400,
  "data": null,
  "message": "Agent name already exists",
  "errors": [
    {
      "field": "name",
      "message": "Agent with name 'helpful-assistant' already exists"
    }
  ]
}
```

### Get Agent by ID

```
GET /api/v1/agents/{agent_id}
```

Retrieves a specific agent by their ID.

**Parameters:**

* `agent_id` (string, required): The agent's unique ID

**Response:**

```json
{
  "status": "success",
  "code": 200,
  "data": {
    "agent": {
      "id": "agent-123",
      "name": "helpful-assistant",
      "description": "A helpful AI assistant",
      "created_at": "2023-01-01T12:00:00Z",
      "updated_at": "2023-01-01T12:00:00Z"
    }
  },
  "message": "Agent retrieved successfully",
  "errors": null
}
```

**Error Response (Agent Not Found):**

```json
{
  "status": "error",
  "code": 404,
  "data": null,
  "message": "Agent not found",
  "errors": [
    {
      "field": "agent_id",
      "message": "Agent with ID 'agent-123' not found"
    }
  ]
}
```

### Update Agent

```
PUT /api/v1/agents/{agent_id}
```

Updates an existing agent. Only the agent ID can be used for updates, not the name.

**Parameters:**

* `agent_id` (string, required): The agent's unique ID

**Request Body:**

```json
{
  "name": "updated-assistant",
  "description": "An updated helpful AI assistant"
}
```

**Request Parameters:**

* `name` (string, optional): New name for the agent
* `description` (string, optional): New description for the agent

**Response (Success):**

```json
{
  "status": "success",
  "code": 200,
  "data": {
    "agent": {
      "id": "agent-123",
      "name": "updated-assistant",
      "description": "An updated helpful AI assistant",
      "created_at": "2023-01-01T12:00:00Z",
      "updated_at": "2023-01-01T12:30:00Z"
    }
  },
  "message": "Agent updated successfully",
  "errors": null
}
```

**Error Response (Agent Not Found):**

```json
{
  "status": "error",
  "code": 404,
  "data": null,
  "message": "Agent not found",
  "errors": [
    {
      "field": "agent_id",
      "message": "Agent with ID 'agent-123' not found"
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
  "message": "Failed to update agent",
  "errors": [
    {
      "field": "general",
      "message": "Database update failed"
    }
  ]
}
```

### Delete Agent

```
DELETE /api/v1/agents/{agent_id}
```

Deletes an agent and all associated sessions. Note that deleting an agent will also remove all sessions where this agent participated.

**Parameters:**

* `agent_id` (string, required): The agent's unique ID

**Response (Success):**

```json
{
  "status": "success",
  "code": 200,
  "data": {
    "agent_id": "agent-123"
  },
  "message": "Agent deleted successfully",
  "errors": null
}
```

**Error Response (Agent Not Found):**

```json
{
  "status": "error",
  "code": 404,
  "data": null,
  "message": "Agent not found",
  "errors": [
    {
      "field": "agent_id",
      "message": "Agent with ID 'agent-123' not found"
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
  "message": "Failed to delete agent",
  "errors": [
    {
      "field": "general",
      "message": "Database delete failed"
    }
  ]
}
```

## Error Handling

The Agents API uses standard HTTP status codes:

* `200 OK`: Successful operation
* `201 Created`: Agent created successfully
* `400 Bad Request`: Invalid request parameters or duplicate name
* `401 Unauthorized`: Missing or invalid API key
* `404 Not Found`: Agent not found
* `500 Internal Server Error`: Server error

All error responses include detailed error messages in the `errors` field following the standard MemFuse API response format.

### Name Validation

Agent names must meet these criteria:

* Must be unique across all agents
* Cannot be empty or null
* Should follow naming conventions (alphanumeric, hyphens, underscores)

### Default Agent

Many MemFuse implementations use a default agent (often named "agent_default") that is automatically created when needed. This agent serves as a fallback for sessions that don't specify a particular agent.