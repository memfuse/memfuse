# Knowledge API

This document describes the Knowledge API endpoints for MemFuse, following the RESTful design principles and resource identification patterns.

## Overview

Knowledge represents stored information content associated with specific users in MemFuse. Each knowledge item contains content that can be used for enriching conversations, providing context, or storing user-specific information. Knowledge items are managed per user and can be created, read, updated, and deleted through the API.

## Resource Identification Pattern

The Knowledge API follows MemFuse's standard resource identification pattern:

- **Query Operations**: Knowledge items are accessed within the context of a user
  - List knowledge: `GET /api/v1/users/{user_id}/knowledge`
  - Read knowledge by IDs: `POST /api/v1/users/{user_id}/knowledge/read`
- **Create Operations**: Knowledge items are added to users with content
- **Update/Delete Operations**: Must use knowledge IDs within user context

## Endpoints

### List Knowledge Items

```
GET /api/v1/users/{user_id}/knowledge
```

Lists all knowledge items for a specific user.

**Parameters:**

- `user_id` (string, required): The user's unique ID

**Response:**

```json
{
  "status": "success",
  "code": 200,
  "data": {
    "knowledge": [
      {
        "id": "knowledge-123",
        "user_id": "user-456",
        "content": "This is important information about the user's preferences.",
        "metadata": {},
        "created_at": "2023-01-01T12:00:00Z"
      },
      {
        "id": "knowledge-124",
        "user_id": "user-456",
        "content": "Another piece of knowledge stored for this user.",
        "metadata": {},
        "created_at": "2023-01-01T12:01:00Z"
      }
    ]
  },
  "message": "Knowledge items retrieved successfully",
  "errors": null
}
```

**Error Response (User Not Found):**

```json
{
  "status": "error",
  "code": 404,
  "data": null,
  "message": "User with ID 'user-456' not found",
  "errors": [
    {
      "field": "user_id",
      "message": "User with ID 'user-456' not found"
    }
  ]
}
```

### Add Knowledge Items

```
POST /api/v1/users/{user_id}/knowledge
```

Adds one or more knowledge items for a specific user.

**Parameters:**

- `user_id` (string, required): The user's unique ID

**Request Body:**

```json
{
  "knowledge": [
    "This is the first piece of knowledge to add.",
    "This is the second piece of knowledge to add."
  ]
}
```

**Request Parameters:**

- `knowledge` (array, required): Array of knowledge content strings to add

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

**Error Response (User Not Found):**

```json
{
  "status": "error",
  "code": 404,
  "data": null,
  "message": "User with ID 'user-456' not found",
  "errors": [
    {
      "field": "user_id",
      "message": "User with ID 'user-456' not found"
    }
  ]
}
```

### Read Knowledge Items by IDs

```
POST /api/v1/users/{user_id}/knowledge/read
```

Retrieves specific knowledge items by their IDs for a user. Only returns knowledge items that belong to the specified user.

**Parameters:**

- `user_id` (string, required): The user's unique ID

**Request Body:**

```json
{
  "knowledge_ids": ["knowledge-123", "knowledge-124", "knowledge-125"]
}
```

**Request Parameters:**

- `knowledge_ids` (array, required): Array of knowledge IDs to retrieve

**Response:**

```json
{
  "status": "success",
  "code": 200,
  "data": {
    "knowledge": [
      {
        "id": "knowledge-123",
        "user_id": "user-456",
        "content": "This is important information about the user's preferences.",
        "metadata": {},
        "created_at": "2023-01-01T12:00:00Z"
      },
      {
        "id": "knowledge-124",
        "user_id": "user-456",
        "content": "Another piece of knowledge stored for this user.",
        "metadata": {},
        "created_at": "2023-01-01T12:01:00Z"
      }
    ]
  },
  "message": "Knowledge items retrieved successfully",
  "errors": null
}
```

**Error Response (User Not Found):**

```json
{
  "status": "error",
  "code": 404,
  "data": null,
  "message": "User with ID 'user-456' not found",
  "errors": [
    {
      "field": "user_id",
      "message": "User with ID 'user-456' not found"
    }
  ]
}
```

### Update Knowledge Items

```
PUT /api/v1/users/{user_id}/knowledge
```

Updates existing knowledge items for a user. Only knowledge items that belong to the specified user can be updated.

**Parameters:**

- `user_id` (string, required): The user's unique ID

**Request Body:**

```json
{
  "knowledge_ids": ["knowledge-123", "knowledge-124"],
  "new_knowledge": [
    "This is the updated content for the first knowledge item.",
    "This is the updated content for the second knowledge item."
  ]
}
```

**Request Parameters:**

- `knowledge_ids` (array, required): Array of knowledge IDs to update
- `new_knowledge` (array, required): Array of new content strings (must match the length of knowledge_ids)

**Response:**

```json
{
  "status": "success",
  "code": 200,
  "data": {
    "knowledge_ids": ["knowledge-123", "knowledge-124"]
  },
  "message": "Knowledge items updated successfully",
  "errors": null
}
```

**Error Response (User Not Found):**

```json
{
  "status": "error",
  "code": 404,
  "data": null,
  "message": "User with ID 'user-456' not found",
  "errors": [
    {
      "field": "user_id",
      "message": "User with ID 'user-456' not found"
    }
  ]
}
```

### Delete Knowledge Items

```
DELETE /api/v1/users/{user_id}/knowledge
```

Deletes specific knowledge items for a user. Only knowledge items that belong to the specified user can be deleted.

**Parameters:**

- `user_id` (string, required): The user's unique ID

**Request Body:**

```json
{
  "knowledge_ids": ["knowledge-123", "knowledge-124"]
}
```

**Request Parameters:**

- `knowledge_ids` (array, required): Array of knowledge IDs to delete

**Response:**

```json
{
  "status": "success",
  "code": 200,
  "data": {
    "knowledge_ids": ["knowledge-123", "knowledge-124"]
  },
  "message": "Knowledge items deleted successfully",
  "errors": null
}
```

**Error Response (User Not Found):**

```json
{
  "status": "error",
  "code": 404,
  "data": null,
  "message": "User with ID 'user-456' not found",
  "errors": [
    {
      "field": "user_id",
      "message": "User with ID 'user-456' not found"
    }
  ]
}
```

## Error Responses

All Knowledge API endpoints return consistent error responses following the MemFuse API standards:

### General Error Response

```json
{
  "status": "error",
  "code": 500,
  "data": null,
  "message": "Failed to perform operation",
  "errors": [
    {
      "field": "general",
      "message": "Internal server error details"
    }
  ]
}
```

## Data Model

Knowledge items in MemFuse follow this data structure:

```json
{
  "id": "knowledge-123",
  "user_id": "user-456",
  "content": "The actual knowledge content as a string",
  "metadata": {},
  "created_at": "2023-01-01T12:00:00Z"
}
```

**Fields:**

- `id` (string): Unique identifier for the knowledge item
- `user_id` (string): Reference to the user who owns this knowledge
- `content` (string): The actual knowledge content
- `metadata` (object): Additional metadata (currently empty object)
- `created_at` (string): ISO 8601 timestamp of when the knowledge was created

## Rate Limiting

The Knowledge API endpoints are subject to rate limiting based on your API key's permissions and usage tier. Rate limit headers are included in all responses:

- `X-RateLimit-Limit`: Maximum requests per time window
- `X-RateLimit-Remaining`: Remaining requests in current window
- `X-RateLimit-Reset`: Time when the rate limit resets

## Usage Examples

### Adding Knowledge for a User

```bash
curl -X POST "https://api.memfuse.com/api/v1/users/user-456/knowledge" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "knowledge": [
      "User prefers technical documentation style",
      "User is interested in AI and machine learning topics"
    ]
  }'
```

### Retrieving All Knowledge for a User

```bash
curl -X GET "https://api.memfuse.com/api/v1/users/user-456/knowledge" \
  -H "X-API-Key: your-api-key"
```

### Reading Specific Knowledge Items

```bash
curl -X POST "https://api.memfuse.com/api/v1/users/user-456/knowledge/read" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "knowledge_ids": ["knowledge-123", "knowledge-124"]
  }'
```

### Updating Knowledge Items

```bash
curl -X PUT "https://api.memfuse.com/api/v1/users/user-456/knowledge" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "knowledge_ids": ["knowledge-123"],
    "new_knowledge": ["Updated: User now prefers concise documentation style"]
  }'
```

### Deleting Knowledge Items

```bash
curl -X DELETE "https://api.memfuse.com/api/v1/users/user-456/knowledge" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "knowledge_ids": ["knowledge-123", "knowledge-124"]
  }'
```
