# Users API

This document describes the Users API endpoints for MemFuse, following the RESTful design principles and resource identification patterns.

## Overview

Users are the primary entities in MemFuse that own sessions, knowledge, and API keys. The Users API provides endpoints for managing user accounts and their associated resources.

## Resource Identification Pattern

The Users API follows MemFuse's standard resource identification pattern:

* **Query Operations**: Resources can be queried by ID or name

  * By ID: `GET /api/v1/users/{user_id}`
  * By name: `GET /api/v1/users?name={name}`
* **Create Operations**: Names must be provided when creating users
* **Update/Delete Operations**: Must use IDs, not names

## Endpoints

### List Users

```
GET /api/v1/users
```

Lists all users accessible to the authenticated API key.

**Response:**

```json
{
  "status": "success",
  "code": 200,
  "data": {
    "users": [
      {
        "id": "user-123",
        "name": "example-user",
        "description": "Example user description",
        "created_at": "2023-01-01T12:00:00Z",
        "updated_at": "2023-01-01T12:00:00Z"
      }
    ]
  },
  "message": "Users retrieved successfully",
  "errors": null
}
```

### Get User by Name

```
GET /api/v1/users?name={name}
```

Retrieves a user by their name. This is primarily used for getting the user ID from a known name.

**Parameters:**

* `name` (string, required): The user's name

**Response:**

```json
{
  "status": "success",
  "code": 200,
  "data": {
    "users": [
      {
        "id": "user-123",
        "name": "example-user",
        "description": "Example user description",
        "created_at": "2023-01-01T12:00:00Z",
        "updated_at": "2023-01-01T12:00:00Z"
      }
    ]
  },
  "message": "User retrieved successfully",
  "errors": null
}
```

**Error Response (User Not Found):**

```json
{
  "status": "error",
  "code": 404,
  "data": null,
  "message": "User not found",
  "errors": [
    {
      "field": "name",
      "message": "User with name 'example-user' not found"
    }
  ]
}
```

### Create User

```
POST /api/v1/users
```

Creates a new user. The user name must be unique.

**Request Body:**

```json
{
  "name": "example-user",
  "description": "Example user description"
}
```

**Parameters:**

* `name` (string, required): Unique name for the user
* `description` (string, optional): User description

**Response:**

```json
{
  "status": "success",
  "code": 201,
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

**Error Response (Duplicate Name):**

```json
{
  "status": "error",
  "code": 400,
  "data": null,
  "message": "User name already exists",
  "errors": [
    {
      "field": "name",
      "message": "User with name 'example-user' already exists"
    }
  ]
}
```

### Get User by ID

```
GET /api/v1/users/{user_id}
```

Retrieves a specific user by their ID.

**Parameters:**

* `user_id` (string, required): The user's unique ID

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
  "message": "User retrieved successfully",
  "errors": null
}
```

**Error Response (User Not Found):**

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

### Update User

```
PUT /api/v1/users/{user_id}
```

Updates an existing user. Only the user ID can be used for updates, not the name.

**Parameters:**

* `user_id` (string, required): The user's unique ID

**Request Body:**

```json
{
  "name": "new-user-name",
  "description": "Updated user description"
}
```

**Request Parameters:**

* `name` (string, optional): New name for the user
* `description` (string, optional): New description for the user

**Response (Success):**

```json
{
  "status": "success",
  "code": 200,
  "data": {
    "user": {
      "id": "user-123",
      "name": "new-user-name",
      "description": "Updated user description",
      "created_at": "2023-01-01T12:00:00Z",
      "updated_at": "2023-01-01T12:30:00Z"
    }
  },
  "message": "User updated successfully",
  "errors": null
}
```

**Error Response (User Not Found):**

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

**Error Response (Update Failed):**

```json
{
  "status": "error",
  "code": 500,
  "data": null,
  "message": "Failed to update user",
  "errors": [
    {
      "field": "general",
      "message": "Database update failed"
    }
  ]
}
```

### Delete User

```
DELETE /api/v1/users/{user_id}
```

Deletes a user and all associated data (sessions, knowledge, API keys).

**Parameters:**

* `user_id` (string, required): The user's unique ID

**Response (Success):**

```json
{
  "status": "success",
  "code": 200,
  "data": {
    "user_id": "user-123"
  },
  "message": "User deleted successfully",
  "errors": null
}
```

**Error Response (User Not Found):**

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

**Error Response (Delete Failed):**

```json
{
  "status": "error",
  "code": 500,
  "data": null,
  "message": "Failed to delete user",
  "errors": [
    {
      "field": "general",
      "message": "Database delete failed"
    }
  ]
}
```

## Error Handling

The Users API uses standard HTTP status codes:

* `200 OK`: Successful operation
* `201 Created`: User created successfully
* `400 Bad Request`: Invalid request parameters or duplicate name
* `401 Unauthorized`: Missing or invalid API key
* `404 Not Found`: User not found
* `500 Internal Server Error`: Server error

All error responses include detailed error messages in the `errors` field following the standard MemFuse API response format.


### Name Validation

User names must meet these criteria:

* Must be unique across all users
* Cannot be empty or null
* Should follow naming conventions (alphanumeric, hyphens, underscores)
