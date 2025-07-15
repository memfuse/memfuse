# Contract Testing Layer Documentation

## Overview

The contract testing layer validates the **API contract** for the MemFuse application. These tests focus on ensuring correct HTTP status codes, response schemas, and API behavior match the documented specification, independent of the underlying business logic implementation.

## Testing Philosophy

Contract tests are designed to:

- **Validate API contracts** - Ensure responses match documented schemas
- **Test HTTP semantics** - Verify correct status codes and headers
- **Isolate API layer** - Use mocks to focus on contract validation
- **Prevent API breaking changes** - Catch schema violations early
- **Document expected behavior** - Serve as living documentation

## Test Infrastructure

### Mock Services

- **Database Service Mock**: In-memory storage for consistent testing
- **Memory Service Mock**: Simulated memory operations
- **API Key Authentication**: Test-specific API keys

### Schema Validation

- **JSON Schema**: Strict validation of response structures
- **Type Checking**: Ensures correct data types
- **Required Fields**: Validates mandatory response fields
- **Additional Properties**: Prevents schema bloat

### Test Client

- **FastAPI TestClient**: Direct API testing
- **Custom Extensions**: Support for special HTTP methods

## API Endpoints Testing Status

### ✅ Users API (`/api/v1/users`)

**Status**: Comprehensive coverage  
**File**: `test_users_api_contract.py`

| Operation        | Endpoint                         | Test Method                              | Status |
| ---------------- | -------------------------------- | ---------------------------------------- | ------ |
| List Users       | `GET /api/v1/users`              | `test_list_users_success_contract`       | ✅     |
| Get User by Name | `GET /api/v1/users/name/{name}`  | `test_get_user_by_name_success_contract` | ✅     |
| Create User      | `POST /api/v1/users`             | `test_create_user_success_contract`      | ✅     |
| Get User by ID   | `GET /api/v1/users/{user_id}`    | `test_get_user_by_id_success_contract`   | ✅     |
| Update User      | `PUT /api/v1/users/{user_id}`    | `test_update_user_success_contract`      | ✅     |
| Delete User      | `DELETE /api/v1/users/{user_id}` | `test_delete_user_success_contract`      | ✅     |

**Error Cases Tested**:

- User not found (404)
- Duplicate user names (409)
- Invalid JSON payload (400)
- Missing API key (401)

---

### ✅ Agents API (`/api/v1/agents`)

**Status**: Comprehensive coverage  
**File**: `test_agents_api_contract.py`

| Operation         | Endpoint                           | Test Method                               | Status |
| ----------------- | ---------------------------------- | ----------------------------------------- | ------ |
| List Agents       | `GET /api/v1/agents`               | `test_list_agents_success_contract`       | ✅     |
| Get Agent by Name | `GET /api/v1/agents/name/{name}`   | `test_get_agent_by_name_success_contract` | ✅     |
| Create Agent      | `POST /api/v1/agents`              | `test_create_agent_success_contract`      | ✅     |
| Get Agent by ID   | `GET /api/v1/agents/{agent_id}`    | `test_get_agent_by_id_success_contract`   | ✅     |
| Update Agent      | `PUT /api/v1/agents/{agent_id}`    | `test_update_agent_success_contract`      | ✅     |
| Delete Agent      | `DELETE /api/v1/agents/{agent_id}` | `test_delete_agent_success_contract`      | ✅     |

**Error Cases Tested**:

- Agent not found (404)
- Duplicate agent names (409)
- Invalid JSON payload (400)

---

### ✅ Sessions API (`/api/v1/sessions`)

**Status**: Comprehensive coverage  
**File**: `test_sessions_api_contract.py`

| Operation           | Endpoint                               | Test Method                                 | Status |
| ------------------- | -------------------------------------- | ------------------------------------------- | ------ |
| List Sessions       | `GET /api/v1/sessions`                 | `test_list_sessions_success_contract`       | ✅     |
| Create Session      | `POST /api/v1/sessions`                | `test_create_session_success_contract`      | ✅     |
| Get Session by ID   | `GET /api/v1/sessions/{session_id}`    | `test_get_session_by_id_success_contract`   | ✅     |
| Get Session by Name | `GET /api/v1/sessions/name/{name}`     | `test_get_session_by_name_success_contract` | ✅     |
| Update Session      | `PUT /api/v1/sessions/{session_id}`    | `test_update_session_success_contract`      | ✅     |
| Delete Session      | `DELETE /api/v1/sessions/{session_id}` | `test_delete_session_success_contract`      | ✅     |

**Error Cases Tested**:

- Session not found (404)
- Invalid user/agent references (400)
- Auto-generated session names
- Session filtering

---

### ✅ Messages API (`/api/v1/sessions/{session_id}/messages`)

**Status**: Comprehensive coverage  
**File**: `test_messages_api_contract.py`

| Operation       | Endpoint                                                     | Test Method                             | Status |
| --------------- | ------------------------------------------------------------ | --------------------------------------- | ------ |
| List Messages   | `GET /api/v1/sessions/{session_id}/messages`                 | `test_list_messages_success_contract`   | ✅     |
| Add Messages    | `POST /api/v1/sessions/{session_id}/messages`                | `test_add_messages_success_contract`    | ✅     |
| Read Messages   | `GET /api/v1/sessions/{session_id}/messages/{message_id}`    | `test_read_messages_success_contract`   | ✅     |
| Update Messages | `PUT /api/v1/sessions/{session_id}/messages/{message_id}`    | `test_update_messages_success_contract` | ✅     |
| Delete Messages | `DELETE /api/v1/sessions/{session_id}/messages/{message_id}` | `test_delete_messages_success_contract` | ✅     |

**Error Cases Tested**:

- Session not found (404)
- Message not found (404)
- Invalid message roles (400)
- Empty message content (400)
- Invalid sorting/pagination parameters

---

### ✅ Knowledge API (`/api/v1/users/{user_id}/knowledge`)

**Status**: Comprehensive coverage  
**File**: `test_knowledge_api_contract.py`

| Operation        | Endpoint                                   | Test Method                              | Status |
| ---------------- | ------------------------------------------ | ---------------------------------------- | ------ |
| List Knowledge   | `GET /api/v1/users/{user_id}/knowledge`    | `test_list_knowledge_success_contract`   | ✅     |
| Add Knowledge    | `POST /api/v1/users/{user_id}/knowledge`   | `test_add_knowledge_success_contract`    | ✅     |
| Read Knowledge   | `GET /api/v1/users/{user_id}/knowledge`    | `test_read_knowledge_success_contract`   | ✅     |
| Update Knowledge | `PUT /api/v1/users/{user_id}/knowledge`    | `test_update_knowledge_success_contract` | ✅     |
| Delete Knowledge | `DELETE /api/v1/users/{user_id}/knowledge` | `test_delete_knowledge_success_contract` | ✅     |

**Error Cases Tested**:

- User not found (404)
- Empty knowledge lists (400)
- Mismatched array lengths (400)
- Non-existent knowledge IDs (400)
- Invalid JSON payload (400)

---

### ✅ Memory API (`/api/v1/users/{user_id}/memory`)

**Status**: Comprehensive coverage  
**File**: `test_memory_api_contract.py`

| Operation    | Endpoint                              | Test Method                          | Status |
| ------------ | ------------------------------------- | ------------------------------------ | ------ |
| Query Memory | `POST /api/v1/users/{user_id}/memory` | `test_query_memory_success_contract` | ✅     |

**Advanced Testing Scenarios**:

- Session-scoped queries
- Agent-filtered queries
- Store type specification (vector, keyword, graph)
- Pagination and top-k results
- Empty result handling
- Parameter validation

**Error Cases Tested**:

- User not found (404)
- Session not found (404)
- Agent not found (404)
- Missing query parameter (400)
- Invalid store types (400)
- Negative top-k values (400)
- Large top-k values (400)

## Common Test Patterns

### Success Response Schema

All successful API responses follow this structure:

```json
{
  "status": "success",
  "code": 200,
  "data": { ... },
  "message": "Operation completed successfully",
  "errors": null
}
```

### Error Response Schema

All error responses follow this structure:

```json
{
  "status": "error",
  "code": 400,
  "data": null,
  "message": "Error description",
  "errors": [
    {
      "field": "field_name",
      "message": "Field-specific error message"
    }
  ]
}
```

### Authentication Testing

- API key required for all endpoints
- Tests for missing/invalid API keys
- Proper 401 Unauthorized responses

## Testing Guidelines

### Adding New Contract Tests

When adding new API endpoints or modifying existing ones:

1. **Create Test Class**: Follow naming pattern `Test{Resource}APIContract`
2. **Define Schemas**: Create JSON schemas for request/response validation
3. **Test Success Cases**: Verify 200/201/204 responses with correct schemas
4. **Test Error Cases**: Cover 400/401/404/409 responses
5. **Test Edge Cases**: Empty lists, boundary conditions, etc.
6. **Update This Document**: Add entry to the appropriate section

### Test Method Naming

- Pattern: `test_{operation}_{scenario}_contract`
- Examples:
  - `test_create_user_success_contract`
  - `test_get_user_not_found_contract`
  - `test_update_user_invalid_json_contract`

### Schema Evolution

- **Backward Compatibility**: Ensure new fields don't break existing schemas
- **Optional Fields**: Mark new fields as optional when possible
- **Deprecation**: Document deprecated fields and removal timelines

## Future Enhancements

### 🔄 Planned Additions

- **Rate Limiting Tests**: Validate API rate limiting behavior
- **Pagination Tests**: Comprehensive pagination scenario testing
- **Bulk Operations**: Test bulk create/update/delete operations
- **Version Compatibility**: Test API versioning headers
- **Content Negotiation**: Test different content types

### 📋 TODO Items

- [ ] Add health check endpoint contract tests
- [ ] Implement API versioning contract validation
- [ ] Add performance constraint validation
- [ ] Create contract test reporting dashboard
- [ ] Add OpenAPI schema validation integration

## Running Contract Tests

```bash
# Run all contract tests
pytest tests/contract/

# Run specific API contract tests
pytest tests/contract/test_users_api_contract.py

# Run with verbose output
pytest tests/contract/ -v

# Run with coverage
pytest tests/contract/ --cov=src/memfuse_core/api
```

## Maintenance Notes

- **Regular Updates**: Review and update schemas when API changes
- **Schema Validation**: Ensure all new endpoints have corresponding contract tests
- **Documentation Sync**: Keep this document updated with test additions/changes
- **Mock Updates**: Update mock services when data models change

---

**Last Updated**: [Current Date]  
**Next Review**: [Schedule regular reviews]  
**Maintainer**: [Team/Individual responsible]
