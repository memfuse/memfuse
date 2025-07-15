# Integration Testing Layer Documentation

## Overview

The integration testing layer validates **real data operations** and **service interactions** for the MemFuse application. Unlike contract tests that use mocks, integration tests use real databases and services to ensure data actually flows correctly through the system.

## Testing Philosophy

Integration tests are designed to:

- **Validate real data operations** - Ensure data persists, retrieves, and scopes correctly
- **Test service boundaries** - Verify components work together with real implementations
- **Isolate failure points** - Use strategic mocking to pinpoint integration issues
- **Ensure data integrity** - Test that operations don't leak between scopes
- **Validate end-to-end flows** - Test complete user journeys with real data

## Test Infrastructure

### Database Setup

- **TimescaleDB**: Real PostgreSQL with pgai extensions
- **Fresh Database**: Each test gets clean database state
- **Cleanup Strategy**: Reset database in setup, preserve for inspection after tests
- **Management**: Use `database_manager.py` for schema operations

### Service Strategy

- **Embedding Service**: Mock for CRUD tests, real for memory/retrieval tests
- **LLM Service**: Mock unless specifically testing LLM functionality
- **Vector Store**: Always use real vector operations with TimescaleDB
- **Database**: Always real for true integration testing

### Test Data

- **MSC-MC10 Dataset**: Use sample from [Hugging Face dataset](https://huggingface.co/datasets/Percena/msc-memfuse-mc10)
- **Taylor Swift Test**: Reference question for validation
- **Test Fixtures**: Reusable conversation data for consistent testing

## Core Integration Testing Pillars

### 1. **Data Persistence Testing** 🎯

**Purpose**: Ensure API calls result in actual database changes

**Test Coverage**:

- Users created via API actually exist in database
- Sessions properly link users and agents with foreign keys
- Messages are stored with correct session association
- Knowledge chunks are persisted with embeddings
- Updates and deletes actually modify/remove database records
- Timestamps and metadata are correctly stored

**Test Strategy**:

- Use real database operations
- Mock embedding/LLM services for CRUD tests
- Validate database state directly after API calls

### 2. **Retrieval Functionality** 🔍

**Purpose**: Ensure memory retrieval works with real data and embeddings

**Test Coverage**:

- Memory queries return relevant results (not empty when data exists)
- Different store types (vector, keyword, graph) return appropriate data
- Embeddings are generated and stored correctly
- Similarity search returns contextually relevant results
- Pagination and top-k limiting work correctly
- Query performance is reasonable

**Test Strategy**:

- Use real embedding service for accurate similarity
- Use real vector operations
- Test with MSC-MC10 conversational data
- Validate retrieval accuracy with known answers

### 3. **Scoping and Data Isolation** 🔒

**Purpose**: Ensure data doesn't leak between users, sessions, or agents

**Test Coverage**:

- User A's memories don't appear in User B's queries
- Session-scoped queries only return session-specific data
- Agent-filtered queries properly isolate by agent
- Knowledge scoping works correctly per user
- Cross-user contamination is prevented
- Memory boundaries are respected

**Test Strategy**:

- Create multiple users/sessions/agents
- Cross-test queries to ensure isolation
- Validate no data leakage occurs

## API Integration Testing Status

### 📋 Users API Integration

**Status**: Planned  
**File**: `test_api_integration.py`

| Test Case             | Description                                      | Priority |
| --------------------- | ------------------------------------------------ | -------- |
| User CRUD Persistence | Verify user creation/update/deletion in database | High     |
| User Uniqueness       | Test duplicate name handling                     | Medium   |
| User Cascade          | Test user deletion cascades properly             | High     |

### 📋 Agents API Integration

**Status**: Planned  
**File**: `test_api_integration.py`

| Test Case              | Description                                       | Priority |
| ---------------------- | ------------------------------------------------- | -------- |
| Agent CRUD Persistence | Verify agent creation/update/deletion in database | High     |
| Agent Uniqueness       | Test duplicate name handling                      | Medium   |
| Agent Cascade          | Test agent deletion cascades properly             | High     |

### 📋 Sessions API Integration

**Status**: Planned  
**File**: `test_api_integration.py`

| Test Case        | Description                                | Priority |
| ---------------- | ------------------------------------------ | -------- |
| Session Creation | Verify session with user/agent links       | High     |
| Session Scoping  | Test session isolation                     | High     |
| Session Cascade  | Test session deletion cascades to messages | High     |

### 📋 Messages API Integration

**Status**: Planned  
**File**: `test_api_integration.py`

| Test Case           | Description                              | Priority |
| ------------------- | ---------------------------------------- | -------- |
| Message Persistence | Verify message storage with session link | High     |
| Message Ordering    | Test message chronological ordering      | Medium   |
| Message Updates     | Test message modification persistence    | Medium   |

### 📋 Knowledge API Integration

**Status**: Planned  
**File**: `test_api_integration.py`

| Test Case           | Description                                    | Priority |
| ------------------- | ---------------------------------------------- | -------- |
| Knowledge Storage   | Verify knowledge chunk storage with embeddings | High     |
| Knowledge Retrieval | Test knowledge search functionality            | High     |
| Knowledge Scoping   | Test user-specific knowledge isolation         | High     |

### 📋 Memory API Integration

**Status**: Planned  
**File**: `test_api_integration.py`

| Test Case         | Description                                 | Priority |
| ----------------- | ------------------------------------------- | -------- |
| Memory Query      | Test memory retrieval with real embeddings  | High     |
| Taylor Swift Test | MSC-MC10 dataset validation                 | High     |
| Store Types       | Test vector/keyword/graph store integration | Medium   |
| Memory Scoping    | Test session/agent filtering                | High     |

## MSC-MC10 Dataset Integration

### Taylor Swift Reference Test

**Purpose**: Validate end-to-end memory retrieval with real conversational data

**Test Flow**:

1. **Setup**: Clean database using `database_manager.py`
2. **Data Loading**: Load MSC-MC10 sample conversation about Taylor Swift
3. **Data Ingestion**: Add conversation to MemFuse via API
4. **Query Testing**: Ask the reference question
5. **Validation**: Verify relevant context is retrieved
6. **Assertion**: Check answer accuracy

**Sample Data Structure**:

```json
{
  "question": "Can you remind me what we talked about regarding Taylor Swift?",
  "expected_context": "Taylor Swift conversation content",
  "haystack_sessions": [
    [
      { "role": "user", "content": "I love Taylor Swift's music..." },
      {
        "role": "assistant",
        "content": "Oh really? What's your favorite song?"
      }
    ]
  ]
}
```

## Test Environment Configuration

### Database Setup

```python
# Use database_manager.py for setup
def setup_fresh_database():
    subprocess.run(["python", "scripts/database_manager.py", "reset"])
    # Verify database is clean and ready
```

### Service Configuration

```python
# Test-specific config with strategic mocking
TEST_CONFIG = {
    "database": "real_timescaledb",
    "embedding_service": "mock_for_crud_real_for_memory",
    "llm_service": "mock_unless_specified",
    "vector_store": "real_always"
}
```

### Test Fixtures

```python
# Reusable test data from MSC-MC10
@pytest.fixture
def msc_mc10_sample():
    return load_json("fixtures/msc_mc10_sample.json")

@pytest.fixture
def taylor_swift_conversation():
    return load_json("fixtures/taylor_swift_test.json")
```

## Future Integration Test Areas

### 🔄 Planned Enhancements

#### **Service Integration Tests**

- **Buffer Service Integration**: Test buffer → database persistence
- **Embedding Pipeline**: Test full embedding generation and storage
- **Memory Layer Integration**: Test hierarchical memory operations
- **Chunking Integration**: Test document chunking and storage

#### **Performance Integration**

- **Query Performance**: Test memory query response times
- **Embedding Performance**: Test embedding generation speed
- **Database Performance**: Test large dataset operations
- **Concurrent Operations**: Test multi-user concurrent access

#### **Error Handling Integration**

- **Database Failures**: Test graceful database failure handling
- **Service Timeouts**: Test service timeout behavior
- **Data Corruption**: Test data integrity validation
- **Recovery Scenarios**: Test system recovery procedures

#### **Configuration Integration**

- **Environment Variables**: Test different configuration scenarios
- **Service Discovery**: Test service endpoint configuration
- **Database Migrations**: Test schema migration handling
- **Feature Flags**: Test feature toggle integration

### 📋 TODO Items

- [ ] Extract MSC-MC10 sample data to fixtures
- [ ] Create Taylor Swift reference test
- [ ] Implement database cleanup fixtures
- [ ] Add test-specific configuration management
- [ ] Create service mocking infrastructure
- [ ] Implement data persistence validation helpers
- [ ] Add retrieval accuracy testing framework
- [ ] Create scoping violation detection tests

## Directory Structure

```
tests/integration/
├── INTEGRATION.md              # This documentation
├── conftest.py                 # Shared fixtures and setup
├── fixtures/                   # Test data files
│   ├── msc_mc10_sample.json   # Sample from MSC-MC10 dataset
│   └── taylor_swift_test.json # Taylor Swift reference test
├── test_api_integration.py     # API-level integration tests
├── test_service_integration.py # Service-level tests (future)
├── test_data_flow.py          # End-to-end data flow tests (future)
└── legacy/                    # Existing tests (preserved)
```

## Running Integration Tests

The integration test framework uses a two-stage approach for optimal performance:

**Stage 1**: Database infrastructure startup (once)
**Stage 2**: Per-test database reset (fast)

### Recommended Workflow

**Option 1: Automatic (Recommended)**

```bash
# Run integration tests - database automatically started
poetry run python scripts/run_tests.py integration
```

**Option 2: Manual Control**

```bash
# 1. Start database services first
poetry run python scripts/memfuse_launcher.py --start-db --optimize-db

# 2. Run integration tests (faster since database is already up)
poetry run pytest tests/integration/ -v
```

### Specific Test Execution

```bash
# Run specific test file
poetry run pytest tests/integration/api/test_users_api_integration.py -v

# Run with verbose output for debugging
poetry run pytest tests/integration/ -v -s

# Run specific test case
poetry run pytest tests/integration/api/test_users_api_integration.py::test_create_user_persistence -v
```

## Development Workflow

### 1. **Infrastructure Setup** (Once)

```bash
# Start database and services (runs in background)
poetry run python scripts/memfuse_launcher.py --start-db --optimize-db

# Verify database status
poetry run python scripts/database_manager.py status
```

### 2. **Test Development** (Iterative)

```bash
# Each test automatically resets database for clean state
poetry run pytest tests/integration/api/test_users_api_integration.py -v

# Manual reset if needed
poetry run python scripts/database_manager.py reset
```

### 3. **Performance Benefits**

The new two-stage approach provides:

- **Faster test execution**: Database container started once, not per test
- **More reliable**: Database ready before any tests run
- **Better separation**: Infrastructure setup vs. test logic
- **Easier debugging**: Clear failure points and preserved state
- **Development friendly**: Quick iteration with database reset

### 4. **Post-Test Analysis**

```bash
# Inspect database state (not cleaned up)
python scripts/database_manager.py status

# Check test data persistence
psql -h localhost -p 5432 -U postgres -d memfuse
```

## Maintenance Guidelines

### **Adding New Integration Tests**

1. **Follow Pillar Structure**: Categorize into persistence, retrieval, or scoping
2. **Use Real Services**: Only mock when specified in service strategy
3. **Clean Database**: Reset in setup, preserve for inspection
4. **Update Documentation**: Add test case to appropriate API section

### **Service Mocking Strategy**

- **CRUD Operations**: Mock embedding/LLM services
- **Memory Operations**: Use real embedding service
- **Vector Operations**: Always real
- **Database Operations**: Always real

### **Test Data Management**

- **Use Fixtures**: Reusable test data in `fixtures/` directory
- **MSC-MC10 Integration**: Reference real conversational data
- **Deterministic Results**: Use consistent test data for reproducible results

---

**Last Updated**: [Current Date]  
**Next Review**: [Schedule regular reviews]  
**Maintainer**: [Team/Individual responsible]
