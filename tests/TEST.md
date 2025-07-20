# MemFuse Testing Guide

This guide explains how to run MemFuse tests effectively, including configuration options for different testing scenarios.

## 📋 Test Structure

MemFuse uses a layered test architecture:

```
tests/
├── smoke/          # Quick health checks
├── contract/       # API contract validation
├── integration/    # Database integration tests
├── unit/           # Unit tests
├── retrieval/      # RAG and retrieval tests
├── e2e/            # End-to-end workflows
├── perf/           # Performance benchmarks
└── slow/           # Comprehensive test suite
```

## 🚀 Quick Start

### Basic Test Execution

```bash
# Run all test layers
poetry run python scripts/run_tests.py

# Run specific test layer
poetry run python scripts/run_tests.py smoke
poetry run python scripts/run_tests.py integration

# Run specific test file
poetry run python scripts/run_tests.py tests/integration/api/test_users_api_integration.py

# Run specific test method
poetry run python scripts/run_tests.py tests/integration/api/test_users_api_integration.py::TestUsersAPIIntegration::test_create_user_persistence
```

### With pytest flags

```bash
# Verbose output with stdout
poetry run python scripts/run_tests.py integration -v -s

# Run tests matching pattern
poetry run python scripts/run_tests.py integration -v -s -k "user"

# Stop on first failure
poetry run python scripts/run_tests.py integration -v -s -x
```

## ⚙️ Configuration Options

### Client Type Configuration

MemFuse tests support two client types:

#### 1. Server Mode (Default)

```bash
poetry run python scripts/run_tests.py --client-type=server integration
```

- ✅ **Tests against actual HTTP server**
- ✅ **Shows requests in server logs**
- ✅ **Most realistic testing**
- ⚠️ **Requires running server**

#### 2. TestClient Mode

```bash
poetry run python scripts/run_tests.py --client-type=testclient integration
```

- ✅ **Faster execution**
- ✅ **Isolated testing**
- ✅ **No server required**
- ⚠️ **No server log visibility**

### Server Management

#### Default: Restart Server (Clean Connections)

```bash
poetry run python scripts/run_tests.py integration
```

- ✅ **Clean database connections**
- ✅ **Reliable test execution**
- ⚠️ **Stops your development server**

#### Keep Server Running (For Monitoring)

```bash
poetry run python scripts/run_tests.py --no-restart-server integration
```

- ✅ **Keep development server running**
- ✅ **Monitor server logs continuously**
- ✅ **Database still gets reset**
- ⚠️ **May cause connection pool conflicts**

## 🔄 Common Workflows

### Development Workflow

**Terminal 1: Start Development Server**

```bash
poetry run python scripts/memfuse_launcher.py
```

**Terminal 2: Run Tests with Monitoring**

```bash
# Test against running server without restart
poetry run python scripts/run_tests.py --no-restart-server integration -v -s

# Test specific functionality
poetry run python scripts/run_tests.py --no-restart-server tests/integration/api/test_users_api_integration.py -v -s
```

### Debugging Failed Tests

```bash
# Run with maximum verbosity
poetry run python scripts/run_tests.py --no-restart-server integration -v -s --tb=long

# Stop on first failure for debugging
poetry run python scripts/run_tests.py --no-restart-server integration -v -s -x

# Run specific failing test
poetry run python scripts/run_tests.py --no-restart-server tests/integration/api/test_users_api_integration.py::TestUsersAPIIntegration::test_create_user_persistence -v -s
```

### Fast Testing (No Server Required)

```bash
# Quick isolated testing
poetry run python scripts/run_tests.py --client-type=testclient integration -v

# Unit-like testing for API endpoints
poetry run python scripts/run_tests.py --client-type=testclient tests/integration/api/test_users_api_integration.py -v
```

### Production-like Testing

```bash
# Clean server restart with fresh connections
poetry run python scripts/run_tests.py --client-type=server integration

# Full end-to-end testing
poetry run python scripts/run_tests.py --client-type=server e2e
```

## 📊 Test Layers Explained

### Smoke Tests

```bash
poetry run python scripts/run_tests.py smoke
```

- **Purpose**: Quick health checks
- **Runtime**: < 30 seconds
- **Coverage**: Basic API endpoints, database connectivity

### Contract Tests

```bash
poetry run python scripts/run_tests.py contract
```

- **Purpose**: API contract validation
- **Runtime**: 1-2 minutes
- **Coverage**: Request/response schemas, error handling

### Integration Tests

```bash
poetry run python scripts/run_tests.py integration
```

- **Purpose**: Database integration
- **Runtime**: 2-5 minutes
- **Coverage**: CRUD operations, data persistence, transactions

### Retrieval Tests

```bash
poetry run python scripts/run_tests.py retrieval
```

- **Purpose**: RAG and search functionality
- **Runtime**: 3-10 minutes
- **Coverage**: Embeddings, vector search, chunking

### E2E Tests

```bash
poetry run python scripts/run_tests.py e2e
```

- **Purpose**: Complete workflows
- **Runtime**: 5-15 minutes
- **Coverage**: User journeys, system integration

### Performance Tests

```bash
poetry run python scripts/run_tests.py perf
```

- **Purpose**: Performance benchmarks
- **Runtime**: 10-30 minutes
- **Coverage**: Load testing, response times, memory usage

## 🛠️ Advanced Usage

### Running Multiple Layers

```bash
# Run up to integration layer
poetry run python scripts/run_tests.py integration

# Run all layers
poetry run python scripts/run_tests.py
```

### Custom pytest Configuration

```bash
# Use custom pytest.ini settings
poetry run python scripts/run_tests.py integration --cov=src --cov-report=html

# Parallel execution (if supported)
poetry run python scripts/run_tests.py integration -n auto

# Custom markers
poetry run python scripts/run_tests.py integration -m "not slow"
```

### Environment Variables

```bash
# Set client type via environment
export MEMFUSE_TEST_CLIENT_TYPE=server
poetry run python scripts/run_tests.py integration

# Override for specific run
MEMFUSE_TEST_CLIENT_TYPE=testclient poetry run python scripts/run_tests.py integration
```

## 🔍 Troubleshooting

### Common Issues

#### "Connection Pool Conflicts"

```bash
# Solution 1: Use clean restart (default)
poetry run python scripts/run_tests.py integration

# Solution 2: Use TestClient mode
poetry run python scripts/run_tests.py --client-type=testclient integration

# Solution 3: Manually restart server
poetry run python scripts/memfuse_launcher.py --recreate-db
```

#### "Server Not Running"

```bash
# Check server status
curl http://localhost:8000/api/v1/health

# Start server
poetry run python scripts/memfuse_launcher.py

# Use TestClient mode as fallback
poetry run python scripts/run_tests.py --client-type=testclient integration
```

#### "Database Connection Failed"

```bash
# Reset database
poetry run python scripts/database_manager.py reset

# Recreate database schema
poetry run python scripts/database_manager.py recreate

# Check database status
poetry run python scripts/database_manager.py status
```

#### "Tests Failing with Stale Data"

```bash
# Database reset is automatic, but you can force it
poetry run python scripts/database_manager.py reset
poetry run python scripts/run_tests.py integration
```

### Debug Mode

```bash
# Maximum verbosity with debugging
poetry run python scripts/run_tests.py --no-restart-server integration -v -s --tb=long --pdb

# Log all HTTP requests
poetry run python scripts/run_tests.py --no-restart-server integration -v -s --log-cli-level=DEBUG
```

## 📋 Test Configuration

### fixtures and Setup

Integration tests use these key fixtures:

- `client`: Configurable HTTP client (server or testclient)
- `headers`: Authentication headers
- `database_connection`: Direct database access
- `test_user_data`: Sample user data
- `integration_helper`: Test utilities

### Environment Setup

Tests automatically:

- ✅ Reset database to clean state
- ✅ Start/check MemFuse server health
- ✅ Configure client based on settings
- ✅ Provide test data and utilities

## 🎯 Best Practices

### For Development

```bash
# Keep server running for monitoring
poetry run python scripts/run_tests.py --no-restart-server integration -v -s
```

### For CI/CD

```bash
# Clean restart for reliability
poetry run python scripts/run_tests.py integration
```

### For Local Testing

```bash
# Fast iteration with TestClient
poetry run python scripts/run_tests.py --client-type=testclient integration -v
```

### For Debugging

```bash
# Maximum visibility
poetry run python scripts/run_tests.py --no-restart-server integration -v -s --tb=long -x
```

## 🔗 Related Documentation

- [Scripts Documentation](../scripts/README.md)
- [API Documentation](../docs/api/)
- [Architecture Documentation](../docs/architecture/)

## 📞 Getting Help

If you encounter issues:

1. **Check server health**: `curl http://localhost:8000/api/v1/health`
2. **Validate database**: `poetry run python scripts/database_manager.py validate`
3. **Try TestClient mode**: `--client-type=testclient`
4. **Check logs**: Keep server running with `--no-restart-server`
5. **Reset environment**: `poetry run python scripts/database_manager.py reset`

For persistent issues, consider recreating the database schema:

```bash
poetry run python scripts/database_manager.py recreate
```
