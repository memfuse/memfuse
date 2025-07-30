# Connection Pool Integration Tests

This directory contains comprehensive integration tests for the PostgreSQL connection pool optimization in MemFuse.

## Test Files

### Core Tests
- **`test_global_connection_manager.py`** - Tests for the GlobalConnectionManager singleton
- **`test_connection_pool_configuration.py`** - Tests for configuration-driven pool settings  
- **`test_connection_leak_prevention.py`** - Tests to verify no connection leaks occur

### Utility Scripts
- **`../connection_monitor.py`** - Monitor PostgreSQL connection count and status
- **`../simple_api_test.py`** - Simple API test to trigger connection usage
- **`../connection_leak_reproduction.py`** - Comprehensive connection leak reproduction test

## Quick Start

### Prerequisites
1. PostgreSQL database running on localhost:5432
2. MemFuse configuration properly set up
3. Poetry environment activated

### Running Tests

#### Individual Tests
```bash
# Test singleton pattern and pool sharing
poetry run python tests/integration/connection_pool/test_global_connection_manager.py

# Test configuration hierarchy
poetry run python tests/integration/connection_pool/test_connection_pool_configuration.py

# Test connection leak prevention
poetry run python tests/integration/connection_pool/test_connection_leak_prevention.py
```

#### Using Test Runner
```bash
# Run specific test file
poetry run python scripts/run_tests.py tests/integration/connection_pool/test_global_connection_manager.py -v

# Run with pytest directly
poetry run pytest tests/integration/connection_pool/ -v -s
```

#### Monitor Connections
```bash
# Single connection check
poetry run python tests/connection_monitor.py

# Continuous monitoring (10s intervals for 60s)
poetry run python tests/connection_monitor.py continuous 10 60
```

## Test Scenarios

### 1. Singleton Pattern Verification
- Verifies GlobalConnectionManager follows singleton pattern
- Tests configuration hierarchy priority
- Validates URL masking for security

### 2. Connection Pool Sharing
- Creates multiple store instances
- Verifies they share the same connection pool
- Checks reference counting and cleanup

### 3. Configuration Testing
- Tests configuration hierarchy (store > database > postgres > defaults)
- Verifies environment variable overrides
- Validates configuration application to actual pools

### 4. Leak Prevention
- Creates and destroys stores rapidly
- Makes multiple API requests
- Monitors connection count for leaks
- Verifies proper cleanup

## Expected Results

### Successful Test Output
```
✅ Singleton pattern verified
✅ All stores are sharing a single connection pool
✅ Configuration values are being used correctly
✅ No connection pools remaining - leak prevention successful
✅ Store instance creation test passed
✅ API requests test passed
```

### Connection Statistics
```
Pool statistics: {
    'postgresql://postgres:***@localhost:5432/memfuse': {
        'min_size': 10, 
        'max_size': 30, 
        'timeout': 30.0, 
        'recycle': 3600, 
        'active_references': 5, 
        'pool_closed': False
    }
}
```

## Troubleshooting

### Common Issues

#### Database Connection Errors
```
connection failed: FATAL: database "memfuse_test" does not exist
```
**Solution**: Tests use mock database URLs that may not exist. This is expected for unit tests.

#### API Connection Errors
```
Cannot connect to API: Connection refused
```
**Solution**: Start MemFuse server before running API tests:
```bash
poetry run memfuse-core &
poetry run python tests/integration/connection_pool/test_connection_leak_prevention.py
```

#### Permission Errors
```
FATAL: password authentication failed
```
**Solution**: Check PostgreSQL credentials in environment variables or config files.

### Debugging

#### Enable Debug Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### Monitor Connections During Tests
```bash
# In one terminal
poetry run python tests/connection_monitor.py continuous 5 300

# In another terminal  
poetry run python tests/integration/connection_pool/test_connection_leak_prevention.py
```

#### Check Pool Statistics
```python
from memfuse_core.services.global_connection_manager import get_global_connection_manager
manager = get_global_connection_manager()
print(manager.get_pool_statistics())
```

## Integration with CI/CD

### Test Commands for CI
```bash
# Quick smoke test
poetry run python tests/integration/connection_pool/test_global_connection_manager.py

# Full integration test (requires database)
poetry run python scripts/run_tests.py tests/integration/connection_pool/test_connection_pool_configuration.py --no-restart-server

# Connection leak test (requires API server)
poetry run python tests/integration/connection_pool/test_connection_leak_prevention.py
```

### Environment Setup for CI
```yaml
env:
  POSTGRES_HOST: localhost
  POSTGRES_PORT: 5432
  POSTGRES_DB: memfuse_test
  POSTGRES_USER: postgres
  POSTGRES_PASSWORD: postgres
  DATABASE_URL: postgresql://postgres:postgres@localhost:5432/memfuse_test
```

## Performance Benchmarks

### Before Optimization
- **Connection Usage**: 2-6 connections per user
- **Pool Count**: Multiple individual pools
- **Leak Rate**: Connections accumulate over time
- **Resource Waste**: Many idle pools

### After Optimization  
- **Connection Usage**: 10-30 connections total (configurable)
- **Pool Count**: 1 shared pool per database
- **Leak Rate**: Zero connection leaks
- **Resource Efficiency**: Shared pools across all users

### Test Metrics
- **Store Creation**: 5 stores → 1 shared pool
- **Connection Increase**: Only 3 connections for 5 stores
- **API Requests**: 10 requests → stable connection count
- **Cleanup**: All pools properly closed

## Contributing

When adding new connection pool tests:

1. **Follow naming convention**: `test_connection_pool_*.py`
2. **Include cleanup**: Always cleanup pools in teardown
3. **Mock external dependencies**: Use mock database URLs when possible
4. **Document expected behavior**: Clear assertions and error messages
5. **Test edge cases**: Error conditions, invalid configs, etc.

## Related Documentation

- [Connection Pool Optimization](../../../docs/optimization/connection_pool_optimization.md)
- [Singleton Design Pattern](../../../docs/optimization/singleton.md)
- [MemFuse Configuration Guide](../../../docs/configuration.md)
