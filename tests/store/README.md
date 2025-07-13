# PgAI Store Tests

This directory contains comprehensive test suites for the pgai store implementation with immediate trigger system and M0 episodic memory layer.

## Test Structure

### Unit Tests
- `test_event_driven_pgai_store.py` - Core functionality tests
- `test_simple_reorganization.py` - File structure validation
- `test_reorganization_validation.py` - Import and compatibility tests
- Mock-based tests that don't require database connections
- Fast execution suitable for CI/CD pipelines

### Integration Tests
- `test_immediate_trigger_comprehensive.py` - **Unified comprehensive test suite**
- End-to-end tests requiring actual PostgreSQL with pgvector
- Real database operations and immediate trigger validation
- Performance and reliability validation (sub-100ms response time)

## Running Tests

### Prerequisites

1. **For Unit Tests Only:**
   ```bash
   # Install test dependencies
   poetry install --with dev
   ```

2. **For Integration Tests:**
   ```bash
   # Start MemFuse pgai environment
   cd memfuse

   # Ensure database is ready
   python scripts/database_manager.py validate
   poetry run memfuse-core
   
   # In another terminal, ensure database is ready
   docker ps | grep postgres
   ```

### Test Execution

#### Unit Tests Only
```bash
# Run all unit tests
pytest tests/store/test_event_driven_pgai_store.py -v

# Run reorganization validation tests
python tests/store/test_simple_reorganization.py
python tests/store/test_reorganization_validation.py

# Run with coverage
pytest tests/store/ --cov=memfuse_core.store --cov-report=html
```

#### Integration Tests (Recommended)
```bash
# Run comprehensive immediate trigger test suite
python tests/integration/test_immediate_trigger_comprehensive.py

# This single test covers:
# - Database schema validation
# - Trigger mechanism testing
# - Performance characteristics
# - Application integration
# - Evidence collection
```

#### Database Management
```bash
# Check database status
python scripts/database_manager.py status

# Validate schema
python scripts/database_manager.py validate

# Reset data (keep schema)
python scripts/database_manager.py reset

# Recreate complete schema
python scripts/database_manager.py recreate
```

## Test Categories

### 1. EventDrivenPgaiStore Tests
- **Initialization**: Store setup with immediate trigger configuration
- **Notification Listener**: PostgreSQL NOTIFY/LISTEN mechanism
- **Worker Pool**: Async worker processing and queue management
- **Retry Logic**: Failed embedding retry with exponential backoff
- **Metrics Collection**: Performance monitoring and statistics
- **Resource Cleanup**: Proper shutdown and resource management

### 2. RetryManager Tests
- **Retry Decision Logic**: When to retry vs. mark as failed
- **Retry Counting**: Independent retry counters per record
- **Status Management**: Tracking retry states (pending/processing/completed/failed)
- **Time-based Retries**: Interval-based retry scheduling
- **Statistics**: Retry performance metrics

### 3. EmbeddingMetrics Tests
- **Event Recording**: Processing success/failure tracking
- **Performance Metrics**: Timing and throughput calculations
- **Statistics Aggregation**: Success rates and error patterns
- **Memory Management**: Bounded history storage

### 4. StoreFactory Tests
- **Store Selection**: Automatic selection between traditional and event-driven
- **Configuration Validation**: Config parsing and validation
- **Backward Compatibility**: Legacy configuration migration

### 5. Integration Tests
- **End-to-End Workflow**: Complete immediate trigger flow
- **Database Operations**: Real PostgreSQL operations
- **Embedding Generation**: Actual model inference
- **Performance Validation**: Latency and throughput verification
- **Retry Mechanism**: Real failure and recovery scenarios

## Test Configuration

### Mock Configuration
```python
mock_config = {
    "database": {
        "postgres": {
            "host": "localhost",
            "port": 5432,
            "database": "test_memfuse",
            "user": "postgres",
            "password": "postgres"
        },
        "pgai": {
            "enabled": True,
            "auto_embedding": True,
            "immediate_trigger": True,
            "max_retries": 3,
            "retry_interval": 1.0,
            "worker_count": 2,
            "queue_size": 100,
            "enable_metrics": True
        }
    }
}
```

### Integration Test Configuration
- Uses actual database connection
- Requires running MemFuse environment
- Tests with real embedding models
- Validates performance characteristics

## Expected Test Results

### Unit Tests
- **Execution Time**: < 30 seconds for full suite
- **Coverage**: > 90% code coverage
- **Assertions**: All functionality properly mocked and tested

### Integration Tests
- **Immediate Trigger Latency**: < 1 second from insert to processing start
- **Processing Time**: < 10 seconds for 3 test embeddings
- **Retry Mechanism**: Proper retry behavior with configurable intervals
- **Success Rate**: > 95% for normal operations

## Performance Benchmarks

### Traditional vs Event-Driven Comparison
| Metric | Traditional (Polling) | Event-Driven (Immediate) | Improvement |
|--------|----------------------|---------------------------|-------------|
| Trigger Latency | 5-15 seconds | < 1 second | 90%+ faster |
| Resource Usage | Continuous polling | On-demand | 70%+ reduction |
| Retry Capability | Basic | Advanced (3x with backoff) | Enhanced |
| Monitoring | Basic logs | Comprehensive metrics | Detailed |

### Test Performance Targets
- **Unit Test Suite**: < 30 seconds
- **Integration Test Suite**: < 60 seconds
- **Memory Usage**: < 100MB during tests
- **Database Connections**: Properly managed and cleaned up

## Troubleshooting

### Common Issues

1. **Integration Tests Failing**
   ```bash
   # Check if MemFuse is running
   curl http://localhost:8000/health
   
   # Check database connectivity
   docker exec -it memfuse-pgai-postgres-1 psql -U postgres -d memfuse -c "SELECT 1;"
   ```

2. **Mock Tests Failing**
   ```bash
   # Check dependencies
   poetry show pytest pytest-asyncio pytest-mock
   
   # Run with verbose output
   pytest tests/store/test_event_driven_pgai_store.py -v -s
   ```

3. **Performance Issues**
   ```bash
   # Check system resources
   top -p $(pgrep -f memfuse-core)
   
   # Monitor database performance
   docker stats memfuse-pgai-postgres-1
   ```

### Debug Mode
```bash
# Run tests with debug logging
PYTHONPATH=/Users/mxue/GitRepos/MemFuse/memfuse/src pytest tests/store/test_event_driven_pgai_store.py -v -s --log-cli-level=DEBUG
```

## Contributing

### Adding New Tests
1. Follow existing test patterns and naming conventions
2. Use appropriate fixtures for setup/teardown
3. Include both positive and negative test cases
4. Add integration tests for new features
5. Update this documentation

### Test Guidelines
- Use descriptive test names that explain the scenario
- Include docstrings for complex test cases
- Mock external dependencies in unit tests
- Use real dependencies in integration tests
- Validate both success and failure paths
- Test edge cases and error conditions

## Continuous Integration

### GitHub Actions Configuration
```yaml
- name: Run Unit Tests
  run: |
    poetry install --with dev
    pytest tests/store/test_event_driven_pgai_store.py -v --cov=memfuse_core.store

- name: Run Integration Tests
  run: |
    docker-compose up -d postgres
    poetry run memfuse-core &
    sleep 30
    pytest tests/store/test_event_driven_pgai_store.py::TestIntegrationEventDrivenPgaiStore --integration -v
```

### Test Reports
- Coverage reports generated in `htmlcov/`
- JUnit XML reports for CI integration
- Performance timing reports
- Integration test results with metrics
