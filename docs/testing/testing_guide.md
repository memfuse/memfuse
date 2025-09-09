# MemFuse Testing Guide

This comprehensive guide covers all aspects of testing in the MemFuse project, including unit tests, integration tests, performance tests, and testing best practices.

## Table of Contents

1. [Test Structure](#test-structure)
2. [Running Tests](#running-tests)
3. [Test Categories](#test-categories)
4. [Performance Testing](#performance-testing)
5. [Memory Leak Detection](#memory-leak-detection)
6. [Testing Best Practices](#testing-best-practices)
7. [Troubleshooting](#troubleshooting)

## Test Structure

### Directory Organization

```
tests/
├── conftest.py                 # Global test configuration and fixtures
├── pytest.ini                 # Pytest configuration
├── unit/                       # Unit tests
│   ├── gateway/               # Gateway component tests
│   ├── buffer/                # QueryBuffer tests
│   ├── persistence/           # Storage and persistence tests
│   ├── observability/         # Monitoring and tracing tests
│   ├── pipeline/              # Pipeline integration tests
│   └── error_scenarios/       # Error handling tests
├── integration/               # Integration tests
│   └── test_advanced_pipeline_scenarios.py
└── performance/               # Performance and load tests
    ├── test_benchmark_pipeline.py
    ├── test_performance_regression.py
    └── test_memory_leak_detection.py
```

### Test Configuration

The project uses pytest with the following key configurations:

```ini
# pytest.ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
asyncio_mode = auto
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    performance: marks tests as performance tests
    integration: marks tests as integration tests
```

## Running Tests

### Basic Test Execution

```bash
# Run all tests
poetry run pytest

# Run with verbose output
poetry run pytest -v

# Run specific test file
poetry run pytest tests/unit/gateway/test_filter_cache.py

# Run specific test class
poetry run pytest tests/unit/gateway/test_filter_cache.py::TestRegexCache

# Run specific test method
poetry run pytest tests/unit/gateway/test_filter_cache.py::TestRegexCache::test_basic_functionality
```

### Test Categories

```bash
# Run only unit tests
poetry run pytest tests/unit/

# Run only integration tests
poetry run pytest tests/integration/ -m integration

# Run only performance tests
poetry run pytest tests/performance/ -m performance

# Skip slow tests
poetry run pytest -m "not slow"

# Run error scenario tests
poetry run pytest tests/unit/error_scenarios/
```

### Parallel Test Execution

```bash
# Run tests in parallel (requires pytest-xdist)
poetry run pytest -n auto

# Run with specific number of workers
poetry run pytest -n 4
```

## Test Categories

### 1. Unit Tests

**Location**: `tests/unit/`

Unit tests focus on individual components and functions in isolation.

#### Gateway Tests
- **Filter Cache**: `test_filter_cache.py` - Caching mechanisms
- **Semantic Validation**: `test_semantic_validation.py` - Content validation
- **Composite Filters**: `test_composite_filters.py` - Filter combinations
- **Sensitive Word Filtering**: `test_sensitive_word_enhanced.py` - Content filtering

#### Buffer Tests
- **QueryBuffer**: Tests for query caching and retrieval optimization

#### Persistence Tests
- **InMemoryStore**: Tests for in-memory storage operations
- **Adapters**: Tests for storage adapters and handlers

#### Observability Tests
- **Tracing**: `test_tracing_fallback.py` - Distributed tracing functionality
- **Metrics**: Tests for metrics collection and reporting

### 2. Integration Tests

**Location**: `tests/integration/`

Integration tests verify component interactions and end-to-end workflows.

```bash
# Run integration tests
poetry run pytest tests/integration/ -v

# Run specific integration scenario
poetry run pytest tests/integration/test_advanced_pipeline_scenarios.py::TestAdvancedPipelineScenarios::test_full_pipeline_with_all_filters -v
```

### 3. Error Scenario Tests

**Location**: `tests/unit/error_scenarios/`

Tests system behavior under various error conditions.

```bash
# Run all error scenario tests
poetry run pytest tests/unit/error_scenarios/ -v

# Test specific error scenarios
poetry run pytest tests/unit/error_scenarios/test_error_handling.py::TestQueryBufferErrorScenarios -v
```

## Performance Testing

### Performance Regression Tests

**Location**: `tests/performance/test_performance_regression.py`

Automated performance regression detection with established baselines.

```bash
# Run performance regression tests
poetry run pytest tests/performance/test_performance_regression.py -m performance -v

# Run specific performance test
poetry run pytest tests/performance/test_performance_regression.py::TestCachePerformanceRegression::test_regex_cache_hit_performance -v
```

#### Performance Baselines

| Operation | Max Duration | Max Memory | Min Throughput |
|-----------|-------------|------------|----------------|
| Regex Cache Hit | 0.1ms | 10MB | 10,000 ops/sec |
| Regex Cache Miss | 50ms | 20MB | 20 ops/sec |
| Content Cache Ops | 1ms | 50MB | 1,000 ops/sec |
| QueryBuffer Small | 10ms | 100MB | 100 ops/sec |
| QueryBuffer Large | 100ms | 500MB | 10 ops/sec |

### Benchmark Tests

**Location**: `tests/performance/test_benchmark_pipeline.py`

Detailed performance benchmarking with metrics collection.

```bash
# Run benchmark tests
poetry run pytest tests/performance/test_benchmark_pipeline.py -m performance -s -v

# Run specific benchmark
poetry run pytest tests/performance/test_benchmark_pipeline.py::TestCachePerformance::test_regex_cache_performance -v -s
```

## Memory Leak Detection

**Location**: `tests/performance/test_memory_leak_detection.py`

Automated memory leak detection for long-running operations.

```bash
# Run memory leak detection tests
poetry run pytest tests/performance/test_memory_leak_detection.py -m performance -v

# Run specific memory leak test
poetry run pytest tests/performance/test_memory_leak_detection.py::TestQueryBufferMemoryLeaks::test_query_buffer_repeated_queries_no_leak -v
```

### Memory Monitoring Features

- **Real-time monitoring**: RSS and VMS memory tracking
- **Trend analysis**: Growth rate calculation and leak detection
- **Threshold-based alerts**: Configurable memory usage thresholds
- **Garbage collection integration**: Post-GC memory verification

## Testing Best Practices

### 1. Test Organization

```python
# Good: Descriptive test names
def test_regex_cache_hit_performance_meets_baseline():
    """Test that regex cache hit performance meets established baseline."""
    pass

# Good: Clear test structure
class TestSemanticValidator:
    """Test cases for SemanticValidator."""
    
    @pytest.fixture
    def validator(self):
        """Create a semantic validator for testing."""
        return SemanticValidator()
    
    def test_validator_initialization(self, validator):
        """Test validator initialization."""
        assert validator is not None
```

### 2. Async Test Patterns

```python
# Good: Proper async test structure
@pytest.mark.asyncio
async def test_async_operation():
    """Test async operation."""
    result = await some_async_function()
    assert result is not None

# Good: Async fixture usage
@pytest.fixture
async def async_store():
    """Create an async store for testing."""
    store = InMemoryStore()
    await store.initialize()
    yield store
    await store.cleanup()
```

### 3. Mock Usage

```python
# Good: Targeted mocking
def test_with_mock_encoder(self, validator):
    """Test with mocked encoder."""
    with patch.object(validator, 'encoder') as mock_encoder:
        mock_encoder.encode.return_value = [[0.1, 0.2, 0.3]]
        result = validator.process("test")
        assert result is not None
```

### 4. Performance Test Patterns

```python
# Good: Performance test with baseline
def test_operation_performance(self, perf_tester):
    """Test operation performance against baseline."""
    def operation():
        return expensive_operation()
    
    results = perf_tester.measure_performance(operation, iterations=100)
    regression_result = perf_tester.check_regression("operation_name", results)
    
    assert regression_result.passed, f"Performance regression: {regression_result.details}"
```

## Troubleshooting

### Common Issues

#### 1. Async Test Failures

**Problem**: `RuntimeError: There is no current event loop in thread 'MainThread'`

**Solution**: 
- Ensure proper async test decoration: `@pytest.mark.asyncio`
- Check that `conftest.py` has proper event loop fixture
- Verify pytest-asyncio is installed and configured

#### 2. Import Errors

**Problem**: `ModuleNotFoundError: No module named 'src.memfuse_core'`

**Solution**:
```bash
# Ensure you're in the project root
cd /path/to/memfuse

# Run tests with poetry
poetry run pytest

# Or activate virtual environment
poetry shell
pytest
```

#### 3. Performance Test Failures

**Problem**: Performance tests failing due to system load

**Solution**:
- Run performance tests on dedicated test environment
- Adjust performance baselines for your hardware
- Use `--tb=short` for cleaner output

#### 4. Memory Leak False Positives

**Problem**: Memory leak tests failing due to system memory pressure

**Solution**:
- Increase memory leak detection thresholds
- Run tests with sufficient available memory
- Check for other memory-intensive processes

### Test Environment Setup

```bash
# Install test dependencies
poetry install --with dev

# Install additional performance testing tools
pip install psutil memory-profiler

# Set up test database (if needed)
createdb memfuse_test

# Run test suite
poetry run pytest --tb=short
```

### Debugging Tests

```bash
# Run with detailed output
poetry run pytest -vvv --tb=long

# Run with pdb on failure
poetry run pytest --pdb

# Run with coverage
poetry run pytest --cov=src --cov-report=html

# Run specific test with output
poetry run pytest tests/unit/gateway/test_filter_cache.py::test_basic_functionality -s -vvv
```

### Performance Profiling

```bash
# Profile test execution
poetry run pytest --profile

# Memory profiling
poetry run pytest --memray

# Generate performance report
poetry run pytest tests/performance/ --benchmark-only --benchmark-json=benchmark.json
```

This testing guide provides comprehensive coverage of all testing aspects in the MemFuse project, ensuring reliable and maintainable code quality.
