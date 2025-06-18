# MemFuse Test Suite

This directory contains comprehensive tests for the MemFuse system following industry best practices.

## 📁 Directory Structure

```
tests/
├── __init__.py                 # Test package initialization
├── conftest.py                 # Pytest configuration and shared fixtures
├── README.md                   # This file
│
├── unit/                       # Unit tests for individual components
│   ├── __init__.py
│   ├── rag/                    # RAG component tests
│   │   ├── __init__.py
│   │   └── chunk/              # Chunking system tests
│   │       ├── __init__.py
│   │       ├── test_base.py                    # ChunkData and ChunkStrategy tests
│   │       ├── test_message_chunk_strategy.py  # MessageChunkStrategy tests
│   │       ├── test_contextual_chunk_strategy.py # ContextualChunkStrategy tests
│   │       └── test_character_chunk_strategy.py  # CharacterChunkStrategy tests
│   ├── services/               # Service layer tests
│   ├── interfaces/             # Interface tests
│   ├── buffer/                 # Buffer system tests
│   └── api/                    # API layer tests
│
├── integration/                # Integration tests for component interactions
│   ├── __init__.py
│   ├── rag/                    # RAG integration tests
│   │   ├── __init__.py
│   │   └── test_contextual_retrieval_integration.py  # Contextual retrieval integration
│   ├── llm/                    # LLM integration tests
│   │   ├── __init__.py
│   │   └── test_llm_integration.py  # LLM provider integration
│   ├── chunking/               # Chunking integration tests
│   │   ├── __init__.py
│   │   └── test_chunking_integration.py  # Advanced chunking integration
│   └── test_chunking_integration.py  # Legacy chunking integration tests
│
└── e2e/                        # End-to-end tests for full system functionality
    ├── __init__.py
    └── test_chunking.py         # E2E chunking functionality tests
```

## 🧪 Test Categories

### Unit Tests (`tests/unit/`)
- Test individual components in isolation
- Fast execution (< 1 second per test)
- Mock external dependencies
- High code coverage focus

### Integration Tests (`tests/integration/`)
- Test component interactions and complete workflows
- Medium execution time (1-10 seconds per test)
- Test real component integration without external dependencies
- Focus on interface contracts and data flow
- **RAG Integration**: Advanced contextual retrieval and three-layer strategies
- **LLM Integration**: LLM provider integration with chunking enhancement
- **Chunking Integration**: Complete chunking workflows with real conversation data

### End-to-End Tests (`tests/e2e/`)
- Test complete user workflows
- Slower execution (10+ seconds per test)
- Test against running system
- Focus on user scenarios

## 🏃‍♂️ Running Tests

### Prerequisites

```bash
# Install test dependencies (using Poetry)
poetry install --with dev

# Or install manually
pip install pytest pytest-asyncio aiohttp pytest-cov

# Ensure MemFuse server is running for E2E tests
python -m memfuse_core.server
```

### Run All Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=src/memfuse_core --cov-report=html
```

### Run Specific Test Categories

```bash
# Run only unit tests
pytest tests/unit/ -m unit

# Run only integration tests
pytest tests/integration/ -m integration

# Run only E2E tests (requires running server)
pytest tests/e2e/ -m e2e

# Run only chunking-related tests
pytest -m chunking

# Run only fast tests (exclude slow tests)
pytest -m "not slow"
```

### Run Specific Test Files

```bash
# Run chunking unit tests
pytest tests/unit/rag/chunk/

# Run specific strategy tests
pytest tests/unit/rag/chunk/test_message_chunk_strategy.py

# Run integration tests
pytest tests/integration/

# Run specific integration test categories
pytest tests/integration/rag/  # RAG integration tests
pytest tests/integration/llm/  # LLM integration tests
pytest tests/integration/chunking/  # Chunking integration tests

# Run E2E tests
pytest tests/e2e/test_chunking.py
```

## 🏷️ Test Markers

Tests are marked with the following markers for easy filtering:

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests  
- `@pytest.mark.e2e` - End-to-end tests
- `@pytest.mark.chunking` - Chunking-related tests
- `@pytest.mark.slow` - Slow-running tests
- `@pytest.mark.api` - API tests
- `@pytest.mark.buffer` - Buffer system tests
- `@pytest.mark.services` - Service layer tests
- `@pytest.mark.rag` - RAG functionality tests

## 🔧 Test Configuration

### Pytest Configuration (`pytest.ini`)

The test suite is configured with:
- Automatic test discovery
- Async test support
- Custom markers
- Timeout settings
- Logging configuration

### Shared Fixtures (`conftest.py`)

Common fixtures available to all tests:
- `sample_messages` - Sample message data
- `sample_message_batch` - Sample message batch data
- `mock_config` - Mock configuration
- `mock_vector_store` - Mock vector store
- `mock_keyword_store` - Mock keyword store
- `mock_graph_store` - Mock graph store
- `mock_memory_service` - Mock memory service
- `mock_buffer_service` - Mock buffer service

## 📊 Test Coverage

### Current Coverage Areas

✅ **Chunking System**
- ChunkData class functionality
- ChunkStrategy abstract base class
- MessageChunkStrategy implementation
- ContextualChunkStrategy implementation
- CharacterChunkStrategy implementation
- Strategy integration with services
- Error handling and edge cases

### Coverage Goals

- **Unit Tests**: >90% code coverage
- **Integration Tests**: All major component interactions
- **E2E Tests**: All user-facing workflows

## 🐛 Debugging Tests

### Running Tests in Debug Mode

```bash
# Run with debug output
pytest -s -v --tb=long

# Run specific test with debugging
pytest tests/unit/rag/chunk/test_base.py::TestChunkData::test_chunk_data_creation -s -v

# Run with pdb on failure
pytest --pdb

# Use test runner script
python tests/run_tests.py --verbose --type unit

# Verify test structure
python tests/verify_structure.py
```

### Common Issues

1. **Import Errors**: Ensure `src/` is in Python path
2. **Async Test Failures**: Check `pytest-asyncio` is installed
3. **E2E Test Failures**: Ensure MemFuse server is running
4. **Mock Issues**: Check fixture dependencies in `conftest.py`

## 📝 Writing New Tests

### Test Naming Convention

- Test files: `test_*.py`
- Test classes: `Test*`
- Test methods: `test_*`

### Example Unit Test

```python
import pytest
from memfuse_core.rag.chunk import MessageChunkStrategy

class TestMyComponent:
    @pytest.fixture
    def component(self):
        return MessageChunkStrategy()
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_my_functionality(self, component):
        """Test description."""
        result = await component.create_chunks([])
        assert result == []
```

### Example Integration Test

```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_component_integration(self, mock_memory_service):
    """Test component integration."""
    result = await mock_memory_service.add_batch([])
    assert "status" in result
```

### Example E2E Test

```python
@pytest.mark.e2e
@pytest.mark.asyncio
async def test_user_workflow(self, http_session, test_user):
    """Test complete user workflow."""
    # Test implementation
    pass
```

## 🚀 Continuous Integration

### GitHub Actions Integration

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.11
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run unit tests
        run: pytest tests/unit/ -m unit
      - name: Run integration tests
        run: pytest tests/integration/ -m integration
```

## 📈 Performance Testing

### Benchmark Tests

```bash
# Run performance tests
pytest tests/integration/ -m "chunking and not slow" --benchmark-only
```

### Memory Usage Testing

```bash
# Run with memory profiling
pytest tests/unit/ --memray
```
