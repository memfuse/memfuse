# MemFuse Test Suite

Comprehensive test suite for MemFuse system with layered architecture and multiple execution modes.

## Test Layers

| Layer | Purpose | Examples |
|-------|---------|----------|
| `smoke/` | Quick health checks | API availability, database connectivity |
| `contract/` | API contract validation | Request/response schemas, error codes |
| `integration/` | Component interactions | Database integration, service communication |
| `unit/` | Individual components | RAG chunking, memory layers, services |
| `retrieval/` | RAG and search | Vector retrieval, semantic search |
| `e2e/` | End-to-end workflows | Complete user journeys |
| `perf/` | Performance benchmarks | Load testing, response times |

## Quick Start

```bash
# Using run_tests.py (recommended)
poetry run python scripts/run_tests.py smoke        # Quick validation
poetry run python scripts/run_tests.py integration  # Full integration tests
poetry run python scripts/run_tests.py --client-type=testclient unit  # Fast unit tests

# Direct pytest usage
poetry run pytest tests/smoke/ -v                   # Smoke tests
poetry run pytest tests/integration/connection_pool/ -v  # Specific component
poetry run pytest -k "test_user" --tb=short        # Pattern matching
```

## Configuration Options

**Client Types**:
- `--client-type=server`: HTTP server (shows requests in logs)
- `--client-type=testclient`: In-process TestClient (faster, isolated)

**Server Management**:
- Default: Restart server after database reset (clean connections)
- `--no-restart-server`: Keep development server running

**Database Options**:
- Automatic database reset for clean test state
- Custom database URL via environment variables

## Direct pytest Usage

```bash
# Prerequisites
poetry install --with dev

# Basic execution
pytest tests/smoke/ -v                    # Quick validation
pytest tests/integration/connection_pool/ -v  # Specific component
pytest --cov=src/memfuse_core --cov-report=html  # With coverage
```

## Test Markers

Key markers for filtering tests:
- `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.e2e`
- `@pytest.mark.chunking`, `@pytest.mark.slow`, `@pytest.mark.api`

## Debugging

```bash
# Debug mode
pytest -s -v --tb=long

# Specific test debugging
pytest tests/unit/rag/chunk/test_base.py::TestChunkData::test_chunk_data_creation -s -v

# Debug on failure
pytest --pdb
```

## Writing Tests

**Naming**: `test_*.py`, `Test*` classes, `test_*` methods

**Example**:
```python
@pytest.mark.unit
@pytest.mark.asyncio
async def test_my_functionality(self, component):
    result = await component.create_chunks([])
    assert result == []
```
