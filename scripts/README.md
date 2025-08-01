# MemFuse Scripts

Core development and deployment utilities for MemFuse.

## Scripts

| Script | Purpose | Key Features |
|--------|---------|--------------|
| `memfuse_launcher.py` | Development launcher | Database startup, health checks, background mode |
| `database_manager.py` | Database operations | Schema management, validation, data reset |
| `run_tests.py` | Test execution | Multi-layer testing, server management |

## Quick Start

```bash
# Development workflow
poetry run python scripts/memfuse_launcher.py              # Start MemFuse + database
poetry run python scripts/database_manager.py status       # Check database
poetry run python scripts/run_tests.py smoke              # Run tests

# Database management
poetry run python scripts/database_manager.py reset       # Clear data, keep schema
poetry run python scripts/database_manager.py recreate    # Rebuild schema (destructive)

# Production deployment
poetry run python scripts/memfuse_launcher.py --background # Background mode
```

## memfuse_launcher.py

**Core Features**: Database startup, health checks, background mode, signal handling

**Key Options**:
```bash
--background          # Production mode (no logs)
--recreate-db         # Force database container recreation
--no-start-db         # Skip database startup
--timeout SECONDS     # Startup timeout
```

**Environment Variables**: `MEMFUSE_START_DB`, `MEMFUSE_BACKGROUND`, `MEMFUSE_TIMEOUT`

## database_manager.py

**Architecture**: Custom pgai-like implementation with PostgreSQL triggers and NOTIFY/LISTEN

**Commands**:
```bash
status      # Container status, tables, record counts, extensions
validate    # Schema validation, trigger configuration, pgvector check
reset       # Clear data, preserve schema (safe for development)
recreate    # Rebuild complete schema (destructive, requires confirmation)
```

**Key Options**: `--container`, `--timeout`, `--retry-count`
**Environment**: `MEMFUSE_DB_CONTAINER`, `MEMFUSE_DB_TIMEOUT`, `MEMFUSE_DB_RETRY_COUNT`

## run_tests.py

**Test Layers**: `smoke`, `contract`, `integration`, `retrieval`, `e2e`, `perf`, `slow`

**Client Types**:
- `--client-type=server` (default): HTTP server, shows requests in logs
- `--client-type=testclient`: In-process, faster, isolated

**Server Management**:
- Default: Restart server after database reset (clean connections)
- `--no-restart-server`: Keep development server running

**Examples**:
```bash
# Layer testing
poetry run python scripts/run_tests.py integration -v

# Specific test with server monitoring
poetry run python scripts/run_tests.py --no-restart-server tests/integration/api/test_users_api_integration.py -v -s

# Custom pytest flags
poetry run python scripts/run_tests.py integration -k "user" --tb=short
```

## Common Workflows

**Development**:
```bash
poetry run python scripts/memfuse_launcher.py           # Start session
poetry run python scripts/database_manager.py reset    # Reset data between tests
poetry run python scripts/run_tests.py smoke           # Validate setup
```

**Troubleshooting**:
```bash
poetry run python scripts/database_manager.py status   # Check database
poetry run python scripts/database_manager.py validate # Validate schema
poetry run python scripts/database_manager.py recreate # Rebuild if needed
poetry run python scripts/memfuse_launcher.py --recreate-db # Restart with fresh DB
```

**Production**:
```bash
poetry run python scripts/memfuse_launcher.py --background # Deploy
poetry run python scripts/run_tests.py smoke              # Validate
```

## Important Notes

**Safety Warnings**:
- `reset`: Clears data, preserves schema
- `recreate`: **DESTROYS ALL DATA** - use with caution
- `--recreate-db`: Recreates Docker container

**Prerequisites**: Docker, Poetry environment, MemFuse dependencies

**Extensions**:
- ✅ **pgvector**: Required for vector operations
- ✅ **timescaledb**: Optional time-series features
- ❌ **pgai**: Not needed - MemFuse has custom implementation

**Troubleshooting**:
- Connection issues: Check container status, use `--recreate-db`
- Schema issues: Run `validate` first, `recreate` as last resort
- Test failures: Ensure database running, reset data, check port 8000
