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

## Performance & Soak Testing

One-shot profiler (seed → run → report):
```bash
# Runs a single profile with progress bar and generates summary.md/html
poetry run python scripts/perf/profile_run.py \
  --host http://localhost:8000 \
  --profile load \
  --users 50 --spawn 10 --runtime 20m

# Optional seeding
poetry run python scripts/perf/profile_run.py --host http://localhost:8000 --profile load \
  --users 50 --spawn 10 --runtime 20m --seed \
  --seed-users 10 --seed-agents 2 --seed-sessions-per-user 3 --seed-messages-per-session 50
```

Soak series harness (matrix of users × durations):
```bash
# Defaults: --host http://localhost:8765, --durations 30m, --users-series 500
poetry run python scripts/perf/soak_series.py \
  --profile soak \
  --users-series 100,500 \
  --durations 30m \
  --spawn 5 \
  --db-interval 60s \
  --snapshot-hourly \
  --allow-failures

# Dry-run to inspect plan
poetry run python scripts/perf/soak_series.py --profile soak --users-series 100,500 --durations 30m --dry-run
 
# Specify DB port for the series (env picked up by inner runs)
POSTGRES_PORT=5432 \
poetry run python scripts/perf/soak_series.py --profile soak --users-series 500 --durations 30m --spawn 5

# Or provide a full DSN
DB_DSN="host=localhost port=5432 dbname=memfuse user=postgres password=postgres" \
poetry run python scripts/perf/soak_series.py --profile soak --users-series 500 --durations 30m
```

Live dashboard (real-time charts):
```bash
# Serve a run directory and open the dashboard at /live.html
poetry run python scripts/perf/live_dashboard.py --dir outputs/perf/<run> --host 127.0.0.1 --port 8088
# Then open http://127.0.0.1:8088/live.html
```

Reports & comparison:
```bash
# Re-generate reports for an existing run directory
poetry run python scripts/perf/aggregate_report.py --run-dir outputs/perf/<run>        # summary.md
poetry run python scripts/perf/aggregate_report.py --run-dir outputs/perf/<run> --format html  # summary.html

# Compare two runs (exit code 2 if regression)
poetry run python scripts/perf/compare_runs.py \
  --run-a outputs/perf/<runA> --run-b outputs/perf/<runB> \
  --max-p95-increase-pct 30 --max-avg-increase-pct 20 --max-fail-rate-pct 1
```

DB sampler defaults & overrides:
- Default DSN used by profile_run: `host=localhost port=54321 dbname=memfuse user=postgres password=postgres`
- Override via `DB_DSN` or `POSTGRES_HOST/PORT/DB/USER/PASSWORD` env vars.

Override DB port examples:
```bash
# One-shot run: override port only (keeps other defaults)
POSTGRES_PORT=5432 \
poetry run python scripts/perf/profile_run.py \
  --host http://localhost:8000 --profile load \
  --users 50 --spawn 10 --runtime 20m

# Or pass a full DSN explicitly
poetry run python scripts/perf/profile_run.py \
  --host http://localhost:8000 --profile load \
  --users 50 --spawn 10 --runtime 20m \
  --db-dsn "host=localhost port=5432 dbname=memfuse user=postgres password=postgres"

# Soak series: set env so each scheduled run uses your port
POSTGRES_PORT=5432 \
poetry run python scripts/perf/soak_series.py \
  --profile soak --users-series 500 --durations 30m --spawn 5 --db-interval 60s

# Or DB_DSN for series
DB_DSN="host=localhost port=5432 dbname=memfuse user=postgres password=postgres" \
poetry run python scripts/perf/soak_series.py --profile soak --users-series 500 --durations 30m
```
