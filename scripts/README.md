# MemFuse Scripts Directory

This directory contains utility scripts for MemFuse development, deployment, and database management.

## 📁 Scripts Overview

| Script | Purpose | Usage |
|--------|---------|-------|
| `memfuse_launcher.py` | **Main development launcher** | Start MemFuse with database |
| `database_manager.py` | **Database management** | Reset, recreate, validate DB |
| `run_tests.py` | **Test execution** | Run different test layers |

## 🚀 Quick Start Guide

### 1. First Time Setup & Daily Development

```bash
# Start MemFuse (database startup and logs enabled by default)
poetry run python scripts/memfuse_launcher.py

# Reset database data (keep schema)
poetry run python scripts/database_manager.py reset

# Check database status
poetry run python scripts/database_manager.py status
```

### 2. Alternative Usage

```bash
# Skip database startup (if already running)
poetry run python scripts/memfuse_launcher.py --no-start-db

# Run in background mode (no logs)
poetry run python scripts/memfuse_launcher.py --background
```

### 3. Troubleshooting

```bash
# Validate database schema
poetry run python scripts/database_manager.py validate

# Force recreate database container
poetry run python scripts/memfuse_launcher.py --recreate-db

# Completely rebuild database schema
poetry run python scripts/database_manager.py recreate
```

## 📋 Detailed Usage

### memfuse_launcher.py - Development Launcher

**Purpose**: Unified launcher for MemFuse development and deployment.

**Features**:
- ✅ Automatic database startup with Docker
- ✅ Connection pool optimization to prevent hanging
- ✅ Database optimization for pgai operations
- ✅ Health checks and connectivity validation
- ✅ Graceful shutdown with signal handling
- ✅ Background mode for production deployment
- ✅ Environment variable configuration support
- ✅ Robust error handling and detailed status messages
- ✅ Configurable timeouts and retry mechanisms
- ✅ Smart defaults with override options

**Usage**:
```bash
# Basic startup (recommended for development)
poetry run python scripts/memfuse_launcher.py

# Force recreate database if needed
poetry run python scripts/memfuse_launcher.py --recreate-db

# Production mode (background)
poetry run python scripts/memfuse_launcher.py --background

# Skip database startup
poetry run python scripts/memfuse_launcher.py --no-start-db

# Skip database optimizations
poetry run python scripts/memfuse_launcher.py --no-optimize-db
```

**Options**:
- `--start-db`: Start database container (default: True)
- `--no-start-db`: Skip starting database container
- `--recreate-db`: Force recreate database container
- `--optimize-db`: Apply database optimizations (default: True)
- `--no-optimize-db`: Skip database optimizations
- `--show-logs`: Show server logs (default: True)
- `--background`: Run server in background (disables logs)
- `--timeout TIMEOUT`: Startup timeout in seconds
- `--version`: Show launcher version

**Environment Variables**:
- `MEMFUSE_START_DB`: Start database container (default: true)
- `MEMFUSE_RECREATE_DB`: Force recreate database container (default: false)
- `MEMFUSE_OPTIMIZE_DB`: Apply database optimizations (default: true)
- `MEMFUSE_SHOW_LOGS`: Show server logs (default: true)
- `MEMFUSE_BACKGROUND`: Run in background mode (default: false)
- `MEMFUSE_TIMEOUT`: Startup timeout in seconds

**Advanced Features**:
- **Smart Configuration**: Environment variables provide defaults, command-line arguments override
- **Health Monitoring**: Automatic health checks for MemFuse server in background mode
- **Robust Error Handling**: Detailed error messages and graceful failure handling
- **Process Management**: Proper signal handling and graceful shutdown
- **Timeout Management**: Configurable timeouts for all operations
- **Status Reporting**: Color-coded status messages with icons for better visibility

### database_manager.py - Database Management

**Purpose**: Comprehensive database operations for MemFuse.

**Architecture Note**: MemFuse implements its own **pgai-like functionality** and does not require TimescaleDB's official pgai extension. Our custom implementation provides event-driven embedding generation through PostgreSQL triggers and NOTIFY/LISTEN mechanisms.

**Features**:
- ✅ Database reset (clear data, keep schema)
- ✅ Schema recreation with complete rebuild
- ✅ Schema validation and health checks
- ✅ Database status reporting
- ✅ Custom pgai-like trigger system setup
- ✅ Vector extension management (pgvector)
- ✅ TimescaleDB integration (optional)
- ✅ Environment variable configuration support
- ✅ Robust error handling and retry mechanisms
- ✅ Configurable timeouts and connection settings
- ✅ Color-coded status messages and detailed reporting

**Commands**:

#### `status` - Database Status
```bash
poetry run python scripts/database_manager.py status
```
- Shows container status
- Lists all tables
- Shows record counts
- Displays installed extensions

#### `validate` - Schema Validation
```bash
poetry run python scripts/database_manager.py validate
```
- Validates m0_episodic table structure
- Checks immediate trigger configuration
- Verifies notification functions
- Confirms vector extension installation

#### `reset` - Clear Data (Keep Schema)
```bash
poetry run python scripts/database_manager.py reset
```
- ⚠️ **Clears all data from m0_episodic table**
- ✅ **Preserves table structure and triggers**
- ✅ **Safe for development data reset**

#### `recreate` - Rebuild Complete Schema
```bash
poetry run python scripts/database_manager.py recreate
```
- ⚠️ **DANGER: Drops ALL existing tables**
- ✅ **Recreates complete database schema**
- ✅ **Rebuilds triggers and functions**
- ✅ **Handles optional extensions gracefully**
- **Requires confirmation prompt**

**Options**:
- `--container CONTAINER`: Database container name (overrides MEMFUSE_DB_CONTAINER)
- `--timeout TIMEOUT`: Command timeout in seconds (overrides MEMFUSE_DB_TIMEOUT)
- `--retry-count COUNT`: Number of retries (overrides MEMFUSE_DB_RETRY_COUNT)
- `--version`: Show database manager version

**Environment Variables**:
- `MEMFUSE_DB_CONTAINER`: Database container name (default: memfuse-pgai-postgres)
- `MEMFUSE_DB_NAME`: Database name (default: memfuse)
- `MEMFUSE_DB_USER`: Database user (default: postgres)
- `MEMFUSE_DB_TIMEOUT`: Command timeout in seconds (default: 60)
- `MEMFUSE_DB_RETRY_COUNT`: Number of retries for failed operations (default: 3)
- `MEMFUSE_DB_RETRY_DELAY`: Delay between retries in seconds (default: 2)

**Advanced Usage**:
```bash
# Use custom timeout
export MEMFUSE_DB_TIMEOUT=120
poetry run python scripts/database_manager.py recreate

# Use different container
poetry run python scripts/database_manager.py --container my-postgres status

# Increase retry count for unreliable connections
poetry run python scripts/database_manager.py --retry-count 5 validate
```

### run_tests.py - Test Execution

**Purpose**: Execute different layers of tests.

**Available Test Layers**:
```bash
poetry run python scripts/run_tests.py smoke        # Quick smoke tests
poetry run python scripts/run_tests.py contract     # Contract tests
poetry run python scripts/run_tests.py integration  # Integration tests
poetry run python scripts/run_tests.py retrieval    # Retrieval tests
poetry run python scripts/run_tests.py e2e          # End-to-end tests
poetry run python scripts/run_tests.py perf         # Performance tests
poetry run python scripts/run_tests.py slow         # Slow/comprehensive tests
```

## 🔄 Common Workflows

### Development Workflow

1. **Start Development Session**:
   ```bash
   poetry run python scripts/memfuse_launcher.py
   ```

2. **Reset Data Between Tests**:
   ```bash
   poetry run python scripts/database_manager.py reset
   ```

3. **Validate Setup**:
   ```bash
   poetry run python scripts/database_manager.py validate
   ```

4. **Run Tests**:
   ```bash
   poetry run python scripts/run_tests.py smoke
   ```

### Troubleshooting Workflow

1. **Check Database Status**:
   ```bash
   poetry run python scripts/database_manager.py status
   ```

2. **Validate Schema**:
   ```bash
   poetry run python scripts/database_manager.py validate
   ```

3. **If Issues Found, Recreate Schema**:
   ```bash
   poetry run python scripts/database_manager.py recreate
   ```

4. **Restart MemFuse**:
   ```bash
   poetry run python scripts/memfuse_launcher.py --recreate-db
   ```

### Production Deployment

1. **Start in Background**:
   ```bash
   poetry run python scripts/memfuse_launcher.py --background
   ```

2. **Validate Deployment**:
   ```bash
   poetry run python scripts/database_manager.py validate
   poetry run python scripts/run_tests.py smoke
   ```

## ⚠️ Important Notes

### Safety Warnings

- **`database_manager.py reset`**: Clears data but preserves schema
- **`database_manager.py recreate`**: **DESTROYS ALL DATA** - use with extreme caution
- **`memfuse_launcher.py --recreate-db`**: Recreates Docker container, may lose data

### Prerequisites

- Docker and Docker Compose installed
- PostgreSQL container running (handled by launcher)
- Poetry environment activated
- MemFuse dependencies installed

### Troubleshooting

**Connection Pool Issues**:
- Use `--optimize-db` flag (enabled by default)
- Check Docker container status
- Restart with `--recreate-db` if needed

**Database Schema Issues**:
- Run `validate` command first
- Use `recreate` command as last resort
- Check container logs: `docker logs memfuse-pgai-postgres`

**Test Failures**:
- Ensure database is running and validated
- Reset data with `reset` command
- Check MemFuse server is running on port 8000

**Launcher Issues**:
- Check environment variables with `--help` to see current defaults
- Use `--version` to verify launcher version
- For debugging, run with explicit flags: `--start-db --show-logs`
- If health checks fail, verify MemFuse server is accessible on port 8000
- Use `--timeout` to adjust startup timeouts if needed

**Database Manager Issues**:
- Use `--version` to verify database manager version
- Check container status first: `poetry run python scripts/database_manager.py status`
- **Extension Notes**:
  - ✅ **pgvector**: Required for vector operations (should always be available)
  - ✅ **timescaledb**: Optional, provides additional time-series features
  - ❌ **pgai**: Not needed - MemFuse has its own pgai-like implementation
- Use `--timeout` to adjust timeouts for slow database operations
- Use `--retry-count` to increase retries for unreliable connections
- Check environment variables with `--help` to see current configuration

## 📋 Quick Reference

### Most Common Commands

```bash
# 🚀 Start development environment
poetry run python scripts/memfuse_launcher.py

# 📊 Check database status
poetry run python scripts/database_manager.py status

# 🔄 Reset data (keep schema)
poetry run python scripts/database_manager.py reset

# ✅ Validate setup
poetry run python scripts/database_manager.py validate

# 🧪 Run quick tests
poetry run python scripts/run_tests.py smoke
```

### Emergency Commands

```bash
# 🆘 Complete system reset
poetry run python scripts/database_manager.py recreate
poetry run python scripts/memfuse_launcher.py --recreate-db

# 🔍 Troubleshoot connection issues
poetry run python scripts/memfuse_launcher.py --no-start-db
```

## 🔗 Related Documentation

- [PgAI Architecture](../docs/architecture/pgai.md)
- [Docker Setup](../docker/README.md)
- [Testing Guide](../tests/README.md)
