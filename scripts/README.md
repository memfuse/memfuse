# MemFuse Scripts Directory

This directory contains utility scripts for MemFuse development, deployment, and database management.

## 📁 Scripts Overview

| Script | Purpose | Usage |
|--------|---------|-------|
| `memfuse_launcher.py` | **Main development launcher** | Start MemFuse with database |
| `database_manager.py` | **Database management** | Reset, recreate, validate DB |
| `run_tests.py` | **Test execution** | Run different test layers |

## 🚀 Quick Start Guide

### 1. First Time Setup

```bash
# Start MemFuse with database setup and optimization
poetry run python scripts/memfuse_launcher.py --start-db --optimize-db
```

### 2. Daily Development

```bash
# Start development server with logs
poetry run python scripts/memfuse_launcher.py --start-db --show-logs

# Reset database data (keep schema)
poetry run python scripts/database_manager.py reset

# Check database status
poetry run python scripts/database_manager.py status
```

### 3. Troubleshooting

```bash
# Validate database schema
poetry run python scripts/database_manager.py validate

# Force recreate database container
poetry run python scripts/memfuse_launcher.py --recreate-db --optimize-db

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

**Usage**:
```bash
# Basic startup (recommended for development)
poetry run python scripts/memfuse_launcher.py --start-db --optimize-db

# Force recreate database if needed
poetry run python scripts/memfuse_launcher.py --recreate-db --optimize-db

# Production mode (background)
poetry run python scripts/memfuse_launcher.py --start-db --background

# Development with full logs
poetry run python scripts/memfuse_launcher.py --start-db --show-logs

# Skip database optimizations
poetry run python scripts/memfuse_launcher.py --start-db --no-optimize-db
```

**Options**:
- `--start-db`: Start database container
- `--recreate-db`: Force recreate database container
- `--optimize-db`: Apply database optimizations (default: True)
- `--show-logs`: Show server logs (default: True)
- `--background`: Run server in background
- `--timeout TIMEOUT`: Startup timeout in seconds

### database_manager.py - Database Management

**Purpose**: Comprehensive database operations for MemFuse.

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
- **Requires confirmation prompt**

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
   poetry run python scripts/memfuse_launcher.py --start-db --show-logs
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
   poetry run python scripts/memfuse_launcher.py --recreate-db --optimize-db
   ```

### Production Deployment

1. **Start in Background**:
   ```bash
   poetry run python scripts/memfuse_launcher.py --start-db --background --optimize-db
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

## 📋 Quick Reference

### Most Common Commands

```bash
# 🚀 Start development environment
poetry run python scripts/memfuse_launcher.py --start-db --show-logs

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
poetry run python scripts/memfuse_launcher.py --recreate-db --optimize-db

# 🔍 Troubleshoot connection issues
poetry run python scripts/memfuse_launcher.py --start-db --optimize-db
```

## 🔗 Related Documentation

- [PgAI Architecture](../docs/architecture/pgai.md)
- [Docker Setup](../docker/README.md)
- [Testing Guide](../tests/README.md)
