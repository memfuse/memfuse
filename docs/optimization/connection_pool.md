# PostgreSQL Connection Pool Architecture & Optimization

## Authoritative, Active Configuration (2025-09)

This section supersedes any legacy references below. It documents the actual code paths and configuration that are in effect in the current codebase.

- **Active pool owner**: `GlobalConnectionManager` (`src/memfuse_core/services/global_connection_manager.py`)
- **Who uses it**: `PgaiStore` (`src/memfuse_core/store/pgai_store/pgai_store.py`) acquires a shared pool from the manager during initialization
- **Pool implementation**: `psycopg_pool.AsyncConnectionPool`
- **Sizing semantics**: `min_size = pool_size`, `max_size = pool_size + max_overflow`, `timeout` is the acquisition timeout supplied to the pool
- **Configuration priority** (highest → lowest):
  1. Environment variables: `POSTGRES_POOL_SIZE`, `POSTGRES_MAX_OVERFLOW`, `POSTGRES_POOL_TIMEOUT`, `POSTGRES_CONNECTION_TIMEOUT`
  2. `store.database.postgres.*`
  3. `database.postgres.*` (default and most common)
  4. Root `postgres.*` (fallback)

Where the pool is created and sized:

```python
## 384:405:src/memfuse_core/services/global_connection_manager.py

logger.info(f"GlobalConnectionManager: Creating pool config from config type={type(config)}")
logger.info(f"GlobalConnectionManager: Raw config keys: {list(config.keys()) if isinstance(config, dict) else 'Not a dict'}")
pool_config = ConnectionPoolConfig.from_memfuse_config(config)
logger.info(f"GlobalConnectionManager: Pool config created: min_size={pool_config.min_size}, max_size={pool_config.max_size}, timeout={pool_config.timeout}, connection_timeout={pool_config.connection_timeout}")
self._pool_configs[db_url] = pool_config

# Create the pool
pool = AsyncConnectionPool(
    db_url,
    min_size=pool_config.min_size,
    max_size=pool_config.max_size,
    open=False,
    configure=self._configure_connection,
    timeout=min(pool_config.timeout, 30.0)
)
```

Who calls it (during store initialization):

```python
## 228:237:src/memfuse_core/store/pgai_store/pgai_store.py

# Use global connection manager (Tier 1 Singleton)
logger.debug("Getting global connection pool...")
connection_manager = get_global_connection_manager()

# Get shared connection pool with store reference for tracking
self.pool = await connection_manager.get_connection_pool(
    db_url=self.db_url,
    config=self.db_config,
    store_ref=self  # Pass self for reference tracking
)
```

Configuration extraction and priority:

```python
## 123:194:src/memfuse_core/services/global_connection_manager.py

@dataclass
class ConnectionPoolConfig:
    min_size: int = 2
    max_size: int = 8
    timeout: float = 15.0
    recycle: int = 1800
    connection_timeout: float = 10.0
    keepalives_idle: int = 300
    keepalives_interval: int = 15
    keepalives_count: int = 3

@classmethod
def from_memfuse_config(cls, config: Dict[str, Any]) -> 'ConnectionPoolConfig':
    # Priority: env > store.database.postgres > database.postgres > postgres > defaults
    env_pool_size = os.environ.get("POSTGRES_POOL_SIZE")
    env_max_overflow = os.environ.get("POSTGRES_MAX_OVERFLOW")
    env_pool_timeout = os.environ.get("POSTGRES_POOL_TIMEOUT")
    env_connection_timeout = os.environ.get("POSTGRES_CONNECTION_TIMEOUT")

    pool_size = int(env_pool_size) if env_pool_size else postgres_config.get("pool_size", 5)
    max_overflow = int(env_max_overflow) if env_max_overflow else postgres_config.get("max_overflow", 15)
    max_size = pool_size + max_overflow
    pool_timeout = float(env_pool_timeout) if env_pool_timeout else postgres_config.get("pool_timeout", 300.0)
    connection_timeout = float(env_connection_timeout) if env_connection_timeout else postgres_config.get("connection_timeout", 60.0)
    return cls(
        min_size=pool_size,
        max_size=max_size,
        timeout=pool_timeout,
        recycle=postgres_config.get("pool_recycle", 7200),
        keepalives_idle=postgres_config.get("keepalives_idle", 600),
        keepalives_interval=postgres_config.get("keepalives_interval", 30),
        keepalives_count=postgres_config.get("keepalives_count", 3)
    )
```

Default effective settings (repository defaults):

```yaml
## 8:23:config/database/default.yaml

postgres:
  host: ${oc.env:POSTGRES_HOST,localhost}
  port: ${oc.env:POSTGRES_PORT,5432}
  database: ${oc.env:POSTGRES_DB,memfuse}
  user: ${oc.env:POSTGRES_USER,postgres}
  password: ${oc.env:POSTGRES_PASSWORD,postgres}
  # Increased connection pool for better concurrency
  pool_size: 20          # Reasonable base pool for streaming
  max_overflow: 30       # Moderate burst capacity (total 50 connections)
  pool_timeout: 2.0      # Fast acquisition timeout
  pool_recycle: 600      # 10-minute recycle
  connection_timeout: 2.0
  keepalives_idle: 60
  keepalives_interval: 10
  keepalives_count: 3
```

To apply config changes at runtime without a full restart, you can force pools to rebuild:

```python
## 834:844:src/memfuse_core/services/global_connection_manager.py

async def reset_global_connection_pools():
    """Reset all global connection pools and clear caches.
    ...
    manager = get_global_connection_manager()
    await manager.reset_all_pools()
```

### Sizing guidance and resource impact

- **Current default total cap = 50** (`20 + 30`). This is mid-range, not conservative.
- **Increase capacity safely**:
  - Prefer increasing `max_overflow` first to raise the cap without inflating always-on idle connections.
  - Example caps:
    - 100 total: keep `pool_size: 20`, set `max_overflow: 80`.
    - 200 total: `pool_size: 30`, `max_overflow: 170` (only if DB resources allow).
- **Memory footprint (DB backend)**: ~5–15 MB per backend connection (idle). Doubling from 50 → 100 may add ~0.5–1.0 GB.
- **Check DB limits**:
  - `SHOW max_connections;` and monitor `pg_stat_activity` to ensure you don’t exhaust the server limit.
- **Semantics reminder**:
  - `pool_size` drives minimum/steady connections (min_size). Raising it increases always-reserved connections.
  - `max_overflow` raises the ceiling for spikes; prefer scaling this first.

### How to change values

- Edit `config/database/default.yaml` → `database.postgres.{pool_size,max_overflow}`.
- Or export env vars (highest priority): `POSTGRES_POOL_SIZE`, `POSTGRES_MAX_OVERFLOW`, `POSTGRES_POOL_TIMEOUT`, `POSTGRES_CONNECTION_TIMEOUT`.
- Then restart the service, or call `reset_global_connection_pools()` to rebuild pools with the new config.

### Verification & monitoring

- App-side pool stats:
  - `GlobalConnectionManager.get_pool_statistics()` shows `min_size`, `max_size`, `active_references`, and closed state per URL.
- DB-side:
  - `SELECT state, count(*) FROM pg_stat_activity GROUP BY 1;`
  - Watch wait events/locks and long-running transactions.

### Upgrade and optimization roadmap

- **PgBouncer in front of Postgres (transaction pooling)**
  - Caps backend connections at the proxy; lets the app handle large spikes without inflating DB processes.
- **Adaptive/dynamic pool sizing**
  - Auto-tune `max_overflow` (and optionally `pool_size`) based on recent acquisition latency, rejection rate, and utilization.
- **Workload-aware pools**
  - Optional split pools (read-mostly vs write-heavy) or per-tenant/critical-path pools if proven beneficial.
- **Acquisition fairness & backpressure**
  - Tune `timeout` and add structured backoff/queuing to smooth spikes.
- **Keepalive and network resilience**
  - Keep current keepalive defaults; review for NAT/gateway environments (shorter idle helps).
- **Operational safeguards**
  - Safety checks against `max_connections` at startup; proactive warnings if configured `max_size` approaches server limits.

— The following sections remain as historical and background context.

## Connection Pool Exhaustion Problem & Resolution

### Critical Issues Encountered

The original implementation suffered from severe connection pool exhaustion causing:

1. **Connection Pool Exhaustion**: PostgreSQL errors "FATAL: sorry, too many clients already"
2. **Rate Limiting Bottlenecks**: 429 Too Many Requests errors under streaming loads
3. **Smart Cleanup Interference**: Background monitoring consuming connections despite configuration
4. **Configuration Fragmentation**: Multiple conflicting configuration files

### Root Cause Analysis

The original architecture suffered from **multi-layer connection pool fragmentation**:

```
API Layer → DatabaseQueueManager → GlobalConnectionManager → SmartConnectionMonitor → PostgresDB
```

Each layer attempted to "optimize" connections but actually created:
- **Resource Competition**: 300 connection requests vs PostgreSQL's 100 connection limit
- **Connection Leakage**: Complex cleanup logic with race conditions
- **Configuration Conflicts**: Settings scattered across multiple files

### Failed Optimization Attempts

#### Attempt 1: Per-Store Individual Pools
**Approach**: Create separate connection pools for each store instance
**Problems Encountered**:
- Resource waste (30+ connections for 5 users)
- Connection limit exhaustion under load
- No sharing between memory layers (M0/M1/M2)
**Verdict**: ❌ Rejected due to scalability issues

#### Attempt 2: Enhanced Queue Management
**Approach**: Improve DatabaseQueueManager with better queueing logic
**Problems Encountered**:
- Added complexity without solving root connection leakage
- Queue bottlenecks under concurrent operations
- Smart cleanup still interfering with connections
**Verdict**: ❌ Rejected as band-aid solution

#### Attempt 3: Smart Monitoring Optimization
**Approach**: Optimize SmartConnectionMonitor to use fewer connections
**Problems Encountered**:
- Background monitoring still consumed connections
- Race conditions between monitoring and application usage
- Health checks broke pool sharing semantics
**Verdict**: ❌ Rejected due to fundamental architectural issues

### Final Solution: Simplified Architecture

**Core Design Philosophy**: **"Simplicity Over Complexity"** - Replace multi-layer management with direct, efficient connection handling.

**New Architecture**:
```
API Layer → SimplifiedConnectionManager → PostgresDB (Direct Pool Access)
```

**Key Improvements**:
- **Direct Pool Access**: Eliminated intermediate queuing and monitoring layers
- **Streaming Optimization**: Large timeout values for sustained loads
- **Minimal Overhead**: Single responsibility for connection management
- **Smart Cleanup Disabled**: Background interference eliminated

### Solution Implementation

#### 1. Unified Configuration System

**File**: `src/memfuse_core/config/unified.py`

```python
@dataclass
class ConnectionPoolConfig:
    """Streaming-optimized connection pool configuration."""
    min_size: int = 20              # Reasonable base pool
    max_size: int = 50              # Total: 20 + 30 overflow
    timeout: float = 30.0           # Generous timeout
    connection_timeout: float = 10.0
    recycle: int = 3600            # 1-hour connection reuse
```

#### 2. Simplified Connection Manager

**File**: `src/memfuse_core/services/simplified_connection_manager.py`

Key features:
- **Direct Pool Access**: No intermediate queuing or monitoring
- **Streaming Optimization**: Large timeout values for sustained loads
- **Minimal Overhead**: Single responsibility for connection management
- **pgvector Support**: Built-in vector extension registration

#### 3. Optimized PostgreSQL Backend

**File**: `src/memfuse_core/database/postgres.py`

```python
async def _execute_with_simplified_connection(self, query: str, params: tuple) -> Any:
    """Execute query with simplified connection management optimized for streaming."""
    conn = None
    try:
        conn = await self.connection_manager.get_connection(self.db_url)
        # Direct execution without complex monitoring
        async with conn.cursor(row_factory=dict_row) as cursor:
            await cursor.execute(query, params)
            # ... handle results
    finally:
        if conn is not None:
            await self.connection_manager.return_connection(conn, self.db_url)
```

### Performance Results

#### Before Optimization
- **Connection Limit**: 300 requests vs 100 PostgreSQL limit
- **Error Rate**: High "too many clients" errors
- **Rate Limiting**: 60 requests/minute causing 429 errors
- **Smart Cleanup**: Background interference every 15 seconds

#### After Optimization
- **Connection Usage**: 50 connections max (well within PostgreSQL limits)
- **Error Rate**: Zero connection exhaustion errors
- **Rate Limiting**: 10,000 requests/minute supporting streaming
- **Background Tasks**: Eliminated monitoring interference

#### Validation Results
```bash
# Stress Test Results
✓ 50 consecutive requests: 100% success rate
✓ No 429 Too Many Requests errors
✓ No "too many clients" PostgreSQL errors
✓ No smart cleanup interference
✓ Server stability: No crashes under load
```

## Architecture Design

### Core Problem
MemFuse's multi-layer memory architecture (M0/M1/M2) with parallel processing creates complex database connection requirements:
- Multiple memory layers accessing database simultaneously
- Per-user isolation requirements vs resource sharing efficiency
- Connection pool sizing for concurrent vector operations
- Health monitoring vs pool sharing semantics

### Design Principles
1. **Global Singleton Pattern**: One connection pool per database URL across all services
2. **Configuration Hierarchy**: Layered configuration with store-specific overrides
3. **Pool Sharing**: Multiple stores share pools while maintaining reference tracking
4. **Health vs Sharing Balance**: Structural health checks without breaking pool sharing

## Configuration Details

### Primary Configuration (Streaming Optimized)

**File**: `config/config.yaml`

```yaml
# Database Configuration - STREAMING OPTIMIZED
database:
  type: "postgres"
  postgres:
    # Connection Pool Settings
    pool_size: 20               # Reasonable base pool
    max_overflow: 30            # Burst capacity (total 50)
    pool_timeout: 30.0          # Connection acquisition timeout
    connection_timeout: 10.0    # Connection establishment timeout
    pool_recycle: 3600          # 1-hour connection reuse
    
    # Stability Settings
    keepalives_idle: 300        # 5-minute idle detection
    keepalives_interval: 30     # 30-second keepalive checks
    keepalives_count: 3         # Failed checks before disconnect
    
    # Disable Legacy Features
    smart_cleanup:
      enabled: false            # Disable smart monitoring

# Server Configuration
server:
  rate_limit_per_minute: 10000  # High limit for streaming

# Service Configuration - SIMPLIFIED
services:
  database_queue:
    enabled: false              # Disable queue management
  connection_monitoring:
    enabled: false              # Disable complex monitoring
```

### Store Configuration

```yaml
store:
  type: "pgai"
  multi_path:
    use_vector: true
    use_graph: false
    use_keyword: true
    fusion_strategy: "rrf"
    vector_weight: 0.7
    graph_weight: 0.0
    keyword_weight: 0.3
  cache_size: 100
```

### Implementation Changes

#### 1. Database Service Simplification

**File**: `src/memfuse_core/services/database_service.py`

```python
# OLD: Complex queue wrapping
cls._instance = QueuedDatabase(base_database)

# NEW: Direct database usage
cls._instance = base_database
```

#### 2. App Service Configuration

**File**: `src/memfuse_core/services/app_service.py`

```python
# Use unified configuration
from ..config.unified import get_unified_config_manager

unified_config = get_unified_config_manager()
config_dict = unified_config.get_config()
```

#### 3. Global Connection Manager Patches

**File**: `src/memfuse_core/services/global_connection_manager.py`

```python
# Force disable smart monitoring
self._monitoring_enabled = False

async def _setup_smart_monitoring(self, db_url: str, config: Dict[str, Any]):
    """Setup smart connection monitoring - DISABLED for streaming optimization"""
    logger.info("Smart monitoring disabled for streaming optimization")
    return
```

## Deployment Guidelines

### Prerequisites

1. **PostgreSQL Configuration**:
   - Default `max_connections = 100` is sufficient
   - TimescaleDB with pgvector extension
   - Docker container: `timescale/timescaledb:latest-pg17`

2. **Environment Setup**:
   ```bash
   # Required environment variables
   POSTGRES_HOST=localhost
   POSTGRES_PORT=5432
   POSTGRES_DB=memfuse
   POSTGRES_USER=postgres
   POSTGRES_PASSWORD=postgres
   ```

### Startup Commands

```bash
# Standard deployment
poetry run python scripts/memfuse_launcher.py

# Development with existing database
poetry run python scripts/memfuse_launcher.py --no-start-db

# Background mode
poetry run python scripts/memfuse_launcher.py --background
```

### Health Verification

```bash
# Basic health check
curl --noproxy localhost http://localhost:8000/api/v1/health

# Streaming load test
for i in {1..50}; do 
  curl -s --noproxy localhost http://localhost:8000/api/v1/health > /dev/null
  echo -n "✓"
  sleep 0.1
done
```

## Troubleshooting and Best Practices

### Common Issues and Solutions

#### Issue: "Connection refused" errors

**Symptoms**: `curl: (7) Failed to connect to localhost port 8000`

**Solution**:
1. Check if server is running: `poetry run python scripts/memfuse_launcher.py --background`
2. Verify port availability: `lsof -i :8000`
3. Check server logs: `tail -f logs/memfuse_core.log`

#### Issue: Rate limiting under load

**Symptoms**: `HTTP/1.1 429 Too Many Requests`

**Solution**:
1. Increase rate limit in config: `rate_limit_per_minute: 20000`
2. Implement client-side request batching
3. Add request backoff/retry logic

#### Issue: Connection pool saturation

**Symptoms**: Slow response times, connection timeouts

**Solution**:
1. Increase pool size: `pool_size: 30, max_overflow: 50`
2. Optimize query performance
3. Implement connection reuse patterns

### Best Practices

#### Configuration Management
1. **Single Source of Truth**: Use `config/config.yaml` for all settings
2. **Environment Overrides**: Use environment variables for deployment-specific values
3. **Validation**: Always validate configuration changes in development

#### Connection Management
1. **Connection Reuse**: Prefer long-lived connections over frequent creation/destruction
2. **Timeout Tuning**: Set generous timeouts for streaming scenarios
3. **Pool Sizing**: Size pools based on actual concurrency needs, not theoretical maximums

#### Performance Optimization
1. **Batch Operations**: Group small operations into larger transactions
2. **Connection Affinity**: Reuse connections for related operations
3. **Query Optimization**: Ensure efficient database queries to reduce connection hold time

### Migration Guide

#### From Previous Versions
1. **Backup Configuration**: Save existing `config/` files
2. **Update Dependencies**: Ensure all required packages are installed
3. **Deploy New Configuration**: Apply unified configuration
4. **Validate Deployment**: Run health checks and load tests
5. **Monitor Performance**: Track metrics for several days post-deployment

#### Rollback Procedure
If issues arise:
1. **Immediate**: Restore previous configuration files
2. **Service Restart**: `poetry run python scripts/memfuse_launcher.py --recreate-db`
3. **Validation**: Confirm service health
4. **Analysis**: Review logs to understand failure cause

## Legacy Architecture Analysis

### Original Solution Comparison

#### Approach 1: Per-Store Individual Pools
**Pros**: Simple isolation, no sharing complexity
**Cons**: Resource waste (30+ connections for 5 users), connection limit exhaustion
**Verdict**: ❌ Rejected due to scalability issues

#### Approach 2: Global Shared Pools (Selected)
**Pros**: Resource efficiency, configurable sizing, proper cleanup
**Cons**: Complexity in reference tracking and health monitoring
**Verdict**: ✅ Selected for optimal resource utilization

#### Approach 3: Per-User Pools
**Pros**: User isolation, moderate resource usage
**Cons**: Still creates multiple pools, doesn't solve M0/M1/M2 sharing
**Verdict**: ❌ Rejected as intermediate solution without full benefits

## Implementation Architecture

### GlobalConnectionManager (Core Component)
```python
class GlobalConnectionManager:
    """
    Singleton connection pool manager implementing:
    - Pool sharing: One pool per database URL
    - Reference tracking: Weak references for automatic cleanup
    - Health monitoring: Structural checks without breaking sharing
    - Configuration hierarchy: Store > database > postgres > defaults
    """
```

**Critical Design Decisions:**

1. **Pool Sharing Strategy**
   - **Challenge**: Multiple stores requesting same database URL
   - **Solution**: Return same pool instance with reference counting
   - **Risk**: Health check failures could break sharing semantics

2. **Health Check Balance**
   - **Original Issue**: Aggressive health checks deleted pools on connectivity failures
   - **Root Cause**: Database connectivity != pool structural health
   - **Solution**: Check `pool.closed` status, not database reachability
   - **Rationale**: Preserve sharing while detecting truly broken pools

3. **Reference Tracking**
   - **Method**: Weak references to store instances
   - **Cleanup**: Automatic when stores are garbage collected
   - **Monitoring**: Track active references per pool for diagnostics

### Configuration Strategy

**Hierarchy Design** (highest to lowest priority):
```yaml
store:
  database:
    postgres:
      pool_size: 15        # Store-specific override
database:
  postgres:
    pool_size: 20          # Database-level setting
    max_overflow: 40       # Additional connections
    pool_timeout: 60.0     # Connection acquisition timeout
    pool_recycle: 7200     # Connection lifecycle (2 hours)
postgres:
  pool_size: 10            # Base configuration
```

**Sizing Strategy for Parallel Processing:**
- **Base**: 10 connections (single-layer operations)
- **Parallel M0/M1/M2**: 20-30 connections (concurrent layer processing)
- **High Load**: 40+ connections with overflow (multiple users + parallel layers)

**Critical Parameters:**
- `pool_timeout`: Must accommodate complex vector operations (60s recommended)
- `pool_recycle`: Balance connection freshness vs overhead (2 hours optimal)
- `keepalives_*`: Prevent connection drops during long operations

## Critical Issues and Solutions

### Issue 1: Pool Sharing vs Health Checks
**Problem**: Health checks broke pool sharing by recreating pools on connectivity failures
**Root Cause**: Conflating database connectivity with pool structural health
**Solution**:
```python
# Check structural health, not database connectivity
if pool.closed:
    # Pool is structurally broken, recreate
    del self._pools[db_url]
else:
    # Pool exists and functional, share it
    return pool
```
**Risk Mitigation**: Monitor pool statistics to detect actual health issues

### Issue 2: Parallel Processing Connection Requirements
**Problem**: M0/M1/M2 parallel processing overwhelmed connection pools
**Analysis**:
- Sequential processing: 5-10 connections sufficient
- Parallel processing: 20-30 connections required
- Multiple users + parallel: 40+ connections needed
**Solution**: Dynamic configuration based on parallel_enabled setting

### Issue 3: Reference Tracking Complexity
**Problem**: Determining when to close shared pools
**Solution**: Weak reference counting with automatic cleanup
**Trade-off**: Slight memory overhead vs robust lifecycle management

## Performance Analysis

### Resource Utilization Comparison
| Scenario | Before (Individual Pools) | After (Shared Pools) | Improvement |
|----------|---------------------------|---------------------|-------------|
| Single User | 6 connections (3 stores × 2) | 2-5 connections | 60% reduction |
| 5 Users | 30 connections | 10-20 connections | 50% reduction |
| Parallel M0/M1/M2 | 45+ connections | 20-30 connections | 40% reduction |

### Scalability Metrics
- **Connection Stability**: Stable count under load (verified in tests)
- **Pool Sharing**: Multiple stores confirmed sharing single pool instance
- **Configuration Responsiveness**: Pool sizing adjusts to parallel_enabled setting
- **Memory Efficiency**: Weak reference tracking prevents memory leaks

### Risk Assessment
**Low Risk**:
- Pool sharing semantics well-tested
- Configuration hierarchy provides flexibility
- Reference tracking prevents resource leaks

**Medium Risk**:
- Complex health check logic requires monitoring
- Parallel processing increases connection requirements

**Mitigation Strategies**:
- Comprehensive test coverage for edge cases
- Pool statistics monitoring in production
- Configurable pool sizing for different deployment scenarios

## Production Configuration

### Recommended Settings
```yaml
# config/database/default.yaml
postgres:
  # Connection pool sizing for parallel M0/M1/M2 processing
  pool_size: 20          # Base connections for concurrent operations
  max_overflow: 40       # Additional connections for peak load
  pool_timeout: 60.0     # Accommodate complex vector operations
  pool_recycle: 7200     # 2-hour connection lifecycle

  # Connection stability
  connection_timeout: 30.0
  keepalives_idle: 600   # 10-minute TCP keepalive
  keepalives_interval: 30
  keepalives_count: 3
```

### Deployment Considerations
- **Development**: pool_size=5, max_overflow=10 (minimal resources)
- **Production**: pool_size=20, max_overflow=40 (parallel processing)
- **High Load**: pool_size=30, max_overflow=60 (multiple concurrent users)

## Monitoring and Testing

### Key Metrics
```python
stats = connection_manager.get_pool_statistics()
# Monitor: active_references, pool_closed, min/max_size utilization
```

### Test Coverage
- **Pool Sharing**: Multiple stores share single pool instance
- **Configuration Hierarchy**: Store > database > postgres precedence
- **Health Checks**: Structural validation without breaking sharing
- **Reference Tracking**: Automatic cleanup on store disposal

### Operational Guidelines
1. **Monitor connection count trends** - Watch for gradual increases indicating leaks
2. **Adjust pool sizing based on parallel_enabled setting** - 20+ for parallel, 10 for sequential
3. **Use pool statistics for capacity planning** - Track active_references vs max_size
4. **Implement graceful shutdown** - Ensure proper pool cleanup during service restart

## Architecture Evolution

### Current State
- Global singleton connection manager
- Configuration-driven pool sizing
- Structural health checks preserving sharing
- Weak reference tracking for cleanup

### Future Considerations
1. **Read/Write Separation**: Separate pools for read-heavy vs write-heavy operations
2. **Dynamic Scaling**: Automatic pool sizing based on load patterns
3. **Multi-Database Support**: Pool management across different database instances
4. **Connection Routing**: Intelligent connection assignment based on operation type

### Lessons Learned
1. **Health checks must preserve sharing semantics** - Database connectivity != pool health
2. **Configuration hierarchy enables flexible deployment** - Different settings per environment
3. **Reference tracking is essential for shared resources** - Prevents both leaks and premature cleanup
4. **Parallel processing significantly impacts connection requirements** - Plan capacity accordingly
