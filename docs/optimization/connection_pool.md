# PostgreSQL Connection Pool Architecture

## Overview

This document describes the PostgreSQL connection pool architecture in MemFuse, including design decisions, implementation strategies, and optimization approaches for handling concurrent database operations across M0/M1/M2 memory layers.

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

### Solution Comparison

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
