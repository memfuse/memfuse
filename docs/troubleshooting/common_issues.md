# MemFuse Troubleshooting Guide

This comprehensive troubleshooting guide covers common issues, their root causes, diagnostic steps, and solutions based on real-world experience with MemFuse deployment and operation.

## Table of Contents

1. [General Troubleshooting](#general-troubleshooting)
2. [Performance Issues](#performance-issues)
3. [Memory and Resource Issues](#memory-and-resource-issues)
4. [Database Connection Issues](#database-connection-issues)
5. [Async Operation Issues](#async-operation-issues)
6. [Testing Issues](#testing-issues)
7. [Deployment Issues](#deployment-issues)
8. [Monitoring and Observability Issues](#monitoring-and-observability-issues)
9. [Configuration Issues](#configuration-issues)
10. [Emergency Procedures](#emergency-procedures)

## General Troubleshooting

### System Health Check

Before diving into specific issues, perform a comprehensive system health check:

```bash
# Check system status
curl -f http://localhost:8000/health

# Check detailed system information
curl -f http://localhost:8000/info

# Check performance metrics
curl -f http://localhost:8000/stats/performance

# Check logs for errors
tail -f logs/memfuse.log | grep -i error

# Check resource usage
htop
df -h
free -h
```

### Log Analysis

MemFuse provides structured logging for effective troubleshooting:

```bash
# Filter logs by severity
grep "ERROR" logs/memfuse.log
grep "WARNING" logs/memfuse.log

# Filter logs by component
grep "gateway" logs/memfuse.log
grep "buffer" logs/memfuse.log
grep "database" logs/memfuse.log

# Filter logs by time range
grep "2024-01-15 10:" logs/memfuse.log

# Search for specific error patterns
grep -E "(timeout|connection|memory)" logs/memfuse.log
```

### Common Log Patterns

| Pattern | Meaning | Action |
|---------|---------|--------|
| `Connection refused` | Database/Redis unavailable | Check service status |
| `Memory usage high` | Memory pressure | Check memory leaks |
| `Timeout` | Operation taking too long | Check performance |
| `Authentication failed` | Invalid credentials | Check API keys |
| `Rate limit exceeded` | Too many requests | Implement backoff |

## Performance Issues

### Slow Response Times

**Symptoms**:
- API responses taking >1 second
- High P95/P99 latencies
- User complaints about slowness

**Diagnostic Steps**:
```bash
# Check current performance metrics
curl http://localhost:8000/stats/performance

# Monitor real-time performance
watch -n 1 'curl -s http://localhost:8000/stats/performance | jq .data.request_stats'

# Check cache hit rates
curl http://localhost:8000/gateway/filters/stats | jq .data.cache_stats

# Profile specific operations
poetry run python -m cProfile -o profile.stats scripts/performance_test.py
```

**Common Causes and Solutions**:

1. **Low Cache Hit Rate**
   ```bash
   # Check cache statistics
   curl http://localhost:8000/gateway/filters/stats
   
   # Solution: Increase cache sizes
   # Edit config/production.yml
   cache:
     regex_cache:
       max_size: 2000  # Increase from 1000
     content_cache:
       max_size: 10000  # Increase from 5000
   ```

2. **Database Query Performance**
   ```sql
   -- Check slow queries
   SELECT query, mean_time, calls 
   FROM pg_stat_statements 
   ORDER BY mean_time DESC 
   LIMIT 10;
   
   -- Check missing indexes
   SELECT schemaname, tablename, attname, n_distinct, correlation 
   FROM pg_stats 
   WHERE schemaname = 'public' 
   ORDER BY n_distinct DESC;
   ```

3. **Memory Pressure**
   ```bash
   # Check memory usage
   free -h
   
   # Check for memory leaks
   poetry run pytest tests/performance/test_memory_leak_detection.py -v
   
   # Solution: Restart service if memory leak detected
   sudo systemctl restart memfuse
   ```

### High CPU Usage

**Symptoms**:
- CPU usage consistently >80%
- System becoming unresponsive
- Slow response times

**Diagnostic Steps**:
```bash
# Check CPU usage by process
top -p $(pgrep -f memfuse)

# Profile CPU usage
poetry run python -m py-spy top --pid $(pgrep -f memfuse)

# Check for CPU-intensive operations
grep "duration_ms" logs/memfuse.log | sort -k3 -nr | head -20
```

**Solutions**:
1. **Optimize Regex Patterns**: Use simpler patterns where possible
2. **Increase Cache Sizes**: Reduce CPU-intensive recomputations
3. **Scale Horizontally**: Add more instances behind load balancer
4. **Optimize Database Queries**: Add indexes, optimize query patterns

### Cache Performance Issues

**Symptoms**:
- Low cache hit rates (<80%)
- Frequent cache evictions
- Inconsistent performance

**Diagnostic Steps**:
```python
# Check cache statistics programmatically
import requests

response = requests.get('http://localhost:8000/gateway/filters/stats')
cache_stats = response.json()['data']['cache_stats']

for cache_name, stats in cache_stats.items():
    hit_rate = stats['hit_rate_percent']
    if hit_rate < 80:
        print(f"Low hit rate for {cache_name}: {hit_rate}%")
```

**Solutions**:
```yaml
# Optimize cache configuration
cache:
  regex_cache:
    max_size: 2000
    ttl: 7200  # Increase TTL
  content_cache:
    max_size: 10000
    ttl: 3600
  quality_cache:
    max_size: 5000
    ttl: 14400
```

## Memory and Resource Issues

### Memory Leaks

**Symptoms**:
- Gradually increasing memory usage
- Out of memory errors
- System becoming unstable

**Diagnostic Steps**:
```bash
# Monitor memory usage over time
while true; do
    echo "$(date): $(ps -o pid,vsz,rss,comm -p $(pgrep -f memfuse))"
    sleep 60
done

# Run memory leak detection
poetry run pytest tests/performance/test_memory_leak_detection.py::TestCacheMemoryLeaks -v

# Check for unclosed resources
lsof -p $(pgrep -f memfuse) | wc -l
```

**Solutions**:
1. **Restart Service**: Immediate fix for memory leaks
   ```bash
   sudo systemctl restart memfuse
   ```

2. **Update Configuration**: Reduce cache sizes temporarily
   ```yaml
   cache:
     regex_cache:
       max_size: 500  # Reduce from 1000
     content_cache:
       max_size: 2500  # Reduce from 5000
   ```

3. **Code Fix**: Implement proper resource cleanup
   ```python
   # Ensure proper async resource management
   async with AsyncResourceManager() as arm:
       # Operations here
       pass  # Resources automatically cleaned up
   ```

### Resource Exhaustion

**Symptoms**:
- "Too many open files" errors
- Database connection errors
- Network connection failures

**Diagnostic Steps**:
```bash
# Check file descriptor usage
lsof -p $(pgrep -f memfuse) | wc -l
ulimit -n

# Check database connections
psql -c "SELECT count(*) FROM pg_stat_activity WHERE application_name = 'memfuse-core';"

# Check network connections
netstat -an | grep :8000 | wc -l
```

**Solutions**:
1. **Increase System Limits**:
   ```bash
   # Edit /etc/security/limits.conf
   memfuse soft nofile 65536
   memfuse hard nofile 65536
   
   # Edit /etc/systemd/system/memfuse.service
   [Service]
   LimitNOFILE=65536
   ```

2. **Optimize Connection Pooling**:
   ```yaml
   database:
     postgres:
       pool_size: 10      # Reduce from 20
       max_overflow: 15   # Reduce from 30
   ```

## Database Connection Issues

### Connection Pool Exhaustion

**Symptoms**:
- "Connection pool exhausted" errors
- Timeouts on database operations
- Slow query performance

**Diagnostic Steps**:
```sql
-- Check active connections
SELECT count(*), state 
FROM pg_stat_activity 
WHERE application_name = 'memfuse-core' 
GROUP BY state;

-- Check long-running queries
SELECT pid, now() - pg_stat_activity.query_start AS duration, query 
FROM pg_stat_activity 
WHERE (now() - pg_stat_activity.query_start) > interval '5 minutes';
```

**Solutions**:
1. **Optimize Connection Pool**:
   ```yaml
   database:
     postgres:
       pool_size: 20
       max_overflow: 30
       pool_timeout: 30
       pool_recycle: 3600
       pool_pre_ping: true
   ```

2. **Kill Long-Running Queries**:
   ```sql
   -- Kill specific query
   SELECT pg_terminate_backend(pid) 
   FROM pg_stat_activity 
   WHERE pid = <problematic_pid>;
   ```

### Database Performance Issues

**Symptoms**:
- Slow query execution
- High database CPU usage
- Query timeouts

**Diagnostic Steps**:
```sql
-- Enable query statistics (if not already enabled)
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Check slowest queries
SELECT query, calls, total_time, mean_time, rows
FROM pg_stat_statements
ORDER BY mean_time DESC
LIMIT 10;

-- Check table sizes
SELECT schemaname, tablename, 
       pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

**Solutions**:
1. **Add Missing Indexes**:
   ```sql
   -- Common indexes for MemFuse
   CREATE INDEX CONCURRENTLY idx_m0_user_session_time 
   ON m0_raw_messages(user_id, session_id, created_at DESC);
   
   CREATE INDEX CONCURRENTLY idx_m1_content_vector 
   ON m1_episodic_memories USING ivfflat (content_embedding vector_cosine_ops);
   ```

2. **Optimize Queries**:
   ```sql
   -- Use EXPLAIN ANALYZE to understand query plans
   EXPLAIN ANALYZE SELECT * FROM m1_episodic_memories 
   WHERE user_id = 'user_123' 
   ORDER BY created_at DESC 
   LIMIT 10;
   ```

## Async Operation Issues

### Event Loop Blocking

**Symptoms**:
- "Event loop is running" errors
- Async operations hanging
- Poor concurrent performance

**Diagnostic Steps**:
```python
# Check for blocking operations in async code
import asyncio
import time

async def check_event_loop():
    loop = asyncio.get_event_loop()
    print(f"Event loop running: {loop.is_running()}")
    print(f"Event loop closed: {loop.is_closed()}")
    
    # Check for pending tasks
    tasks = asyncio.all_tasks(loop)
    print(f"Pending tasks: {len(tasks)}")
    for task in tasks:
        print(f"Task: {task}")

# Run the check
asyncio.run(check_event_loop())
```

**Solutions**:
1. **Fix Event Loop Management**:
   ```python
   # Use proper event loop fixtures in tests
   @pytest.fixture(scope="function")
   def event_loop():
       policy = asyncio.get_event_loop_policy()
       loop = policy.new_event_loop()
       asyncio.set_event_loop(loop)
       yield loop
       
       # Proper cleanup
       try:
           pending = asyncio.all_tasks(loop)
           if pending:
               for task in pending:
                   task.cancel()
               loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
       finally:
           loop.close()
   ```

2. **Use Async Resource Managers**:
   ```python
   async def safe_operation():
       async with AsyncResourceManager() as arm:
           task = arm.track_task(some_async_operation())
           result = await task
           return result
   ```

### Task Cancellation Issues

**Symptoms**:
- Tasks not being cancelled properly
- Resource leaks from uncancelled tasks
- Hanging operations

**Solutions**:
```python
# Proper task cancellation
async def cancel_tasks_safely():
    tasks = asyncio.all_tasks()
    for task in tasks:
        if not task.done():
            task.cancel()
    
    # Wait for cancellation to complete
    await asyncio.gather(*tasks, return_exceptions=True)
```

## Testing Issues

### Async Test Failures

**Symptoms**:
- "RuntimeError: There is no current event loop in thread 'MainThread'"
- Tests passing individually but failing in batch
- Inconsistent test results

**Solutions**:
1. **Use Proper Event Loop Fixtures**:
   ```python
   # In conftest.py
   @pytest.fixture(scope="function")
   def event_loop():
       """Create a new event loop for each test function."""
       policy = asyncio.get_event_loop_policy()
       loop = policy.new_event_loop()
       asyncio.set_event_loop(loop)
       yield loop
       
       # Clean up
       try:
           pending = asyncio.all_tasks(loop)
           if pending:
               for task in pending:
                   task.cancel()
               loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
       except Exception:
           pass
       finally:
           loop.close()
   ```

2. **Run Tests with Proper Configuration**:
   ```bash
   # Use asyncio mode
   poetry run pytest --asyncio-mode=auto tests/

   # Run with verbose output for debugging
   poetry run pytest -v -s tests/unit/test_async_operations.py
   ```

### Test Database Issues

**Symptoms**:
- Tests failing due to database state
- Inconsistent test results
- Database connection errors in tests

**Solutions**:
1. **Use Test Database Isolation**:
   ```python
   @pytest.fixture(scope="function")
   async def test_db():
       # Create test database
       test_db_name = f"test_memfuse_{uuid.uuid4().hex[:8]}"
       await create_test_database(test_db_name)
       
       yield test_db_name
       
       # Cleanup
       await drop_test_database(test_db_name)
   ```

2. **Reset Database State Between Tests**:
   ```python
   @pytest.fixture(autouse=True)
   async def reset_database():
       # Clear all tables
       await clear_all_tables()
       
       # Reset sequences
       await reset_sequences()
   ```

## Deployment Issues

### Docker Container Issues

**Symptoms**:
- Container failing to start
- Health checks failing
- Port binding issues

**Diagnostic Steps**:
```bash
# Check container logs
docker logs memfuse-container

# Check container resource usage
docker stats memfuse-container

# Check port bindings
docker port memfuse-container

# Inspect container configuration
docker inspect memfuse-container
```

**Solutions**:
1. **Fix Health Check Configuration**:
   ```dockerfile
   HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
     CMD curl -f http://localhost:8000/health || exit 1
   ```

2. **Proper Environment Configuration**:
   ```yaml
   # docker-compose.yml
   environment:
     - MEMFUSE_ENV=production
     - DATABASE_URL=postgresql://user:pass@db:5432/memfuse
     - REDIS_URL=redis://redis:6379/0
   ```

### Kubernetes Deployment Issues

**Symptoms**:
- Pods failing to start
- Service discovery issues
- Load balancing problems

**Diagnostic Steps**:
```bash
# Check pod status
kubectl get pods -l app=memfuse

# Check pod logs
kubectl logs -l app=memfuse --tail=100

# Check service configuration
kubectl describe service memfuse-service

# Check ingress configuration
kubectl describe ingress memfuse-ingress
```

**Solutions**:
1. **Fix Resource Limits**:
   ```yaml
   resources:
     requests:
       memory: "512Mi"
       cpu: "250m"
     limits:
       memory: "1Gi"
       cpu: "500m"
   ```

2. **Configure Proper Health Checks**:
   ```yaml
   livenessProbe:
     httpGet:
       path: /health
       port: 8000
     initialDelaySeconds: 30
     periodSeconds: 10
   
   readinessProbe:
     httpGet:
       path: /health
       port: 8000
     initialDelaySeconds: 5
     periodSeconds: 5
   ```

## Emergency Procedures

### System Recovery

**High Memory Usage**:
```bash
# Immediate actions
sudo systemctl restart memfuse
docker restart memfuse-container  # If using Docker

# Monitor recovery
watch -n 5 'free -h && ps aux | grep memfuse'
```

**Database Connection Issues**:
```bash
# Restart database connections
sudo systemctl restart postgresql
sudo systemctl restart redis

# Clear connection pools
curl -X POST http://localhost:8000/admin/clear-pools
```

**Complete System Failure**:
```bash
# Stop all services
sudo systemctl stop memfuse
sudo systemctl stop postgresql
sudo systemctl stop redis

# Check disk space
df -h

# Check system logs
journalctl -u memfuse --since "1 hour ago"

# Restart services in order
sudo systemctl start postgresql
sudo systemctl start redis
sudo systemctl start memfuse

# Verify system health
curl http://localhost:8000/health
```

### Data Recovery

**Database Corruption**:
```bash
# Stop MemFuse
sudo systemctl stop memfuse

# Restore from backup
pg_restore -d memfuse /path/to/backup.sql

# Restart services
sudo systemctl start memfuse
```

**Cache Corruption**:
```bash
# Clear Redis cache
redis-cli FLUSHALL

# Restart MemFuse to rebuild caches
sudo systemctl restart memfuse
```

This troubleshooting guide provides comprehensive coverage of common issues and their solutions, based on real-world experience with MemFuse deployment and operation.
