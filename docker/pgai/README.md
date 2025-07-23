# MemFuse PgAI Docker Configuration

This directory contains Docker configurations for running MemFuse with PostgreSQL + pgai + pgvector extensions, optimized for immediate trigger system and vector operations.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MemFuse PgAI Stack                      │
├─────────────────────────────────────────────────────────────┤
│  MemFuse App (Port 8000)                                   │
│  ├── EventDrivenPgaiStore                                  │
│  ├── Immediate Trigger System                              │
│  └── all-MiniLM-L6-v2 Embedding Model                     │
├─────────────────────────────────────────────────────────────┤
│  PostgreSQL 17 + pgai + pgvector (Port 5432)              │
│  ├── pgvector: Vector operations & HNSW indexes           │
│  ├── pgai: AI workflow helpers                            │
│  ├── Immediate triggers: NOTIFY/LISTEN system             │
│  └── Optimized configuration for vector workloads         │
├─────────────────────────────────────────────────────────────┤
│  Optional Services                                          │
│  ├── pgAdmin (Port 8080) - Database management            │
│  └── Redis (Port 6379) - Caching layer                    │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- 8GB+ RAM (for embedding model)
- 10GB+ disk space

### 1. Build and Start
```bash
# Build pgai Docker image
./docker/scripts/build-pgai.sh build

# Start the environment
./docker/scripts/build-pgai.sh up

# Check status
./docker/scripts/build-pgai.sh status
```

### 2. Verify Installation
```bash
# Test all components
./docker/scripts/build-pgai.sh test

# Open PostgreSQL shell
./docker/scripts/build-pgai.sh shell
```

### 3. Access Services
- **MemFuse API**: http://localhost:8000
- **API Health**: http://localhost:8000/api/v1/health
- **pgAdmin** (optional): http://localhost:8080
- **PostgreSQL**: localhost:5432

## 📁 Directory Structure

```
docker/pgai/
├── Dockerfile                    # PostgreSQL + pgai + pgvector image
├── postgresql.conf               # Optimized PostgreSQL configuration
├── pg_hba.conf                   # Authentication configuration
├── init-scripts/                 # Database initialization scripts
│   ├── 01-init-extensions.sh     # Extensions and schema setup
│   └── 02-init-immediate-triggers.sh # Immediate trigger system
└── README.md                     # This documentation
```

## 🔧 Configuration

### Environment Variables (.env)
```bash
# Database Connection
POSTGRES_HOST=postgres-pgai
POSTGRES_PORT=5432
POSTGRES_DB=memfuse
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres

# MemFuse Settings
EMBEDDING_MODEL=all-MiniLM-L6-v2
PGAI_IMMEDIATE_TRIGGER=true
PGAI_AUTO_EMBEDDING=true

# Optional: External LLM
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini
```

### PostgreSQL Optimizations
- **Memory**: 256MB shared_buffers, 1GB effective_cache_size
- **Vector Operations**: Optimized work_mem for vector queries
- **HNSW Indexes**: Fast similarity search
- **Immediate Triggers**: NOTIFY/LISTEN optimization
- **Autovacuum**: Tuned for vector tables

## 🧪 Testing

### Manual Testing
```bash
# 1. Test database connection
docker exec memfuse-pgai-postgres psql -U postgres -d memfuse -c "SELECT version();"

# 2. Test extensions
docker exec memfuse-pgai-postgres psql -U postgres -d memfuse -c "
SELECT extname, extversion 
FROM pg_extension 
WHERE extname IN ('vector', 'pgai');"

# 3. Test immediate triggers
docker exec memfuse-pgai-postgres psql -U postgres -d memfuse -c "
SELECT * FROM get_trigger_system_status();"

# 4. Test MemFuse API
curl http://localhost:8000/api/v1/health
```

### Automated Testing
```bash
# Run comprehensive test suite
./docker/scripts/build-pgai.sh test
```

## 📊 Performance Monitoring

### Database Metrics
```sql
-- Check table sizes
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables 
WHERE tablename = 'm0_raw';

-- Check index usage
SELECT 
    indexrelname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes 
WHERE relname = 'm0_raw';

-- Check vector index performance
EXPLAIN (ANALYZE, BUFFERS) 
SELECT id, content, embedding <=> '[0.1,0.2,...]'::vector as distance
FROM m0_raw
ORDER BY embedding <=> '[0.1,0.2,...]'::vector 
LIMIT 10;
```

### Container Metrics
```bash
# Monitor resource usage
docker stats memfuse-pgai-postgres memfuse-pgai-app

# Check logs
docker logs memfuse-pgai-postgres --tail=100
docker logs memfuse-pgai-app --tail=100
```

## 🔍 Troubleshooting

### Common Issues

1. **Build Failures**
   ```bash
   # Clean build without cache
   ./docker/scripts/build-pgai.sh build --no-cache
   
   # Check Docker resources
   docker system df
   docker system prune -a
   ```

2. **Extension Loading Issues**
   ```bash
   # Check extension installation
   docker exec memfuse-pgai-postgres psql -U postgres -d memfuse -c "
   SELECT * FROM pg_available_extensions WHERE name IN ('vector', 'pgai');"
   ```

3. **Memory Issues**
   ```bash
   # Increase Docker memory limit to 8GB+
   # Check container memory usage
   docker stats --no-stream
   ```

4. **Connection Issues**
   ```bash
   # Check network connectivity
   docker network ls
   docker network inspect memfuse-pgai-network
   ```

### Logs and Debugging
```bash
# View all logs
./docker/scripts/build-pgai.sh logs

# View specific service logs
docker logs memfuse-pgai-postgres
docker logs memfuse-pgai-app

# Debug database initialization
docker exec memfuse-pgai-postgres cat /var/log/postgresql/postgresql-*.log
```

## 🧹 Cleanup

```bash
# Stop services
./docker/scripts/build-pgai.sh down

# Clean up all resources (WARNING: removes data)
./docker/scripts/build-pgai.sh clean
```

## 🔗 Related Documentation

- [MemFuse PgAI Architecture](../../docs/architecture/pgai.md)
- [Docker Main README](../README.md)
- [pgvector Documentation](https://github.com/pgvector/pgvector)
- [pgai Documentation](https://github.com/timescale/pgai)
