# MemFuse Production Deployment Guide

This guide provides comprehensive instructions for deploying MemFuse in production environments with enterprise-grade reliability, security, and performance.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [System Requirements](#system-requirements)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Security](#security)
6. [Monitoring & Observability](#monitoring--observability)
7. [Performance Tuning](#performance-tuning)
8. [High Availability](#high-availability)
9. [Backup & Recovery](#backup--recovery)
10. [Troubleshooting](#troubleshooting)

## Prerequisites

### Software Requirements

- **Python**: 3.11+ (recommended: 3.11.13)
- **PostgreSQL**: 14+ with pgvector and pgai extensions
- **TimescaleDB**: 2.11+ (for time-series data)
- **Redis**: 7.0+ (for caching and session management)
- **Docker**: 24.0+ (for containerized deployment)
- **Kubernetes**: 1.28+ (for orchestrated deployment)

### Hardware Requirements

#### Minimum Requirements
- **CPU**: 4 cores (8 threads)
- **RAM**: 16GB
- **Storage**: 100GB SSD
- **Network**: 1Gbps

#### Recommended Requirements
- **CPU**: 8 cores (16 threads)
- **RAM**: 32GB
- **Storage**: 500GB NVMe SSD
- **Network**: 10Gbps

#### High-Performance Requirements
- **CPU**: 16+ cores (32+ threads)
- **RAM**: 64GB+
- **Storage**: 1TB+ NVMe SSD with RAID 10
- **Network**: 25Gbps+

## System Requirements

### Operating System
- **Linux**: Ubuntu 22.04 LTS, RHEL 9, or CentOS Stream 9
- **Container**: Alpine Linux 3.18+ (for Docker images)

### Database Configuration

#### PostgreSQL with Extensions
```sql
-- Install required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgai;
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Configure memory settings
ALTER SYSTEM SET shared_buffers = '8GB';
ALTER SYSTEM SET effective_cache_size = '24GB';
ALTER SYSTEM SET maintenance_work_mem = '2GB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;
ALTER SYSTEM SET random_page_cost = 1.1;
ALTER SYSTEM SET effective_io_concurrency = 200;

-- Reload configuration
SELECT pg_reload_conf();
```

#### TimescaleDB Optimization
```sql
-- Create hypertables for time-series data
SELECT create_hypertable('m0_raw_messages', 'created_at');
SELECT create_hypertable('m1_episodic_memories', 'created_at');
SELECT create_hypertable('system_metrics', 'timestamp');

-- Set compression policies
SELECT add_compression_policy('m0_raw_messages', INTERVAL '7 days');
SELECT add_compression_policy('m1_episodic_memories', INTERVAL '30 days');
SELECT add_compression_policy('system_metrics', INTERVAL '1 day');

-- Set retention policies
SELECT add_retention_policy('m0_raw_messages', INTERVAL '1 year');
SELECT add_retention_policy('system_metrics', INTERVAL '90 days');
```

## Installation

### Docker Deployment (Recommended)

#### 1. Create Docker Compose Configuration

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  memfuse-api:
    image: memfuse/memfuse-core:latest
    ports:
      - "8000:8000"
    environment:
      - MEMFUSE_ENV=production
      - DATABASE_URL=postgresql://memfuse:${DB_PASSWORD}@postgres:5432/memfuse_prod
      - REDIS_URL=redis://redis:6379/0
      - LOG_LEVEL=INFO
      - METRICS_ENABLED=true
      - TRACING_ENABLED=true
    volumes:
      - ./config:/app/config:ro
      - ./logs:/app/logs
    depends_on:
      - postgres
      - redis
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  postgres:
    image: timescale/timescaledb-ha:pg15-latest
    environment:
      - POSTGRES_DB=memfuse_prod
      - POSTGRES_USER=memfuse
      - POSTGRES_PASSWORD=${DB_PASSWORD}
      - POSTGRES_INITDB_ARGS=--encoding=UTF-8 --lc-collate=C --lc-ctype=C
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init-scripts:/docker-entrypoint-initdb.d:ro
    ports:
      - "5432:5432"
    restart: unless-stopped
    command: >
      postgres
      -c shared_buffers=2GB
      -c effective_cache_size=6GB
      -c maintenance_work_mem=512MB
      -c checkpoint_completion_target=0.9
      -c wal_buffers=16MB
      -c default_statistics_target=100
      -c random_page_cost=1.1
      -c effective_io_concurrency=200
      -c max_connections=200

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
      - ./redis.conf:/usr/local/etc/redis/redis.conf:ro
    command: redis-server /usr/local/etc/redis/redis.conf
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
      - ./logs/nginx:/var/log/nginx
    depends_on:
      - memfuse-api
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=90d'
      - '--web.enable-lifecycle'
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
      - GF_INSTALL_PLUGINS=grafana-piechart-panel
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards:ro
      - ./grafana/datasources:/etc/grafana/provisioning/datasources:ro
    depends_on:
      - prometheus
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:
```

#### 2. Environment Configuration

```bash
# .env.prod
DB_PASSWORD=your_secure_database_password
GRAFANA_PASSWORD=your_secure_grafana_password
MEMFUSE_SECRET_KEY=your_secret_key_for_jwt_signing
ENCRYPTION_KEY=your_32_byte_encryption_key
```

#### 3. Deploy with Docker Compose

```bash
# Create necessary directories
mkdir -p config logs ssl grafana/{dashboards,datasources}

# Copy configuration files
cp config/production/* config/

# Start services
docker-compose -f docker-compose.prod.yml up -d

# Check service health
docker-compose -f docker-compose.prod.yml ps
```

### Kubernetes Deployment

#### 1. Create Namespace and Secrets

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: memfuse-prod
---
apiVersion: v1
kind: Secret
metadata:
  name: memfuse-secrets
  namespace: memfuse-prod
type: Opaque
stringData:
  database-password: "your_secure_database_password"
  redis-password: "your_secure_redis_password"
  secret-key: "your_secret_key_for_jwt_signing"
  encryption-key: "your_32_byte_encryption_key"
```

#### 2. Deploy Database

```yaml
# k8s/postgres.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
  namespace: memfuse-prod
spec:
  serviceName: postgres
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: timescale/timescaledb-ha:pg15-latest
        env:
        - name: POSTGRES_DB
          value: memfuse_prod
        - name: POSTGRES_USER
          value: memfuse
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: memfuse-secrets
              key: database-password
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
  volumeClaimTemplates:
  - metadata:
      name: postgres-storage
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 100Gi
      storageClassName: fast-ssd
---
apiVersion: v1
kind: Service
metadata:
  name: postgres
  namespace: memfuse-prod
spec:
  selector:
    app: postgres
  ports:
  - port: 5432
    targetPort: 5432
  type: ClusterIP
```

#### 3. Deploy MemFuse Application

```yaml
# k8s/memfuse-app.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: memfuse-api
  namespace: memfuse-prod
spec:
  replicas: 3
  selector:
    matchLabels:
      app: memfuse-api
  template:
    metadata:
      labels:
        app: memfuse-api
    spec:
      containers:
      - name: memfuse-api
        image: memfuse/memfuse-core:latest
        ports:
        - containerPort: 8000
        env:
        - name: MEMFUSE_ENV
          value: "production"
        - name: DATABASE_URL
          value: "postgresql://memfuse:$(DATABASE_PASSWORD)@postgres:5432/memfuse_prod"
        - name: DATABASE_PASSWORD
          valueFrom:
            secretKeyRef:
              name: memfuse-secrets
              key: database-password
        - name: SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: memfuse-secrets
              key: secret-key
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: memfuse-api
  namespace: memfuse-prod
spec:
  selector:
    app: memfuse-api
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP
```

## Configuration

### Production Configuration Files

#### Main Configuration (`config/production.yaml`)

```yaml
# MemFuse Production Configuration
environment: production
debug: false

# Database Configuration
database:
  postgres:
    host: postgres
    port: 5432
    database: memfuse_prod
    user: memfuse
    password: ${DATABASE_PASSWORD}
    pool_size: 20
    max_overflow: 30
    pool_timeout: 30
    pool_recycle: 3600
  
  pgai:
    enabled: true
    auto_embedding: true
    immediate_trigger: false
    max_retries: 3
    retry_interval: 2.0
    worker_count: 4
    queue_size: 1000
    enable_metrics: true

# Cache Configuration
cache:
  redis:
    url: redis://redis:6379/0
    max_connections: 100
    retry_on_timeout: true
    socket_timeout: 5
    socket_connect_timeout: 5
  
  memory:
    max_size: 10000
    ttl: 3600
    cleanup_interval: 300

# Gateway Configuration
gateway:
  enabled: true
  max_request_size: 10485760  # 10MB
  timeout: 30
  
  filters:
    inbound:
      - name: request_validator
        enabled: true
        config:
          max_content_length: 1000000
          required_fields: ["content"]
      
      - name: rate_limiter
        enabled: true
        config:
          requests_per_minute: 1000
          burst_size: 100
      
      - name: input_sanitizer
        enabled: true
        config:
          strip_html: true
          normalize_unicode: true
    
    outbound:
      - name: sensitive_word
        enabled: true
        config:
          action: "mask"
          patterns_file: "sensitive_words.txt"
      
      - name: max_length
        enabled: true
        config:
          max_length: 5000
          action: "truncate"

# Metrics & Monitoring
metrics:
  enabled: true
  prometheus:
    enabled: true
    port: 9090
    path: /metrics
  
  custom_metrics:
    - name: request_duration
      type: histogram
      description: "Request processing duration"
    
    - name: memory_operations
      type: counter
      description: "Memory layer operations"

# Tracing
tracing:
  enabled: true
  service_name: memfuse-prod
  service_version: "1.0.0"
  exporter_type: jaeger
  jaeger_endpoint: http://jaeger:14268/api/traces
  sample_rate: 0.1
  include_request_body: false
  include_response_body: false

# Logging
logging:
  level: INFO
  format: json
  file: /app/logs/memfuse.log
  max_size: 100MB
  backup_count: 10
  
  loggers:
    memfuse_core: INFO
    uvicorn: WARNING
    sqlalchemy: WARNING

# Security
security:
  cors:
    enabled: true
    allow_origins: ["https://your-domain.com"]
    allow_methods: ["GET", "POST", "PUT", "DELETE"]
    allow_headers: ["*"]
    allow_credentials: true
  
  authentication:
    enabled: true
    jwt_secret: ${SECRET_KEY}
    jwt_algorithm: HS256
    jwt_expiration: 3600
  
  encryption:
    enabled: true
    key: ${ENCRYPTION_KEY}
    algorithm: AES-256-GCM

# Performance
performance:
  async_workers: 4
  max_concurrent_requests: 1000
  request_timeout: 30
  keepalive_timeout: 5
  
  memory:
    max_memory_usage: 80  # Percentage
    gc_threshold: 70
    
  cache:
    preload_common_queries: true
    cache_warming_enabled: true
```

## Security

### SSL/TLS Configuration

```nginx
# nginx.conf
server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    
    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options DENY always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    
    location / {
        proxy_pass http://memfuse-api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
        
        # Buffer settings
        proxy_buffering on;
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;
    }
}
```

### Network Security

```bash
# Firewall rules (UFW)
ufw default deny incoming
ufw default allow outgoing
ufw allow ssh
ufw allow 80/tcp
ufw allow 443/tcp
ufw allow from 10.0.0.0/8 to any port 5432  # Database access from internal network only
ufw enable
```

## Monitoring & Observability

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "memfuse_rules.yml"

scrape_configs:
  - job_name: 'memfuse-api'
    static_configs:
      - targets: ['memfuse-api:8000']
    metrics_path: /metrics
    scrape_interval: 10s
    
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']
    
  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']
    
  - job_name: 'node'
    static_configs:
      - targets: ['node-exporter:9100']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### Grafana Dashboards

```json
{
  "dashboard": {
    "title": "MemFuse Production Dashboard",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(memfuse_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(memfuse_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "memfuse_memory_usage_bytes / 1024 / 1024",
            "legendFormat": "Memory Usage (MB)"
          }
        ]
      },
      {
        "title": "Cache Hit Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(memfuse_cache_hits_total[5m]) / rate(memfuse_cache_requests_total[5m]) * 100",
            "legendFormat": "Cache Hit Rate %"
          }
        ]
      }
    ]
  }
}
```

## Performance Tuning

### Database Optimization

```sql
-- Index optimization for common queries
CREATE INDEX CONCURRENTLY idx_m0_raw_user_session
ON m0_raw_messages(user_id, session_id, created_at DESC);

CREATE INDEX CONCURRENTLY idx_m1_episodic_content_vector
ON m1_episodic_memories USING ivfflat (content_embedding vector_cosine_ops)
WITH (lists = 1000);

CREATE INDEX CONCURRENTLY idx_m2_semantic_tags
ON m2_semantic_memories USING gin(tags);

-- Vacuum and analyze regularly
SELECT cron.schedule('vacuum-analyze', '0 2 * * *', 'VACUUM ANALYZE;');

-- Update statistics
SELECT cron.schedule('update-stats', '0 3 * * 0',
  'ANALYZE m0_raw_messages, m1_episodic_memories, m2_semantic_memories;');
```

### Application Performance

```python
# config/performance_tuning.py
PERFORMANCE_CONFIG = {
    # Connection pooling
    "database": {
        "pool_size": 20,
        "max_overflow": 30,
        "pool_timeout": 30,
        "pool_recycle": 3600,
        "pool_pre_ping": True
    },

    # Async settings
    "async": {
        "max_workers": 8,
        "max_concurrent_requests": 1000,
        "request_timeout": 30,
        "keepalive_timeout": 5
    },

    # Cache optimization
    "cache": {
        "redis_pool_size": 50,
        "memory_cache_size": 10000,
        "cache_ttl": 3600,
        "preload_cache": True
    },

    # Memory management
    "memory": {
        "max_memory_usage": 80,  # Percentage
        "gc_threshold": 70,
        "gc_interval": 300
    }
}
```

### System-Level Tuning

```bash
# /etc/sysctl.conf
# Network optimization
net.core.somaxconn = 65535
net.core.netdev_max_backlog = 5000
net.ipv4.tcp_max_syn_backlog = 65535
net.ipv4.tcp_keepalive_time = 600
net.ipv4.tcp_keepalive_intvl = 60
net.ipv4.tcp_keepalive_probes = 10

# Memory optimization
vm.swappiness = 10
vm.dirty_ratio = 15
vm.dirty_background_ratio = 5

# File descriptor limits
fs.file-max = 2097152

# Apply changes
sysctl -p
```

## High Availability

### Load Balancer Configuration

```nginx
# nginx-lb.conf
upstream memfuse_backend {
    least_conn;
    server memfuse-api-1:8000 max_fails=3 fail_timeout=30s;
    server memfuse-api-2:8000 max_fails=3 fail_timeout=30s;
    server memfuse-api-3:8000 max_fails=3 fail_timeout=30s;
    keepalive 32;
}

server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://memfuse_backend;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Health check
        proxy_next_upstream error timeout invalid_header http_500 http_502 http_503 http_504;
        proxy_connect_timeout 5s;
        proxy_send_timeout 10s;
        proxy_read_timeout 10s;
    }

    location /health {
        access_log off;
        proxy_pass http://memfuse_backend;
        proxy_connect_timeout 2s;
        proxy_send_timeout 2s;
        proxy_read_timeout 2s;
    }
}
```

### Database High Availability

```yaml
# PostgreSQL HA with Patroni
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres-ha
spec:
  serviceName: postgres-ha
  replicas: 3
  selector:
    matchLabels:
      app: postgres-ha
  template:
    metadata:
      labels:
        app: postgres-ha
    spec:
      containers:
      - name: postgres
        image: postgres:15-alpine
        env:
        - name: PATRONI_SCOPE
          value: postgres-cluster
        - name: PATRONI_NAME
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        - name: PATRONI_POSTGRESQL_DATA_DIR
          value: /var/lib/postgresql/data
        - name: PATRONI_POSTGRESQL_PGPASS
          value: /tmp/pgpass
        - name: PATRONI_RESTAPI_LISTEN
          value: 0.0.0.0:8008
        - name: PATRONI_RESTAPI_CONNECT_ADDRESS
          valueFrom:
            fieldRef:
              fieldPath: status.podIP
        ports:
        - containerPort: 5432
        - containerPort: 8008
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
  volumeClaimTemplates:
  - metadata:
      name: postgres-storage
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 100Gi
```

## Backup & Recovery

### Database Backup Strategy

```bash
#!/bin/bash
# backup_script.sh

# Configuration
DB_HOST="postgres"
DB_NAME="memfuse_prod"
DB_USER="memfuse"
BACKUP_DIR="/backups"
RETENTION_DAYS=30

# Create backup directory
mkdir -p $BACKUP_DIR

# Full backup
BACKUP_FILE="$BACKUP_DIR/memfuse_full_$(date +%Y%m%d_%H%M%S).sql.gz"
pg_dump -h $DB_HOST -U $DB_USER -d $DB_NAME | gzip > $BACKUP_FILE

# Verify backup
if [ $? -eq 0 ]; then
    echo "Backup completed successfully: $BACKUP_FILE"

    # Upload to S3 (optional)
    aws s3 cp $BACKUP_FILE s3://your-backup-bucket/memfuse/

    # Clean old backups
    find $BACKUP_DIR -name "memfuse_full_*.sql.gz" -mtime +$RETENTION_DAYS -delete
else
    echo "Backup failed!"
    exit 1
fi

# Point-in-time recovery setup
pg_basebackup -h $DB_HOST -U $DB_USER -D $BACKUP_DIR/base_backup -Ft -z -P
```

### Application State Backup

```python
# backup_app_state.py
import asyncio
import json
from datetime import datetime
from src.memfuse_core.services.backup_service import BackupService

async def backup_application_state():
    """Backup application state including caches and configurations."""
    backup_service = BackupService()

    backup_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "cache_state": await backup_service.export_cache_state(),
        "configuration": await backup_service.export_configuration(),
        "metrics": await backup_service.export_metrics_state(),
        "user_sessions": await backup_service.export_active_sessions()
    }

    backup_file = f"memfuse_state_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"

    with open(f"/backups/{backup_file}", 'w') as f:
        json.dump(backup_data, f, indent=2)

    print(f"Application state backup completed: {backup_file}")

if __name__ == "__main__":
    asyncio.run(backup_application_state())
```

## Troubleshooting

### Common Issues and Solutions

#### 1. High Memory Usage

```bash
# Check memory usage
free -h
ps aux --sort=-%mem | head -10

# Check MemFuse memory usage
docker stats memfuse-api

# Solutions:
# - Reduce cache sizes in configuration
# - Increase garbage collection frequency
# - Scale horizontally with more instances
```

#### 2. Database Connection Issues

```sql
-- Check active connections
SELECT count(*) FROM pg_stat_activity;

-- Check connection limits
SHOW max_connections;

-- Kill long-running queries
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE state = 'active'
AND query_start < now() - interval '5 minutes';
```

#### 3. Performance Degradation

```bash
# Check system resources
top
iotop
netstat -i

# Check application logs
docker logs memfuse-api --tail=100

# Check database performance
SELECT query, mean_time, calls
FROM pg_stat_statements
ORDER BY mean_time DESC
LIMIT 10;
```

### Health Check Endpoints

```python
# Health check implementation
@app.get("/health")
async def health_check():
    """Comprehensive health check."""
    checks = {
        "database": await check_database_connection(),
        "redis": await check_redis_connection(),
        "memory": check_memory_usage(),
        "disk": check_disk_space(),
        "cache": check_cache_health()
    }

    all_healthy = all(checks.values())
    status_code = 200 if all_healthy else 503

    return JSONResponse(
        status_code=status_code,
        content={
            "status": "healthy" if all_healthy else "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "checks": checks
        }
    )
```

### Monitoring Alerts

```yaml
# alerting_rules.yml
groups:
- name: memfuse_alerts
  rules:
  - alert: MemFuseHighErrorRate
    expr: rate(memfuse_errors_total[5m]) > 0.1
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "High error rate detected"
      description: "Error rate is {{ $value }} errors per second"

  - alert: MemFuseHighMemoryUsage
    expr: memfuse_memory_usage_percent > 90
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "High memory usage"
      description: "Memory usage is {{ $value }}%"

  - alert: MemFuseDatabaseConnectionFailed
    expr: memfuse_database_connection_status == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Database connection failed"
      description: "Cannot connect to database"
```

This comprehensive deployment guide provides all the necessary components for a production-ready MemFuse deployment with enterprise-grade reliability, security, and performance monitoring.
