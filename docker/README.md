# MemFuse Docker Configuration

This directory contains Docker configurations and management scripts for MemFuse deployment across different environments.

## 📁 Directory Structure

```
docker/
├── app/
│   ├── Dockerfile           # Application container definition
│   └── .dockerignore        # Docker build ignore patterns
├── compose/
│   ├── docker-compose.dev.yml    # Development environment
│   ├── docker-compose.prod.yml   # Production environment
│   ├── docker-compose.test.yml   # Testing environment
│   ├── docker-compose.local.yml  # Local development
│   └── docker-compose.pgai.yml   # PostgreSQL + pgai + pgvector
├── pgai/
│   ├── Dockerfile               # PostgreSQL 17 + pgai + pgvector
│   ├── postgresql.conf          # Optimized PostgreSQL configuration
│   ├── pg_hba.conf             # Authentication configuration
│   ├── init-scripts/           # Database initialization scripts
│   └── README.md               # PgAI documentation
├── scripts/
│   ├── build.sh             # Build management script
│   ├── build-pgai.sh        # PgAI environment management script
│   ├── deploy.sh            # Deployment management script
│   └── health-check.sh      # Health monitoring script
└── README.md                # This documentation
```

## 🚀 Quick Start

### Development Environment
```bash
# Start development environment
make docker-dev

# Or manually
docker-compose -f docker/compose/docker-compose.dev.yml up -d
```

### PgAI Environment (PostgreSQL + pgai + pgvector)
```bash
# Build and start pgai environment
./docker/scripts/build-pgai.sh build
./docker/scripts/build-pgai.sh up

# Test the environment
./docker/scripts/build-pgai.sh test

# View status and logs
./docker/scripts/build-pgai.sh status
./docker/scripts/build-pgai.sh logs
```

### Production Environment
```bash
# Build and deploy production
make docker-prod

# Or manually
docker-compose -f docker/compose/docker-compose.prod.yml up -d
```

## 🔧 Management Scripts

### Build Script (`scripts/build.sh`)
Build Docker images with various options:

```bash
# Basic build
./docker/scripts/build.sh

# Production build with tag
./docker/scripts/build.sh -e prod -t v1.0.0

# Multi-platform build and push
./docker/scripts/build.sh --platform linux/amd64,linux/arm64 --push

# Build without cache
./docker/scripts/build.sh --no-cache
```

**Options:**
- `-e, --env`: Environment (dev|prod|test|local)
- `-t, --tag`: Docker image tag
- `-p, --push`: Push to registry
- `--no-cache`: Build without cache
- `--platform`: Target platform(s)

### Deploy Script (`scripts/deploy.sh`)
Manage deployments across environments:

```bash
# Start development environment
./docker/scripts/deploy.sh -e dev -a up

# Stop production environment
./docker/scripts/deploy.sh -e prod -a down

# Restart with build
./docker/scripts/deploy.sh -e dev -a restart -b

# View logs
./docker/scripts/deploy.sh -e prod -a logs
```

**Options:**
- `-e, --env`: Environment (dev|prod|test|local)
- `-a, --action`: Action (up|down|restart|logs|status)
- `-b, --build`: Build before deployment
- `-p, --pull`: Pull latest images
- `-f, --foreground`: Run in foreground

### Health Check Script (`scripts/health-check.sh`)
Monitor service health:

```bash
# Basic health check
./docker/scripts/health-check.sh

# Verbose production check
./docker/scripts/health-check.sh -e prod -v

# JSON output for monitoring
./docker/scripts/health-check.sh -j > health-report.json
```

**Options:**
- `-e, --env`: Environment to check
- `-t, --timeout`: Timeout in seconds
- `-v, --verbose`: Verbose output
- `-j, --json`: JSON output format

## 🌍 Environment Configurations

### Development (`docker-compose.dev.yml`)
- **Purpose**: Local development with hot reload
- **Features**: 
  - Source code mounting for live changes
  - Debug logging enabled
  - PostgreSQL on port 5432
  - MemFuse on port 8000

### Production (`docker-compose.prod.yml`)
- **Purpose**: Production deployment
- **Features**:
  - Environment variable configuration
  - Persistent volumes
  - Health checks and restart policies
  - Logging configuration
  - Optional Redis and Nginx services

### Testing (`docker-compose.test.yml`)
- **Purpose**: Automated testing and CI/CD
- **Features**:
  - Isolated test database
  - Test runner services
  - Coverage reporting
  - Integration test profiles

### Local (`docker-compose.local.yml`)
- **Purpose**: Lightweight local development
- **Features**:
  - Minimal resource usage
  - Optional pgAdmin for database management
  - Optional Redis for caching experiments
  - Different ports to avoid conflicts

### PgAI (`docker-compose.pgai.yml`)
- **Purpose**: PostgreSQL with pgai + pgvector extensions
- **Features**:
  - PostgreSQL 17 with pgai and pgvector extensions
  - Real-time embedding generation with immediate triggers
  - Optimized for vector operations and AI workflows
  - Optional pgAdmin and Redis services
  - Comprehensive initialization scripts
- **Documentation**: See [docker/pgai/README.md](pgai/README.md)

## 📊 Service Ports

| Environment | PostgreSQL | MemFuse | Additional |
|-------------|------------|---------|------------|
| Development | 5432       | 8000    | -          |
| Production  | 5432       | 8000    | 80, 443 (Nginx) |
| Testing     | 5432       | 8000    | -          |
| Local       | 5432       | 8000    | 8080 (pgAdmin) |
| PgAI        | 5432       | 8000    | 8080 (pgAdmin), 6379 (Redis) |

## 🔐 Environment Variables

### Required for Production
```bash
POSTGRES_PASSWORD=your_secure_password
OPENAI_API_KEY=your_openai_key
```

### Optional Configuration
```bash
POSTGRES_DB=memfuse
POSTGRES_USER=postgres
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
OPENAI_MODEL=gpt-4o-mini
LOG_LEVEL=INFO
MEMFUSE_PORT=8000
```

## 🛠️ Makefile Integration

The project Makefile includes Docker commands:

```bash
# Build Docker image
make docker-build

# Start development environment
make docker-dev

# Start production environment  
make docker-prod

# Start test environment
make docker-test

# Clean Docker resources
make docker-clean

# Health check
make docker-health
```

## 🔍 Troubleshooting

### Common Issues

1. **Port Conflicts**
   ```bash
   # Check what's using the port
   lsof -i :5432
   
   # Use different environment
   docker-compose -f docker/compose/docker-compose.local.yml up -d
   ```

2. **Permission Issues**
   ```bash
   # Fix script permissions
   chmod +x docker/scripts/*.sh
   ```

3. **Build Failures**
   ```bash
   # Clean build without cache
   ./docker/scripts/build.sh --no-cache
   
   # Clean Docker system
   docker system prune -a
   ```

4. **Database Connection Issues**
   ```bash
   # Check PostgreSQL health
   ./docker/scripts/health-check.sh -v
   
   # View database logs
   docker-compose -f docker/compose/docker-compose.dev.yml logs postgres
   ```

### Health Monitoring

```bash
# Quick health check
./docker/scripts/health-check.sh

# Detailed status
./docker/scripts/deploy.sh -a status

# Service logs
./docker/scripts/deploy.sh -a logs
```

## 📚 Best Practices

1. **Environment Separation**: Always use appropriate environment configurations
2. **Resource Management**: Monitor resource usage in production
3. **Security**: Use strong passwords and secure environment variables
4. **Monitoring**: Regular health checks and log monitoring
5. **Backup**: Regular database backups for production
6. **Updates**: Keep Docker images and dependencies updated

## 🔄 Migration from Root Directory

If migrating from root directory Docker files:

1. **Old commands** → **New commands**:
   ```bash
   # Old
   docker-compose up -d
   
   # New
   make docker-dev
   # or
   docker-compose -f docker/compose/docker-compose.dev.yml up -d
   ```

2. **Update CI/CD pipelines** to use new paths
3. **Update documentation** references to Docker files

## 📞 Support

For Docker-related issues:
1. Check this documentation
2. Review service logs
3. Run health checks
4. Check the main project README for general setup
