# 🐳 MemFuse Docker Setup

This guide explains how to containerize and run MemFuse using Docker.

## 📋 Prerequisites

- Docker installed on your system
- Docker Compose (optional, for easier orchestration)
- At least 4GB of RAM (for ML models)
- Internet connection (for downloading models on first run)

## 🚀 Quick Start

### Option 1: Using the Setup Script (Recommended)

The easiest way to get started:

```bash
# Make the script executable (if not already)
chmod +x docker-setup.sh

# Run with pip-only build (most reliable)
./docker-setup.sh pip

# Or run with Poetry (if you prefer)
./docker-setup.sh dev

# Or run with Docker Compose
./docker-setup.sh compose
```

### Option 2: Manual Docker Commands

```bash
# Build and run with pip-only Dockerfile (recommended)
docker build -f Dockerfile.pip -t memfuse:pip .
docker run -d -p 8000:8000 --name memfuse-pip memfuse:pip

# Or build with Poetry
docker build -t memfuse:dev .
docker run -d -p 8000:8000 --name memfuse-dev memfuse:dev
```

### Option 3: Docker Compose

```bash
# Start all services
docker-compose up -d --build

# View logs
docker-compose logs -f memfuse

# Stop services
docker-compose down
```

## 📁 Available Dockerfiles

### 1. `Dockerfile.pip` (Recommended)

- **Use case**: Development and production
- **Pros**: Faster build, more reliable dependency installation
- **Cons**: Less control over dependency versions
- **Build time**: ~7-8 minutes

### 2. `Dockerfile` (Poetry-based)

- **Use case**: Development with exact dependency control
- **Pros**: Exact dependency versions, Poetry lock file support
- **Cons**: Slower build, potential timeout issues
- **Build time**: ~10-15 minutes

### 3. `Dockerfile.prod` (Multi-stage)

- **Use case**: Production deployment
- **Pros**: Smaller image size, non-root user, security optimized
- **Cons**: Longer build time
- **Build time**: ~12-18 minutes

## 🔧 Configuration

### Environment Variables

You can override configuration using environment variables:

```bash
docker run -d \
  -p 8000:8000 \
  -e MEMFUSE_HOST=0.0.0.0 \
  -e MEMFUSE_PORT=8000 \
  --name memfuse \
  memfuse:pip
```

### Volume Mounts

The container uses several directories that you may want to persist:

```bash
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \          # Persistent data storage
  -v $(pwd)/logs:/app/logs \          # Application logs
  -v $(pwd)/config:/app/config:ro \   # Configuration files (read-only)
  --name memfuse \
  memfuse:pip
```

## 📊 Monitoring and Health Checks

### Health Check Endpoint

The application provides a health check at:

```
GET http://localhost:8000/api/v1/health
```

### Viewing Logs

```bash
# View container logs
docker logs memfuse-pip

# Follow logs in real-time
docker logs -f memfuse-pip

# With Docker Compose
docker-compose logs -f memfuse
```

### Container Status

```bash
# Check if container is running
docker ps

# Check container health
docker inspect memfuse-pip | grep Health -A 10
```

## 🛠️ Troubleshooting

### Build Issues

**Problem**: Poetry timeout during dependency installation

```bash
# Solution: Use pip-only build
./docker-setup.sh pip
```

**Problem**: Out of memory during build

```bash
# Solution: Increase Docker memory limit or use production build
./docker-setup.sh prod
```

### Runtime Issues

**Problem**: Application not starting

```bash
# Check logs
docker logs memfuse-pip

# Check if port is available
lsof -i :8000
```

**Problem**: Health check failing

```bash
# Test health endpoint manually
curl http://localhost:8000/api/v1/health

# Check if application is fully loaded (models take time)
docker logs memfuse-pip | grep "All services initialized"
```

### Performance Issues

**Problem**: Slow model loading

- **Cause**: ML models are downloaded and loaded on first run
- **Solution**: Wait for initialization to complete (~3-5 minutes)
- **Check**: Look for "All services initialized successfully" in logs

## 🔄 Development Workflow

### Development Setup

```bash
# Start development container with live reload
./docker-setup.sh dev

# Mount source code for development (optional)
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/src:/app/src \
  -v $(pwd)/config:/app/config \
  --name memfuse-dev \
  memfuse:dev
```

### Testing Changes

```bash
# Rebuild and restart
./docker-setup.sh cleanup
./docker-setup.sh pip

# Or with Docker Compose
docker-compose down
docker-compose up -d --build
```

## 🚀 Production Deployment

### Production Build

```bash
# Use production Dockerfile for optimized image
./docker-setup.sh prod

# Or build manually
docker build -f Dockerfile.prod -t memfuse:prod .
```

### Production Configuration

```yaml
# docker-compose.prod.yml
version: "3.8"
services:
  memfuse:
    build:
      context: .
      dockerfile: Dockerfile.prod
    ports:
      - "8000:8000"
    volumes:
      - memfuse_data:/app/data
      - memfuse_logs:/app/logs
    environment:
      - MEMFUSE_HOST=0.0.0.0
      - MEMFUSE_PORT=8000
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G

volumes:
  memfuse_data:
  memfuse_logs:
```

### Security Considerations

- Production image runs as non-root user
- Minimal runtime dependencies
- Read-only configuration mounts
- Health checks enabled
- Resource limits configured

## 📚 API Documentation

Once running, access the API documentation at:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- OpenAPI JSON: http://localhost:8000/openapi.json

## 🧹 Cleanup

```bash
# Stop and remove containers
./docker-setup.sh cleanup

# Remove images (optional)
docker rmi memfuse:pip memfuse:dev memfuse:prod

# Remove volumes (optional - will delete data!)
docker volume prune
```

## 📞 Support

If you encounter issues:

1. Check the logs: `docker logs <container-name>`
2. Verify health endpoint: `curl http://localhost:8000/api/v1/health`
3. Try the pip-only build: `./docker-setup.sh pip`
4. Check available memory: `docker stats`

For more help, refer to the main MemFuse documentation or open an issue on GitHub.
