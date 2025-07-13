# MemFuse Application Docker Image
#
# This Dockerfile creates a complete MemFuse application image with all dependencies
# including support for pgai integration and vector operations.
#
# Build: docker build -t memfuse/app:latest -f docker/app/Dockerfile .
# Run: docker run -d --name memfuse-app -p 8000:8000 memfuse/app:latest

FROM python:3.11-slim

# Metadata
LABEL maintainer="MemFuse Team"
LABEL description="MemFuse application with pgai and vector support"
LABEL version="0.2.1"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    PIP_RETRIES=5

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY pyproject.toml poetry.lock* ./

# Install Poetry
RUN pip install poetry

# Configure Poetry
RUN poetry config virtualenvs.create false \
    && poetry config virtualenvs.in-project false

# Install Python dependencies
RUN poetry install --no-dev --no-interaction --no-ansi

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/config

# Set proper permissions
RUN chmod +x /app/docker/scripts/*.sh 2>/dev/null || true

# Create non-root user
RUN groupadd -r memfuse && useradd -r -g memfuse memfuse
RUN chown -R memfuse:memfuse /app
USER memfuse

# Expose the port the app runs on
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Default command
CMD ["poetry", "run", "memfuse-core"]