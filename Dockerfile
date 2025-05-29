# Use Python 3.11 slim image as base
FROM python:3.11-slim

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
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install Poetry with timeout and retry settings
RUN pip install --timeout=100 --retries=5 poetry

# Configure Poetry: Don't create virtual environment, install dependencies globally
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=0 \
    POETRY_CACHE_DIR=/tmp/poetry_cache \
    POETRY_HTTP_TIMEOUT=300 \
    POETRY_INSTALLER_MAX_WORKERS=1

# Set work directory
WORKDIR /app

# Copy Poetry configuration files
COPY pyproject.toml poetry.lock* ./

# Configure Poetry settings for better reliability
RUN poetry config virtualenvs.create false \
    && poetry config installer.max-workers 1 \
    && poetry config http-basic.pypi.timeout 300

# Install dependencies with retry logic
RUN for i in {1..3}; do \
        echo "Attempt $i to install dependencies..." && \
        poetry install --only=main --no-root --timeout=300 && break || \
        (echo "Attempt $i failed, retrying..." && sleep 30); \
    done && \
    rm -rf $POETRY_CACHE_DIR

# Copy project files
COPY src/ ./src/
COPY config/ ./config/
COPY README.md ./

# Install the package in editable mode
RUN pip install --timeout=100 --retries=5 -e .

# Create data directory
RUN mkdir -p data logs

# Expose the port the app runs on
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Run the application
CMD ["python", "-m", "memfuse_core"] 