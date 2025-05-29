# Alternative Dockerfile using pip instead of Poetry
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

# Set work directory
WORKDIR /app

# Copy project files first
COPY pyproject.toml ./
COPY src/ ./src/
COPY config/ ./config/
COPY README.md ./

# Install the package with all dependencies using pip
# This extracts dependencies from pyproject.toml and installs them
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