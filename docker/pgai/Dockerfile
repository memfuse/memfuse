# MemFuse PostgreSQL Docker Image with pgvector
#
# This Dockerfile creates a PostgreSQL 17 image with pgvector extension
# optimized for MemFuse's vector operations and embedding storage.
#
# Build: docker build -t memfuse/postgres-vector:latest -f docker/pgai/Dockerfile .
# Run: docker run -d --name memfuse-postgres -p 5432:5432 -e POSTGRES_PASSWORD=postgres memfuse/postgres-vector:latest

FROM postgres:17

# Metadata
LABEL maintainer="MemFuse Team"
LABEL description="PostgreSQL 17 with pgvector extension for MemFuse"
LABEL version="1.0.0"

# Environment variables
ENV POSTGRES_DB=memfuse
ENV POSTGRES_USER=postgres
ENV POSTGRES_PASSWORD=postgres

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    postgresql-server-dev-17 \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install pgvector extension
RUN cd /tmp \
    && git clone --branch v0.8.0 https://github.com/pgvector/pgvector.git \
    && cd pgvector \
    && make clean \
    && make OPTFLAGS="" \
    && make install \
    && cd / \
    && rm -rf /tmp/pgvector

# Install essential Python packages for vector operations
RUN python3 -m pip install --upgrade pip --break-system-packages \
    && python3 -m pip install --break-system-packages \
        psycopg2-binary \
        numpy

# Copy initialization scripts
COPY docker/pgai/init-scripts/ /docker-entrypoint-initdb.d/

# Copy configuration files
COPY docker/pgai/postgresql.conf /etc/postgresql/postgresql.conf
COPY docker/pgai/pg_hba.conf /etc/postgresql/pg_hba.conf

# Set proper permissions
RUN chmod +x /docker-entrypoint-initdb.d/*.sh

# Configure PostgreSQL for vector operations
RUN echo "shared_preload_libraries = 'vector'" >> /usr/share/postgresql/postgresql.conf.sample

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD pg_isready -U $POSTGRES_USER -d $POSTGRES_DB

# Expose PostgreSQL port
EXPOSE 5432

# Use the default PostgreSQL entrypoint
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["postgres", "-c", "config_file=/etc/postgresql/postgresql.conf"]
