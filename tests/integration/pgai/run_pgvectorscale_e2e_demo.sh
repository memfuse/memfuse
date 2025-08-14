#!/bin/bash

# MemFuse pgvectorscale End-to-End Integration Demo
# ==================================================
# 
# This script demonstrates the complete MemFuse memory layer architecture using pgvectorscale
# with StreamingDiskANN for high-performance vector similarity search.
#
# USAGE:
#   bash ./tests/integration/pgai/run_pgvectorscale_e2e_demo.sh [OPTIONS]
#
# OPTIONS:
#   --cleanup    Clean up Docker containers and volumes
#   --verify     Verify existing data integrity
#   --help       Show this help message
#
# REQUIREMENTS:
#   - Docker and Docker Compose installed
#   - Python 3.8+ with required packages (sentence-transformers, psycopg2-binary, numpy)
#   - At least 4GB available memory for Docker
#
# ARCHITECTURE:
#   - M0 Layer: Raw streaming messages with metadata
#   - M1 Layer: Intelligent chunking with embeddings
#   - pgvectorscale: StreamingDiskANN for optimized vector search
#   - Normalized similarity scores (0-1 range) for comparison
#
# EXECUTION FLOW:
#   1. Start pgvectorscale database container
#   2. Initialize schema and StreamingDiskANN indexes
#   3. Generate streaming conversation data
#   4. Process M0 → M1 chunking with embeddings
#   5. Execute similarity search queries with normalized scores
#   6. Validate data integrity and lineage
#
# PERFORMANCE FEATURES:
#   - StreamingDiskANN: 2-5x faster than HNSW
#   - SBQ compression: 75% memory reduction
#   - Memory-optimized storage layout
#   - Incremental index updates for streaming data

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../" && pwd)"
DOCKER_COMPOSE_FILE="${PROJECT_ROOT}/tests/integration/pgai/docker-compose.pgvectorscale.yml"
CONTAINER_NAME="pgvectorscale-e2e"
DB_NAME="memfuse"
DB_USER="postgres"
DB_PASSWORD="postgres"
DB_PORT="5432"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

log_header() {
    echo -e "${PURPLE}$1${NC}"
    echo "============================================================"
}

# Help function
show_help() {
    cat << EOF
MemFuse pgvectorscale End-to-End Integration Demo

USAGE:
    $0 [OPTIONS]

OPTIONS:
    --cleanup    Clean up Docker containers and volumes
    --verify     Verify existing data integrity
    --help       Show this help message

DESCRIPTION:
    This script demonstrates the complete MemFuse memory layer architecture
    using pgvectorscale with StreamingDiskANN for high-performance vector
    similarity search with normalized scores (0-1 range).

REQUIREMENTS:
    - Docker and Docker Compose
    - Python 3.8+ with sentence-transformers, psycopg2-binary, numpy
    - At least 4GB available memory

EXAMPLES:
    # Run complete end-to-end demo
    $0

    # Verify existing data
    $0 --verify

    # Clean up environment
    $0 --cleanup

ARCHITECTURE:
    M0 Layer: Raw streaming messages with metadata
    M1 Layer: Intelligent chunking with embeddings
    pgvectorscale: StreamingDiskANN for optimized vector search
    Normalized similarity scores for cross-system comparison

EOF
}

# Cleanup function
cleanup_environment() {
    log_header "🧹 Cleaning Up Environment"
    
    if [ -f "$DOCKER_COMPOSE_FILE" ]; then
        log_info "Stopping and removing containers..."
        docker-compose -f "$DOCKER_COMPOSE_FILE" down -v --remove-orphans || true
    fi
    
    # Remove any dangling containers
    if docker ps -a --format "table {{.Names}}" | grep -q "$CONTAINER_NAME"; then
        log_info "Removing container: $CONTAINER_NAME"
        docker rm -f "$CONTAINER_NAME" || true
    fi
    
    log_success "Environment cleaned up successfully"
}

# Check prerequisites
check_prerequisites() {
    log_header "📋 Checking Prerequisites"
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
    log_success "Docker is available"
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed or not in PATH"
        exit 1
    fi
    log_success "Docker Compose is available"
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed or not in PATH"
        exit 1
    fi
    log_success "Python 3 is available"
    
    # Check required files
    local required_files=(
        "$DOCKER_COMPOSE_FILE"
        "${PROJECT_ROOT}/tests/integration/pgai/init-pgvectorscale.sql"
        "${PROJECT_ROOT}/tests/integration/pgai/pgvectorscale_e2e_demo.py"
    )
    
    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            log_error "Required file not found: $file"
            exit 1
        fi
    done
    log_success "All required files are present"
}

# Wait for database to be ready
wait_for_database() {
    log_info "Waiting for pgvectorscale database to be ready..."
    
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if docker exec "$CONTAINER_NAME" pg_isready -U "$DB_USER" -d "$DB_NAME" &> /dev/null; then
            log_success "Database is ready"
            return 0
        fi
        
        echo -n "."
        sleep 2
        ((attempt++))
    done
    
    log_error "Database failed to start within $((max_attempts * 2)) seconds"
    return 1
}

# Start database environment
start_database() {
    log_header "🚀 Starting pgvectorscale Database Environment"
    
    # Ensure data directory exists
    local data_dir="${PROJECT_ROOT}/data/volumes"
    if [ ! -d "$data_dir" ]; then
        log_info "Creating data directory: $data_dir"
        mkdir -p "$data_dir"
        log_success "Data directory created"
    fi
    
    log_info "Starting pgvectorscale container with StreamingDiskANN support..."
    docker-compose -f "$DOCKER_COMPOSE_FILE" up -d
    
    wait_for_database
    
    # Verify pgvectorscale extension
    log_info "Verifying pgvectorscale extension..."
    local extension_check=$(docker exec "$CONTAINER_NAME" psql -U "$DB_USER" -d "$DB_NAME" -t -c "SELECT COUNT(*) FROM pg_available_extensions WHERE name = 'vectorscale';")
    
    if [ "$(echo $extension_check | tr -d ' ')" = "1" ]; then
        log_success "pgvectorscale extension is available"
    else
        log_error "pgvectorscale extension is not available"
        exit 1
    fi
}

# Initialize database schema
initialize_database() {
    log_header "🔧 Initializing Database Schema"
    
    log_info "Creating extensions and schema..."
    docker exec "$CONTAINER_NAME" psql -U "$DB_USER" -d "$DB_NAME" -f /docker-entrypoint-initdb.d/init-pgvectorscale.sql
    
    log_success "Database schema initialized successfully"
}

# Run Python demo
run_python_demo() {
    log_header "🎬 Running End-to-End Python Demo"
    
    log_info "Executing pgvectorscale E2E demonstration..."
    
    # Set environment variables for Python script
    export PGVECTORSCALE_HOST="localhost"
    export PGVECTORSCALE_PORT="$DB_PORT"
    export PGVECTORSCALE_DB="$DB_NAME"
    export PGVECTORSCALE_USER="$DB_USER"
    export PGVECTORSCALE_PASSWORD="$DB_PASSWORD"
    
    # Run the Python demo script
    cd "$PROJECT_ROOT"
    poetry run python tests/integration/pgai/pgvectorscale_e2e_demo.py
    
    log_success "Python demo completed successfully"
}

# Verify data integrity
verify_data_integrity() {
    log_header "🔍 Verifying Data Integrity"
    
    log_info "Checking database status..."
    docker exec "$CONTAINER_NAME" psql -U "$DB_USER" -d "$DB_NAME" -c "
        SELECT 
            'M0 Messages' as layer,
            COUNT(*) as count,
            pg_size_pretty(pg_total_relation_size('m0_raw')) as size
        FROM m0_raw
        UNION ALL
        SELECT 
            'M1 Chunks' as layer,
            COUNT(*) as count,
            pg_size_pretty(pg_total_relation_size('m1_episodic')) as size
        FROM m1_episodic
        UNION ALL
        SELECT 
            'M1 Embeddings' as layer,
            COUNT(*) as count,
            'N/A' as size
        FROM m1_episodic WHERE embedding IS NOT NULL;
    "
    
    log_info "Checking StreamingDiskANN index configuration..."
    docker exec "$CONTAINER_NAME" psql -U "$DB_USER" -d "$DB_NAME" -c "
        SELECT 
            indexname,
            'StreamingDiskANN (pgvectorscale)' as index_type,
            indexdef
        FROM pg_indexes 
        WHERE tablename = 'm1_episodic' 
        AND indexdef LIKE '%diskann%';
    "
    
    log_info "Checking data lineage relationships..."
    docker exec "$CONTAINER_NAME" psql -U "$DB_USER" -d "$DB_NAME" -c "
        SELECT 
            COUNT(DISTINCT chunk_id) as total_chunks,
            AVG(array_length(m0_message_ids, 1)) as avg_m0_per_chunk,
            MIN(array_length(m0_message_ids, 1)) as min_m0_per_chunk,
            MAX(array_length(m0_message_ids, 1)) as max_m0_per_chunk
        FROM m1_episodic;
    "
    
    log_success "Data integrity verification completed"
}

# Main execution function
main() {
    case "${1:-}" in
        --help)
            show_help
            exit 0
            ;;
        --cleanup)
            cleanup_environment
            exit 0
            ;;
        --verify)
            verify_data_integrity
            exit 0
            ;;
        "")
            # Run full demo
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
    
    log_header "🎯 MemFuse pgvectorscale End-to-End Integration Demo"
    log_info "Starting complete end-to-end demonstration..."
    
    check_prerequisites
    start_database
    initialize_database
    run_python_demo
    verify_data_integrity
    
    log_header "🎉 Demo Completed Successfully"
    log_success "All components verified and working correctly!"
    log_info "Container '$CONTAINER_NAME' is still running for further testing"
    log_info "Connect: docker exec -it $CONTAINER_NAME psql -U $DB_USER -d $DB_NAME"
    log_info "Cleanup: $0 --cleanup"
}

# Execute main function with all arguments
main "$@"
