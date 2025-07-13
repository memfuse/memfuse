#!/bin/bash
# MemFuse PgAI Docker Build Script
# This script builds and manages the pgai-enabled Docker environment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DOCKER_DIR="$PROJECT_ROOT/docker"

# Default values
ENVIRONMENT="dev"
BUILD_CACHE="true"
PULL_LATEST="false"
VERBOSE="false"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    cat << EOF
MemFuse PgAI Docker Build Script

Usage: $0 [OPTIONS] COMMAND

Commands:
    build       Build the pgai Docker image
    up          Start the pgai environment
    down        Stop the pgai environment
    restart     Restart the pgai environment
    logs        Show logs from pgai services
    status      Show status of pgai services
    clean       Clean up pgai Docker resources
    test        Test the pgai environment
    shell       Open shell in postgres container

Options:
    -e, --env ENV           Environment (dev|prod|test) [default: dev]
    -n, --no-cache          Build without cache
    -p, --pull              Pull latest base images
    -v, --verbose           Verbose output
    -h, --help              Show this help message

Examples:
    $0 build                    # Build pgai image
    $0 up                       # Start pgai environment
    $0 up --env prod            # Start production environment
    $0 test                     # Test pgai functionality
    $0 shell                    # Open postgres shell
    $0 clean                    # Clean up resources

EOF
}

# Function to build pgai image
build_pgai() {
    print_status "Building MemFuse PgAI Docker image..."
    
    local build_args=""
    if [ "$BUILD_CACHE" = "false" ]; then
        build_args="--no-cache"
    fi
    
    if [ "$PULL_LATEST" = "true" ]; then
        build_args="$build_args --pull"
    fi
    
    cd "$PROJECT_ROOT"
    
    print_status "Building pgai PostgreSQL image..."
    docker build $build_args -t memfuse/pgai:latest -f docker/pgai/Dockerfile .
    
    print_status "Building MemFuse application image..."
    docker build $build_args -t memfuse/app:latest -f docker/app/Dockerfile .
    
    print_success "PgAI Docker images built successfully!"
}

# Function to start pgai environment
start_pgai() {
    print_status "Starting MemFuse PgAI environment..."
    
    cd "$PROJECT_ROOT"
    
    # Check if .env file exists
    if [ ! -f ".env" ]; then
        print_warning ".env file not found. Creating default .env file..."
        cat > .env << EOF
# MemFuse PgAI Environment Variables
POSTGRES_HOST=postgres-pgai
POSTGRES_PORT=5432
POSTGRES_DB=memfuse
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres

# Optional: OpenAI API (if using external LLM)
# OPENAI_API_KEY=your_api_key_here
# OPENAI_MODEL=gpt-4o-mini

# Embedding Model
EMBEDDING_MODEL=all-MiniLM-L6-v2

# PgAI Settings
PGAI_IMMEDIATE_TRIGGER=true
PGAI_AUTO_EMBEDDING=true
EOF
        print_success "Default .env file created"
    fi
    
    docker-compose -f docker/compose/docker-compose.pgai.yml up -d
    
    print_success "PgAI environment started!"
    print_status "Services:"
    docker-compose -f docker/compose/docker-compose.pgai.yml ps
}

# Function to stop pgai environment
stop_pgai() {
    print_status "Stopping MemFuse PgAI environment..."
    
    cd "$PROJECT_ROOT"
    docker-compose -f docker/compose/docker-compose.pgai.yml down
    
    print_success "PgAI environment stopped!"
}

# Function to restart pgai environment
restart_pgai() {
    print_status "Restarting MemFuse PgAI environment..."
    stop_pgai
    start_pgai
}

# Function to show logs
show_logs() {
    print_status "Showing PgAI environment logs..."
    
    cd "$PROJECT_ROOT"
    docker-compose -f docker/compose/docker-compose.pgai.yml logs -f --tail=100
}

# Function to show status
show_status() {
    print_status "PgAI Environment Status:"
    
    cd "$PROJECT_ROOT"
    docker-compose -f docker/compose/docker-compose.pgai.yml ps
    
    print_status "Container Health:"
    docker ps --filter "name=memfuse-pgai" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
}

# Function to clean up
clean_pgai() {
    print_warning "This will remove all pgai containers, networks, and volumes!"
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Cleaning up PgAI Docker resources..."
        
        cd "$PROJECT_ROOT"
        
        # Stop and remove containers
        docker-compose -f docker/compose/docker-compose.pgai.yml down -v --remove-orphans
        
        # Remove images
        docker rmi memfuse/pgai:latest memfuse/app:latest 2>/dev/null || true
        
        # Remove dangling images
        docker image prune -f
        
        print_success "PgAI Docker resources cleaned up!"
    else
        print_status "Cleanup cancelled."
    fi
}

# Function to test pgai environment
test_pgai() {
    print_status "Testing MemFuse PgAI environment..."
    
    cd "$PROJECT_ROOT"
    
    # Check if containers are running
    if ! docker-compose -f docker/compose/docker-compose.pgai.yml ps | grep -q "Up"; then
        print_error "PgAI environment is not running. Start it first with: $0 up"
        exit 1
    fi
    
    print_status "Testing PostgreSQL connection..."
    docker exec memfuse-pgai-postgres psql -U postgres -d memfuse -c "SELECT version();"
    
    print_status "Testing pgvector extension..."
    docker exec memfuse-pgai-postgres psql -U postgres -d memfuse -c "SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';"
    
    print_status "Testing pgai extension..."
    docker exec memfuse-pgai-postgres psql -U postgres -d memfuse -c "SELECT extname, extversion FROM pg_extension WHERE extname = 'pgai';"
    
    print_status "Testing m0_episodic table..."
    docker exec memfuse-pgai-postgres psql -U postgres -d memfuse -c "SELECT COUNT(*) FROM m0_episodic;"
    
    print_status "Testing immediate trigger system..."
    docker exec memfuse-pgai-postgres psql -U postgres -d memfuse -c "SELECT * FROM get_trigger_system_status();"
    
    print_status "Testing MemFuse API health..."
    if curl -f http://localhost:8000/api/v1/health >/dev/null 2>&1; then
        print_success "MemFuse API is healthy!"
    else
        print_warning "MemFuse API health check failed. Check logs with: $0 logs"
    fi
    
    print_success "PgAI environment test completed!"
}

# Function to open postgres shell
open_shell() {
    print_status "Opening PostgreSQL shell..."
    docker exec -it memfuse-pgai-postgres psql -U postgres -d memfuse
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -n|--no-cache)
            BUILD_CACHE="false"
            shift
            ;;
        -p|--pull)
            PULL_LATEST="true"
            shift
            ;;
        -v|--verbose)
            VERBOSE="true"
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        build|up|down|restart|logs|status|clean|test|shell)
            COMMAND="$1"
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Check if command is provided
if [ -z "${COMMAND:-}" ]; then
    print_error "No command provided"
    show_usage
    exit 1
fi

# Execute command
case $COMMAND in
    build)
        build_pgai
        ;;
    up)
        start_pgai
        ;;
    down)
        stop_pgai
        ;;
    restart)
        restart_pgai
        ;;
    logs)
        show_logs
        ;;
    status)
        show_status
        ;;
    clean)
        clean_pgai
        ;;
    test)
        test_pgai
        ;;
    shell)
        open_shell
        ;;
    *)
        print_error "Unknown command: $COMMAND"
        show_usage
        exit 1
        ;;
esac
