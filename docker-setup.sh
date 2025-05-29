#!/bin/bash

# MemFuse Docker Setup Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    print_warning "docker-compose not found, trying docker compose..."
    if ! docker compose version &> /dev/null; then
        print_error "Neither docker-compose nor 'docker compose' is available. Please install Docker Compose."
        exit 1
    fi
    COMPOSE_CMD="docker compose"
else
    COMPOSE_CMD="docker-compose"
fi

# Function to build and run with development Dockerfile (Poetry)
run_dev() {
    print_status "Building MemFuse with development Dockerfile (Poetry)..."
    docker build -t memfuse:dev .
    
    print_status "Starting MemFuse development container..."
    docker run -d \
        --name memfuse-dev \
        -p 8000:8000 \
        -v "$(pwd)/data:/app/data" \
        -v "$(pwd)/logs:/app/logs" \
        -v "$(pwd)/config:/app/config:ro" \
        memfuse:dev
    
    print_status "MemFuse is running at http://localhost:8000"
    print_status "Use 'docker logs memfuse-dev' to view logs"
    print_status "Use 'docker stop memfuse-dev' to stop the container"
}

# Function to build and run with pip-only Dockerfile
run_pip() {
    print_status "Building MemFuse with pip-only Dockerfile..."
    docker build -f Dockerfile.pip -t memfuse:pip .
    
    print_status "Starting MemFuse pip container..."
    docker run -d \
        --name memfuse-pip \
        -p 8000:8000 \
        -v "$(pwd)/data:/app/data" \
        -v "$(pwd)/logs:/app/logs" \
        -v "$(pwd)/config:/app/config:ro" \
        memfuse:pip
    
    print_status "MemFuse is running at http://localhost:8000"
    print_status "Use 'docker logs memfuse-pip' to view logs"
    print_status "Use 'docker stop memfuse-pip' to stop the container"
}

# Function to build and run with production Dockerfile
run_prod() {
    print_status "Building MemFuse with production Dockerfile..."
    docker build -f Dockerfile.prod -t memfuse:prod .
    
    print_status "Starting MemFuse production container..."
    docker run -d \
        --name memfuse-prod \
        -p 8000:8000 \
        -v "$(pwd)/data:/app/data" \
        -v "$(pwd)/logs:/app/logs" \
        -v "$(pwd)/config:/app/config:ro" \
        memfuse:prod
    
    print_status "MemFuse is running at http://localhost:8000"
    print_status "Use 'docker logs memfuse-prod' to view logs"
    print_status "Use 'docker stop memfuse-prod' to stop the container"
}

# Function to run with docker-compose
run_compose() {
    print_status "Building and starting MemFuse with Docker Compose..."
    $COMPOSE_CMD up -d --build
    
    print_status "MemFuse is running at http://localhost:8000"
    print_status "Use '$COMPOSE_CMD logs' to view logs"
    print_status "Use '$COMPOSE_CMD down' to stop all services"
}

# Function to stop and clean up containers
cleanup() {
    print_status "Stopping and removing containers..."
    
    # Stop and remove dev container if it exists
    if docker ps -a --format "table {{.Names}}" | grep -q "memfuse-dev"; then
        docker stop memfuse-dev 2>/dev/null || true
        docker rm memfuse-dev 2>/dev/null || true
    fi
    
    # Stop and remove pip container if it exists
    if docker ps -a --format "table {{.Names}}" | grep -q "memfuse-pip"; then
        docker stop memfuse-pip 2>/dev/null || true
        docker rm memfuse-pip 2>/dev/null || true
    fi
    
    # Stop and remove prod container if it exists
    if docker ps -a --format "table {{.Names}}" | grep -q "memfuse-prod"; then
        docker stop memfuse-prod 2>/dev/null || true
        docker rm memfuse-prod 2>/dev/null || true
    fi
    
    # Stop docker-compose services
    $COMPOSE_CMD down 2>/dev/null || true
    
    print_status "Cleanup completed"
}

# Create necessary directories
mkdir -p data logs

# Main script logic
case "${1:-}" in
    "dev")
        cleanup
        run_dev
        ;;
    "pip")
        cleanup
        run_pip
        ;;
    "prod")
        cleanup
        run_prod
        ;;
    "compose")
        run_compose
        ;;
    "cleanup"|"clean")
        cleanup
        ;;
    "help"|"-h"|"--help")
        echo "MemFuse Docker Setup Script"
        echo "Usage: $0 [dev|pip|prod|compose|cleanup|help]"
        echo ""
        echo "Commands:"
        echo "  dev      - Build and run with development Dockerfile (Poetry)"
        echo "  pip      - Build and run with pip-only Dockerfile (faster build)"
        echo "  prod     - Build and run with production Dockerfile (multi-stage)"
        echo "  compose  - Build and run with Docker Compose"
        echo "  cleanup  - Stop and remove all containers"
        echo "  help     - Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0 dev     # Start development container with Poetry"
        echo "  $0 pip     # Start container using pip-only (recommended for build issues)"
        echo "  $0 prod    # Start production container"
        echo "  $0 compose # Start with Docker Compose"
        echo "  $0 cleanup # Clean up containers"
        echo ""
        print_info "If you're experiencing timeout issues, try:"
        print_info "  $0 pip     # Uses pip directly, often more reliable"
        ;;
    *)
        print_warning "No command specified. Trying pip-only build for better reliability."
        print_status "Use '$0 help' to see all available options."
        cleanup
        run_pip
        ;;
esac 