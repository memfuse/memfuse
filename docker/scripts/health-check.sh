#!/bin/bash
# MemFuse Health Check Script
# Comprehensive health monitoring for all services

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
ENVIRONMENT="dev"
TIMEOUT=30
VERBOSE=false
JSON_OUTPUT=false

# Help function
show_help() {
    echo "MemFuse Health Check Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -e, --env ENVIRONMENT    Environment to check (dev|prod|test|local) [default: dev]"
    echo "  -t, --timeout SECONDS    Timeout for health checks [default: 30]"
    echo "  -v, --verbose            Verbose output"
    echo "  -j, --json               Output in JSON format"
    echo "  -h, --help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -e prod -v"
    echo "  $0 --env dev --timeout 60"
    echo "  $0 -j > health-report.json"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -t|--timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -j|--json)
            JSON_OUTPUT=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# Validate environment
case $ENVIRONMENT in
    dev|prod|test|local)
        ;;
    *)
        echo -e "${RED}Invalid environment: $ENVIRONMENT${NC}"
        echo "Valid environments: dev, prod, test, local"
        exit 1
        ;;
esac

# Set compose file and ports based on environment
COMPOSE_FILE="docker/compose/docker-compose.${ENVIRONMENT}.yml"
case $ENVIRONMENT in
    dev)
        POSTGRES_PORT=5432
        MEMFUSE_PORT=8000
        ;;
    prod)
        POSTGRES_PORT=5432
        MEMFUSE_PORT=8000
        ;;
    test)
        POSTGRES_PORT=5433
        MEMFUSE_PORT=8000
        ;;
    local)
        POSTGRES_PORT=5434
        MEMFUSE_PORT=8001
        ;;
esac

# Change to project root
cd "$(dirname "$0")/../.."

# Initialize health status
OVERALL_HEALTH="healthy"
HEALTH_RESULTS=()

# Function to check service health
check_service_health() {
    local service_name=$1
    local check_command=$2
    local description=$3
    
    if [ "$VERBOSE" = true ] && [ "$JSON_OUTPUT" = false ]; then
        echo -e "${BLUE}Checking $description...${NC}"
    fi
    
    local start_time=$(date +%s)
    local status="unhealthy"
    local message=""
    
    # Try the health check with timeout
    if timeout $TIMEOUT bash -c "$check_command" >/dev/null 2>&1; then
        status="healthy"
        message="Service is responding correctly"
    else
        status="unhealthy"
        message="Service is not responding or unhealthy"
        OVERALL_HEALTH="unhealthy"
    fi
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    # Store result
    HEALTH_RESULTS+=("{\"service\":\"$service_name\",\"status\":\"$status\",\"message\":\"$message\",\"duration\":$duration}")
    
    if [ "$JSON_OUTPUT" = false ]; then
        if [ "$status" = "healthy" ]; then
            echo -e "${GREEN}✅ $description: $status (${duration}s)${NC}"
        else
            echo -e "${RED}❌ $description: $status (${duration}s)${NC}"
        fi
        
        if [ "$VERBOSE" = true ]; then
            echo -e "${BLUE}   Message: $message${NC}"
        fi
    fi
}

# Function to get container status
get_container_status() {
    local service_name=$1
    docker-compose -f "$COMPOSE_FILE" ps -q "$service_name" 2>/dev/null | xargs docker inspect --format='{{.State.Status}}' 2>/dev/null || echo "not_found"
}

# Start health checks
if [ "$JSON_OUTPUT" = false ]; then
    echo -e "${BLUE}🏥 MemFuse Health Check${NC}"
    echo -e "${BLUE}Environment: ${YELLOW}$ENVIRONMENT${NC}"
    echo -e "${BLUE}Timeout: ${YELLOW}${TIMEOUT}s${NC}"
    echo ""
fi

# Check if compose file exists
if [ ! -f "$COMPOSE_FILE" ]; then
    echo -e "${RED}❌ Compose file not found: $COMPOSE_FILE${NC}"
    exit 1
fi

# Check Docker Compose services status
if [ "$JSON_OUTPUT" = false ]; then
    echo -e "${BLUE}📊 Container Status:${NC}"
    docker-compose -f "$COMPOSE_FILE" ps
    echo ""
fi

# Check PostgreSQL health
check_service_health "postgres" \
    "pg_isready -h localhost -p $POSTGRES_PORT -U postgres" \
    "PostgreSQL Database"

# Check MemFuse API health
check_service_health "memfuse" \
    "curl -f -s http://localhost:$MEMFUSE_PORT/api/v1/health" \
    "MemFuse API"

# Check database connectivity from MemFuse
check_service_health "database_connection" \
    "curl -f -s http://localhost:$MEMFUSE_PORT/api/v1/health | grep -q 'database.*ok'" \
    "Database Connection"

# Additional checks for production environment
if [ "$ENVIRONMENT" = "prod" ]; then
    # Check disk space
    check_service_health "disk_space" \
        "[ \$(df / | tail -1 | awk '{print \$5}' | sed 's/%//') -lt 90 ]" \
        "Disk Space (<90%)"
    
    # Check memory usage
    check_service_health "memory_usage" \
        "[ \$(free | grep Mem | awk '{print (\$3/\$2) * 100.0}' | cut -d. -f1) -lt 90 ]" \
        "Memory Usage (<90%)"
fi

# Output results
if [ "$JSON_OUTPUT" = true ]; then
    echo "{"
    echo "  \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\","
    echo "  \"environment\": \"$ENVIRONMENT\","
    echo "  \"overall_status\": \"$OVERALL_HEALTH\","
    echo "  \"checks\": ["
    
    for i in "${!HEALTH_RESULTS[@]}"; do
        echo "    ${HEALTH_RESULTS[$i]}"
        if [ $i -lt $((${#HEALTH_RESULTS[@]} - 1)) ]; then
            echo ","
        fi
    done
    
    echo "  ]"
    echo "}"
else
    echo ""
    echo -e "${BLUE}📋 Health Check Summary:${NC}"
    if [ "$OVERALL_HEALTH" = "healthy" ]; then
        echo -e "${GREEN}✅ Overall Status: HEALTHY${NC}"
        exit 0
    else
        echo -e "${RED}❌ Overall Status: UNHEALTHY${NC}"
        exit 1
    fi
fi
