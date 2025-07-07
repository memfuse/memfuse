#!/bin/bash
# MemFuse Docker Deployment Script
# Supports different environments and deployment strategies

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
ENVIRONMENT="dev"
ACTION="up"
DETACH=true
BUILD=false
PULL=false

# Help function
show_help() {
    echo "MemFuse Docker Deployment Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -e, --env ENVIRONMENT    Environment to deploy (dev|prod|test|local) [default: dev]"
    echo "  -a, --action ACTION      Action to perform (up|down|restart|logs|status) [default: up]"
    echo "  -f, --foreground         Run in foreground (not detached)"
    echo "  -b, --build              Build images before deployment"
    echo "  -p, --pull               Pull latest images before deployment"
    echo "  -h, --help               Show this help message"
    echo ""
    echo "Actions:"
    echo "  up        Start services"
    echo "  down      Stop and remove services"
    echo "  restart   Restart services"
    echo "  logs      Show service logs"
    echo "  status    Show service status"
    echo ""
    echo "Examples:"
    echo "  $0 -e prod -a up -b"
    echo "  $0 --env dev --action logs"
    echo "  $0 -e test -a down"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -a|--action)
            ACTION="$2"
            shift 2
            ;;
        -f|--foreground)
            DETACH=false
            shift
            ;;
        -b|--build)
            BUILD=true
            shift
            ;;
        -p|--pull)
            PULL=true
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

# Validate action
case $ACTION in
    up|down|restart|logs|status)
        ;;
    *)
        echo -e "${RED}Invalid action: $ACTION${NC}"
        echo "Valid actions: up, down, restart, logs, status"
        exit 1
        ;;
esac

# Set compose file
COMPOSE_FILE="docker/compose/docker-compose.${ENVIRONMENT}.yml"

echo -e "${BLUE}🚀 MemFuse Deployment${NC}"
echo -e "${BLUE}Environment: ${YELLOW}$ENVIRONMENT${NC}"
echo -e "${BLUE}Action: ${YELLOW}$ACTION${NC}"
echo -e "${BLUE}Compose File: ${YELLOW}$COMPOSE_FILE${NC}"

# Change to project root
cd "$(dirname "$0")/../.."

# Check if compose file exists
if [ ! -f "$COMPOSE_FILE" ]; then
    echo -e "${RED}❌ Compose file not found: $COMPOSE_FILE${NC}"
    exit 1
fi

# Base docker-compose command
COMPOSE_CMD="docker-compose -f $COMPOSE_FILE"

# Execute action
case $ACTION in
    up)
        echo -e "${YELLOW}Starting services...${NC}"
        
        # Pull images if requested
        if [ "$PULL" = true ]; then
            echo -e "${BLUE}📥 Pulling latest images...${NC}"
            $COMPOSE_CMD pull
        fi
        
        # Build if requested
        if [ "$BUILD" = true ]; then
            echo -e "${BLUE}🔨 Building images...${NC}"
            $COMPOSE_CMD build
        fi
        
        # Start services
        if [ "$DETACH" = true ]; then
            $COMPOSE_CMD up -d
        else
            $COMPOSE_CMD up
        fi
        
        if [ "$DETACH" = true ]; then
            echo -e "${GREEN}✅ Services started successfully!${NC}"
            echo -e "${BLUE}📊 Service Status:${NC}"
            $COMPOSE_CMD ps
        fi
        ;;
        
    down)
        echo -e "${YELLOW}Stopping services...${NC}"
        $COMPOSE_CMD down -v
        echo -e "${GREEN}✅ Services stopped successfully!${NC}"
        ;;
        
    restart)
        echo -e "${YELLOW}Restarting services...${NC}"
        $COMPOSE_CMD restart
        echo -e "${GREEN}✅ Services restarted successfully!${NC}"
        echo -e "${BLUE}📊 Service Status:${NC}"
        $COMPOSE_CMD ps
        ;;
        
    logs)
        echo -e "${BLUE}📋 Service Logs:${NC}"
        $COMPOSE_CMD logs -f --tail=100
        ;;
        
    status)
        echo -e "${BLUE}📊 Service Status:${NC}"
        $COMPOSE_CMD ps
        echo ""
        echo -e "${BLUE}🔍 Health Status:${NC}"
        $COMPOSE_CMD exec postgres pg_isready -U postgres || echo "PostgreSQL not ready"
        $COMPOSE_CMD exec memfuse curl -f http://localhost:8000/api/v1/health || echo "MemFuse not ready"
        ;;
esac

echo -e "${GREEN}🎉 Deployment operation completed!${NC}"
