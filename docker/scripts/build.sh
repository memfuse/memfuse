#!/bin/bash
# MemFuse Docker Build Script
# Supports multi-environment builds and tag management

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
ENVIRONMENT="dev"
TAG="latest"
PUSH=false
CACHE=true
PLATFORM=""

# Help function
show_help() {
    echo "MemFuse Docker Build Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -e, --env ENVIRONMENT    Environment to build for (dev|prod|test|local) [default: dev]"
    echo "  -t, --tag TAG           Docker image tag [default: latest]"
    echo "  -p, --push              Push image to registry after build"
    echo "  --no-cache              Build without using cache"
    echo "  --platform PLATFORM     Target platform (e.g., linux/amd64,linux/arm64)"
    echo "  -h, --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -e prod -t v1.0.0 -p"
    echo "  $0 --env dev --tag latest"
    echo "  $0 --platform linux/amd64,linux/arm64 --push"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -t|--tag)
            TAG="$2"
            shift 2
            ;;
        -p|--push)
            PUSH=true
            shift
            ;;
        --no-cache)
            CACHE=false
            shift
            ;;
        --platform)
            PLATFORM="$2"
            shift 2
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

# Set image name based on environment
IMAGE_NAME="memfuse:${TAG}-${ENVIRONMENT}"

echo -e "${BLUE}🐳 Building MemFuse Docker Image${NC}"
echo -e "${BLUE}Environment: ${YELLOW}$ENVIRONMENT${NC}"
echo -e "${BLUE}Tag: ${YELLOW}$TAG${NC}"
echo -e "${BLUE}Image: ${YELLOW}$IMAGE_NAME${NC}"

# Change to project root
cd "$(dirname "$0")/../.."

# Build command
BUILD_CMD="docker build -f docker/app/Dockerfile"

# Add cache option
if [ "$CACHE" = false ]; then
    BUILD_CMD="$BUILD_CMD --no-cache"
fi

# Add platform option
if [ -n "$PLATFORM" ]; then
    BUILD_CMD="$BUILD_CMD --platform $PLATFORM"
fi

# Add tag and context
BUILD_CMD="$BUILD_CMD -t $IMAGE_NAME ."

echo -e "${YELLOW}Executing: $BUILD_CMD${NC}"

# Execute build
if eval $BUILD_CMD; then
    echo -e "${GREEN}✅ Build completed successfully!${NC}"
    echo -e "${GREEN}Image: $IMAGE_NAME${NC}"
else
    echo -e "${RED}❌ Build failed!${NC}"
    exit 1
fi

# Push if requested
if [ "$PUSH" = true ]; then
    echo -e "${BLUE}🚀 Pushing image to registry...${NC}"
    if docker push "$IMAGE_NAME"; then
        echo -e "${GREEN}✅ Push completed successfully!${NC}"
    else
        echo -e "${RED}❌ Push failed!${NC}"
        exit 1
    fi
fi

# Show image info
echo -e "${BLUE}📊 Image Information:${NC}"
docker images "$IMAGE_NAME" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"

echo -e "${GREEN}🎉 Build process completed!${NC}"
