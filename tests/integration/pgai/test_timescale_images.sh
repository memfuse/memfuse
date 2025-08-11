#!/bin/bash

# Test script to verify pgvectorscale installation capability across different TimescaleDB images

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test images
IMAGES=(
    # "timescale/timescaledb-ha:pg16"
    "timescale/timescaledb-ha:pg17"
    # "timescale/timescaledb:latest-pg17"
)

# Function to test an image
test_image() {
    local image=$1
    local container_name="test-pgvectorscale-$(echo $image | tr '/:' '-')"
    
    echo -e "${BLUE}🧪 Testing image: $image${NC}"
    echo "Container name: $container_name"
    
    # Clean up any existing container
    docker rm -f $container_name 2>/dev/null || true
    
    # Start container
    echo "Starting container..."
    docker run -d \
        --name $container_name \
        -e POSTGRES_PASSWORD=postgres \
        -e POSTGRES_DB=testdb \
        -p 5433:5432 \
        $image
    
    # Wait for PostgreSQL to be ready
    echo "Waiting for PostgreSQL to be ready..."
    for i in {1..30}; do
        if docker exec $container_name pg_isready -U postgres -d testdb >/dev/null 2>&1; then
            echo "PostgreSQL is ready"
            break
        fi
        if [ $i -eq 30 ]; then
            echo -e "${RED}❌ PostgreSQL failed to start within 30 seconds${NC}"
            docker logs $container_name
            docker rm -f $container_name
            return 1
        fi
        sleep 1
    done
    
    # Test extension availability and installation
    echo "Testing extension availability..."
    
    # Check available extensions
    echo "Available vector-related extensions:"
    docker exec $container_name psql -U postgres -d testdb -c \
        "SELECT name, default_version, comment FROM pg_available_extensions WHERE name LIKE '%vector%' OR name LIKE '%ai%';" \
        2>/dev/null || echo "No vector/ai extensions found"
    
    # Test pgai installation
    echo "Testing pgai extension installation..."
    if docker exec $container_name psql -U postgres -d testdb -c \
        "CREATE EXTENSION IF NOT EXISTS ai CASCADE;" 2>/dev/null; then
        echo -e "${GREEN}✅ pgai extension installed successfully${NC}"
        
        # Check pgai version
        docker exec $container_name psql -U postgres -d testdb -c \
            "SELECT extname, extversion FROM pg_extension WHERE extname = 'ai';" 2>/dev/null || true
    else
        echo -e "${RED}❌ pgai extension installation failed${NC}"
    fi
    
    # Test pgvectorscale installation
    echo "Testing pgvectorscale extension installation..."
    if docker exec $container_name psql -U postgres -d testdb -c \
        "CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE;" 2>/dev/null; then
        echo -e "${GREEN}✅ pgvectorscale extension installed successfully${NC}"
        
        # Check pgvectorscale version
        docker exec $container_name psql -U postgres -d testdb -c \
            "SELECT extname, extversion FROM pg_extension WHERE extname = 'vectorscale';" 2>/dev/null || true
            
        # Test basic functionality
        echo "Testing basic pgvectorscale functionality..."
        docker exec $container_name psql -U postgres -d testdb -c \
            "CREATE TABLE test_vectors (id SERIAL PRIMARY KEY, embedding vector(3)); 
             INSERT INTO test_vectors (embedding) VALUES ('[1,2,3]'), ('[4,5,6]');
             CREATE INDEX ON test_vectors USING diskann (embedding);
             SELECT id, embedding FROM test_vectors ORDER BY embedding <-> '[1,2,3]' LIMIT 1;" \
            2>/dev/null && echo -e "${GREEN}✅ pgvectorscale basic functionality works${NC}" \
            || echo -e "${YELLOW}⚠️  pgvectorscale basic functionality test failed${NC}"
    else
        echo -e "${RED}❌ pgvectorscale extension installation failed${NC}"
        echo "Error details:"
        docker exec $container_name psql -U postgres -d testdb -c \
            "CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE;" 2>&1 || true
    fi
    
    # Clean up
    echo "Cleaning up container..."
    docker rm -f $container_name >/dev/null 2>&1
    
    echo -e "${BLUE}📋 Test completed for $image${NC}"
    echo "----------------------------------------"
}

# Main execution
echo -e "${BLUE}🎯 TimescaleDB Images pgvectorscale Installation Test${NC}"
echo "============================================================"
echo "Testing pgvectorscale installation capability across different TimescaleDB images"
echo ""

# Test each image
for image in "${IMAGES[@]}"; do
    test_image "$image"
    echo ""
done

echo -e "${GREEN}🎉 All tests completed!${NC}"
