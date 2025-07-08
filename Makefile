# MemFuse Makefile

.PHONY: help test test-unit test-integration test-e2e test-chunking test-quick test-coverage clean
.PHONY: docker-build docker-dev docker-prod docker-test docker-local docker-clean docker-health

# Default target
help:
	@echo "MemFuse Commands:"
	@echo ""
	@echo "Testing:"
	@echo "  make test           - Run all tests"
	@echo "  make test-unit      - Run unit tests only"
	@echo "  make test-integration - Run integration tests only"
	@echo "  make test-e2e       - Run end-to-end tests only"
	@echo "  make test-chunking  - Run chunking-related tests only"
	@echo "  make test-quick     - Run quick tests (unit, no slow)"
	@echo "  make test-coverage  - Run tests with coverage report"
	@echo "  make clean          - Clean test artifacts"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build   - Build Docker image"
	@echo "  make docker-dev     - Start development environment"
	@echo "  make docker-prod    - Start production environment"
	@echo "  make docker-test    - Start test environment"
	@echo "  make docker-local   - Start local development environment"
	@echo "  make docker-clean   - Clean Docker resources"
	@echo "  make docker-health  - Check service health"

# Run all tests
test:
	@echo "🧪 Running all tests..."
	poetry run python -m pytest tests/ -v

# Run unit tests only
test-unit:
	@echo "🧪 Running unit tests..."
	poetry run python -m pytest tests/unit/ -v -m unit

# Run integration tests only
test-integration:
	@echo "🧪 Running integration tests..."
	poetry run python -m pytest tests/integration/ -v -m integration

# Run end-to-end tests only (requires running server)
test-e2e:
	@echo "🧪 Running end-to-end tests..."
	@echo "⚠️  Make sure MemFuse server is running!"
	poetry run python -m pytest tests/e2e/ -v -m e2e

# Run chunking-related tests only
test-chunking:
	@echo "🧪 Running chunking tests..."
	poetry run python -m pytest tests/ -v -m chunking

# Run quick tests (unit tests, no slow tests)
test-quick:
	@echo "🧪 Running quick tests..."
	poetry run python -m pytest tests/unit/ -v -m "unit and not slow"

# Run tests with coverage
test-coverage:
	@echo "🧪 Running tests with coverage..."
	poetry run python -m pytest tests/ -v --cov=src/memfuse_core --cov-report=html --cov-report=term-missing
	@echo "📊 Coverage report generated in htmlcov/index.html"

# Clean test artifacts
clean:
	@echo "🧹 Cleaning test artifacts..."
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "✅ Cleaned test artifacts"

# Install test dependencies
install-test-deps:
	@echo "📦 Installing test dependencies..."
	poetry install --with dev

# Lint tests
lint-tests:
	@echo "🔍 Linting test files..."
	python -m flake8 tests/ --max-line-length=120
	python -m black tests/ --check

# Docker Commands

# Build Docker image
docker-build:
	@echo "🐳 Building Docker image..."
	./docker/scripts/build.sh

# Start development environment
docker-dev:
	@echo "🐳 Starting development environment..."
	./docker/scripts/deploy.sh -e dev -a up

# Start production environment
docker-prod:
	@echo "🐳 Starting production environment..."
	./docker/scripts/deploy.sh -e prod -a up

# Start test environment
docker-test:
	@echo "🐳 Starting test environment..."
	./docker/scripts/deploy.sh -e test -a up

# Start local development environment
docker-local:
	@echo "🐳 Starting local development environment..."
	./docker/scripts/deploy.sh -e local -a up

# Clean Docker resources
docker-clean:
	@echo "🐳 Cleaning Docker resources..."
	./docker/scripts/deploy.sh -e dev -a down
	./docker/scripts/deploy.sh -e prod -a down
	./docker/scripts/deploy.sh -e test -a down
	./docker/scripts/deploy.sh -e local -a down
	docker system prune -f

# Check service health
docker-health:
	@echo "🏥 Checking service health..."
	./docker/scripts/health-check.sh -v

# Format tests
format-tests:
	@echo "🎨 Formatting test files..."
	python -m black tests/
	python -m isort tests/

# Run specific test file
# Usage: make test-file FILE=tests/unit/rag/chunk/test_base.py
test-file:
	@echo "🧪 Running specific test file: $(FILE)"
	python -m pytest $(FILE) -v

# Run tests with specific marker
# Usage: make test-marker MARKER=chunking
test-marker:
	@echo "🧪 Running tests with marker: $(MARKER)"
	python -m pytest tests/ -v -m $(MARKER)

# Verify test structure
verify-tests:
	@echo "🔍 Verifying test structure..."
	python tests/verify_structure.py

# Debug specific test
# Usage: make debug-test TEST=tests/unit/rag/chunk/test_base.py::TestChunkData::test_chunk_data_creation
debug-test:
	@echo "🐛 Debugging test: $(TEST)"
	python -m pytest $(TEST) -v -s --tb=long

# Performance test
test-performance:
	@echo "🚀 Running performance tests..."
	python -m pytest tests/integration/ -v -m "chunking and not slow" --durations=10

# Test with parallel execution
test-parallel:
	@echo "🧪 Running tests in parallel..."
	python -m pytest tests/ -v -n auto

# Continuous testing (watch for changes)
test-watch:
	@echo "👀 Watching for changes and running tests..."
	python -m pytest tests/ -v --looponfail

# Generate test report
test-report:
	@echo "📊 Generating test report..."
	python -m pytest tests/ --html=test_report.html --self-contained-html
	@echo "📊 Test report generated: test_report.html"
