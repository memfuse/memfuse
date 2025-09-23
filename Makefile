# MemFuse Makefile

# Run each recipe in a single shell so that env/vars persist across lines
.ONESHELL:

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
	@echo ""
	@echo "Performance:"
	@echo "  make perf-api-smoke   - Run smoke profile + DB metrics"
	@echo "  make perf-api-load    - Run load profile + DB metrics"
	@echo "  make perf-api-stress  - Run stress profile + DB metrics"
	@echo "  make perf-api-spike   - Run spike profile + DB metrics"
	@echo "  make perf-api-soak    - Run soak profile + DB metrics"
	@echo "  make perf-seed        - Seed dataset via REST (users/agents/sessions/messages)"
	@echo "  make perf-compare     - Compare two perf runs (RUN_A, RUN_B)"
	@echo "  make perf-scaling     - Run multi-worker scaling harness (uvicorn workers)"
	@echo "      Vars: HOST (default http://localhost:8000), USERS, SPAWN, RUNTIME, DB_INTERVAL (default 5s)"

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

# Generate summary for an existing perf run directory
# Usage: make perf-report RUN_DIR=outputs/perf/load-YYYYMMDD-HHMMSS
perf-report:
	@if [ -z "$(RUN_DIR)" ]; then echo "RUN_DIR is required (e.g., outputs/perf/load-YYYYMMDD-HHMMSS)"; exit 1; fi
	poetry run python scripts/perf/aggregate_report.py --run-dir $(RUN_DIR) --host $(HOST)
	poetry run python scripts/perf/aggregate_report.py --run-dir $(RUN_DIR) --host $(HOST) --format html

# Compare two runs for regressions
# Usage: make perf-compare RUN_A=outputs/perf/load-<tsA> RUN_B=outputs/perf/load-<tsB> [P95=30 AVG=20 FAIL=1]
perf-compare:
	@if [ -z "$(RUN_A)" ] || [ -z "$(RUN_B)" ]; then echo "RUN_A and RUN_B are required"; exit 1; fi
	@P95=$${P95:-30}; AVG=$${AVG:-20}; FAIL=$${FAIL:-1}; \
	poetry run python scripts/perf/compare_runs.py --run-a $(RUN_A) --run-b $(RUN_B) \
	  --max-p95-increase-pct $$P95 --max-avg-increase-pct $$AVG --max-fail-rate-pct $$FAIL

# One-shot profiler (seed -> run -> report) with progress/ETA
# Usage: make perf-profile [HOST=... PROFILE=load USERS=50 SPAWN=10 RUNTIME=20m DB_INTERVAL=5s]
# Optional seeding: SEED=true [SEED_USERS=... SEED_AGENTS=... SEED_SESSIONS_PER_USER=... SEED_MESSAGES_PER_SESSION=... SEED_MSG_SIZE_PROFILE=mixed SEED_CONCURRENCY=6 PER_AGENT_SESSIONS=true]
perf-profile:
	@HOST=$${HOST:-$(HOST)}; PROFILE=$${PROFILE:-load}; USERS=$${USERS:-50}; SPAWN=$${SPAWN:-10}; RUNTIME=$${RUNTIME:-20m}; DB_INTERVAL=$${DB_INTERVAL:-$(DB_INTERVAL)}; \
	SEED_FLAG=""; if [ "$$SEED" = "true" ]; then SEED_FLAG="--seed"; fi; \
	ALLOW_FAIL_FLAG=""; if [ "$$ALLOW_FAIL" = "true" ]; then ALLOW_FAIL_FLAG="--allow-failures"; fi; \
	EXTRA_SEED=""; if [ "$$PER_AGENT_SESSIONS" = "true" ]; then EXTRA_SEED="--per-agent-sessions"; fi; \
	poetry run python scripts/perf/profile_run.py \
	  --host "$$HOST" --profile "$$PROFILE" --users "$$USERS" --spawn "$$SPAWN" --runtime "$$RUNTIME" --db-interval "$$DB_INTERVAL" \
	  $$ALLOW_FAIL_FLAG \
	  $$SEED_FLAG \
	  --seed-users "$${SEED_USERS:-3}" \
	  --seed-agents "$${SEED_AGENTS:-1}" \
	  --seed-sessions-per-user "$${SEED_SESSIONS_PER_USER:-2}" \
	  --seed-messages-per-session "$${SEED_MESSAGES_PER_SESSION:-20}" \
	  --seed-msg-size-profile "$${SEED_MSG_SIZE_PROFILE:-mixed}" \
	  --seed-concurrency "$${SEED_CONCURRENCY:-6}" \
	  $$EXTRA_SEED

# Multi-worker scaling harness
# Usage: make perf-scaling [WORKERS="1,2,4" PORT=8010 PROFILE=load USERS=50 SPAWN=10 RUNTIME=20m]
perf-scaling:
	@WORKERS=$${WORKERS:-"1,2,4"}; PORT=$${PORT:-8010}; PROFILE=$${PROFILE:-load}; \
	USERS=$${USERS:-50}; SPAWN=$${SPAWN:-10}; RUNTIME=$${RUNTIME:-20m}; \
	poetry run python scripts/perf/worker_scaling.py --workers $$WORKERS --port $$PORT --profile $$PROFILE --users $$USERS --spawn $$SPAWN --runtime $$RUNTIME

# ------------------------------------------------------------
# Performance targets (Locust + DB metrics sampler)
# ------------------------------------------------------------

HOST ?= http://localhost:8000
DB_INTERVAL ?= 5s

define RUN_PERF
	@echo "🚀 Running perf profile: $(1)"
	@echo "# Stable timestamp computed at parse time and passed as $(5)"
	@RUN_PATH=outputs/perf/$(1)-$(5) && \
	HOST=$${HOST:-$(HOST)} && \
	USERS=$${USERS:-$(2)} && SPAWN=$${SPAWN:-$(3)} && RUNTIME=$${RUNTIME:-$(4)} && \
	mkdir -p "$$RUN_PATH" && \
	poetry run python scripts/perf/capture_run_metadata.py --run-dir "$$RUN_PATH" --host "$$HOST" && \
	echo "Run dir: $$RUN_PATH" && \
	echo "Host: $$HOST  Users: $$USERS  Spawn: $$SPAWN  Runtime: $$RUNTIME  DB interval: $(DB_INTERVAL)" && \
	(PROFILE=$(1) poetry run python scripts/perf/db_metrics.py --interval $(DB_INTERVAL) --duration "$$RUNTIME" --out "$$RUN_PATH/db_metrics.jsonl" &) && \
	DB_PID=$$! && \
	PROFILE=$(1) poetry run locust -f tests/performance/api/locustfile.py --host="$$HOST" --users "$$USERS" --spawn-rate "$$SPAWN" --run-time "$$RUNTIME" --headless --csv="$$RUN_PATH/locust" --csv-full-history && \
	poetry run python scripts/perf/aggregate_report.py --run-dir "$$RUN_PATH" --host "$$HOST" && \
	poetry run python scripts/perf/aggregate_report.py --run-dir "$$RUN_PATH" --host "$$HOST" --format html && \
	wait $$DB_PID && \
	echo "✅ Done. Artifacts in $$RUN_PATH"
endef

perf-api-smoke:
	$(call RUN_PERF,smoke,$(or $(USERS),5),$(or $(SPAWN),2),$(or $(RUNTIME),3m),$(shell date +%Y%m%d-%H%M%S))

perf-api-load:
	$(call RUN_PERF,load,$(or $(USERS),50),$(or $(SPAWN),10),$(or $(RUNTIME),20m),$(shell date +%Y%m%d-%H%M%S))

perf-api-stress:
	$(call RUN_PERF,stress,$(or $(USERS),100),$(or $(SPAWN),20),$(or $(RUNTIME),25m),$(shell date +%Y%m%d-%H%M%S))

perf-api-spike:
	$(call RUN_PERF,spike,$(or $(USERS),30),$(or $(SPAWN),30),$(or $(RUNTIME),6m),$(shell date +%Y%m%d-%H%M%S))

perf-api-soak:
	$(call RUN_PERF,soak,$(or $(USERS),20),$(or $(SPAWN),5),$(or $(RUNTIME),2h),$(shell date +%Y%m%d-%H%M%S))

# ------------------------------------------------------------
# Dataset seeding (TICKET-005)
# ------------------------------------------------------------

API_PREFIX ?= /api/v1
ENTITY_PREFIX ?= perf

SEED_USERS ?= 3
SEED_AGENTS ?= 1
SEED_SESSIONS_PER_USER ?= 2
SEED_MESSAGES_PER_SESSION ?= 20
SEED_MSG_SIZE_PROFILE ?= mixed
SEED_CONCURRENCY ?= 6

# Set PER_AGENT_SESSIONS=true to create sessions_per_user per agent
ifeq ($(PER_AGENT_SESSIONS),true)
  PER_AGENT_FLAG := --per-agent-sessions
else
  PER_AGENT_FLAG :=
endif

perf-seed:
	@echo "🌱 Seeding dataset..."
	@echo "Host: $(HOST)  Users: $(SEED_USERS)  Agents: $(SEED_AGENTS)  Sessions/user: $(SEED_SESSIONS_PER_USER)  Messages/session: $(SEED_MESSAGES_PER_SESSION)  Size: $(SEED_MSG_SIZE_PROFILE)  Concurrency: $(SEED_CONCURRENCY)  Per-agent-sessions: $(PER_AGENT_SESSIONS)"
	@API_KEY_ARG=""; \
	if [ -n "$(API_KEY)" ]; then API_KEY_ARG="--api-key $(API_KEY)"; fi; \
	poetry run python scripts/perf/seed_data.py \
	  --base-url $(HOST) \
	  --api-prefix $(API_PREFIX) $$API_KEY_ARG \
	  --entity-prefix $(ENTITY_PREFIX) \
	  --users $(SEED_USERS) \
	  --agents $(SEED_AGENTS) \
	  --sessions-per-user $(SEED_SESSIONS_PER_USER) \
	  --messages-per-session $(SEED_MESSAGES_PER_SESSION) \
	  --msg-size-profile $(SEED_MSG_SIZE_PROFILE) \
	  $(PER_AGENT_FLAG) \
	  --concurrency $(SEED_CONCURRENCY)
	@echo "✅ Seeding complete"
