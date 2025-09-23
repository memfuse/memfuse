# API Load Testing (Locust) – Base Suite

This suite provides a Locust-based load harness for MemFuse REST APIs. It creates a user, agent, and session per virtual user (VU), then exercises messages, queries, and chunk reads with a configurable mix.

## Prerequisites
- MemFuse server running (default: `http://localhost:8000`) with DB up.
- Optional: API key auth disabled (default). If enabled, set `API_KEY` env.
- Locust installed in your env: `pip install locust`

## Environment Variables
- `BASE_URL` – Server base URL (default `http://localhost:8000`).
- `API_PREFIX` – API prefix (default `/api/v1`).
- `API_KEY` – If auth is on, pass the bearer token value.
- `ENTITY_PREFIX` – Name prefix for users/agents (default `perf`).
- `MSG_SIZE_PROFILE` – `short|mixed|long` (default `mixed`).
- `WAIT_MIN_MS`, `WAIT_MAX_MS` – Think time between tasks (default 100–500ms).

## Running
Basic (UI):
```
locust -f tests/performance/api/locustfile.py --host=http://localhost:8000
```

Headless examples:
```
# Smoke (~5 VUs for 2m)
BASE_URL=http://localhost:8000 locust -f tests/performance/api/locustfile.py \
  --host=$BASE_URL --users 5 --spawn-rate 2 --run-time 2m --headless

# Load (~50 VUs for 15m) with a named profile
PROFILE=load BASE_URL=http://localhost:8000 locust -f tests/performance/api/locustfile.py \
  --host=$BASE_URL --users 50 --spawn-rate 10 --run-time 15m --headless

# Stress (adjust users/step outside via CLI; weights/think-time from profile)
PROFILE=stress BASE_URL=http://localhost:8000 locust -f tests/performance/api/locustfile.py \
  --host=$BASE_URL --users 100 --spawn-rate 20 --run-time 25m --headless

# Soak profile (2h suggested runtime)
PROFILE=soak BASE_URL=http://localhost:8000 locust -f tests/performance/api/locustfile.py \
  --host=$BASE_URL --users 20 --spawn-rate 5 --run-time 2h --headless
```

## Endpoints Covered
- Users: create/get by name
- Agents: create/get by name
- Sessions: create
- Messages: add/list/update/delete
- User Query: `/users/{user_id}/query`
- Chunks: `/sessions/{session_id}/chunks`

## Notes
- Suite reads API prefix from `API_PREFIX` (defaults to `/api/v1`).
- If API keys are enabled, set `API_KEY` and the suite will send
  `Authorization: Bearer <API_KEY>`.
- Payloads follow the models in `src/memfuse_core/models/api.py`.

## Seeding a Dataset (TICKET-005)
Before longer runs, pre-seed a realistic dataset to reduce cold-start effects and stabilize caches:

Examples:
```
# Small dataset
poetry run python scripts/perf/seed_data.py \
  --base-url http://localhost:8000 --users 3 --agents 1 \
  --sessions-per-user 2 --messages-per-session 10

# Larger dataset (mixed message sizes, moderate concurrency)
poetry run python scripts/perf/seed_data.py \
  --base-url http://localhost:8000 --users 10 --agents 2 \
  --sessions-per-user 3 --messages-per-session 50 \
  --msg-size-profile mixed --concurrency 8
```
If API keys are enabled, add `--api-key YOUR_TOKEN` (header defaults to `Authorization`).

### Profiles
- Set `PROFILE` env to one of: `smoke`, `load`, `stress`, `spike`, `soak`.
- You can also point to a custom JSON file via `PROFILE_PATH=/path/to/custom.json`.
- Profiles control task weights, think-time, and message size mix. Concurrency and
  runtime are still driven by Locust CLI flags (`--users`, `--spawn-rate`, `--run-time`).
- Each profile also includes a `suggested` block with recommended CLI values for
  users/spawn-rate/run-time. You can copy those into your command if desired.
