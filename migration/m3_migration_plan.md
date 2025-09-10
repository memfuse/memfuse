# M3 Migration Plan (Phase A → B)

This document details the Option A first (fast MVP-aligned feature parity), then consolidate to the existing M3 tables (Option B) in a later phase.

Author: MemFuse Core Team
Status: Approved for Phase A planning

## Goals
- Bring MVP-style procedural memory (M3) and multi-agent orchestration into MemFuse Core with minimal risk.
- Do not modify `memfuse_mvp/` files (reference only).
- Avoid altering current `messages` schema. Use an auxiliary reference table for workflow logging.
- Unify LLM invocation across MemFuse Core, compatible with MVP semantics.
- Place new RAG logic under `src/memfuse_core/rag` following MVP’s approach.

## Constraints and Agreements
- Tags: Use current API `metadata` (and/or auxiliary reference table) instead of query-param-based `tag=m3` for persisted signals.
- Messages table: do not alter. Use a new table that references messages for workflow annotations (temporary, migratable later).
- LLM: Prefer current project-compatible invocation; expose a thin abstraction compatible with MVP’s `ChatLLM` semantics.
- RAG: Mirror MVP technical approach; implement under `src/memfuse_core/rag/`.
- Tests: Add unit tests that pass immediately (docs/config/schema presence and sanity checks) without external services.
- Docs: Update `docs/` with M3 overview, API usage, and run instructions.

## Phase A (MVP-compatible, minimal risk)

### A1. Configuration
- Add Hydra/env-driven knobs (no code yet):
  - `m3.enabled`: master switch (already present in `config/memory/default.yaml`)
  - `m3.procedural_top_k` (default: 5)
  - `m3.procedural_reuse_threshold` (default: 0.9)
  - `m3.planner_max_attempts` (default: 3)
  - `m3.runs_base_dir` (default: `runs`)
- Keep values in config; implementation will read them later.

### A2. Schemas (temporary, non-destructive)
- New tables (SQL proposals in `migration/schemas/m3_phaseA.sql`):
  1) `message_workflows`
     - id (TEXT PK)
     - message_id (TEXT, FK to messages.id)
     - workflow_id (TEXT NULL)
     - step_index (INT NULL)
     - tags (TEXT[] DEFAULT `{}`)
     - metadata (JSONB DEFAULT `{}`)
     - created_at/updated_at
  2) `procedural_memory`
     - workflow_id (TEXT PK)
     - trigger_embedding (VECTOR(384) or configurable)
     - trigger_pattern (TEXT NULL)
     - successful_workflow (JSONB)
     - usage_count (INT DEFAULT 1)
     - created_at/updated_at
  3) `procedural_lessons`
     - lesson_id (TEXT PK)
     - trigger_embedding (VECTOR(384))
     - goal_text (TEXT)
     - agent (TEXT)
     - status (TEXT CHECK IN('success','fail'))
     - error (TEXT NULL)
     - fix_summary (TEXT NULL)
     - working_params (JSONB)
     - created_at/updated_at

Notes:
- No changes to `messages` in Phase A.
- We will provide migration scripts later for Phase B consolidation.

### A3. LLM Unification (design-only in Phase A)
- Define a thin interface compatible with MVP `ChatLLM` (completion_json, chat) and current Core runtime.
- Target: OpenAI-compatible HTTP client (base URL/model/key from env), offline fallback for tests if needed.
- Keep in `src/memfuse_core/llm/` (to be implemented in the coding phase).

### A4. RAG Placement
- Implement MVP-like RAG service under `src/memfuse_core/rag/` with:
  - ingestion (chunking+embedding+store)
  - retrieval (session-aware, top-k)
  - context building
  - chat invocation via unified LLM interface
- In Phase A, mirror MVP behavior; later align with Core vector store/hybrid buffer.

### A5. API Routing (design-only in Phase A)
- Messages API: extend creation/chat flow to recognize M3 via `metadata` (e.g., `{"tag": "m3"}`) and route to orchestrator when enabled.
- New Query API: `POST /api/v1/users/{user_id}/query` with `metadata.tag == 'm3'` in body to focus on workflow/lessons retrieval.
- For persistence of workflow logs, write entries into `message_workflows` (not the `messages` table) referencing the created assistant/system message IDs.

### A6. Observability
- Persist plan/trace/reflection artifacts into `runs/{timestamp}/{session_id}` (configurable via `runs_base_dir`).

### A7. Tests (already added in this patch)
- Validate presence and integrity of docs, config, and schema plans.
- No network or DB required.

### A8. Docs (already added in this patch)
- Overview, API usage, dev workflow, and next steps under `docs/m3/`.

## Phase B (Consolidation into existing M3 tables)

- Map `procedural_memory` → `m3_procedural` (embed `trigger_embedding` into `embedding`, transform workflows into `pattern_content/steps`).
- Map `procedural_lessons` → `m3_behaviors` or a new `m3_lessons` for dedicated experience tracking.
- Provide one-off migration SQL and a dual-read window; then switch Orchestrator to `m3_*` tables.

## Deliverables Checklist
- [x] Plan docs and schemas (this phase)
- [x] Passing tests verifying presence
- [ ] Orchestrator/Agents/RAG implementation (next phase)
- [ ] API wiring (messages/query) with metadata-based M3 routing
- [ ] E2E validation script

## Acceptance Criteria (Phase A)
- Tests under `tests/m3` pass with `poetry run pytest -q tests/m3`.
- Docs explain design decisions, constraints, and execution steps.
- No changes to `memfuse_mvp/`.
- No changes to `messages` table schema in this phase.
