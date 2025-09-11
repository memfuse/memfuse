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
- New tables documented in `docs/m3/schema_phaseA.md`.

### A3. LLM Unification (design-only in Phase A)
- Define a thin interface compatible with MVP `ChatLLM` (completion_json, chat) and current Core runtime.

### A4. RAG Placement
- Implement MVP-like RAG service under `src/memfuse_core/rag/`.

### A5. API Routing (design-only in Phase A)
- Messages API uses `metadata.tag == 'm3'` to trigger orchestration.
- Add a query endpoint for M3 workflows/lessons scan.

### A6. Observability
- Persist plan/trace/reflection artifacts under `runs/`.

### A7. Tests and Docs
- Include config/docs/tests as part of Phase A changes.

## Phase B (Consolidation)
- Migrate to `m3_*` tables per `docs/m3/schema_phaseB.md`.

## Acceptance Criteria
- Tests under `tests/m3` pass locally without external services.
- Docs explain design decisions, constraints, and execution steps.
- No changes to `memfuse_mvp/` are required.
- No changes to `messages` table schema in Phase A.
