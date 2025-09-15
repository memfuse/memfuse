# M3 Overview

M3 = Procedural Memory & Multi‑Agent Orchestration. It enables MemFuse to:

- Plan a user goal into steps (agents), execute with retries, and learn lessons.
- Reuse successful workflows for similar future tasks (procedural memory).
- Keep a lightweight guidance channel to inform new tasks.

This branch integrates M3 while enforcing a consistent API Response Schema and metadata flow across layers (M0/M1/M2/M3).

## Key Concepts
- Planner: decomposes a goal into ordered steps.
- Agents: tools like RAGQueryAgent, WebSearchAgent, DatabaseQueryAgent, ShellCommandAgent, ReportGenerationAgent.
- Orchestrator: reuse‑or‑plan‑execute‑learn; logs workflows/lessons.
- Procedural Store: persists `procedural_memory`, `procedural_lessons`, and `message_workflows`.

## Triggers & Modes
- Write path (ADD):
  - If a user message contains `metadata.task` and `metadata.task_eos: true`, Gateway triggers the M3 Orchestrator for that task, processes history for the same task, and logs workflows/lessons.
  - This is the only path that triggers orchestration (to avoid side effects in queries).
- Read path (QUERY):
  - If query includes `metadata.task`, Gateway can enrich results with compact guidance text (`metadata.m3_guidance`) summarizing reusable workflows/lessons.
  - Controlled by `config/m3/default.yaml:m3.enable_query_guidance` (default: false).

## Data & Schema Across Layers
- M0 (`m0_raw`): raw messages; now includes JSONB `metadata` to persist request metadata (e.g., task/mode). Indexed via GIN.
- M1 (`m1_episodic`): chunked episodic memory; includes JSONB `metadata` for request lineage. Indexed via GIN.
- M2 (`m2_semantic`): semantic facts; already includes JSONB `metadata`.
- M3 (procedural):
  - `procedural_memory`: reusable workflows (workflow_id, trigger_embedding, successful_workflow, usage_count).
  - `procedural_lessons`: success/fail lessons with trigger_embedding for similarity.
  - `message_workflows`: links message ids to workflows and step index.

## API Response Schema (Core)
- Required top‑level: `status`, `code`, `data`, `message`, `errors`.
- `data` contains `results` (array) and `total` (int). API layer strips internal echoes.
- Each result requires: `id`, `relevance_score`, `memory_type`, `created_at`, `updated_at`, `metadata`.
- Episodic vs Semantic:
  - Episodic: has `content`; no `fact`.
  - Semantic: has `fact = {text, triples}`; no `content`; `derived_from` lives in `metadata`.
- Metadata: requires `user_id`, `agent_id`, `session_id`, `session_name`, `scope`.
  - `scope`: `in_session`/`cross_session` when request carries `session_id`, else null.
  - Forbidden: `level`, `retrieval`, `source`.

## Config
- `config/m3/default.yaml`:
  - `enable_workflow_reuse` (default true): controls write‑path orchestration reuse.
  - `enable_lesson_learning` (default true): enable reflection system.
  - `enable_query_guidance` (default false): enable query‑time guidance enrichment.

## Differences vs memfuse_mvp
- MVP lacked metadata persistence and strict response schema. In this branch:
  - Gateway processors standardize/migrate MVP‑style fields to the new contract.
  - M0/M1 now persist request metadata JSONB for lineage/use.
  - M3 uses structured tables for workflows and lessons.

