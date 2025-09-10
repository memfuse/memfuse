# M3 Overview

This document provides a high-level overview of the M3 (Procedural Memory & Multi-Agent Orchestration) integration for MemFuse Core.

## What is M3?
- A workflow-oriented capability that plans complex tasks into steps (agents), executes them with retries and parameterization, then learns from successes/failures.
- Stores:
  - Successful workflows for future reuse (procedural memory)
  - Lessons (success snippets, fix patterns)

## Phase A vs Phase B
- Phase A: MVP-compatible, minimal risk, add new auxiliary tables, orchestrator, and APIs (without altering `messages` schema).
- Phase B: Consolidate Phase A data into existing `m3_*` tables already present in the store.

## Key Concepts
- Planner → plan steps
- AgentExecutor → parameter propose/execute/judge/retry
- Orchestrator → reuse-or-plan-execute-learn
- RAGService → retrieval-powered context for steps that require knowledge

## What ships in Phase A (docs/tests only in this patch)
- Migration plan, schema proposals, and tests for presence.
- Next patch: orchestrator, agents, RAG, and API wiring.
