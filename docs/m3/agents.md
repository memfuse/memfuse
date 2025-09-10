# M3 Agents and Executor

This document summarizes the built-in agents and the AgentExecutor abstraction.

## AgentExecutor
- Location: `src/memfuse_core/m3/executor.py`
- Responsibilities:
  - Lessons-informed parameter seeding (uses recent successful lesson params)
  - Retry attempts with simple success heuristics per agent
  - Per-step artifact writing: `step_{idx}_{agent}.json`
  - Returns executed steps and outcomes (`success`, `attempts`, `duration_ms`, `error`)

The orchestrator composes steps and delegates step execution to the executor.

## Built-in Agents

- `RAGQueryAgent` (default)
  - Uses the minimal RAG service to retrieve chunks and produce an answer.
  - Success: non-empty `answer` and no `error`.

- `ReportGenerationAgent` (default)
  - Summarizes inputs into a concise report via LLM.
  - Success: non-empty `report` and no `error`.

- `WebSearchAgent`
  - Uses DuckDuckGo Instant Answer API (no key required) to fetch top results.
  - Input: `{ "query": str, "max_results": int? }`
  - Output: `{ "results": [{title, url, snippet}], "provider": "duckduckgo" }`
  - Success: at least one result present.
  - Testability: `_fetch_json` is easily monkeypatched for offline tests.

- `DatabaseQueryAgent`
  - Read-only: allows `SELECT` queries via DatabaseService.
  - Output: `{ headers: [...], rows: [...] }` (limited for safety)
  - Success: presence of `rows` or `headers`.

- `ShellCommandAgent`
  - Disabled by default. Enable via `ALLOW_SHELL_AGENT=true`.
  - Only allows safe commands such as `echo`.
  - Output: `{ exit, output }` or `{ error }` when disabled or rejected.

## Artifacts
- Each step writes a JSON file under `runs/{timestamp}/{session_id}/` including:
  - input (excluding context), output, attempts, success, duration_ms
- The orchestrator also writes `input.json`, `plan.json`, `pre_lessons.json`, `reflection.json`, `report.txt`.

