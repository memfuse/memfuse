# LLM Unification Plan

Goal: Provide a thin LLM interface compatible with MVP `ChatLLM` while integrating cleanly with MemFuse Core.

## Interface (to be implemented)
- Module: `src/memfuse_core/llm/`
- Class: `ChatLLM`
  - `completion_json(system_prompt: str, user_prompt: str) -> str`
  - `chat(system_prompt: str, messages: list[dict]) -> str`

## Transport
- Default: OpenAI-compatible HTTP API
  - Env: `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_COMPATIBLE_MODEL`
- Fallback: Minimal offline behaviors for tests (e.g., echo/report summarizer) when key/base_url is missing.

## Reasoning
- MVP’s Planner/Executor depend on predictable JSON completions.
- Consolidating LLM invocation lets both RAG and Orchestrator share one path, easing provider swaps.

## Next Steps
- Implement `ChatLLM` in Phase A coding step.
- Add provider config validation and helpful error messages.
