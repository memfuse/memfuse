# M3 Phase B Consolidation Plan (Documentation)

This document proposes how to consolidate Phase A auxiliary tables into the unified `m3_*` schemas. This is a reference plan and is not executed by the application.

```sql
-- Phase B: Consolidation of Phase A procedural tables into target m3_* schemas

-- This script documents the intended transformation from Phase A auxiliary tables
-- (procedural_memory, procedural_lessons, message_workflows) into the unified
-- m3_procedural, m3_skills, m3_behaviors schema already present in the project.

-- NOTE: This is a proposal and not executed automatically by the application.
-- Operators should review and adapt indexes/constraints based on environment.

-- 1) Migrate procedural_memory -> m3_procedural
--    - workflow_id -> id
--    - successful_workflow.plan -> steps
--    - successful_workflow.goal -> pattern_content
--    - usage_count -> execution_count
--    - embedding: copy trigger_embedding into m3_procedural.embedding

-- Example (requires jsonb extraction as per your data):
-- INSERT INTO m3_procedural (id, pattern_content, steps, confidence, execution_count, embedding, needs_embedding, metadata)
-- SELECT
--   pm.workflow_id AS id,
--   COALESCE(pm.successful_workflow->>'goal', '') AS pattern_content,
--   COALESCE(pm.successful_workflow->'plan', '[]'::jsonb) AS steps,
--   0.8 AS confidence,
--   pm.usage_count AS execution_count,
--   pm.trigger_embedding AS embedding,
--   FALSE AS needs_embedding,
--   jsonb_build_object('source', 'phaseA') AS metadata
-- FROM procedural_memory pm
-- ON CONFLICT (id) DO UPDATE SET
--   steps = EXCLUDED.steps,
--   pattern_content = EXCLUDED.pattern_content,
--   execution_count = GREATEST(m3_procedural.execution_count, EXCLUDED.execution_count),
--   embedding = COALESCE(EXCLUDED.embedding, m3_procedural.embedding),
--   updated_at = CURRENT_TIMESTAMP;

-- 2) Migrate procedural_lessons -> m3_behaviors (or new dedicated m3_lessons)
--    This example targets m3_behaviors with minimal mapping.
-- INSERT INTO m3_behaviors (id, behavior_name, behavior_description, metadata, embedding, needs_embedding)
-- SELECT
--   pl.lesson_id,
--   COALESCE(pl.agent, 'agent'),
--   COALESCE(pl.fix_summary, ''),
--   jsonb_build_object('status', pl.status, 'working_params', pl.working_params),
--   pl.trigger_embedding,
--   FALSE
-- FROM procedural_lessons pl
-- ON CONFLICT (id) DO NOTHING;

-- 3) Migrate message_workflows references into m3_procedural metadata links where applicable.
--    (Optional) Maintain a separate link table if needed for history.

-- 4) After validation, update application read path to use m3_* tables and phase out Phase A tables.
```

