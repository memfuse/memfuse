-- Phase A: Non-destructive M3 schemas (do not alter existing tables)

-- 1) message_workflows (auxiliary table referencing messages)
CREATE TABLE IF NOT EXISTS message_workflows (
    id TEXT PRIMARY KEY,
    message_id TEXT NOT NULL,
    workflow_id TEXT,
    step_index INT,
    tags TEXT[] DEFAULT '{}',
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Optional: we can add a FK once we stabilize the messages table namespace
-- ALTER TABLE message_workflows
--   ADD CONSTRAINT fk_message_workflows_message
--   FOREIGN KEY (message_id) REFERENCES messages (id) ON DELETE CASCADE;

CREATE INDEX IF NOT EXISTS idx_message_workflows_message_id ON message_workflows (message_id);
CREATE INDEX IF NOT EXISTS idx_message_workflows_workflow_id ON message_workflows (workflow_id);
CREATE INDEX IF NOT EXISTS idx_message_workflows_tags ON message_workflows USING GIN (tags);
CREATE INDEX IF NOT EXISTS idx_message_workflows_metadata_gin ON message_workflows USING GIN (metadata);

-- 2) procedural_memory (MVP-compatible)
CREATE TABLE IF NOT EXISTS procedural_memory (
    workflow_id TEXT PRIMARY KEY,
    trigger_embedding VECTOR(384),
    trigger_pattern TEXT,
    successful_workflow JSONB NOT NULL,
    usage_count INT DEFAULT 1,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_procedural_memory_trigger_embedding ON procedural_memory USING diskann (trigger_embedding vector_cosine_ops);

-- 3) procedural_lessons (MVP-compatible)
CREATE TABLE IF NOT EXISTS procedural_lessons (
    lesson_id TEXT PRIMARY KEY,
    trigger_embedding VECTOR(384),
    goal_text TEXT,
    agent TEXT,
    status TEXT CHECK (status IN ('success', 'fail')),
    error TEXT,
    fix_summary TEXT,
    working_params JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_procedural_lessons_trigger_embedding ON procedural_lessons USING diskann (trigger_embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_procedural_lessons_agent ON procedural_lessons (agent);
