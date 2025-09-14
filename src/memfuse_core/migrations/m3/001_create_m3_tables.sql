-- M3 Phase A migration: Create procedural memory tables
-- This migration creates the core M3 tables for procedural memory and multi-agent orchestration

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";

-- Message workflows table: tracks which messages belong to which M3 workflows
CREATE TABLE IF NOT EXISTS message_workflows (
    id TEXT PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    message_id TEXT NOT NULL,
    workflow_id TEXT,
    step_index INT,
    tags TEXT[] DEFAULT '{}',
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Procedural memory table: stores successful workflows for reuse
CREATE TABLE IF NOT EXISTS procedural_memory (
    workflow_id TEXT PRIMARY KEY,
    trigger_embedding VECTOR(384),
    trigger_pattern TEXT,
    successful_workflow JSONB NOT NULL,
    usage_count INT DEFAULT 1,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Procedural lessons table: stores lessons learned from workflow execution
CREATE TABLE IF NOT EXISTS procedural_lessons (
    lesson_id TEXT PRIMARY KEY DEFAULT uuid_generate_v4()::text,
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

-- Create indexes for message_workflows
CREATE INDEX IF NOT EXISTS idx_message_workflows_message_id 
    ON message_workflows (message_id);

CREATE INDEX IF NOT EXISTS idx_message_workflows_workflow_id 
    ON message_workflows (workflow_id);

CREATE INDEX IF NOT EXISTS idx_message_workflows_tags 
    ON message_workflows USING GIN (tags);

CREATE INDEX IF NOT EXISTS idx_message_workflows_metadata_gin 
    ON message_workflows USING GIN (metadata);

CREATE INDEX IF NOT EXISTS idx_message_workflows_created_at 
    ON message_workflows (created_at DESC);

-- Create indexes for procedural_memory
CREATE INDEX IF NOT EXISTS idx_procedural_memory_usage_count 
    ON procedural_memory (usage_count DESC);

CREATE INDEX IF NOT EXISTS idx_procedural_memory_created_at 
    ON procedural_memory (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_procedural_memory_trigger_pattern 
    ON procedural_memory (trigger_pattern);

-- Create vector similarity index for procedural_memory (best-effort)
-- This may fail if diskann is not available, but the table will still work
DO $$
BEGIN
    CREATE INDEX IF NOT EXISTS idx_procedural_memory_trigger_embedding 
        ON procedural_memory USING diskann (trigger_embedding vector_cosine_ops);
EXCEPTION WHEN OTHERS THEN
    -- Fall back to basic vector index if diskann is not available
    BEGIN
        CREATE INDEX IF NOT EXISTS idx_procedural_memory_trigger_embedding 
            ON procedural_memory USING ivfflat (trigger_embedding vector_cosine_ops);
    EXCEPTION WHEN OTHERS THEN
        -- Log the issue but don't fail the migration
        RAISE NOTICE 'Vector indexing not available, procedural memory will use sequential scans';
    END;
END$$;

-- Create indexes for procedural_lessons
CREATE INDEX IF NOT EXISTS idx_procedural_lessons_agent 
    ON procedural_lessons (agent);

CREATE INDEX IF NOT EXISTS idx_procedural_lessons_status 
    ON procedural_lessons (status);

CREATE INDEX IF NOT EXISTS idx_procedural_lessons_created_at 
    ON procedural_lessons (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_procedural_lessons_goal_text 
    ON procedural_lessons USING gin (to_tsvector('english', goal_text));

-- Create vector similarity index for procedural_lessons (best-effort)
DO $$
BEGIN
    CREATE INDEX IF NOT EXISTS idx_procedural_lessons_trigger_embedding 
        ON procedural_lessons USING diskann (trigger_embedding vector_cosine_ops);
EXCEPTION WHEN OTHERS THEN
    -- Fall back to basic vector index if diskann is not available
    BEGIN
        CREATE INDEX IF NOT EXISTS idx_procedural_lessons_trigger_embedding 
            ON procedural_lessons USING ivfflat (trigger_embedding vector_cosine_ops);
    EXCEPTION WHEN OTHERS THEN
        -- Log the issue but don't fail the migration
        RAISE NOTICE 'Vector indexing not available for lessons, will use sequential scans';
    END;
END$$;

-- Create trigger to update updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply update triggers to all M3 tables
DROP TRIGGER IF EXISTS update_message_workflows_updated_at ON message_workflows;
CREATE TRIGGER update_message_workflows_updated_at
    BEFORE UPDATE ON message_workflows
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_procedural_memory_updated_at ON procedural_memory;
CREATE TRIGGER update_procedural_memory_updated_at
    BEFORE UPDATE ON procedural_memory
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_procedural_lessons_updated_at ON procedural_lessons;
CREATE TRIGGER update_procedural_lessons_updated_at
    BEFORE UPDATE ON procedural_lessons
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Create a view for workflow statistics
CREATE OR REPLACE VIEW workflow_statistics AS
SELECT 
    pm.workflow_id,
    pm.trigger_pattern,
    pm.usage_count,
    pm.created_at as first_used,
    pm.updated_at as last_used,
    COUNT(mw.id) as message_count,
    COALESCE(lesson_stats.success_count, 0) as success_lessons,
    COALESCE(lesson_stats.fail_count, 0) as fail_lessons
FROM procedural_memory pm
LEFT JOIN message_workflows mw ON pm.workflow_id = mw.workflow_id
LEFT JOIN (
    SELECT 
        goal_text,
        COUNT(CASE WHEN status = 'success' THEN 1 END) as success_count,
        COUNT(CASE WHEN status = 'fail' THEN 1 END) as fail_count
    FROM procedural_lessons 
    GROUP BY goal_text
) lesson_stats ON pm.trigger_pattern = lesson_stats.goal_text
GROUP BY pm.workflow_id, pm.trigger_pattern, pm.usage_count, pm.created_at, pm.updated_at, 
         lesson_stats.success_count, lesson_stats.fail_count;

-- Create a view for task performance metrics
CREATE OR REPLACE VIEW task_performance_metrics AS
SELECT 
    goal_text as task_name,
    agent,
    COUNT(*) as total_attempts,
    COUNT(CASE WHEN status = 'success' THEN 1 END) as success_count,
    COUNT(CASE WHEN status = 'fail' THEN 1 END) as fail_count,
    ROUND(
        COUNT(CASE WHEN status = 'success' THEN 1 END)::numeric / 
        COUNT(*)::numeric * 100, 2
    ) as success_rate,
    MIN(created_at) as first_attempt,
    MAX(created_at) as last_attempt
FROM procedural_lessons
GROUP BY goal_text, agent
ORDER BY success_rate DESC, total_attempts DESC;

-- Insert migration record
INSERT INTO schema_migrations (version, applied_at, description) 
VALUES ('001', CURRENT_TIMESTAMP, 'Create M3 procedural memory tables')
ON CONFLICT (version) DO NOTHING;